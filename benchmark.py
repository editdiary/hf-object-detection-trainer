"""
Multi-model object detection benchmark.

Loads COCO-format prediction files (produced by inference_hf.py / inference_yolo.py),
evaluates them against YOLO-format ground truth using pycocotools COCOeval,
and produces a comparison table + CSV.

Usage:
    python benchmark.py \
        --results_dir inference_results/ \
        --gt_dir /path/to/test/images \
        --conf_threshold 0.25 \
        --output_csv benchmark_summary.csv
"""

import argparse
import csv
import glob
import json
import os

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.metrics import compute_precision_recall_f1


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark multiple object detection models against YOLO GT"
    )
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing *_results.json files")
    parser.add_argument("--gt_dir", required=True,
                        help="Path to images directory (YOLO labels derived via images/→labels/ swap)")
    parser.add_argument("--conf_threshold", type=float, default=0.25,
                        help="Confidence threshold for P/R/F1 (default: 0.25)")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for P/R/F1 matching (default: 0.5)")
    parser.add_argument("--output_csv", default="benchmark_summary.csv",
                        help="Path for CSV output (default: benchmark_summary.csv)")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# Step 1: Discover prediction files
# ──────────────────────────────────────────────────────────────
def discover_models(results_dir):
    """
    Scan results_dir for prediction JSON files.

    Returns list of dicts: {name, results_path, images_path, categories_path}
    """
    all_json = sorted(glob.glob(os.path.join(results_dir, "*.json")))

    # Filter out companion files
    result_files = [
        f for f in all_json
        if not f.endswith("_images.json") and not f.endswith("_categories.json")
    ]

    models = []
    for rpath in result_files:
        base, ext = os.path.splitext(rpath)
        images_path = f"{base}_images{ext}"
        categories_path = f"{base}_categories{ext}"

        if not os.path.exists(images_path):
            print(f"⚠ Skipping {rpath}: missing {os.path.basename(images_path)}")
            continue
        if not os.path.exists(categories_path):
            print(f"⚠ Skipping {rpath}: missing {os.path.basename(categories_path)}")
            continue

        # Derive model name from filename
        fname = os.path.basename(base)
        # Strip common suffixes like _results, _predictions
        for suffix in ("_results", "_predictions", "_preds"):
            if fname.endswith(suffix):
                fname = fname[: -len(suffix)]
                break

        models.append({
            "name": fname,
            "results_path": rpath,
            "images_path": images_path,
            "categories_path": categories_path,
        })

    return models


# ──────────────────────────────────────────────────────────────
# Step 2: Build COCO ground truth
# ──────────────────────────────────────────────────────────────
def build_coco_gt(images_info, categories_info, gt_dir):
    """
    Build a COCO-format ground truth dict from YOLO label files.

    Args:
        images_info:     list of {id, file_name, width, height}
        categories_info: list of {id, name}
        gt_dir:          path to images directory (labels derived via images/→labels/ swap)

    Returns:
        COCO object loaded from the GT dict
    """
    annotations = []
    ann_id = 1
    sep = os.sep

    for img_info in images_info:
        file_name = img_info["file_name"]
        img_id = img_info["id"]
        W = img_info["width"]
        H = img_info["height"]

        # Find the actual image path (may be in subdirectory)
        candidates = glob.glob(os.path.join(gt_dir, "**", file_name), recursive=True)
        if not candidates:
            continue
        img_path = candidates[0]

        # Derive label path: images/ → labels/ swap
        if f"{sep}images{sep}" in img_path:
            label_path = img_path.replace(f"{sep}images{sep}", f"{sep}labels{sep}")
        else:
            label_path = img_path.replace("images", "labels")
        label_path = label_path.rsplit(".", 1)[0] + ".txt"

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                # Normalized cxcywh → pixel xywh (COCO format)
                x = (cx - bw / 2) * W
                y = (cy - bh / 2) * H
                w = bw * W
                h = bh * H

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                })
                ann_id += 1

    gt_dict = {
        "images": images_info,
        "annotations": annotations,
        "categories": categories_info,
    }

    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()
    return coco_gt


# ──────────────────────────────────────────────────────────────
# Step 3: Build cached predictions for P/R/F1
# ──────────────────────────────────────────────────────────────
def build_cached_predictions(predictions, images_info, gt_dir):
    """
    Convert COCO-format predictions and YOLO GT into the tensor format
    expected by compute_precision_recall_f1.
    """
    # Index images by id
    img_by_id = {img["id"]: img for img in images_info}

    # Group predictions by image_id
    preds_by_img = {}
    for pred in predictions:
        img_id = pred["image_id"]
        preds_by_img.setdefault(img_id, []).append(pred)

    sep = os.sep
    cached = []

    for img_info in images_info:
        img_id = img_info["id"]
        W = img_info["width"]
        H = img_info["height"]
        file_name = img_info["file_name"]

        # Load GT from YOLO label
        candidates = glob.glob(os.path.join(gt_dir, "**", file_name), recursive=True)
        gt_boxes_norm = []
        gt_labels = []
        if candidates:
            img_path = candidates[0]
            if f"{sep}images{sep}" in img_path:
                label_path = img_path.replace(f"{sep}images{sep}", f"{sep}labels{sep}")
            else:
                label_path = img_path.replace("images", "labels")
            label_path = label_path.rsplit(".", 1)[0] + ".txt"

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        # Normalized xyxy
                        x1 = cx - bw / 2
                        y1 = cy - bh / 2
                        x2 = cx + bw / 2
                        y2 = cy + bh / 2
                        gt_boxes_norm.append([x1, y1, x2, y2])
                        gt_labels.append(cls_id)

        if gt_boxes_norm:
            t_boxes = torch.tensor(gt_boxes_norm, dtype=torch.float32)
            t_labels = torch.tensor(gt_labels, dtype=torch.long)
        else:
            t_boxes = torch.zeros((0, 4), dtype=torch.float32)
            t_labels = torch.zeros((0,), dtype=torch.long)

        # Predictions for this image
        img_preds = preds_by_img.get(img_id, [])
        if img_preds:
            p_scores = torch.tensor([p["score"] for p in img_preds], dtype=torch.float32)
            p_labels = torch.tensor([p["category_id"] for p in img_preds], dtype=torch.long)
            # COCO bbox [x,y,w,h] pixel → normalized xyxy
            p_boxes = []
            for p in img_preds:
                bx, by, bw, bh = p["bbox"]
                p_boxes.append([bx / W, by / H, (bx + bw) / W, (by + bh) / H])
            p_boxes = torch.tensor(p_boxes, dtype=torch.float32)
        else:
            p_scores = torch.zeros((0,), dtype=torch.float32)
            p_labels = torch.zeros((0,), dtype=torch.long)
            p_boxes = torch.zeros((0, 4), dtype=torch.float32)

        cached.append({
            "obj_scores": p_scores,
            "obj_labels": p_labels,
            "p_boxes_xyxy": p_boxes,
            "t_boxes_xyxy": t_boxes,
            "t_labels": t_labels,
        })

    return cached


# ──────────────────────────────────────────────────────────────
# Step 4: Evaluate a single model
# ──────────────────────────────────────────────────────────────
def evaluate_model(model_info, coco_gt, gt_dir, conf_threshold, iou_threshold):
    """
    Evaluate one model's predictions.

    Returns dict with: name, mAP50_95, mAP50, precision, recall, f1
    """
    with open(model_info["results_path"], "r") as f:
        predictions = json.load(f)
    with open(model_info["images_path"], "r") as f:
        images_info = json.load(f)

    # --- COCOeval for mAP ---
    if predictions:
        # Strip 'id' field if present (loadRes expects image_id, category_id, bbox, score)
        clean_preds = []
        for p in predictions:
            clean_preds.append({
                "image_id": p["image_id"],
                "category_id": p["category_id"],
                "bbox": p["bbox"],
                "score": p["score"],
            })
        coco_dt = coco_gt.loadRes(clean_preds)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map50_95 = coco_eval.stats[0]  # AP @ IoU=0.50:0.95
        map50 = coco_eval.stats[1]     # AP @ IoU=0.50
    else:
        map50_95 = 0.0
        map50 = 0.0

    # --- P/R/F1 via threshold-based matching ---
    cached = build_cached_predictions(predictions, images_info, gt_dir)
    precision, recall, f1 = compute_precision_recall_f1(
        cached, iou_threshold=iou_threshold, conf_threshold=conf_threshold
    )

    return {
        "name": model_info["name"],
        "mAP50_95": round(map50_95, 4),
        "mAP50": round(map50, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ──────────────────────────────────────────────────────────────
# Output formatting
# ──────────────────────────────────────────────────────────────
def print_table(rows):
    """Print a formatted comparison table."""
    headers = ["Model", "mAP50-95", "mAP50", "Precision", "Recall", "F1"]
    col_widths = [
        max(len(headers[0]), max(len(r["name"]) for r in rows)) + 2,
        10, 8, 11, 8, 8,
    ]

    def sep_line(left, mid, right, fill="─"):
        parts = [fill * w for w in col_widths]
        return left + mid.join(parts) + right

    def row_line(vals):
        cells = []
        for i, v in enumerate(vals):
            if i == 0:
                cells.append(f" {v:<{col_widths[i]-1}}")
            else:
                cells.append(f"{v:>{col_widths[i]-1}} ")
        return "│" + "│".join(cells) + "│"

    print()
    print(sep_line("┌", "┬", "┐"))
    print(row_line(headers))
    print(sep_line("├", "┼", "┤"))
    for r in rows:
        print(row_line([
            r["name"],
            f"{r['mAP50_95']:.4f}",
            f"{r['mAP50']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
        ]))
    print(sep_line("└", "┴", "┘"))
    print()


def write_csv(rows, output_path):
    """Write benchmark results to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mAP50_95", "mAP50", "precision", "recall", "f1"])
        for r in rows:
            writer.writerow([
                r["name"], r["mAP50_95"], r["mAP50"],
                r["precision"], r["recall"], r["f1"],
            ])


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. Discover models
    models = discover_models(args.results_dir)
    if not models:
        print(f"No valid prediction files found in {args.results_dir}")
        print("Expected: <name>.json with companion <name>_images.json and <name>_categories.json")
        return
    print(f"Found {len(models)} model(s): {', '.join(m['name'] for m in models)}")

    # 2. Build COCO GT using first model's image/category metadata
    with open(models[0]["images_path"], "r") as f:
        images_info = json.load(f)
    with open(models[0]["categories_path"], "r") as f:
        categories_info = json.load(f)

    print(f"Building COCO ground truth from {len(images_info)} images...")
    coco_gt = build_coco_gt(images_info, categories_info, args.gt_dir)
    print(f"  GT annotations: {len(coco_gt.dataset['annotations'])}")

    # 3. Evaluate each model
    results = []
    for model_info in models:
        print(f"\nEvaluating: {model_info['name']}")
        row = evaluate_model(
            model_info, coco_gt, args.gt_dir,
            args.conf_threshold, args.iou_threshold,
        )
        results.append(row)

    # 4. Sort by mAP50-95 descending
    results.sort(key=lambda r: r["mAP50_95"], reverse=True)

    # 5. Output
    print_table(results)
    write_csv(results, args.output_csv)
    print(f"Saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
