"""
Multi-model object detection benchmark.

Loads COCO-format prediction files from per-model subfolders,
evaluates them against YOLO-format ground truth using pycocotools COCOeval,
and produces a comparison table + CSV.

Usage:
    python benchmark.py
    python benchmark.py --results_dir val_inference_results --max_dets 300
"""

import argparse
import csv
import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.config import Config


# ──────────────────────────────────────────────────────────────
# Step 1: Discover model subfolders
# ──────────────────────────────────────────────────────────────
def discover_models(results_dir):
    """
    Scan results_dir for model subfolders containing the required JSON triplet.

    Expected structure:
        results_dir/{model_name}/{model_name}_results.json
        results_dir/{model_name}/{model_name}_results_images.json
        results_dir/{model_name}/{model_name}_results_categories.json

    Returns list of dicts: {name, results_path, images_path, categories_path}
    """
    models = []
    for entry in sorted(os.listdir(results_dir)):
        sub = os.path.join(results_dir, entry)
        if not os.path.isdir(sub):
            continue

        results_path = os.path.join(sub, f"{entry}_results.json")
        images_path = os.path.join(sub, f"{entry}_results_images.json")
        categories_path = os.path.join(sub, f"{entry}_results_categories.json")

        missing = [
            os.path.basename(p) for p in [results_path, images_path, categories_path]
            if not os.path.exists(p)
        ]
        if missing:
            print(f"Skipping {entry}: missing {', '.join(missing)}")
            continue

        models.append({
            "name": entry,
            "results_path": results_path,
            "images_path": images_path,
            "categories_path": categories_path,
        })
    return models


# ──────────────────────────────────────────────────────────────
# Step 2: Build COCO ground truth from YOLO labels
# ──────────────────────────────────────────────────────────────
def build_coco_gt(images_info, categories_info, labels_dir):
    """
    Build a COCO-format ground truth from YOLO label files.

    Args:
        images_info:     list of {id, file_name, width, height}
        categories_info: list of {id, name}
        labels_dir:      path to directory containing YOLO .txt label files

    Returns:
        COCO object loaded from the GT dict
    """
    annotations = []
    ann_id = 1

    for img in images_info:
        stem = os.path.splitext(img["file_name"])[0]
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        if not os.path.exists(label_path):
            continue

        W, H = img["width"], img["height"]
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
                w, h = bw * W, bh * H
                annotations.append({
                    "id": ann_id,
                    "image_id": img["id"],
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
# Step 3: Extract all metrics from COCOeval
# ──────────────────────────────────────────────────────────────
def extract_metrics_from_cocoeval(coco_eval):
    """
    Extract mAP50-95, mAP50, and max-F1 (with P/R) from COCOeval's
    accumulated precision array.

    We compute directly from eval["precision"] rather than using
    summarize()'s stats, because stats[0] hardcodes maxDets=100 which
    may not be in our custom maxDets list.

    The precision array has shape (T, R, K, A, M):
        T = IoU thresholds (0.50, 0.55, ..., 0.95)
        R = 101 recall thresholds (0.00, 0.01, ..., 1.00)
        K = categories, A = area ranges, M = maxDets settings

    Returns:
        (mAP50_95, mAP50, precision, recall, f1)
    """
    precision_array = coco_eval.eval["precision"]   # (T, R, K, A, M)
    recall_thresholds = coco_eval.params.recThrs     # [0.0, 0.01, ..., 1.0]

    # All IoU thresholds, area='all' (0), maxDets=last (-1)
    p_all = precision_array[:, :, 0, 0, -1]  # (T, 101)

    # mAP50-95: mean AP across all IoU thresholds
    aps = []
    for t in range(p_all.shape[0]):
        valid = p_all[t] > -1
        if valid.any():
            aps.append(p_all[t][valid].mean())
    map50_95 = float(np.mean(aps)) if aps else 0.0

    # mAP50: AP at IoU=0.50 (index 0)
    valid50 = p_all[0] > -1
    map50 = float(p_all[0][valid50].mean()) if valid50.any() else 0.0

    # Max F1 at IoU=0.50, area='all', maxDets=last
    p_at_r = p_all[0]  # (101,)
    valid = p_at_r > -1
    if valid.any():
        r_vals = recall_thresholds[valid]
        p_vals = p_at_r[valid]
        f1_vals = np.where(
            (p_vals + r_vals) > 0,
            2 * p_vals * r_vals / (p_vals + r_vals),
            0.0,
        )
        best_idx = int(np.argmax(f1_vals))
        best_p, best_r, best_f1 = float(p_vals[best_idx]), float(r_vals[best_idx]), float(f1_vals[best_idx])
    else:
        best_p = best_r = best_f1 = 0.0

    return map50_95, map50, best_p, best_r, best_f1


# ──────────────────────────────────────────────────────────────
# Step 4: Evaluate a single model
# ──────────────────────────────────────────────────────────────
def evaluate_model(model_info, coco_gt, max_dets):
    """
    Evaluate one model's predictions using COCOeval.

    Returns dict with: name, mAP50_95, mAP50, precision, recall, f1, num_preds
    """
    with open(model_info["results_path"], "r") as f:
        predictions = json.load(f)

    if not predictions:
        return {
            "name": model_info["name"], "mAP50_95": 0.0, "mAP50": 0.0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "num_preds": 0,
        }

    # Strip 'id' field — loadRes expects image_id, category_id, bbox, score only
    clean = [{
        "image_id": p["image_id"], "category_id": p["category_id"],
        "bbox": p["bbox"], "score": p["score"],
    } for p in predictions]

    coco_dt = coco_gt.loadRes(clean)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.maxDets = max_dets
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map50_95, map50, precision, recall, f1 = extract_metrics_from_cocoeval(coco_eval)

    return {
        "name": model_info["name"],
        "mAP50_95": round(map50_95, 4),
        "mAP50": round(map50, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "num_preds": len(predictions),
    }


# ──────────────────────────────────────────────────────────────
# Output formatting
# ──────────────────────────────────────────────────────────────
def print_table(rows):
    """Print a formatted comparison table."""
    headers = ["Model", "mAP50-95", "mAP50", "Precision", "Recall", "F1", "Preds"]
    col_widths = [
        max(len(headers[0]), max(len(r["name"]) for r in rows)) + 2,
        10, 8, 11, 8, 8, 8,
    ]

    def sep_line(left, mid, right, fill="─"):
        return left + mid.join(fill * w for w in col_widths) + right

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
            str(r["num_preds"]),
        ]))
    print(sep_line("└", "┴", "┘"))
    print()


def write_csv(rows, output_path):
    """Write benchmark results to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mAP50_95", "mAP50", "precision", "recall", "f1", "num_preds"])
        for r in rows:
            writer.writerow([
                r["name"], r["mAP50_95"], r["mAP50"],
                r["precision"], r["recall"], r["f1"], r["num_preds"],
            ])


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark object detection models against YOLO GT using COCOeval"
    )
    parser.add_argument("--results_dir", default="val_inference_results",
                        help="Directory with model subfolders (default: val_inference_results)")
    parser.add_argument("--max_dets", type=int, default=300,
                        help="Max detections per image for COCOeval (default: 300)")
    parser.add_argument("--output_csv", default="benchmark_summary.csv",
                        help="Path for CSV output (default: benchmark_summary.csv)")
    args = parser.parse_args()

    # Derive GT labels path from data config
    data_cfg = Config.load_data_config()
    labels_dir = os.path.join(
        data_cfg["base_path"],
        data_cfg["val_path"].replace("images", "labels"),
    )

    # 1. Discover models
    models = discover_models(args.results_dir)
    if not models:
        print(f"No model subfolders found in {args.results_dir}")
        return
    print(f"Found {len(models)} model(s): {', '.join(m['name'] for m in models)}")

    # 2. Build COCO GT using first model's image/category metadata
    with open(models[0]["images_path"], "r") as f:
        images_info = json.load(f)
    with open(models[0]["categories_path"], "r") as f:
        categories_info = json.load(f)

    print(f"Building COCO ground truth from {len(images_info)} images...")
    coco_gt = build_coco_gt(images_info, categories_info, labels_dir)
    print(f"  GT annotations: {len(coco_gt.dataset['annotations'])}")

    # 3. Evaluate each model
    max_dets = [1, 10, args.max_dets]
    results = []
    for model_info in models:
        print(f"\nEvaluating: {model_info['name']}")
        row = evaluate_model(model_info, coco_gt, max_dets)
        results.append(row)

    # 4. Sort by mAP50-95 descending and output
    results.sort(key=lambda r: r["mAP50_95"], reverse=True)
    print_table(results)
    write_csv(results, args.output_csv)
    print(f"Saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
