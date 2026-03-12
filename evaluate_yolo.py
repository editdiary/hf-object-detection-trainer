"""
Standalone YOLO evaluation script.
Evaluates ultralytics YOLO models using the same metric functions
as the HF-based evaluate.py, producing identical output artifacts.

Usage:
    python evaluate_yolo.py --model_dir runs/.../best.pt --split val
"""
import os
import csv
import json
import glob
import math
import random
import argparse

import torch
import torchvision.ops as ops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from ultralytics import YOLO

from src.config import Config
from src.metrics import compute_precision_recall_f1
from src.visualization import generate_pr_curve, generate_confusion_matrix


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model using project metrics.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to YOLO .pt weight file")
    parser.add_argument("--data_yaml", type=str, default=Config.DATA_YAML_PATH,
                        help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO inference resolution")
    parser.add_argument("--conf_threshold", type=float, default=Config.CONF_MATRIX_CONF_THRESHOLD,
                        help="Confidence threshold for confusion matrix / P/R/F1")
    parser.add_argument("--iou_threshold", type=float, default=Config.IOU_THRESHOLD,
                        help="IoU threshold for TP/FP matching")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save per-image visualizations and JSON prediction files")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# Ground-truth loader
# ──────────────────────────────────────────────────────────────
def load_yolo_gt(image_dir):
    """
    Load ground-truth from YOLO-format .txt label files.

    Returns:
        list of (img_path, gt_boxes_xyxy_norm, gt_labels) tuples
        where gt_boxes_xyxy_norm is (M, 4) tensor in normalized xyxy,
        and gt_labels is (M,) int tensor.
    """
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
    image_files.sort()

    samples = []
    sep = os.sep
    for img_path in image_files:
        # images → labels path swap
        if f"{sep}images{sep}" in img_path:
            label_path = img_path.replace(f"{sep}images{sep}", f"{sep}labels{sep}")
        else:
            label_path = img_path.replace("images", "labels")
        label_path = label_path.rsplit(".", 1)[0] + ".txt"

        gt_boxes = []
        gt_labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        # normalized cxcywh → normalized xyxy
                        x1 = cx - bw / 2
                        y1 = cy - bh / 2
                        x2 = cx + bw / 2
                        y2 = cy + bh / 2
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_labels.append(cls_id)

        if gt_boxes:
            boxes_t = torch.tensor(gt_boxes, dtype=torch.float32)
            labels_t = torch.tensor(gt_labels, dtype=torch.long)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)

        samples.append((img_path, boxes_t, labels_t))

    return samples


# ──────────────────────────────────────────────────────────────
# Inference visualization (YOLO-specific)
# ──────────────────────────────────────────────────────────────
def visualize_yolo_inference_samples(model, samples, output_dir, id2label, imgsz,
                                     num_samples=Config.VIS_NUM_SAMPLES,
                                     conf_threshold=Config.VIS_CONF_THRESHOLD):
    """
    Sample random images, run YOLO inference, and plot GT (green dashed)
    vs. predicted (red solid) bounding boxes.
    """
    actual_num_samples = min(num_samples, len(samples))
    chosen = random.sample(samples, actual_num_samples)

    cols = min(actual_num_samples, 2)
    rows = math.ceil(actual_num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))

    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (ax, (img_path, gt_boxes_norm, gt_labels)) in enumerate(
        zip(axes[:actual_num_samples], chosen)
    ):
        raw_img = Image.open(img_path).convert("RGB")
        W, H = raw_img.size
        ax.imshow(raw_img)

        # GT boxes (green dashed) — convert normalized xyxy to pixel
        for box, label_idx in zip(gt_boxes_norm, gt_labels):
            x1, y1, x2, y2 = box.tolist()
            px1, py1, px2, py2 = x1 * W, y1 * H, x2 * W, y2 * H
            rect = patches.Rectangle(
                (px1, py1), px2 - px1, py2 - py1,
                linewidth=2, edgecolor="lime", facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)
            ax.text(px1, py1 - 5, f"GT: {id2label[label_idx.item()]}",
                    color="black", fontsize=8, weight="bold",
                    bbox=dict(facecolor="lime", alpha=0.7, edgecolor="none", pad=1))

        # YOLO predictions (red solid)
        results = model.predict(img_path, imgsz=imgsz, verbose=False)[0]
        for box, conf, cls_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            if conf.item() < conf_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="red", facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{id2label[int(cls_id.item())]}: {conf:.2f}",
                    color="white", fontsize=8, weight="bold",
                    bbox=dict(facecolor="red", alpha=0.7, edgecolor="none", pad=1))

        ax.axis("off")
        ax.set_title(f"Sample {idx + 1}")

    for ax in axes[actual_num_samples:]:
        ax.axis("off")

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    save_path = os.path.join(output_dir, "inference_samples_raw.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🖼️ Inference sample visualization: {save_path}")


# ──────────────────────────────────────────────────────────────
# Per-image predictions (optional --save_predictions)
# ──────────────────────────────────────────────────────────────
def save_per_image_predictions(model, samples, output_dir, id2label, imgsz, conf_threshold):
    """Save per-image JSON + visualization for every image."""
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for img_path, _, _ in tqdm(samples, desc="Saving predictions"):
        results = model.predict(img_path, imgsz=imgsz, verbose=False)[0]
        raw_img = Image.open(img_path).convert("RGB")

        scores = results.boxes.conf.tolist()
        labels = results.boxes.cls.int().tolist()
        boxes = results.boxes.xyxy.tolist()

        # JSON
        predictions = [
            {"label": id2label[lbl], "score": round(sc, 4),
             "box": [round(v, 2) for v in box]}
            for sc, lbl, box in zip(scores, labels, boxes)
            if sc >= conf_threshold
        ]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(pred_dir, f"{stem}_pred.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"image": img_path, "predictions": predictions}, f,
                      indent=2, ensure_ascii=False)

        # Visualization
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(raw_img)
        for sc, lbl, (x1, y1, x2, y2) in zip(scores, labels, boxes):
            if sc < conf_threshold:
                continue
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="red", facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{id2label[lbl]}: {sc:.2f}",
                    color="white", fontsize=8, weight="bold",
                    bbox=dict(facecolor="red", alpha=0.7, edgecolor="none", pad=1))
        ax.axis("off")
        plt.tight_layout(pad=0)
        vis_path = os.path.join(pred_dir, f"{stem}_pred.jpg")
        plt.savefig(vis_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"💾 Per-image predictions saved ({len(samples)} images): {pred_dir}")


# ──────────────────────────────────────────────────────────────
# Unique output directory helper
# ──────────────────────────────────────────────────────────────
def _get_unique_output_dir(base_dir: str) -> str:
    if not os.path.exists(base_dir):
        return base_dir
    version = 1
    while True:
        candidate = f"{base_dir}_{version}"
        if not os.path.exists(candidate):
            return candidate
        version += 1


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    print(f"✅ Loading YOLO model from: {args.model_dir}")

    # 1. Load data config & model
    data_cfg = Config.load_data_config(args.data_yaml)
    model = YOLO(args.model_dir)

    id2label = {i: name for i, name in enumerate(data_cfg["names"])}

    # 2. Resolve split image directory
    split_key = f"{args.split}_full_path"
    if split_key not in data_cfg:
        print(f"❌ '{args.split}' split not found in data.yaml")
        return
    split_image_dir = data_cfg[split_key]

    # 3. Load ground truth
    samples = load_yolo_gt(split_image_dir)
    if not samples:
        print(f"❌ No images found in {split_image_dir}")
        return
    print(f"📦 Loaded {len(samples)} images from '{args.split}' split")

    # 4. Output directory (next to model weights)
    model_parent = os.path.dirname(os.path.normpath(args.model_dir))
    base_output_dir = os.path.join(model_parent, f"eval_{args.split}")
    output_dir = _get_unique_output_dir(base_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # 5. Single-pass inference
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="faster_coco_eval")
    metric.warn_on_many_detections = False
    cached_predictions = []

    print(f"🚀 Running inference on {len(samples)} images (imgsz={args.imgsz})...")
    for img_path, gt_boxes_norm, gt_labels in tqdm(samples, desc="Evaluating"):
        result = model.predict(img_path, imgsz=args.imgsz, conf=0.001, iou=0.6, verbose=False)[0]

        # Extract predictions in pixel coords
        p_boxes_pixel = result.boxes.xyxy.cpu().float()        # (N, 4)
        p_scores = result.boxes.conf.cpu().float()             # (N,)
        p_labels = result.boxes.cls.cpu().long()               # (N,)

        # Normalize pixel boxes by original image size
        orig_h, orig_w = result.orig_shape  # (H, W)
        scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=torch.float32)
        p_boxes_norm = p_boxes_pixel / scale  # (N, 4) normalized xyxy

        # Cache entry (unfiltered, for confusion matrix / PR / F1)
        cached_predictions.append({
            "obj_scores": p_scores,
            "obj_labels": p_labels,
            "p_boxes_xyxy": p_boxes_norm,
            "t_boxes_xyxy": gt_boxes_norm,
            "t_labels": gt_labels,
        })

        # mAP update (filtered by MAP_SCORE_THRESHOLD)
        map_keep = p_scores > Config.MAP_SCORE_THRESHOLD
        metric.update(
            [{"boxes": p_boxes_norm[map_keep],
              "scores": p_scores[map_keep],
              "labels": p_labels[map_keep]}],
            [{"boxes": gt_boxes_norm,
              "labels": gt_labels}],
        )

    # 6. Compute metrics
    results = metric.compute()
    precision, recall, f1 = compute_precision_recall_f1(
        cached_predictions,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
    )

    # 7. Print results
    col_w = 22
    print("\n" + "=" * 50)
    print("📊 [Evaluation Results]")
    print("=" * 50)
    print(f"  {'Metric':<{col_w}} {'Value':>8}")
    print(f"  {'-'*col_w}  {'-'*8}")
    print(f"  {'mAP (IoU 0.50:0.95)':<{col_w}} {results['map'].item():>8.4f}")
    print(f"  {'mAP @ IoU=0.50':<{col_w}} {results['map_50'].item():>8.4f}")
    print(f"  {'mAP @ IoU=0.75':<{col_w}} {results['map_75'].item():>8.4f}")
    print(f"  {'-'*col_w}  {'-'*8}")
    print(f"  {'Precision (IoU=0.5)':<{col_w}} {precision:>8.4f}")
    print(f"  {'Recall    (IoU=0.5)':<{col_w}} {recall:>8.4f}")
    print(f"  {'F1-score  (IoU=0.5)':<{col_w}} {f1:>8.4f}")
    print("=" * 50)

    # 8. Save results.csv
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["mAP_50_95", round(results["map"].item(), 4)])
        writer.writerow(["mAP_50", round(results["map_50"].item(), 4)])
        writer.writerow(["mAP_75", round(results["map_75"].item(), 4)])
        writer.writerow(["precision", round(precision, 4)])
        writer.writerow(["recall", round(recall, 4)])
        writer.writerow(["f1", round(f1, 4)])
    print(f"📄 Metrics saved to: {csv_path}")

    # 9. Generate visualizations (reuse existing functions)
    generate_pr_curve(cached_predictions, output_dir)
    generate_confusion_matrix(
        cached_predictions=cached_predictions,
        output_dir=output_dir,
        id2label=id2label,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
    )
    visualize_yolo_inference_samples(
        model=model,
        samples=samples,
        output_dir=output_dir,
        id2label=id2label,
        imgsz=args.imgsz,
    )

    # 10. Optional per-image predictions
    if args.save_predictions:
        save_per_image_predictions(
            model=model,
            samples=samples,
            output_dir=output_dir,
            id2label=id2label,
            imgsz=args.imgsz,
            conf_threshold=args.conf_threshold,
        )

    print(f"\n✅ All evaluation results saved to: {output_dir}")


if __name__ == "__main__":
    main()
