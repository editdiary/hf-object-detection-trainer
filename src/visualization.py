"""
Post-training visualization utilities:
  - Training curves (loss, mAP, P/R/F1)
  - Precision-Recall curve
  - Confusion matrix
  - Inference sample grid (GT vs. predictions)
  - Augmented training sample grid (sanity check)
"""
import os
import glob
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image

import torch
import torchvision.ops as ops
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.config import Config
from src.metrics import match_boxes


def plot_training_results(csv_path, output_dir):
    """Generate a 2x2 training curve plot from training_metrics.csv."""
    if not os.path.exists(csv_path):
        print(f"⚠️ {csv_path} 파일을 찾을 수 없어 시각화를 건너뜁니다.")
        return

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_kwargs = {'marker': 'o', 'markersize': 4, 'linewidth': 1.5}

    # (0, 0) Training & Evaluation Loss — Linear Scale
    if 'train_loss' in df.columns and 'eval_loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', **plot_kwargs)
        axes[0, 0].plot(df['epoch'], df['eval_loss'], label='Eval Loss', **plot_kwargs)
        axes[0, 0].set_title('Training & Evaluation Loss (Linear Scale)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    # (0, 1) Training & Evaluation Loss — Log Scale
    if 'train_loss' in df.columns and 'eval_loss' in df.columns:
        axes[0, 1].plot(df['epoch'], df['train_loss'], label='Train Loss', **plot_kwargs)
        axes[0, 1].plot(df['epoch'], df['eval_loss'], label='Eval Loss', **plot_kwargs)
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('Training & Evaluation Loss (Log Scale)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss (log)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    # (1, 0) mAP Metrics
    if 'mAP_50' in df.columns and 'mAP_50_95' in df.columns:
        axes[1, 0].plot(df['epoch'], df['mAP_50'], label='mAP@.50', **plot_kwargs)
        axes[1, 0].plot(df['epoch'], df['mAP_50_95'], label='mAP@.50-.95', **plot_kwargs)
        axes[1, 0].set_title('mAP Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    # (1, 1) Precision, Recall & F1-Score
    if 'Precision' in df.columns and 'Recall' in df.columns:
        axes[1, 1].plot(df['epoch'], df['Precision'], label='Precision', **plot_kwargs)
        axes[1, 1].plot(df['epoch'], df['Recall'], label='Recall', **plot_kwargs)
        if 'F1_score' in df.columns:
            axes[1, 1].plot(df['epoch'], df['F1_score'], label='F1-Score', **plot_kwargs)
        axes[1, 1].set_title('Precision, Recall & F1-Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "results_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 학습 추이 그래프 완료: {save_path}")


def generate_pr_curve(cached_predictions, output_dir):
    """Generate a Precision-Recall curve using torchmetrics (COCO-compliant).

    Uses the same backend, threshold, and logic as DetectionMetricsCallback so
    the mAP@0.5 label on the chart is identical to the value written to the CSV.
    """
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="faster_coco_eval")
    metric.warn_on_many_detections = False

    for item in cached_predictions:
        keep = item["obj_scores"] > Config.MAP_SCORE_THRESHOLD
        metric.update(
            [{"boxes": item["p_boxes_xyxy"][keep],
              "scores": item["obj_scores"][keep],
              "labels": item["obj_labels"][keep]}],
            [{"boxes": item["t_boxes_xyxy"],
              "labels": item["t_labels"]}],
        )

    results = metric.compute()
    metric.reset()
    map_50 = results["map_50"].item()
    map_5095 = results["map"].item()

    if "precision" not in results or results["precision"].numel() == 0:
        print("⚠️  PR curve skipped: 'precision' not available from metric backend.")
        return

    # results["precision"]: (T, R, K, A, M)
    #   T = IoU thresholds (10 values: 0.50, 0.55, …, 0.95)
    #   R = 101 recall points [0.00, 0.01, …, 1.00]
    #   K = number of classes
    #   A = area ranges  (index 0 → all areas)
    #   M = max detections (index 2 → 100)
    recall_thresholds = torch.linspace(0, 1, results["precision"].shape[1])

    # mAP@0.5 curve: IoU threshold index 0 → 0.50, average over classes
    prec_50 = results["precision"][0, :, :, 0, 2]  # (R=101, K)
    valid_50 = prec_50 >= 0
    mean_prec_50 = (prec_50.clamp(min=0) * valid_50).sum(dim=1) / valid_50.sum(dim=1).clamp(min=1)

    # mAP@0.5:0.95 curve: average over all T thresholds and K classes jointly
    prec_5095 = results["precision"][:, :, :, 0, 2]  # (T=10, R=101, K)
    valid_5095 = prec_5095 >= 0
    mean_prec_5095 = (
        (prec_5095.clamp(min=0) * valid_5095).sum(dim=(0, 2))
        / valid_5095.sum(dim=(0, 2)).clamp(min=1)
    )  # (R=101,)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(recall_thresholds.numpy(), mean_prec_50.numpy(),
             color='blue', lw=2, label=f'all classes {map_50:.3f} mAP@0.5')
    plt.plot(recall_thresholds.numpy(), mean_prec_5095.numpy(),
             color='orange', lw=2, linestyle='--', label=f'all classes {map_5095:.3f} mAP@0.5:0.95')
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "BoxPR_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📈 BoxPR_curve 완료: {save_path}")


def generate_confusion_matrix(cached_predictions, output_dir, id2label,
                               iou_threshold=Config.IOU_THRESHOLD,
                               conf_threshold=Config.CONF_MATRIX_CONF_THRESHOLD):
    """Generate an object-detection confusion matrix from pre-computed single-pass predictions."""
    num_classes = len(id2label)
    bg_class_idx = num_classes
    class_names = [id2label[i] for i in range(num_classes)] + ["background"]
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    for item in cached_predictions:
        obj_scores = item["obj_scores"]
        obj_labels = item["obj_labels"]
        p_boxes_xyxy = item["p_boxes_xyxy"]
        t_boxes_xyxy = item["t_boxes_xyxy"]
        t_labels = item["t_labels"]

        keep = obj_scores > conf_threshold
        p_scores = obj_scores[keep]
        p_labels = obj_labels[keep]
        p_boxes = p_boxes_xyxy[keep]

        sort_idx = torch.argsort(p_scores, descending=True)
        p_labels = p_labels[sort_idx]
        p_boxes = p_boxes[sort_idx]

        pred_results, matched_gt = match_boxes(
            p_boxes, p_labels, t_boxes_xyxy, t_labels, iou_threshold=iou_threshold
        )

        for (best_iou, best_gt_idx), p_label in zip(pred_results, p_labels):
            if best_iou >= iou_threshold:
                conf_matrix[t_labels[best_gt_idx].item(), p_label.item()] += 1
            else:
                conf_matrix[bg_class_idx, p_label.item()] += 1

        for gt_idx, t_label in enumerate(t_labels):
            if gt_idx not in matched_gt:
                conf_matrix[t_label.item(), bg_class_idx] += 1

    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"🧮 Confusion Matrix 완료: {save_path}")


def visualize_inference_samples(model, processor, data_cfg, device, output_dir,
                                id2label, num_samples=Config.VIS_NUM_SAMPLES,
                                conf_threshold=Config.VIS_CONF_THRESHOLD,
                                img_dir=None):
    """
    Sample random images, run inference, and plot ground-truth
    (green dashed) vs. predicted (red solid) bounding boxes.

    img_dir: override the image directory (defaults to data_cfg['val_full_path']).
    """
    model.eval()

    val_img_dir = img_dir if img_dir is not None else data_cfg['val_full_path']
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_files.extend(glob.glob(os.path.join(val_img_dir, '**', ext), recursive=True))

    if not image_files:
        print(f"⚠️ {val_img_dir} 에서 이미지를 찾을 수 없어 시각화를 건너뜁니다.")
        return

    actual_num_samples = min(num_samples, len(image_files))
    sample_paths = random.sample(image_files, actual_num_samples)

    cols = min(actual_num_samples, 2)
    rows = math.ceil(actual_num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))

    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (ax, img_path) in enumerate(zip(axes[:actual_num_samples], sample_paths)):
        raw_img = Image.open(img_path).convert("RGB")
        W, H = raw_img.size
        ax.imshow(raw_img)

        # Ground truth (green dashed)
        sep = os.sep
        if f"{sep}images{sep}" in img_path:
            label_path = img_path.replace(f"{sep}images{sep}", f"{sep}labels{sep}")
        else:
            label_path = img_path.replace("images", "labels")
        label_path = label_path.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, bw, bh = map(float, parts[1:5])
                        xmin = (cx - bw / 2) * W
                        ymin = (cy - bh / 2) * H
                        rect = patches.Rectangle(
                            (xmin, ymin), bw * W, bh * H,
                            linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
                        )
                        ax.add_patch(rect)

        # Model inference
        inputs = processor(images=raw_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([raw_img.size[::-1]]).to(device)
        img_proc = getattr(processor, "image_processor", processor)
        results = img_proc.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_threshold
        )[0]

        # Predictions (red solid)
        for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            xmin, ymin, xmax, ymax = box.tolist()
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"{id2label[label_idx.item()]}: {score:.2f}",
                    color='white', fontsize=8, weight='bold',
                    bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', pad=1))

        ax.axis('off')
        ax.set_title(f"Sample {idx + 1}")

    for ax in axes[actual_num_samples:]:
        ax.axis('off')

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    save_path = os.path.join(output_dir, "inference_samples_raw.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🖼️ 원본 이미지 추론 시각화 완료: {save_path}")


def visualize_training_samples(dataset, processor, output_dir, id2label,
                               num_samples=Config.VIS_NUM_SAMPLES):
    """
    Visualize augmented training samples before training starts (sanity check).
    Denormalizes pixel values and draws ground-truth boxes in cyan.
    """
    num_samples = min(num_samples, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)

    cols = min(num_samples, 2)
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))

    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    img_proc = getattr(processor, "image_processor", processor)
    do_normalize = getattr(img_proc, "do_normalize", False)

    if do_normalize:
        image_mean = getattr(img_proc, "image_mean", [0.485, 0.456, 0.406])
        image_std = getattr(img_proc, "image_std", [0.229, 0.224, 0.225])
    else:
        image_mean = [0.0, 0.0, 0.0]
        image_std = [1.0, 1.0, 1.0]

    mean = np.array(image_mean).reshape(3, 1, 1)
    std = np.array(image_std).reshape(3, 1, 1)

    for idx, ax in zip(sample_indices, axes[:num_samples]):
        item = dataset[idx]
        img_tensor = item["pixel_values"].cpu().numpy()

        img_np = np.clip((img_tensor * std) + mean, 0, 1)
        img_np = np.transpose(img_np, (1, 2, 0))
        H, W, _ = img_np.shape
        ax.imshow(img_np)

        target_boxes = ops.box_convert(item["labels"]["boxes"], in_fmt="cxcywh", out_fmt="xyxy")
        target_labels = item["labels"]["class_labels"]

        for box, label_idx in zip(target_boxes, target_labels):
            xmin, ymin, xmax, ymax = box.tolist()
            xmin, xmax = xmin * W, xmax * W
            ymin, ymax = ymin * H, ymax * H
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='cyan', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"{id2label[label_idx.item()]}",
                    color='black', fontsize=10, weight='bold',
                    bbox=dict(facecolor='cyan', alpha=0.7, edgecolor='none', pad=1))

        ax.axis('off')
        ax.set_title(f"Augmented Train Sample {idx}")

    for ax in axes[num_samples:]:
        ax.axis('off')

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    save_path = os.path.join(output_dir, "train_samples_augmented.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🖼️ [Sanity Check] 학습 전 데이터 증강 샘플 시각화 완료: {save_path}")
