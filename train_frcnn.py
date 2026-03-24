"""
Faster R-CNN (ResNet50-FPN v2) fine-tuning script.

Standalone PyTorch training loop that reuses the same YOLO-format dataset,
augmentation pipeline, and visualization utilities as the HF Trainer-based
train.py, but targets torchvision's Faster R-CNN instead of RT-DETR/DETR.
"""

import os
import csv
import glob
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.config import Config
from src.metrics import match_boxes
from src.visualization import plot_training_results, generate_pr_curve, generate_confusion_matrix


# =========================================================
# 1. FRCNN-specific Config
# =========================================================
class FRCNNConfig:
    """Faster R-CNN fine-tuning hyperparameters."""
    # Optimizer
    OPTIMIZER = "sgd"
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005

    # Scheduler
    WARMUP_EPOCHS = 1
    SCHEDULER = "cosine"  # cosine annealing after warmup

    # Training
    BATCH_SIZE = 4
    GRAD_ACCUMULATION_STEPS = 4  # effective batch = 16
    EPOCHS = 100
    MAX_GRAD_NORM = 5.0
    NUM_WORKERS = Config.NUM_WORKERS
    SEED = Config.SEED

    # Model
    IMAGE_SIZE = Config.IMAGE_SIZE  # 640
    FREEZE_BACKBONE = False

    # Early stopping
    EARLY_STOP_PATIENCE = 20

    # Evaluation thresholds (reuse from Config)
    MAP_SCORE_THRESHOLD = Config.MAP_SCORE_THRESHOLD
    PR_SCORE_THRESHOLD = Config.PR_SCORE_THRESHOLD
    IOU_THRESHOLD = Config.IOU_THRESHOLD
    PR_CURVE_MIN_SCORE = Config.PR_CURVE_MIN_SCORE
    CONF_MATRIX_CONF_THRESHOLD = Config.CONF_MATRIX_CONF_THRESHOLD

    # Visualization
    VIS_NUM_SAMPLES = Config.VIS_NUM_SAMPLES
    VIS_CONF_THRESHOLD = Config.VIS_CONF_THRESHOLD

    # Output directory
    BASE_SAVE_DIR = Config.BASE_SAVE_DIR
    PROJECT_NAME = "frcnn_r50fpnv2"
    EXPERIMENT_NAME = "repeat-exp_seed42_"

    @classmethod
    def get_output_dir(cls):
        project_dir = os.path.join(cls.BASE_SAVE_DIR, cls.PROJECT_NAME)
        os.makedirs(project_dir, exist_ok=True)
        counter = 1
        while True:
            exp_dir_name = f"{cls.EXPERIMENT_NAME}{counter}"
            full_path = os.path.join(project_dir, exp_dir_name)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
                return full_path
            counter += 1


# =========================================================
# 2. Dataset: YOLO → torchvision Faster R-CNN targets
# =========================================================
class YoloFRCNNDataset(Dataset):
    """Reads YOLO-format labels and returns (image_tensor, target_dict)
    compatible with torchvision Faster R-CNN."""

    def __init__(self, image_dir, class_names, transform=None):
        self.image_dir = image_dir
        self.class_names = class_names
        self.transform = transform

        self.id2label = {i: name for i, name in enumerate(class_names)}
        self.label2id = {name: i for i, name in enumerate(class_names)}

        self.image_files = sorted(
            f for f in glob.glob(os.path.join(image_dir, "*.*"))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]

        # Label file path (images → labels)
        label_path = os.path.splitext(image_path)[0] + ".txt"
        if f"{os.sep}images{os.sep}" in label_path:
            label_path = label_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
        elif "images" in label_path:
            label_path = label_path.replace("images", "labels")

        image = Image.open(image_path).convert("RGB")
        w_orig, h_orig = image.size
        image_np = np.array(image)

        # Parse YOLO normalized cx,cy,w,h → COCO absolute [x_min, y_min, w, h]
        boxes_coco = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    abs_w, abs_h = bw * w_orig, bh * h_orig
                    abs_cx, abs_cy = cx * w_orig, cy * h_orig
                    x_min = abs_cx - abs_w / 2
                    y_min = abs_cy - abs_h / 2
                    boxes_coco.append([x_min, y_min, abs_w, abs_h])
                    labels.append(class_id)

        # Apply Albumentations (format='coco')
        if self.transform:
            transformed = self.transform(
                image=image_np, bboxes=boxes_coco, category_ids=labels
            )
            image_np = transformed['image']
            boxes_coco = list(transformed['bboxes'])
            labels = list(transformed['category_ids'])

        # Convert image to tensor [0, 1]
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        # COCO [x,y,w,h] → xyxy and apply label+1 shift (0=background in torchvision)
        boxes_xyxy = []
        valid_labels = []
        for (x, y, w, h), lbl in zip(boxes_coco, labels):
            x1, y1, x2, y2 = x, y, x + w, y + h
            if x2 > x1 + 1 and y2 > y1 + 1:  # filter degenerate
                boxes_xyxy.append([x1, y1, x2, y2])
                valid_labels.append(lbl + 1)  # +1: YOLO class 0 → label 1

        if len(boxes_xyxy) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
            labels_tensor = torch.tensor(valid_labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
        }
        return image_tensor, target


# =========================================================
# 3. Collate
# =========================================================
def frcnn_collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


# =========================================================
# 4. Model loading
# =========================================================
def load_frcnn_model(num_classes, image_size, freeze_backbone=False):
    """Load pretrained Faster R-CNN and replace the classification head."""
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

    # Replace box predictor for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Match image size to existing pipeline
    model.transform.min_size = (image_size,)
    model.transform.max_size = image_size

    if freeze_backbone:
        frozen = 0
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
                frozen += 1
        print(f"Backbone frozen: {frozen} parameters excluded from training.")

    return model


# =========================================================
# 5. Training loop
# =========================================================
def train_one_epoch(model, loader, optimizer, scaler, device, grad_accum_steps, max_grad_norm,
                    amp_enabled=True, amp_dtype=torch.float16):
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Train", leave=False)
    for step, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values()) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1
        pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

    return total_loss / max(num_batches, 1)


# =========================================================
# 6. Evaluation loop
# =========================================================
@torch.no_grad()
def evaluate_one_epoch(model, loader, device, id2label, cfg):
    """Run evaluation: compute val loss, mAP, P/R/F1, and cache predictions."""
    num_classes = len(id2label)

    # --- Val loss (model.train + no_grad) ---
    model.train()
    val_loss_sum = 0.0
    val_batches = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        val_loss_sum += sum(v.item() for v in loss_dict.values())
        val_batches += 1
    val_loss = val_loss_sum / max(val_batches, 1)

    # --- Predictions (model.eval) ---
    model.eval()
    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox",
        backend="faster_coco_eval", extended_summary=True,
    )
    metric.warn_on_many_detections = False

    cached_predictions = []
    tp = fp = fn = 0

    for images, targets in tqdm(loader, desc="Eval", leave=False):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for pred, gt in zip(outputs, targets):
            scores = pred["scores"].cpu()
            pred_labels = pred["labels"].cpu()
            pred_boxes = pred["boxes"].cpu()
            gt_boxes = gt["boxes"]  # already on CPU
            gt_labels = gt["labels"]

            # Shift labels back: torchvision label 1 → YOLO class 0
            pred_labels_shifted = pred_labels - 1
            gt_labels_shifted = gt_labels - 1

            # mAP computation (low threshold)
            keep_map = scores > cfg.MAP_SCORE_THRESHOLD
            metric.update(
                [{"boxes": pred_boxes[keep_map], "scores": scores[keep_map],
                  "labels": pred_labels_shifted[keep_map]}],
                [{"boxes": gt_boxes, "labels": gt_labels_shifted}],
            )

            # Cache for PR curve / confusion matrix
            cached_predictions.append({
                "obj_scores": scores,
                "obj_labels": pred_labels_shifted,
                "p_boxes_xyxy": pred_boxes,
                "t_boxes_xyxy": gt_boxes,
                "t_labels": gt_labels_shifted,
            })

            # P/R/F1 (higher threshold)
            keep_pr = scores >= cfg.PR_SCORE_THRESHOLD
            s_filt = scores[keep_pr]
            l_filt = pred_labels_shifted[keep_pr]
            b_filt = pred_boxes[keep_pr]
            order = s_filt.argsort(descending=True)
            l_filt, b_filt = l_filt[order], b_filt[order]

            n_gt = len(gt_labels_shifted)
            if len(l_filt) == 0:
                fn += n_gt
                continue
            if n_gt == 0:
                fp += len(l_filt)
                continue

            _, matched_gt = match_boxes(
                b_filt, l_filt, gt_boxes, gt_labels_shifted, cfg.IOU_THRESHOLD
            )
            image_tp = len(matched_gt)
            tp += image_tp
            fp += len(l_filt) - image_tp
            fn += n_gt - image_tp

    results = metric.compute()
    metric.reset()
    map_50 = results["map_50"].item()
    map_5095 = results["map"].item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "val_loss": val_loss,
        "mAP_50": map_50,
        "mAP_50_95": map_5095,
        "Precision": precision,
        "Recall": recall,
        "F1_score": f1,
    }
    return metrics, cached_predictions


# =========================================================
# 7. Early Stopping
# =========================================================
class EarlyStopping:
    def __init__(self, patience=20, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value):
        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.counter = 0
            return True  # new best
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# =========================================================
# 8. Visualization helpers (FRCNN-specific)
# =========================================================
def visualize_training_samples_frcnn(dataset, output_dir, id2label, num_samples=8):
    """Show augmented training samples with GT boxes (FRCNN dataset returns [0,1] tensors)."""
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    cols = min(num_samples, 2)
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax_idx, data_idx in enumerate(indices):
        img_tensor, target = dataset[data_idx]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[ax_idx].imshow(img_np)

        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.tolist()
            # Reverse label shift for display
            class_name = id2label.get(label.item() - 1, f"cls{label.item()}")
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='cyan', facecolor='none'
            )
            axes[ax_idx].add_patch(rect)
            axes[ax_idx].text(
                x1, y1 - 5, class_name, color='black', fontsize=10, weight='bold',
                bbox=dict(facecolor='cyan', alpha=0.7, edgecolor='none', pad=1),
            )
        axes[ax_idx].axis('off')
        axes[ax_idx].set_title(f"Augmented Train Sample {data_idx}")

    for ax in axes[num_samples:]:
        ax.axis('off')

    plt.tight_layout(pad=0.5)
    save_path = os.path.join(output_dir, "train_samples_augmented.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Train augmentation samples saved: {save_path}")


def visualize_inference_samples_frcnn(model, data_cfg, device, output_dir,
                                      id2label, num_samples=8, conf_threshold=0.4):
    """Run inference on random val images and draw GT (green dashed) + pred (red solid) boxes."""
    model.eval()

    val_img_dir = data_cfg['val_full_path']
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_files.extend(glob.glob(os.path.join(val_img_dir, '**', ext), recursive=True))
    if not image_files:
        print(f"No images found in {val_img_dir}, skipping inference visualization.")
        return

    actual = min(num_samples, len(image_files))
    sample_paths = random.sample(image_files, actual)

    cols = min(actual, 2)
    rows = math.ceil(actual / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (ax, img_path) in enumerate(zip(axes[:actual], sample_paths)):
        raw_img = Image.open(img_path).convert("RGB")
        W, H = raw_img.size
        ax.imshow(raw_img)

        # Ground truth (green dashed)
        sep = os.sep
        label_path = os.path.splitext(img_path)[0] + '.txt'
        if f"{sep}images{sep}" in label_path:
            label_path = label_path.replace(f"{sep}images{sep}", f"{sep}labels{sep}")
        elif "images" in label_path:
            label_path = label_path.replace("images", "labels")
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, bw, bh = map(float, parts[1:5])
                        xmin = (cx - bw / 2) * W
                        ymin = (cy - bh / 2) * H
                        rect = patches.Rectangle(
                            (xmin, ymin), bw * W, bh * H,
                            linewidth=2, edgecolor='lime', facecolor='none', linestyle='--',
                        )
                        ax.add_patch(rect)

        # Model inference
        img_tensor = torch.from_numpy(np.array(raw_img)).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            preds = model([img_tensor.to(device)])[0]

        for score, label_idx, box in zip(preds["scores"], preds["labels"], preds["boxes"]):
            if score.item() < conf_threshold:
                continue
            x1, y1, x2, y2 = box.cpu().tolist()
            class_name = id2label.get(label_idx.item() - 1, f"cls{label_idx.item()}")
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none',
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 - 5, f"{class_name}: {score:.2f}",
                color='white', fontsize=8, weight='bold',
                bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', pad=1),
            )
        ax.axis('off')
        ax.set_title(f"Sample {idx + 1}")

    for ax in axes[actual:]:
        ax.axis('off')

    plt.tight_layout(pad=0.5)
    save_path = os.path.join(output_dir, "inference_samples_raw.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Inference visualization saved: {save_path}")


# =========================================================
# 9. CSV logging
# =========================================================
def init_csv(path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "eval_loss",
            "mAP_50", "mAP_50_95", "Jaccard_Index",
            "Precision", "Recall", "F1_score",
        ])


def append_csv(path, epoch, train_loss, metrics):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, f"{train_loss:.6f}", f"{metrics['val_loss']:.6f}",
            f"{metrics['mAP_50']:.6f}", f"{metrics['mAP_50_95']:.6f}",
            "",  # Jaccard_Index placeholder for CSV compatibility
            f"{metrics['Precision']:.6f}", f"{metrics['Recall']:.6f}",
            f"{metrics['F1_score']:.6f}",
        ])


# =========================================================
# 10. Main
# =========================================================
def main():
    cfg = FRCNNConfig

    # 1. Seed
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    # 2. Data config
    data_cfg = Config.load_data_config()
    class_names = data_cfg['names']
    num_classes = len(class_names) + 1  # +1 for background
    id2label = {i: name for i, name in enumerate(class_names)}
    print(f"Dataset: {data_cfg['format']} | Classes: {class_names} | num_classes(+bg): {num_classes}")

    # 3. Datasets & DataLoaders
    train_transform = Config.get_train_transforms()
    train_dataset = YoloFRCNNDataset(
        image_dir=data_cfg['train_full_path'],
        class_names=class_names,
        transform=train_transform,
    )
    val_dataset = YoloFRCNNDataset(
        image_dir=data_cfg['val_full_path'],
        class_names=class_names,
        transform=None,
    )
    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=frcnn_collate_fn,
        pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=frcnn_collate_fn,
        pin_memory=True, persistent_workers=True,
    )

    # 4. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_frcnn_model(num_classes, cfg.IMAGE_SIZE, cfg.FREEZE_BACKBONE)
    model.to(device)
    print(f"Model loaded on {device} | Image size: {cfg.IMAGE_SIZE}")

    # 5. Optimizer + Scheduler + Scaler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=cfg.LEARNING_RATE,
        momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY,
    )

    # Warmup (linear) for first epoch, then cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS - cfg.WARMUP_EPOCHS, eta_min=1e-6,
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.001, end_factor=1.0, total_iters=cfg.WARMUP_EPOCHS,
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[cfg.WARMUP_EPOCHS],
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_amp = torch.cuda.is_available()
    # bf16 doesn't need loss scaling; fp16 does
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not use_bf16)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"AMP: {'bf16' if use_bf16 else 'fp16' if use_amp else 'disabled'}")

    # 6. Early stopping + output dir
    early_stopper = EarlyStopping(patience=cfg.EARLY_STOP_PATIENCE, mode="max")
    output_dir = cfg.get_output_dir()
    best_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    csv_path = os.path.join(output_dir, "training_metrics.csv")
    init_csv(csv_path)

    # 7. Visualize augmented training samples
    print("Visualizing augmented training samples...")
    visualize_training_samples_frcnn(
        train_dataset, output_dir, id2label, num_samples=cfg.VIS_NUM_SAMPLES,
    )

    # 8. Training loop
    print(f"\nStarting training for {cfg.EPOCHS} epochs...")
    print(f"  Batch size: {cfg.BATCH_SIZE} x {cfg.GRAD_ACCUMULATION_STEPS} accum = {cfg.BATCH_SIZE * cfg.GRAD_ACCUMULATION_STEPS} effective")
    best_map50 = 0.0
    last_cached_predictions = None

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            cfg.GRAD_ACCUMULATION_STEPS, cfg.MAX_GRAD_NORM,
            amp_enabled=use_amp, amp_dtype=amp_dtype,
        )

        # Eval
        metrics, cached_predictions = evaluate_one_epoch(
            model, val_loader, device, id2label, cfg,
        )

        # Step LR scheduler
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{cfg.EPOCHS} ({elapsed:.0f}s) | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f} | "
            f"mAP@50: {metrics['mAP_50']:.4f} | mAP@50-95: {metrics['mAP_50_95']:.4f} | "
            f"P: {metrics['Precision']:.4f} R: {metrics['Recall']:.4f} F1: {metrics['F1_score']:.4f}"
        )

        append_csv(csv_path, epoch, train_loss, metrics)

        # Early stopping check + best model save
        is_best = early_stopper.step(metrics['mAP_50'])
        if is_best:
            best_map50 = metrics['mAP_50']
            torch.save(model.state_dict(), os.path.join(best_model_dir, "model.pth"))
            last_cached_predictions = cached_predictions
            print(f"  -> New best mAP@50: {best_map50:.4f} (model saved)")

        if early_stopper.should_stop:
            print(f"Early stopping triggered at epoch {epoch} (patience={cfg.EARLY_STOP_PATIENCE})")
            break

    # 9. Load best model
    best_ckpt = os.path.join(best_model_dir, "model.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        print(f"\nLoaded best model (mAP@50={best_map50:.4f})")
    model.eval()

    # If best model was the last epoch, we already have cached predictions;
    # otherwise re-run evaluation to get fresh predictions for viz
    if last_cached_predictions is None:
        _, last_cached_predictions = evaluate_one_epoch(
            model, val_loader, device, id2label, cfg,
        )

    # 10. Post-training visualization
    print("\n" + "=" * 50)
    print("Post-training visualization...")

    # Training curves
    plot_training_results(csv_path, output_dir)

    # PR curve
    generate_pr_curve(last_cached_predictions, output_dir)

    # Confusion matrix
    generate_confusion_matrix(
        cached_predictions=last_cached_predictions,
        output_dir=output_dir,
        id2label=id2label,
        iou_threshold=cfg.IOU_THRESHOLD,
        conf_threshold=cfg.CONF_MATRIX_CONF_THRESHOLD,
    )

    # Inference visualization
    print("Running inference on random val samples...")
    visualize_inference_samples_frcnn(
        model, data_cfg, device, output_dir,
        id2label, num_samples=cfg.VIS_NUM_SAMPLES,
        conf_threshold=cfg.VIS_CONF_THRESHOLD,
    )

    # 11. Summary
    print("=" * 50)
    print(f"Training complete!")
    print(f"  Best mAP@50: {best_map50:.4f}")
    print(f"  Output: {output_dir}")
    print(f"  Best model: {best_model_dir}")


if __name__ == "__main__":
    main()
