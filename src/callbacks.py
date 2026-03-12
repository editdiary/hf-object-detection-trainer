"""
Custom Trainer callbacks for the object detection training pipeline.
"""
import os
import csv

import torch
import torchvision.ops as ops
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import TrainerCallback
from torch.utils.data import DataLoader

from src.config import Config
from src.metrics import get_model_probs, extract_scores_and_labels, match_boxes, accepts_pixel_mask


class DetectionMetricsCallback(TrainerCallback):
    """
    Called after each evaluation epoch to compute mAP, Precision, Recall,
    F1-Score, and Jaccard Index, then append the results to a CSV file.
    """

    def __init__(self, eval_dataset, collate_fn, output_dir, device="cuda", model=None):
        self.output_csv = os.path.join(output_dir, "training_metrics.csv")
        self.device = device
        self.best_metrics = {}
        self.accepts_pixel_mask = accepts_pixel_mask(model) if model is not None else True

        # Build the DataLoader once so worker processes aren't respawned every epoch.
        # persistent_workers=True keeps the worker pool alive between evaluation calls,
        # matching the same setting used by the Trainer's own DataLoader.
        self.dataloader = DataLoader(
            eval_dataset,
            batch_size=Config.BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=Config.NUM_WORKERS > 0,
        )

        os.makedirs(output_dir, exist_ok=True)
        with open(self.output_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "eval_loss",
                "mAP_50", "mAP_50_95",
                "Jaccard_Index", "Precision", "Recall", "F1_score",
            ])

    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        model.eval()

        map_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            backend="faster_coco_eval",
        )
        map_metric.warn_on_many_detections = False

        num_target_classes = len(model.config.id2label)
        total_tp, total_fp, total_fn = 0, 0, 0

        # --- Diagnostic: track raw score statistics ---
        all_max_scores = []

        try:
            for batch in self.dataloader:
                pixel_values = batch["pixel_values"].to(self.device)
                pixel_mask = batch.get("pixel_mask")
                if pixel_mask is not None:
                    pixel_mask = pixel_mask.to(self.device)

                with torch.no_grad():
                    if self.accepts_pixel_mask:
                        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
                    else:
                        outputs = model(pixel_values=pixel_values)

                logits = outputs.logits
                pred_boxes = outputs.pred_boxes
                probs = get_model_probs(logits, num_target_classes)

                preds, targets = [], []

                for i in range(len(batch["labels"])):
                    obj_scores, obj_labels = extract_scores_and_labels(probs[i], num_target_classes)

                    pred_box_xyxy = ops.box_convert(pred_boxes[i], in_fmt="cxcywh", out_fmt="xyxy")
                    target_box_xyxy = ops.box_convert(
                        batch["labels"][i]["boxes"].to(self.device), in_fmt="cxcywh", out_fmt="xyxy"
                    )
                    target_labels = batch["labels"][i]["class_labels"].to(self.device)

                    # --- Diagnostic: collect per-image top score ---
                    all_max_scores.append(obj_scores.max().item() if len(obj_scores) > 0 else 0.0)

                    # mAP: discard very low-confidence predictions for speed
                    keep_map = obj_scores > Config.MAP_SCORE_THRESHOLD
                    preds.append({
                        "boxes": pred_box_xyxy[keep_map],
                        "scores": obj_scores[keep_map],
                        "labels": obj_labels[keep_map],
                    })
                    targets.append({"boxes": target_box_xyxy, "labels": target_labels})

                    # Precision / Recall / F1
                    # Sort by confidence descending so higher-confidence predictions
                    # get priority when claiming GT boxes in greedy matching.
                    keep_pr = obj_scores > Config.PR_SCORE_THRESHOLD
                    p_scores_pr = obj_scores[keep_pr]
                    p_boxes = pred_box_xyxy[keep_pr]
                    p_labels = obj_labels[keep_pr]

                    sort_idx = torch.argsort(p_scores_pr, descending=True)
                    p_boxes = p_boxes[sort_idx]
                    p_labels = p_labels[sort_idx]

                    pred_results, matched_gt = match_boxes(
                        p_boxes, p_labels, target_box_xyxy, target_labels,
                        iou_threshold=Config.IOU_THRESHOLD
                    )

                    for best_iou, _ in pred_results:
                        if best_iou >= Config.IOU_THRESHOLD:
                            total_tp += 1
                        else:
                            total_fp += 1

                    total_fn += len(target_box_xyxy) - len(matched_gt)

                map_metric.update(preds, targets)

            eval_result = map_metric.compute()
            mAP_50_95 = eval_result["map"].item()
            mAP_50 = eval_result["map_50"].item()
        finally:
            map_metric.reset()

        # --- Diagnostic: print raw score statistics ---
        if all_max_scores:
            import statistics
            score_mean = statistics.mean(all_max_scores)
            score_max = max(all_max_scores)
            score_median = statistics.median(all_max_scores)
            above_pr = sum(1 for s in all_max_scores if s >= Config.PR_SCORE_THRESHOLD)
            print(
                f"\n  📊 [Score Diagnostic] Epoch {state.epoch:.0f}: "
                f"per-image max score — mean={score_mean:.4f}, median={score_median:.4f}, "
                f"global_max={score_max:.4f} | "
                f"images with max_score >= PR_THRESHOLD({Config.PR_SCORE_THRESHOLD}): "
                f"{above_pr}/{len(all_max_scores)}"
            )

        no_preds = (total_tp + total_fp) == 0
        if no_preds:
            print(
                f"\n  ⚠️  [Epoch {state.epoch:.0f}] PR_SCORE_THRESHOLD({Config.PR_SCORE_THRESHOLD}) 이상의 예측이 없습니다. "
                "PR_SCORE_THRESHOLD를 낮추거나 더 많은 에포크를 학습하세요."
            )
        precision = total_tp / (total_tp + total_fp) if not no_preds else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        jaccard_index = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

        epoch = state.epoch

        train_loss = 0.0
        for log in reversed(state.log_history):
            if "loss" in log:
                train_loss = log["loss"]
                break

        eval_loss = metrics.get("eval_loss", 0.0)

        with open(self.output_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                f"{epoch:.2f}", f"{train_loss:.4f}", f"{eval_loss:.4f}",
                f"{mAP_50:.4f}", f"{mAP_50_95:.4f}",
                f"{jaccard_index:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1_score:.4f}",
            ])

        is_best = (
            not self.best_metrics
            or eval_loss < self.best_metrics.get("eval_loss", float('inf'))
        )
        if is_best:
            self.best_metrics = {
                "epoch": epoch,
                "mAP_50_95": mAP_50_95,
                "mAP_50": mAP_50,
                "eval_loss": eval_loss,
                "jaccard_index": jaccard_index,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }
            print(f"\n🌟 [Best Model 갱신!] Epoch {epoch:.2f}")
            print(f"   -> Eval Loss: {eval_loss:.4f} (최저!) | mAP(50-95): {mAP_50_95:.4f} | mAP@50: {mAP_50:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        print("\n🎉 [Training Complete] 최종 Best Model 성적표:")
        print(f"  - Epoch:       {self.best_metrics.get('epoch', 0):.2f}")
        print(f"  - Eval Loss:   {self.best_metrics.get('eval_loss', 0):.4f}")
        print(f"  - mAP (50-95): {self.best_metrics.get('mAP_50_95', 0):.4f}")
        print(f"  - mAP@.50:     {self.best_metrics.get('mAP_50', 0):.4f}")
        print(f"  - Jaccard Idx: {self.best_metrics.get('jaccard_index', 0):.4f}")
        print(f"  - Precision:   {self.best_metrics.get('precision', 0):.4f}")
        print(f"  - Recall:      {self.best_metrics.get('recall', 0):.4f}")
        print(f"  - F1-Score:    {self.best_metrics.get('f1_score', 0):.4f}")
        print(f"💾 상세 에포크별 기록은 '{self.output_csv}' 파일에 저장되어 있습니다.")
