"""
collate_fn이나 시각화 함수 등 보조 도구들을 모아둡니다.
"""

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.ops as ops
from torchvision.ops import box_convert
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import TrainerCallback
from torch.utils.data import DataLoader

from configs.config import Config

def get_collate_fn(processor):
    """
    Processor에 따라 collate_fn을 생성하여 반환하는 함수
    """
    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # 1. 현재 배치 내에서 가장 큰 이미지의 높이(H)와 너비(W) 찾기
        max_h = max(img.shape[1] for img in pixel_values)
        max_w = max(img.shape[2] for img in pixel_values)

        padded_pixel_values = []
        pixel_masks = []
        
        for img in pixel_values:
            c, h, w = img.shape
            
            # 2. 이미지 패딩 (가장 큰 크기에 맞춰 빈 공간을 0으로 채움)
            padded_img = torch.zeros((c, max_h, max_w), dtype=img.dtype)
            padded_img[:, :h, :w] = img
            padded_pixel_values.append(padded_img)
            
            # 3. 마스크 생성 (실제 이미지가 있는 곳은 1, 패딩된 빈 공간은 0)
            mask = torch.zeros((max_h, max_w), dtype=torch.long)
            mask[:h, :w] = 1
            pixel_masks.append(mask)
            
        return {
            "pixel_values": torch.stack(padded_pixel_values),
            "pixel_mask": torch.stack(pixel_masks),
            "labels": labels
        }

    return collate_fn

class MAPLoggingCallback(TrainerCallback):
    """
    1 Epoch 검증(Evaluation)이 끝날 때마다 호출되어 
    Loss(Train/Eval), mAP(50, 75, 50-95), Acc, P, R, F1을 계산하고 CSV로 저장하는 커스텀 콜백
    """
    def __init__(self, eval_dataset, collate_fn, output_dir, device="cuda"):
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn
        self.output_csv = os.path.join(output_dir, "training_metrics.csv")
        self.device = device
        self.best_metrics = {}
        
        # CSV 파일 초기화 및 첫 줄(헤더) 확장
        os.makedirs(output_dir, exist_ok=True)
        with open(self.output_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "eval_loss", 
                             "mAP_50", "mAP_75", "mAP_50_95", 
                             "Accuracy", "Precision", "Recall", "F1_score"])

    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        model.eval()

        # mAP 계산용 객체
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="faster_coco_eval")
        # [수정 후] 일꾼 8명 고용 및 GPU 전송 가속화!
        dataloader = DataLoader(
            self.eval_dataset, 
            batch_size=args.eval_batch_size, 
            collate_fn=self.collate_fn,
            num_workers=Config.NUM_WORKERS,        # CPU 코어를 활용해 데이터 미리 준비
            pin_memory=True       # 데이터를 GPU로 넘기는 속도 부스트
        )

        # P, R, F1 계산을 위한 카운터
        total_tp, total_fp, total_fn = 0, 0, 0

        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(self.device)
            pixel_mask = batch.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(self.device)
            
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            
            logits = outputs.logits
            pred_boxes = outputs.pred_boxes
            probs = logits.softmax(-1)
            
            preds, targets = [], []
            
            for i in range(len(batch["labels"])):
                # 예측값 처리 (마지막 인덱스는 '배경'이므로 제외)
                obj_scores, obj_labels = probs[i, :, :-1].max(dim=-1)
                
                # 1. mAP용 데이터
                # 계산 속도 최적화를 위해 점수가 10% 미만인 가짜 박스는 버림
                keep_map = obj_scores > 0.10
                
                # [중심x, 중심y, 너비, 높이] -> [좌상단x, 좌상단y, 우하단x, 우하단y] 변환
                pred_box_xyxy = ops.box_convert(pred_boxes[i], in_fmt="cxcywh", out_fmt="xyxy")
                
                preds.append({
                    "boxes": pred_box_xyxy[keep_map],
                    "scores": obj_scores[keep_map],
                    "labels": obj_labels[keep_map]
                })
                
                # 정답(Target) 처리
                target_box_xyxy = ops.box_convert(batch["labels"][i]["boxes"].to(self.device), in_fmt="cxcywh", out_fmt="xyxy")
                target_labels = batch["labels"][i]["class_labels"].to(self.device)

                targets.append({
                    "boxes": target_box_xyxy,
                    "labels": target_labels
                })

                # 2. P, R, F1 계산 로직
                keep_pr = obj_scores > 0.5
                p_boxes = pred_box_xyxy[keep_pr]
                p_labels = obj_labels[keep_pr]
                
                matched_gt = set()
                # 예측 박스들을 순회하며 실제 박스(Target)와 매칭
                for p_box, p_label in zip(p_boxes, p_labels):
                    best_iou, best_gt_idx = 0, -1
                    
                    for gt_idx, (t_box, t_label) in enumerate(zip(target_box_xyxy, target_labels)):
                        if gt_idx in matched_gt or p_label != t_label:
                            continue
                        # IoU (겹치는 면적) 계산
                        iou = ops.box_iou(p_box.unsqueeze(0), t_box.unsqueeze(0)).item()
                        if iou > best_iou:
                            best_iou, best_gt_idx = iou, gt_idx
                            
                    if best_iou >= 0.5:
                        total_tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        total_fp += 1
                        
                total_fn += len(target_box_xyxy) - len(matched_gt)
            
            metric.update(preds, targets)
        
        # 전체 지표 계산
        eval_result = metric.compute()
        mAP_50_95 = eval_result["map"].item()  # 0.50부터 0.95까지의 평균
        mAP_50 = eval_result["map_50"].item()
        mAP_75 = eval_result["map_75"].item()

        # P, R, F1, Acc 계산 (0으로 나누기 방지)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
        
        epoch = state.epoch

        # Train Loss 가져오기
        train_loss = 0.0
        for log in reversed(state.log_history):
            if "loss" in log:
                train_loss = log["loss"]
                break
        
        eval_loss = metrics.get("eval_loss", 0.0)
        
        # CSV에 저장
        with open(self.output_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{epoch:.2f}", f"{train_loss:.4f}", f"{eval_loss:.4f}", 
                             f"{mAP_50:.4f}", f"{mAP_75:.4f}", f"{mAP_50_95:.4f}",
                             f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1_score:.4f}"])
        
        # =====================================================================
        # [수정] Best 모델 갱신 여부 판단 (eval_loss가 '낮아질 때' 갱신!)
        # =====================================================================
        is_best = False
        # 이전 기록이 없거나, 현재 eval_loss가 기존 베스트 eval_loss보다 작으면 갱신
        if not self.best_metrics or eval_loss < self.best_metrics.get("eval_loss", float('inf')):
            is_best = True
            self.best_metrics = {
                "epoch": epoch, "mAP_50_95": mAP_50_95, "mAP_50": mAP_50, "eval_loss": eval_loss,
                "precision": precision, "recall": recall, "f1_score": f1_score
            }
        
        # 기록을 경신했을 때만 심플하게 화면에 띄워줍니다!
        if is_best:
            print(f"\n🌟 [Best Model 갱신!] Epoch {epoch:.2f}")
            print(f"   -> Eval Loss: {eval_loss:.4f} (최저!) | mAP(50-95): {mAP_50_95:.4f} | mAP@50: {mAP_50:.4f}")
        
        metric.reset()  # 다음 계산을 위해 메모리 초기화
    
    def on_train_end(self, args, state, control, **kwargs):
        # 학습이 끝나면 종합 성적표를 화려하게 출력합니다.
        print("\n🎉 [Training Complete] 최종 Best Model 성적표 (mAP_50_95 기준):")
        print(f"  - Epoch:       {self.best_metrics.get('epoch', 0):.2f}")
        print(f"  - Eval Loss:   {self.best_metrics.get('eval_loss', 0):.4f}")
        print(f"  - mAP (50-95): {self.best_metrics.get('mAP_50_95', 0):.4f}")
        print(f"  - mAP@.50:     {self.best_metrics.get('mAP_50', 0):.4f}")
        print(f"  - Accuracy:    {self.best_metrics.get('accuracy', 0):.4f}")
        print(f"  - Precision:   {self.best_metrics.get('precision', 0):.4f}")
        print(f"  - Recall:      {self.best_metrics.get('recall', 0):.4f}")
        print(f"  - F1-Score:    {self.best_metrics.get('f1_score', 0):.4f}")
        print(f"💾 상세 에포크별 기록은 '{self.output_csv}' 파일에 저장되어 있습니다.")

# TODO 여기 시각화 함수도 추가되어야 할 것 (e.g., PR_curve)