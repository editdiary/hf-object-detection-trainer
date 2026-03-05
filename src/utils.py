"""
collate_fn이나 시각화 함수 등 보조 도구들을 모아둡니다.
"""
import os
import csv
import glob
from PIL import Image
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score

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
        metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            backend="faster_coco_eval",
        )
        metric.warn_on_many_detections = False
        
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
            probs = logits.sigmoid()
            
            preds, targets = [], []
            
            for i in range(len(batch["labels"])):
                # =========================================================
                # [수정 전] 무조건 맨 마지막(배경)을 빼버림
                # obj_scores, obj_labels = probs[i, :, :-1].max(dim=-1)
                # =========================================================
                
                # =========================================================
                # [수정 후] 모델의 출력 개수와 우리가 가진 정답 클래스 개수를 비교
                # =========================================================
                num_pred_classes = probs.shape[-1]
                num_target_classes = len(model.config.id2label)

                if num_pred_classes > num_target_classes:
                    # YOLOS처럼 배경 클래스가 더 포함된 경우 (마지막 인덱스는 '배경'이므로 제외)
                    obj_scores, obj_labels = probs[i, :, :-1].max(dim=-1)
                else:
                    # RT-DETR처럼 배경 없이 정답 클래스만 있는 경우 (전체에서 최대값 찾기)
                    obj_scores, obj_labels = probs[i, :, :].max(dim=-1)
                
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

# TODO 시각화 함수는 대략적으로 구현이 되어 있는데, 완벽하진 않은 거 같음
# 코드 한 번씩 눈으로 보면서 이상한 건 없는지 확인 한 번 할 것
# =====================================================================
# [추가] 시각화 도구 모음 (학습 종료 후 train.py에서 한 번 호출됨)
# =====================================================================

def plot_training_results(csv_path, output_dir):
    """ training_metrics.csv를 기반으로 1x3 추이 그래프 생성 """
    if not os.path.exists(csv_path):
        print(f"⚠️ {csv_path} 파일을 찾을 수 없어 시각화를 건너뜁니다.")
        return

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_kwargs = {'marker': 'o', 'markersize': 4, 'linewidth': 1.5}

    if 'train_loss' in df.columns and 'eval_loss' in df.columns:
        axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', **plot_kwargs)
        axes[0].plot(df['epoch'], df['eval_loss'], label='Eval Loss', **plot_kwargs)
        axes[0].set_title('Training & Evaluation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)

    if 'mAP_50' in df.columns and 'mAP_50_95' in df.columns:
        axes[1].plot(df['epoch'], df['mAP_50'], label='mAP@.50', **plot_kwargs)
        axes[1].plot(df['epoch'], df['mAP_50_95'], label='mAP@.50-.95', **plot_kwargs)
        axes[1].set_title('mAP Metrics')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)

    if 'Precision' in df.columns and 'Recall' in df.columns:
        axes[2].plot(df['epoch'], df['Precision'], label='Precision', **plot_kwargs)
        axes[2].plot(df['epoch'], df['Recall'], label='Recall', **plot_kwargs)
        if 'F1_score' in df.columns:
            axes[2].plot(df['epoch'], df['F1_score'], label='F1-Score', **plot_kwargs)
        axes[2].set_title('Precision, Recall & F1-Score')
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "results_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 학습 추이 그래프 완료: {save_path}")

def generate_pr_curve(model, dataloader, device, output_dir):
    """ Best Model을 이용해 정밀한 BoxPR_curve를 생성 """
    model.eval()
    y_true = []
    y_scores = []
    
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        probs = logits.sigmoid()
        
        for i in range(len(batch["labels"])):
            num_pred_classes = probs.shape[-1]
            num_target_classes = len(model.config.id2label)
            
            if num_pred_classes > num_target_classes:
                # YOLOS처럼 배경 클래스가 더 포함된 경우 (맨 마지막 제외)
                obj_scores, obj_labels = probs[i, :, :-1].max(dim=-1)
            else:
                # RT-DETR처럼 배경 없이 정답 클래스만 있는 경우 (전체에서 최대값 찾기)
                obj_scores, obj_labels = probs[i, :, :].max(dim=-1)
            
            # PR Curve를 위해 신뢰도가 1% 이상인 박스들은 모두 끌어모음
            keep = obj_scores > 0.01 
            p_scores = obj_scores[keep]
            p_labels = obj_labels[keep]
            p_boxes = ops.box_convert(pred_boxes[i][keep], in_fmt="cxcywh", out_fmt="xyxy")
            
            # 높은 점수순으로 정렬 (PR Curve의 핵심)
            sort_idx = torch.argsort(p_scores, descending=True)
            p_scores = p_scores[sort_idx]
            p_labels = p_labels[sort_idx]
            p_boxes = p_boxes[sort_idx]
            
            t_boxes = ops.box_convert(batch["labels"][i]["boxes"].to(device), in_fmt="cxcywh", out_fmt="xyxy")
            t_labels = batch["labels"][i]["class_labels"].to(device)
            matched_gt = set()
            
            for p_box, p_label, p_score in zip(p_boxes, p_labels, p_scores):
                best_iou, best_gt_idx = 0, -1
                for gt_idx, (t_box, t_label) in enumerate(zip(t_boxes, t_labels)):
                    if gt_idx in matched_gt or p_label != t_label:
                        continue
                    iou = ops.box_iou(p_box.unsqueeze(0), t_box.unsqueeze(0)).item()
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, gt_idx
                
                if best_iou >= 0.5:  # 제대로 맞춘 경우 (True Positive)
                    y_true.append(1)
                    y_scores.append(p_score.item())
                    matched_gt.add(best_gt_idx)
                else:                # 잘못 짚은 경우 (False Positive)
                    y_true.append(0)
                    y_scores.append(p_score.item())
            
            # 아예 못 찾은 정답(False Negative)은 점수 0점으로 처리
            fn_count = len(t_boxes) - len(matched_gt)
            for _ in range(fn_count):
                y_true.append(1)
                y_scores.append(0.0)

    # Scikit-Learn으로 PR 곡선 면적 및 좌표 계산
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    # 네가 보여준 YOLO 스타일로 그리기!
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(recall, precision, color='blue', lw=3, label=f'all classes {ap:.3f} mAP@0.5')
    
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.5)
    # 범례를 그래프 밖 우측 상단으로 빼기
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "BoxPR_curve.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"📈 BoxPR_curve 완료: {save_path}")

def generate_confusion_matrix(model, dataloader, device, output_dir, id2label, iou_threshold=0.5, conf_threshold=0.1):
    """ Best Model을 이용해 객체 탐지용 Confusion Matrix를 생성 """
    model.eval()
    
    # 클래스 세팅: 실제 클래스들 + 'background'
    num_classes = len(id2label)
    bg_class_idx = num_classes
    class_names = [id2label[i] for i in range(num_classes)] + ["background"]
    
    # (클래스 수 + 1) x (클래스 수 + 1) 크기의 0 행렬 생성
    # 행(Row): 실제 정답(True), 열(Column): 모델 예측(Predicted)
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        probs = logits.sigmoid()
        
        for i in range(len(batch["labels"])):
            # 모델(YOLOS vs RT-DETR)에 따른 슬라이싱 분기 처리
            num_pred_classes = probs.shape[-1]
            if num_pred_classes > num_classes:
                obj_scores, obj_labels = probs[i, :, :-1].max(dim=-1)
            else:
                obj_scores, obj_labels = probs[i, :, :].max(dim=-1)
            
            # Confidence Threshold 이상만 남기기
            keep = obj_scores > conf_threshold
            p_scores = obj_scores[keep]
            p_labels = obj_labels[keep]
            p_boxes = ops.box_convert(pred_boxes[i][keep], in_fmt="cxcywh", out_fmt="xyxy")
            
            # 점수순으로 정렬
            sort_idx = torch.argsort(p_scores, descending=True)
            p_labels = p_labels[sort_idx]
            p_boxes = p_boxes[sort_idx]
            
            t_boxes = ops.box_convert(batch["labels"][i]["boxes"].to(device), in_fmt="cxcywh", out_fmt="xyxy")
            t_labels = batch["labels"][i]["class_labels"].to(device)
            
            matched_gt = set()
            
            # 1. 예측 박스(Pred) 기준으로 정답(Target)과 매칭
            for p_idx, (p_box, p_label) in enumerate(zip(p_boxes, p_labels)):
                best_iou, best_gt_idx = 0, -1
                for gt_idx, (t_box, t_label) in enumerate(zip(t_boxes, t_labels)):
                    if gt_idx in matched_gt or p_label != t_label:
                        continue
                    iou = ops.box_iou(p_box.unsqueeze(0), t_box.unsqueeze(0)).item()
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, gt_idx
                
                if best_iou >= iou_threshold:
                    # True Positive (제대로 찾음)
                    conf_matrix[t_labels[best_gt_idx].item(), p_label.item()] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    # False Positive (배경인데 객체라고 오해함 -> 정답이 배경)
                    conf_matrix[bg_class_idx, p_label.item()] += 1
            
            # 2. 매칭되지 않은 남은 정답(Target) 처리
            for gt_idx, t_label in enumerate(t_labels):
                if gt_idx not in matched_gt:
                    # False Negative (객체인데 못 찾고 배경 취급함 -> 예측이 배경)
                    conf_matrix[t_label.item(), bg_class_idx] += 1

    # 시각화 (Seaborn Heatmap)
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

def visualize_inference_samples(model, processor, data_cfg, device, output_dir, id2label, num_samples=4, conf_threshold=0.4):
    """
    [완성판] Validation 원본 폴더에서 이미지를 가져와 원본 화질로 다이나믹 그리드(자동 줄바꿈) 시각화합니다.
    """
    model.eval()
    
    # 1. Validation 이미지 폴더에서 이미지 파일 목록 긁어오기
    val_img_dir = data_cfg['val_full_path']
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_files.extend(glob.glob(os.path.join(val_img_dir, '**', ext), recursive=True))
        
    if not image_files:
        print(f"⚠️ {val_img_dir} 에서 이미지를 찾을 수 없어 시각화를 건너뜁니다.")
        return

    # 무작위 샘플링
    actual_num_samples = min(num_samples, len(image_files))
    sample_paths = random.sample(image_files, actual_num_samples)
    
    # =========================================================
    # [수정] 가로로만 길어지지 않게 다이나믹 그리드(2D) 계산
    # =========================================================
    cols = min(actual_num_samples, 4)  # 한 줄에 최대 4장 (원하면 3, 5로 변경 가능)
    rows = math.ceil(actual_num_samples / cols) # 필요한 줄(행) 수 계산
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    
    # axes를 무조건 1차원 리스트로 평평하게 펴주기 (반복문에서 쓰기 쉽게)
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    # =========================================================

    for idx, (ax, img_path) in enumerate(zip(axes[:actual_num_samples], sample_paths)):
        # 2. 원본 이미지 로드 (PIL)
        raw_img = Image.open(img_path).convert("RGB")
        W, H = raw_img.size
        ax.imshow(raw_img)
        
        # 3. 정답(Ground Truth) 그리기 (초록색 점선)
        label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, bw, bh = map(float, parts[1:5])
                        xmin = (cx - bw / 2) * W
                        ymin = (cy - bh / 2) * H
                        box_w, box_h = bw * W, bh * H
                        
                        rect = patches.Rectangle((xmin, ymin), box_w, box_h, 
                                                 linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
                        ax.add_patch(rect)

        # 4. 모델 추론
        inputs = processor(images=raw_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            
        # 5. 후처리
        target_sizes = torch.tensor([raw_img.size[::-1]]).to(device)
        img_proc = getattr(processor, "image_processor", processor)
        results = img_proc.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=conf_threshold)[0]

        # 6. 예측 결과 그리기 (빨간색 실선)
        for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            xmin, ymin, xmax, ymax = box.tolist()
            
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"{id2label[label_idx.item()]}: {score:.2f}", 
                    color='white', fontsize=8, weight='bold',
                    bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', pad=1))
            
        ax.axis('off')
        ax.set_title(f"Sample {idx+1}") # 0번부터 말고 1번부터 표시되게 살짝 수정!

    # =========================================================
    # [추가] 그려지지 않고 남은 하얀색 빈 프레임 숨기기
    # =========================================================
    for ax in axes[actual_num_samples:]:
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "inference_samples_raw.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️ 원본 이미지 추론 시각화 완료: {save_path}")

def visualize_training_samples(dataset, processor, output_dir, id2label, num_samples=4):
    """
    [학습 전 점검용] Data Augmentation과 Preprocessing이 모두 끝난 
    Train Dataset의 텐서를 가져와서 사람이 볼 수 있는 이미지로 복원하여 저장합니다.
    """
    # 1. 시각화할 샘플 무작위 추출
    num_samples = min(num_samples, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)

    # =========================================================
    # [핵심 수정] 가로로만 길어지지 않게 다이나믹 그리드(2D) 계산
    # =========================================================
    cols = min(num_samples, 4)  # 한 줄에 최대 4장까지만! (원하면 3, 5로 변경 가능)
    rows = math.ceil(num_samples / cols) # 필요한 줄(행) 수 자동 계산
    
    # 세로 길이도 rows에 비례해서 늘어나도록 설정
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    # axes가 1D든 2D든 무조건 평평한 1차원 리스트로 만들어서 다루기 쉽게 변환
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 2. Processor에서 정규화(Normalization) 여부와 수치 동적 추출
    img_proc = getattr(processor, "image_processor", processor)
    
    # [핵심 수정] 모델이 진짜로 정규화를 썼는지 확인 (기본값 False로 안전하게)
    do_normalize = getattr(img_proc, "do_normalize", False)
    
    if do_normalize:
        image_mean = getattr(img_proc, "image_mean", [0.485, 0.456, 0.406])
        image_std = getattr(img_proc, "image_std", [0.229, 0.224, 0.225])
    else:
        # 정규화를 안 하는 모델이면 억지로 복원하지 않도록 기본값(0, 1) 세팅!
        image_mean = [0.0, 0.0, 0.0]
        image_std = [1.0, 1.0, 1.0]
    
    mean = np.array(image_mean).reshape(3, 1, 1)
    std = np.array(image_std).reshape(3, 1, 1)

    # 기존처럼 axes와 zip으로 묶어서 그리기 (단, 넘치는 빈칸은 제외하고!)
    for idx, ax in zip(sample_indices, axes[:num_samples]):
        # 3. 데이터셋에서 증강/전처리가 완료된 텐서 꺼내기
        item = dataset[idx]
        img_tensor = item["pixel_values"].cpu().numpy()
        
        # 4. 정규화 해제 (Denormalization) 및 차원 변경 (C, H, W -> H, W, C)
        img_np = (img_tensor * std) + mean
        img_np = np.clip(img_np, 0, 1)
        img_np = np.transpose(img_np, (1, 2, 0))
        
        H, W, _ = img_np.shape
        ax.imshow(img_np)

        # 5. 증강 과정에서 살아남은(?) 정답 박스(Ground Truth) 그리기
        target_boxes = ops.box_convert(item["labels"]["boxes"], in_fmt="cxcywh", out_fmt="xyxy")
        target_labels = item["labels"]["class_labels"]
        
        for box, label_idx in zip(target_boxes, target_labels):
            xmin, ymin, xmax, ymax = box.tolist()
            # 0~1 상대 좌표를 실제 이미지 크기에 맞게 변환
            xmin, xmax, ymin, ymax = xmin * W, xmax * W, ymin * H, ymax * H
            
            # 훈련용 정답 박스는 눈에 띄게 '시안색(Cyan)'으로 표시
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)
            
            # 박스 라벨
            ax.text(xmin, ymin - 5, f"{id2label[label_idx.item()]}", 
                    color='black', fontsize=10, weight='bold',
                    bbox=dict(facecolor='cyan', alpha=0.7, edgecolor='none', pad=1))
        
        ax.axis('off')
        ax.set_title(f"Augmented Train Sample {idx}")
    
    # 만약 8장을 뽑아서 3x3 그리드(총 9칸)가 만들어졌다면, 
    # 그림이 안 그려진 마지막 1칸의 하얀색 빈 프레임(테두리와 축)을 안 보이게 숨겨줌
    for ax in axes[num_samples:]:
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "train_samples_augmented.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🖼️ [Sanity Check] 학습 전 데이터 증강 샘플 시각화 완료: {save_path}")