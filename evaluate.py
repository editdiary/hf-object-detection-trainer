import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.ops as ops
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

# --- 우리가 만든 모듈 임포트 ---
from configs.config import Config
from src.dataset import create_dataset
from src.utils import get_collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Hugging Face Object Detection model.")
    # 평가할 모델이 저장된 폴더 경로를 인자로 받습니다.
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the saved best_model directory")
    # 평가할 데이터 분할 (기본값: val) - yaml 파일에 test가 있다면 'test'로 지정 가능
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate on")
    # Confusion Matrix를 그릴 때 0.4(40%) 이상 확신하는 박스만 예측으로 인정합니다.
    parser.add_argument("--conf_threshold", type=float, default=0.4, help="Confidence threshold for Confusion Matrix")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for Confusion Matrix matching")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    print(f"✅ Loading model from: {args.model_dir}")

    # 1. 설정 및 데이터셋 로드
    data_cfg = Config.load_data_config()
    
    # [수정] 사용자의 dataset.py에 맞춰 create_dataset 사용
    # AutoImageProcessor를 이용해 저장된 모델의 프로세서 정보 로드
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    
    eval_dataset = create_dataset(data_cfg, processor=processor, split=args.split)
    if eval_dataset is None:
        print(f"❌ '{args.split}' 데이터셋을 찾을 수 없습니다. data.yaml 구성을 확인하세요.")
        return
        
    id2label = eval_dataset.id2label
    label2id = eval_dataset.label2id
    
    # 2. 모델 로드
    model = AutoModelForObjectDetection.from_pretrained(
        args.model_dir, 
        id2label=id2label, 
        label2id=label2id
    )
    model.to(device)
    model.eval()

    # 3. 데이터로더 설정 (Config.NUM_WORKERS 활용)
    collate_fn = get_collate_fn(processor)
    dataloader = DataLoader(
        eval_dataset, 
        batch_size=Config.BATCH_SIZE, 
        collate_fn=collate_fn, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    # 4. 평가 지표 (mAP) 및 Confusion Matrix 초기화
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="faster_coco_eval")
    metric.warn_on_many_detections = False # 귀찮은 100개 초과 탐지 경고 끄기

    num_classes = len(id2label)
    bg_class_idx = num_classes
    class_names = [id2label[i] for i in range(num_classes)] + ["background"]
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    # 5. 평가 루프 시작
    print(f"🚀 Starting evaluation on '{args.split}' split ({len(eval_dataset)} images)...")
    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)
            
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        
        # 🔥 [핵심 수학적 버그 수정] softmax 대신 독립 확률 sigmoid 사용!
        probs = logits.sigmoid() 
        
        preds_list = []
        targets_list = []
        
        for i in range(len(batch["labels"])):
            # --- 타겟(정답) 정리 ---
            t_boxes = ops.box_convert(batch["labels"][i]["boxes"].to(device), in_fmt="cxcywh", out_fmt="xyxy")
            t_labels = batch["labels"][i]["class_labels"].to(device)
            targets_list.append({"boxes": t_boxes, "labels": t_labels})
            
            # --- 예측 결과 정리 ---
            num_pred_classes = probs.shape[-1]
            if num_pred_classes > num_classes:
                obj_scores, obj_labels = probs[i, :, :-1].max(dim=-1)
            else:
                obj_scores, obj_labels = probs[i, :, :].max(dim=-1)
                
            p_boxes = ops.box_convert(pred_boxes[i], in_fmt="cxcywh", out_fmt="xyxy")
            
            # mAP 계산용 (너무 낮은 점수는 버려 속도/메모리 최적화)
            map_keep = obj_scores > 0.01
            preds_list.append({
                "boxes": p_boxes[map_keep],
                "scores": obj_scores[map_keep],
                "labels": obj_labels[map_keep]
            })
            
            # --- Confusion Matrix 계산 로직 ---
            cm_keep = obj_scores > args.conf_threshold
            cm_scores = obj_scores[cm_keep]
            cm_labels = obj_labels[cm_keep]
            cm_boxes = p_boxes[cm_keep]
            
            # 점수순 정렬
            sort_idx = torch.argsort(cm_scores, descending=True)
            cm_labels = cm_labels[sort_idx]
            cm_boxes = cm_boxes[sort_idx]
            
            matched_gt = set()
            
            # 예측 박스 기준 정답 매칭
            for p_idx, (p_box, p_label) in enumerate(zip(cm_boxes, cm_labels)):
                best_iou, best_gt_idx = 0, -1
                for gt_idx, (t_box, t_label) in enumerate(zip(t_boxes, t_labels)):
                    if gt_idx in matched_gt or p_label != t_label:
                        continue
                    iou = ops.box_iou(p_box.unsqueeze(0), t_box.unsqueeze(0)).item()
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, gt_idx
                
                if best_iou >= args.iou_threshold:
                    conf_matrix[t_labels[best_gt_idx].item(), p_label.item()] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    conf_matrix[bg_class_idx, p_label.item()] += 1 # False Positive
            
            # 매칭 안 된 남은 정답 처리
            for gt_idx, t_label in enumerate(t_labels):
                if gt_idx not in matched_gt:
                    conf_matrix[t_label.item(), bg_class_idx] += 1 # False Negative
                    
        # mAP 업데이트
        metric.update(preds_list, targets_list)

    # 6. 결과 출력 및 저장
    print("\n" + "="*50)
    print("📊 [Evaluation Results]")
    results = metric.compute()
    print(f" - mAP (50-95) : {results['map'].item():.4f}")
    print(f" - mAP @ 0.50  : {results['map_50'].item():.4f}")
    print(f" - mAP @ 0.75  : {results['map_75'].item():.4f}")
    
    # 7. Confusion Matrix 이미지 저장
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})
    
    plt.title(f'Confusion Matrix (Conf > {args.conf_threshold})', fontsize=16)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    
    # 모델 폴더 안에 바로 저장
    cm_save_path = os.path.join(args.model_dir, f"eval_{args.split}_confusion_matrix.png")
    plt.savefig(cm_save_path)
    plt.close()
    
    print(f"🧮 Confusion Matrix 저장 완료: {cm_save_path}")
    print("="*50)

if __name__ == "__main__":
    main()