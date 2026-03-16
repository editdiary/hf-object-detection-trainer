import os

# --- [필수] 라이브러리 버그로 인한 시끄러운 경고 끄기 ---
import warnings
# timm 백본 로딩 시 발생하는 불필요한 경고를 무시합니다.
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
# -----------------------------------------------------

import torch
import torchvision.ops as ops
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, set_seed, EarlyStoppingCallback

# 우리가 만든 모듈 임포트
from src.config import Config
from src.dataset import create_dataset
from src.model import load_model, load_processor
from src.collate import get_collate_fn
from src.callbacks import DetectionMetricsCallback, LossComponentCallback
from src.metrics import get_model_probs, extract_scores_and_labels, accepts_pixel_mask
from src.visualization import (
    plot_training_results,
    generate_pr_curve,
    generate_confusion_matrix,
    visualize_inference_samples,
    visualize_training_samples,
)

def _patch_matcher_debug(model, max_steps):
    """Monkey-patch the RTDetrHungarianMatcher to log cost matrix statistics
    for the first `max_steps` training steps."""
    import logging
    logger = logging.getLogger("matcher_debug")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[MatcherDebug] %(message)s"))
        logger.addHandler(handler)

    # Find the matcher inside the model's loss function
    criterion = getattr(model, "criterion", None)
    if criterion is None:
        logger.warning("No criterion found on model — matcher debug skipped.")
        return
    matcher = getattr(criterion, "matcher", None)
    if matcher is None:
        logger.warning("No matcher found on criterion — matcher debug skipped.")
        return

    logger.info(
        f"Matcher costs — class: {matcher.class_cost}, "
        f"bbox: {matcher.bbox_cost}, giou: {matcher.giou_cost}"
    )

    original_forward = matcher.forward.__wrapped__ if hasattr(matcher.forward, '__wrapped__') else matcher.forward
    step_counter = {"n": 0}

    @torch.no_grad()
    def _debug_forward(outputs, targets):
        import torch.nn.functional as F
        from transformers.loss.loss_for_object_detection import generalized_box_iou
        from transformers.image_transforms import center_to_corners_format

        step_counter["n"] += 1
        step = step_counter["n"]

        if step <= max_steps:
            batch_size, num_queries = outputs["logits"].shape[:2]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)
            target_ids = torch.cat([v["class_labels"] for v in targets])
            target_bbox = torch.cat([v["boxes"] for v in targets])

            # Classification cost
            if matcher.use_focal_loss:
                out_prob = F.sigmoid(outputs["logits"].flatten(0, 1))
                out_prob = out_prob[:, target_ids]
                neg_cost_class = (1 - matcher.alpha) * (out_prob**matcher.gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = matcher.alpha * ((1 - out_prob) ** matcher.gamma) * (-(out_prob + 1e-8).log())
                class_cost = pos_cost_class - neg_cost_class
            else:
                out_prob = outputs["logits"].flatten(0, 1).softmax(-1)
                class_cost = -out_prob[:, target_ids]

            bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)
            giou_cost = -generalized_box_iou(
                center_to_corners_format(out_bbox),
                center_to_corners_format(target_bbox),
            )

            logger.info(
                f"Step {step}/{max_steps} | "
                f"class_cost: {class_cost.mean().item():.4f} (weighted: {matcher.class_cost * class_cost.mean().item():.4f}) | "
                f"bbox_cost: {bbox_cost.mean().item():.4f} (weighted: {matcher.bbox_cost * bbox_cost.mean().item():.4f}) | "
                f"giou_cost: {giou_cost.mean().item():.4f} (weighted: {matcher.giou_cost * giou_cost.mean().item():.4f})"
            )

        return original_forward(outputs, targets)

    matcher.forward = _debug_forward


class LossLoggingTrainer(Trainer):
    """Trainer that captures per-step loss component breakdown from model outputs.
    Only used when Config.DEBUG_LOSS is enabled."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        # Store individual loss components on the model for the callback to read
        if hasattr(outputs, "loss_dict") and outputs.loss_dict is not None:
            model._last_loss_dict = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in outputs.loss_dict.items()
            }
        else:
            model._last_loss_dict = {}

        return (loss, outputs) if return_outputs else loss


def main():
    # 1. 시드 설정 (재현성)
    set_seed(Config.SEED)

    # 2. YAML 파일에서 데이터셋 정보 로드
    data_cfg = Config.load_data_config()
    print(f"Dataset Format: {data_cfg['format']}")
    
    print(f"Loading Processor: {Config.MODEL_CHECKPOINT}")

    # 3. 데이터셋 준비를 위해 Processor 먼저 로드 (Dataset 초기화용)
    processor = load_processor(Config.MODEL_CHECKPOINT)

    # =========================================================
    # [추가] 이번 실험을 위한 고유한 폴더 경로 생성 (예: runs/chamoe/yolos1)
    # =========================================================
    current_output_dir = Config.get_output_dir()
    print(f"📁 이번 실험 결과는 다음 경로에 안전하게 저장됩니다: {current_output_dir}")

    # [Mod] config.py에서 정의한 증강 파이프라인
    train_transform = Config.get_train_transforms()

    # 4. 데이터셋 생성 (Factory 함수 사용)
    train_dataset = create_dataset(
        data_cfg=data_cfg, 
        split='train', 
        processor=processor,
        transform=train_transform # Train에만 증강 적용
    )
    
    eval_dataset = create_dataset(
        data_cfg=data_cfg, 
        split='val', 
        processor=processor,
        transform=None # 검증/평가할 때는 원본 그대로
    )

    print(f"Data Loaded - Train: {len(train_dataset)}, Val: {len(eval_dataset)}")
    
    # 5. 모델 로드 (Label 정보 주입)
    print(f"Loading Model: {Config.MODEL_CHECKPOINT}")
    model = load_model(
        Config.MODEL_CHECKPOINT,
        train_dataset.id2label,
        train_dataset.label2id,
        config=Config
    )

    # 5-1. Backbone 가중치 동결 (선택)
    if Config.FREEZE_BACKBONE:
        frozen_count = 0
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
                frozen_count += 1
        print(f"🧊 Backbone 동결 완료: {frozen_count}개 파라미터가 학습에서 제외됩니다.")

    # 5-2. Matcher 디버그 패치 (첫 N 스텝의 cost matrix 평균 출력)
    if Config.DEBUG_MATCHER_STEPS > 0:
        _patch_matcher_debug(model, Config.DEBUG_MATCHER_STEPS)
        print(f"🔍 Matcher Debug 활성화: 첫 {Config.DEBUG_MATCHER_STEPS} 스텝의 cost matrix 통계를 출력합니다.")

    # 6. 학습 인자 설정
    # bf16은 Ampere 이상 GPU에서 fp16보다 수치적으로 안정적 (overflow 없음, 동일 속도)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    training_args = TrainingArguments(
        # 기본 학습 설정 (Basic)
        output_dir=current_output_dir,      # [Mod] 동적으로 생성된 경로 사용
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,

        # 하드웨어 및 속도 최적화 (Performance)
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_pin_memory=True,
        dataloader_num_workers=Config.NUM_WORKERS,
        dataloader_persistent_workers=True,  # 에포크 간 worker 재생성 오버헤드 방지

        # 학습률 및 규제 (Optimization)
        optim=Config.OPTIM,
        learning_rate=Config.LEARNING_RATE,
        max_grad_norm=Config.MAX_GRAD_NORM,
        weight_decay=Config.WEIGHT_DECAY,
        lr_scheduler_type=Config.LR_SCHEDULER_TYPE,
        warmup_ratio=0.02,                  # 전체 스텝의 2%를 warmup으로 사용 (초기 학습 안정화)

        # 저장 및 평가 전략 (Strategy)
        eval_strategy="epoch", # 매 Epoch마다 검증
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,        # 성능이 가장 높으면 Best Model 따로 저장
        metric_for_best_model="eval_loss",  # 의도적 설계: 검증 손실 기반 일반화 성능을 기준으로 best model 선택
        greater_is_better=False,

        # 데이터 처리
        remove_unused_columns=False,
    )

    # 7. 콜백 초기화
    device = "cuda" if torch.cuda.is_available() else "cpu"
    collate_fn = get_collate_fn(processor)
    map_callback = DetectionMetricsCallback(
        eval_dataset=eval_dataset,
        collate_fn=collate_fn,
        output_dir=current_output_dir,      # [Mod] CSV 파일도 같은 곳에 저장
        device=device,
        model=model,
    )
    # 8. Early Stopping 콜백 조건부 생성
    callbacks = [map_callback]

    if Config.DEBUG_LOSS:
        callbacks.append(LossComponentCallback())
        print("🔍 DEBUG_LOSS 활성화: 매 에포크마다 개별 Loss 항목(VFL, BBox, GIoU 등)을 출력합니다.")

    if Config.USE_EARLY_STOPPING:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=Config.EARLY_STOP_PATIENCE,
            early_stopping_threshold=Config.EARLY_STOP_THRESHOLD
        )
        callbacks.append(early_stopping)
        print(f"🛑 Early Stopping 활성화 (Patience: {Config.EARLY_STOP_PATIENCE})")

    # 9. Trainer 초기화
    TrainerClass = LossLoggingTrainer if Config.DEBUG_LOSS else Trainer
    trainer = TrainerClass(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks    # [Add] 커스텀 콜백 연동
    )

    # =========================================================
    # [추가] 학습 시작 전, 증강된 데이터가 모델에게 어떻게 보이는지 확인!
    # =========================================================
    print("📸 학습 시작 전, 증강(Augmentation)이 적용된 Train 데이터 샘플을 시각화합니다...")
    visualize_training_samples(
        dataset=train_dataset,
        processor=processor,
        output_dir=current_output_dir,
        id2label=train_dataset.id2label,
        num_samples=Config.VIS_NUM_SAMPLES,
    )
    
    # 10. 학습 시작
    print("Starting Training...")
    trainer.train()
    model.eval()  # load_best_model_at_end=True가 best checkpoint를 로드하므로 즉시 eval 모드로 전환

    # 11. 모델 저장 (Processor 설정 포함)
    final_save_path = os.path.join(current_output_dir, "best_model")
    trainer.save_model(final_save_path)
    processor.save_pretrained(final_save_path)
    print(f"Training Finished! Best Model saved at: {final_save_path}")

    # =========================================================
    # 12. 학습 종료 후 시각화
    # =========================================================
    print("\n" + "="*50)
    print("🎨 [Visualization] 저장된 데이터를 바탕으로 시각화를 시작합니다...")

    # (1) 학습 추이 그래프
    csv_path = os.path.join(current_output_dir, "training_metrics.csv")
    plot_training_results(csv_path, current_output_dir)

    # (2) 평가용 DataLoader (load_best_model_at_end=True로 model이 이미 best weights 포함)
    final_eval_loader = DataLoader(
        eval_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # (3) 단일 패스 추론 — PR Curve + Confusion Matrix에서 결과 공유
    print("🚀 단일 패스 추론으로 PR Curve와 Confusion Matrix 데이터를 수집합니다...")
    num_target_classes = len(train_dataset.id2label)
    _accepts_mask = accepts_pixel_mask(model)
    cached_predictions = []
    for batch in tqdm(final_eval_loader, desc="Inference"):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)
        with torch.no_grad():
            if _accepts_mask:
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            else:
                outputs = model(pixel_values=pixel_values)
        if not hasattr(outputs, 'pred_boxes'):
            raise AttributeError("Model output missing 'pred_boxes'. Only DETR-style models with box regression heads are supported.")
        probs = get_model_probs(outputs.logits, num_target_classes)
        for i in range(len(batch["labels"])):
            obj_scores, obj_labels = extract_scores_and_labels(probs[i], num_target_classes)
            cached_predictions.append({
                "obj_scores": obj_scores.cpu(),
                "obj_labels": obj_labels.cpu(),
                "p_boxes_xyxy": ops.box_convert(
                    outputs.pred_boxes[i], in_fmt="cxcywh", out_fmt="xyxy"
                ).cpu(),
                "t_boxes_xyxy": ops.box_convert(
                    batch["labels"][i]["boxes"], in_fmt="cxcywh", out_fmt="xyxy"
                ),
                "t_labels": batch["labels"][i]["class_labels"],
            })

    generate_pr_curve(cached_predictions, current_output_dir)
    generate_confusion_matrix(
        cached_predictions=cached_predictions,
        output_dir=current_output_dir,
        id2label=train_dataset.id2label,
    )

    # (4) 추론 시각화 (원본 이미지에 박스 그리기)
    print("📸 Best Model로 검증 원본 데이터를 랜덤 샘플링 후 직접 추론해 봅니다...")
    visualize_inference_samples(
        model=model,
        processor=processor,
        data_cfg=data_cfg,
        device=device,
        output_dir=current_output_dir,
        id2label=train_dataset.id2label,
    )

    print("="*50)
    print(f"✅ 모든 학습 및 시각화 프로세스가 완료되었습니다! ({current_output_dir} 확인)")

if __name__ == "__main__":
    main()