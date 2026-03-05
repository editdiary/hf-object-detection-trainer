import os
import sys

# --- [필수] 라이브러리 버그로 인한 시끄러운 경고 끄기 ---
import warnings
# timm 백본 로딩 시 발생하는 불필요한 경고를 무시합니다.
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
# -----------------------------------------------------

import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, set_seed, EarlyStoppingCallback
import albumentations as A  # [Add] Albumentations 임포트

# 우리가 만든 모듈 임포트
from configs.config import Config
from src.dataset import create_dataset
from src.model import load_model, load_processor
from src.utils import get_collate_fn, MAPLoggingCallback, plot_training_results, generate_pr_curve, generate_confusion_matrix, visualize_inference_samples, visualize_training_samples

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
        train_dataset.label2id
    )

    # 6. 학습 인자 설정
    training_args = TrainingArguments(
        # 기본 학습 설정 (Basic)
        output_dir=current_output_dir,      # [Mod] 동적으로 생성된 경로 사용
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,

        # 하드웨어 및 속도 최적화 (Performance)
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        dataloader_num_workers=Config.NUM_WORKERS,

        # 학습률 및 규제 (Optimization)
        optim=Config.OPTIM,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        lr_scheduler_type=Config.LR_SCHEDULER_TYPE,

        # 저장 및 평가 전략 (Strategy)
        eval_strategy="epoch", # 매 Epoch마다 검증
        logging_strategy="epoch",
        #logging_steps=Config.LOGGING_STEPS,
        save_strategy="epoch",
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,        # 성능이 가장 높으면 Best Model 따로 저장
        metric_for_best_model="eval_loss",  # CHECK 선택할 수 있는 옵션 또 뭐 있나 확인할 것
        greater_is_better=False,

        # 데이터 처리
        remove_unused_columns=False,
    )

    # 7. 콜백 초기화
    device = "cuda" if torch.cuda.is_available() else "cpu"
    map_callback = MAPLoggingCallback(
        eval_dataset=eval_dataset,
        collate_fn=get_collate_fn(processor),
        output_dir=current_output_dir,      # [Mod] CSV 파일도 같은 곳에 저장
        device=device
    )

    # 8. Early Stopping 콜백 조건부 생성
    callbacks = [map_callback]

    if Config.USE_EARLY_STOPPING:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=Config.EARLY_STOP_PATIENCE, 
            early_stopping_threshold=Config.EARLY_STOP_THRESHOLD
        )
        callbacks.append(early_stopping)
        print(f"🛑 Early Stopping 활성화 (Patience: {Config.EARLY_STOP_PATIENCE})")
    
    # 9. Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=get_collate_fn(processor),    # CHECK get_collate_fn도 무슨 기능 하는 건지 자세히 살펴볼 것
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
        num_samples=4
    )
    
    # 10. 학습 시작
    print("Starting Training...")
    trainer.train()

    # 11. 모델 저장 (Processor 설정 포함)
    # [Mod] current_output_dir를 사용하도록 수정
    final_save_path = os.path.join(current_output_dir, "best_model")
    trainer.save_model(final_save_path)
    processor.save_pretrained(final_save_path)
    print(f"Training Finished! Best Model saved at: {final_save_path}")

    # =========================================================
    # [추가] 12. 학습 종료 후 시각화 (Results & PR Curve)
    # =========================================================
    print("\n" + "="*50)
    print("🎨 [Visualization] 저장된 데이터를 바탕으로 시각화를 시작합니다...")
    
    # (1) 학습 추이 그래프 그리기
    csv_path = os.path.join(current_output_dir, "training_metrics.csv")
    plot_training_results(csv_path, current_output_dir)

    # (2) PR Curve 그리기 (Best Model로 최종 평가)
    print("🚀 Best Model을 로드하여 최종 PR Curve를 평가합니다...")
    best_model = load_model(final_save_path, train_dataset.id2label, train_dataset.label2id)
    best_model.to(device)
    
    # 평가용 DataLoader 세팅 (병목 없이 빠르게!)
    final_eval_loader = DataLoader(
        eval_dataset, 
        batch_size=Config.BATCH_SIZE, 
        collate_fn=get_collate_fn(processor),
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    generate_pr_curve(best_model, final_eval_loader, device, current_output_dir)

    # [추가] 3. Confusion Matrix 그리기
    print("🚀 Best Model을 로드하여 Confusion Matrix를 생성합니다...")
    generate_confusion_matrix(
        model=best_model, 
        dataloader=final_eval_loader, 
        device=device, 
        output_dir=current_output_dir,
        id2label=train_dataset.id2label
    )

    # =========================================================
    # [수정] 4. 실제 원본 사진에 추론 결과(Bounding Box) 그려보기
    # =========================================================
    print("📸 Best Model로 검증 원본 데이터를 랜덤 샘플링 후 직접 추론해 봅니다...")
    visualize_inference_samples(
        model=best_model,
        processor=processor,
        data_cfg=data_cfg,        # [변경] dataset 대신 data_cfg(경로 정보)를 넘김
        device=device,
        output_dir=current_output_dir,
        id2label=train_dataset.id2label,
        num_samples=4,
        conf_threshold=0.4
    )
    
    print("="*50)
    print(f"✅ 모든 학습 및 시각화 프로세스가 완료되었습니다! ({current_output_dir} 확인)")

if __name__ == "__main__":
    main()