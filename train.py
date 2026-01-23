import os
import sys

# --- [필수] 라이브러리 버그로 인한 시끄러운 경고 끄기 ---
import warnings
# timm 백본 로딩 시 발생하는 불필요한 경고를 무시합니다.
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
# -----------------------------------------------------

import torch
from transformers import TrainingArguments, Trainer, set_seed

# 우리가 만든 모듈 임포트
from configs.config import Config
from src.dataset import create_dataset
from src.utils import get_collate_fn
from src.model import load_model, load_processor

def main():
    # 1. 시드 설정 (재현성)
    set_seed(Config.SEED)

    # 2. YAML 파일에서 데이터셋 정보 로드
    data_cfg = Config.load_data_config()
    print(f"Dataset Format: {data_cfg['format']}")
    
    print(f"Loading Processor: {Config.MODEL_CHECKPOINT}")

    # 3. 데이터셋 준비를 위해 Processor 먼저 로드 (Dataset 초기화용)
    processor = load_processor(Config.MODEL_CHECKPOINT)

    # 4. 데이터셋 생성 (Factory 함수 사용!)
    train_dataset = create_dataset(
        data_cfg=data_cfg, 
        split='train', 
        processor=processor
    )
    
    eval_dataset = create_dataset(
        data_cfg=data_cfg, 
        split='val', 
        processor=processor
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
        output_dir=Config.OUTPUT_DIR,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,

        # 하드웨어 및 속도 최적화 (Performance)
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=True,

        # 학습률 및 규제 (Optimization)
        optim=Config.OPTIM,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        lr_scheduler_type=Config.LR_SCHEDULER_TYPE,

        # 저장 및 평가 전략 (Strategy)
        #save_steps=Config.SAVE_STEPS,
        #logging_steps=Config.LOGGING_STEPS,
        eval_strategy="epoch", # 매 Epoch마다 검증
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,        # 성능이 가장 높으면 Best Model 따로 저장
        metric_for_best_model="eval_loss",  # CHECK 선택할 수 있는 옵션 또 뭐 있나 확인할 것

        # 데이터 처리
        remove_unused_columns=False,
    )
    
    # 7. Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=get_collate_fn(processor),    # CHECK get_collate_fn도 무슨 기능 하는 건지 자세히 살펴볼 것
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # 8. 학습 시작
    print("Starting Training...")
    trainer.train()
    
    # 9. 모델 저장 (Processor 설정 포함)
    trainer.save_model(f"{Config.OUTPUT_DIR}/final_model")
    processor.save_pretrained(f"{Config.OUTPUT_DIR}/final_model")
    print("Training Finished and Model Saved!")

if __name__ == "__main__":
    main()