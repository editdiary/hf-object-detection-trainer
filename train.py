"""
모듈들을 조립해서 학습을 실행합니다.
코드가 훨씬 깔끔해집니다.
"""

import torch
from torch.utils.data import random_split
from transformers import TrainingArguments, Trainer, set_seed

# 우리가 만든 모듈 임포트
from configs.config import Config
from src.dataset import CVATDetrDataset
from src.utils import get_collate_fn
from src.model import load_model_and_processor

def main():
    # 1. 시드 설정 (재현성)
    set_seed(Config.seed)
    
    print(f"Loading Model: {Config.MODEL_CHECKPOINT}")
    
    # 2. 데이터셋 준비를 위해 임시로 Processor 먼저 로드 (Dataset 초기화용)
    # (실제로는 model.py 함수를 조금 수정하거나, 여기서 먼저 processor만 불러와도 됨)
    _, processor = load_model_and_processor(Config.MODEL_CHECKPOINT, {}, {})
    
    # 3. 데이터셋 생성
    full_dataset = CVATDetrDataset(
        image_dir=Config.IMAGE_DIR,
        annotation_file=Config.ANNOTATION_FILE,
        processor=processor
    )
    
    # TODO 지금은 임시로 random_split 하도록 되어 있는데,
    #        애초에 train/validation/test로 구분된 데이터를 받도록 수정하자.
    # 4. 데이터셋 분할 (Train/Val/Test)
    total_size = len(full_dataset)
    train_size = int(Config.TRAIN_RATIO * total_size)
    val_size = int(Config.VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, eval_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"Data Split - Train: {len(train_dataset)}, Val: {len(eval_dataset)}, Test: {len(test_dataset)}")
    
    # 5. 모델 로드 (Label 정보 주입)
    model, _ = load_model_and_processor(
        Config.MODEL_CHECKPOINT, 
        full_dataset.id2label, 
        full_dataset.label2id
    )

    # TODO 각 인자들의 의미와 무엇을 세팅할 수 있는지 정리할 것
    # 6. 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,
        fp16=torch.cuda.is_available(),
        save_steps=Config.SAVE_STEPS,
        logging_steps=Config.LOGGING_STEPS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        evaluation_strategy="epoch", # 매 Epoch마다 검증
        save_strategy="epoch",
        load_best_model_at_end=True,        # 성능이 가장 높으면 Best Model 따로 저장
        metric_for_best_model="eval_loss"   # CHECK 선택할 수 있는 옵션 또 뭐 있나 확인할 것
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