"""
학습된 모델(final_model 또는 체크포인트)을 불러와서 Test Set에 대해 평가합니다.
"""

# TODO Train 전부 구현 잘 되면 살펴보고 수정할 것

import torch
import os
from torch.utils.data import random_split
from transformers import DetrForObjectDetection, DetrImageProcessor, Trainer

from configs.config import Config
from src.dataset import CVATDetrDataset
from src.utils import get_collate_fn

def main():
    # 1. 학습된 모델 경로 확인
    model_path = f"{Config.OUTPUT_DIR}/final_model"
    if not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
        return

    print(f"Loading trained model from: {model_path}")
    
    # 2. 모델 & 프로세서 로드 (학습된 가중치)
    model = DetrForObjectDetection.from_pretrained(model_path)
    processor = DetrImageProcessor.from_pretrained(model_path)
    
    # 3. 데이터셋 로드 및 분할 (train.py와 동일한 시드로 쪼개야 Test Set이 보존됨!)
    # *중요: 실제 연구에선 Test Set 파일 목록을 따로 저장해두는 것이 안전합니다.
    full_dataset = CVATDetrDataset(
        image_dir=Config.IMAGE_DIR,
        annotation_file=Config.ANNOTATION_FILE,
        processor=processor
    )
    
    total_size = len(full_dataset)
    train_size = int(Config.TRAIN_RATIO * total_size)
    val_size = int(Config.VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size
    
    # 시드 고정 필수!
    torch.manual_seed(Config.seed)
    _, _, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"Starting Evaluation on Test Set ({len(test_dataset)} images)...")
    
    # 4. Trainer를 이용한 평가 (Trainer는 평가 기능도 강력합니다)
    trainer = Trainer(
        model=model,
        data_collator=get_collate_fn(processor),
        eval_dataset=test_dataset
    )
    
    # 5. 평가 실행
    metrics = trainer.evaluate()
    print("Evaluation Results:")
    print(metrics)

if __name__ == "__main__":
    main()