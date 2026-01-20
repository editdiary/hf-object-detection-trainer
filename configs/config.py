import os

"""
모든 설정을 관리하는 파일입니다.
나중에 실험할 때는 이 파일만 수정하면 됩니다.
"""

# TODO 코드 구현하면서 설정으로 뺄 수 있는 건 여기로 배서 수정해둘 것

class Config:
    # 1. 경로 설정
    BASE_DIR = "/home/leedh/바탕화면/EDL_exp/dataset_cvat" # 사용자 경로
    IMAGE_DIR = os.path.join(BASE_DIR, "images")
    ANNOTATION_FILE = os.path.join(BASE_DIR, "annotations.xml")
    OUTPUT_DIR = "./runs/detr-chamoe-result"
    
    # 2. 모델 설정
    MODEL_CHECKPOINT = "facebook/detr-resnet-50" # 나중에 다른 모델로 교체 가능
    
    # 3. 데이터셋 설정
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    NUM_WORKERS = 2 # 데이터 로더 워커 수
    
    # 4. 학습 하이퍼파라미터 (TrainingArguments)
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    
    # 5. 로깅 및 저장 설정
    SAVE_STEPS = 50
    LOGGING_STEPS = 10
    SAVE_TOTAL_LIMIT = 2
    
    # 6. 디바이스 설정
    seed = 42