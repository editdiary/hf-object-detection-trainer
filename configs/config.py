import os
import yaml

"""
모든 설정을 관리하는 파일입니다.
나중에 실험할 때는 이 파일만 수정하면 됩니다.
"""

# TODO 코드 구현하면서 설정으로 뺄 수 있는 건 여기로 빼서 수정해둘 것

class Config:
    # 1. Data YAML 파일 경로 (이것만 바꾸면 데이터셋 교체 끝!)
    DATA_YAML_PATH = "C:/Users/henho/OneDrive/Desktop/hf-object-detection-trainer/data/dataset_yolo_split/data.yaml"
    
    # 2. 모델 설정
    MODEL_CHECKPOINT = "facebook/detr-resnet-50" # 나중에 다른 모델로 교체 가능
    OUTPUT_DIR = "./runs/detr-chamoe-result"
    
    # 3. 학습 하이퍼파라미터 (TrainingArguments)
    BATCH_SIZE = 8
    EPOCHS = 10
    OPTIM = "adamw_torch"   # 가능한 값: 'adamw_torch', 'sgd', 'adafactor' 등
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    LR_SCHEDULER_TYPE = "cosine"  # 가능한 값: 'linear', 'cosine', 'polynomial' 등
    NUM_WORKERS = 8
    
    # 4. 로깅 및 저장 설정
    SAVE_STEPS = 50
    LOGGING_STEPS = 10
    SAVE_TOTAL_LIMIT = 2
    SEED = 42

    # --- [New] YAML 로드 헬퍼 함수 ---
    @staticmethod
    def load_data_config(yaml_path=None):
        if yaml_path is None:
            yaml_path = Config.DATA_YAML_PATH
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        # 기본 경로 (path)가 있으면 결합, 없으면 현재 경로 기준
        base_path = data.get('path', '')

        # 포맷 정보 읽기 (기본값은 yolo)
        dataset_format = data.get('format', 'yolo')
        
        config = {
            'base_path': base_path, # 편의를 위해 base_path 저장
            'train_path': data['train'], # 폴더명 or 파일명
            'val_path': data['val'],
            'test_path': data.get('test', None),    # test는 없을 수도 있음
            'nc': data['nc'],
            'names': data['names'],
            'format': dataset_format, # 포맷 정보 반환
            'image_dir': data.get('image_dir', 'images') # CVAT 등에서 쓸 공통 이미지 폴더명
        }

        # 경로 결합 (Absolute Path 만들기)
        # YOLO는 train이 폴더지만, CVAT는 xml 파일일 수 있으므로 상황에 맞게 join
        config['train_full_path'] = os.path.join(base_path, config['train_path'])
        config['val_full_path'] = os.path.join(base_path, config['val_path'])
        if config['test_path']:
            config['test_full_path'] = os.path.join(base_path, config['test_path'])
        
        return config