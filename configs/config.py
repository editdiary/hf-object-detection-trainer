import os
import yaml
import albumentations as A

"""
모든 설정을 관리하는 파일입니다.
나중에 실험할 때는 이 파일만 수정하면 됩니다.
"""

# TODO 코드 구현하면서 설정으로 뺄 수 있는 건 여기로 빼서 수정해둘 것

class Config:
    # 1. Data YAML 파일 경로 (이것만 바꾸면 데이터셋 교체 끝!)
    DATA_YAML_PATH = os.path.join("data", "99_exp_dataset", "data.yaml")
    
    # 2. 모델 및 실험 설정
    #MODEL_CHECKPOINT = "facebook/detr-resnet-50" # 다른 모델로 교체 가능
    MODEL_CHECKPOINT = "PekingU/rtdetr_v2_r18vd"
    #MODEL_CHECKPOINT = "PekingU/rtdetr_v2_r50vd"

    # [Mod] 고정된 OUTPUT_DIR을 지우고 프로젝트와 실험 이름으로 분리
    PROJECT_NAME = "rtdetr_v2_r18-chamoe"
    EXPERIMENT_NAME = "real-train"
    BASE_SAVE_DIR = "./runs" # 최상위 저장 폴더

    # 3. 학습 하이퍼파라미터 (TrainingArguments)
    BATCH_SIZE = 16
    EPOCHS = 200
    OPTIM = "adamw_torch"   # 가능한 값: 'adamw_torch', 'sgd', 'adafactor' 등
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    LR_SCHEDULER_TYPE = "cosine"  # 가능한 값: 'linear', 'cosine', 'polynomial' 등
    NUM_WORKERS = 12
    
    # 4. 로깅 및 저장 설정
    #SAVE_STEPS = 50
    LOGGING_STEPS = 50
    SAVE_TOTAL_LIMIT = 1
    SEED = 42

    # [추가] 조기 종료(Early Stopping) 설정
    USE_EARLY_STOPPING = True       # 조기 종료 사용 여부
    EARLY_STOP_PATIENCE = 40        # 성능 개선이 없을 때 기다릴 epoch 수
    EARLY_STOP_THRESHOLD = 0.0      # 최소 개선 수치

    # =========================================================
    # [추가] 실험 폴더 자동 넘버링 생성기
    # =========================================================
    @classmethod
    def get_output_dir(cls):
        """
        runs/PROJECT_NAME/EXPERIMENT_NAME1, 2, 3... 형태로 
        중복되지 않는 새 폴더 경로를 찾아 반환하고 생성합니다.
        """
        # 예: ./runs/chamoe-detection
        project_dir = os.path.join(cls.BASE_SAVE_DIR, cls.PROJECT_NAME)
        os.makedirs(project_dir, exist_ok=True)
        
        counter = 1
        while True:
            # 예: yolos-training1
            exp_dir_name = f"{cls.EXPERIMENT_NAME}{counter}"
            full_path = os.path.join(project_dir, exp_dir_name)
            
            # 해당 폴더가 없으면 거기로 확정!
            if not os.path.exists(full_path):
                os.makedirs(full_path) # 폴더를 미리 생성해 둠
                return full_path
            
            # 이미 있으면 번호를 1 올려서 다시 확인
            counter += 1

    # =========================================================
    # [추가] 데이터 증강(Augmentation) 파이프라인 통합 관리
    # =========================================================
    @staticmethod
    def get_train_transforms():
        """
        다양한 Albumentations 증강 기법을 여기서 한 번에 관리합니다.
        필요 없는 기법은 확률(p)을 0으로 하거나 주석 처리하면 됩니다.
        """
        return A.Compose([
            # 1. 공간적 변환 (위치, 회전, 크기)
            A.RandomResizedCrop(
                size=(640, 640),      # 잘라낸 후 다시 맞출 크기 (모델 입력 크기에 맞춰 조절)
                scale=(0.7, 1.0),     # 원본 이미지의 60% ~ 100% 면적을 랜덤하게 선택해서 자름
                ratio=(0.75, 1.33),   # 가로세로 비율 유지 범위
                p=0.3                 # 너무 자주 하면 배경 학습이 부족할 수 있으니 30% 정도 추천
            ),
            A.HorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
            A.Affine(       # 이동, 크기 조절, 회전을 동시에 적용
                scale=(0.9, 1.1),             # 크기 조절
                translate_percent=(-0.1, 0.1), # 상하좌우 이동
                rotate=(-10, 10),             # 회전 정도
                p=0.4                         # 적용 확률
            ),

            # 2. 색상 및 조명 변환 (야외 환경 모사)
            A.RandomBrightnessContrast(
                brightness_limit=0.3,   # 밝기 조절
                contrast_limit=0.2,     # 대비 조절
                p=0.4
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2), # 그림자/강한 빛 디테일 살리기
            A.RandomGamma(gamma_limit=(90, 110), p=0.2),           # 전체적인 빛 노출 조절
            A.HueSaturationValue(
                hue_shift_limit=5,      # 색조 조절 (색상 보존을 위해 조금만 수정)
                sat_shift_limit=20,     # 채도(진하기) 조절 범위
                val_shift_limit=15,     # 명도 조절 범위
                p=0.5
            ),

            # 3. 화질 저하 및 가려짐 (노이즈, 흔들림, 장애물)
            A.MotionBlur(blur_limit=5, p=0.2),                     # 움직임 흔들림
            A.GaussNoise(std_range=(5.0 / 255, 15.0 / 255), p=0.1), # 약간의 노이즈 추가 (노이즈 범위 명시)
            A.CoarseDropout(
                num_holes_range=(2, 6), 
                hole_height_range=(8, 24), 
                hole_width_range=(8, 24), 
                fill=0, 
                p=0.2
            ),
            
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], clip=True, min_visibility=0.2))

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