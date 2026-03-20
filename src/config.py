"""
모든 설정을 관리하는 파일입니다.
나중에 실험할 때는 이 파일만 수정하면 됩니다.
"""
import os
import yaml
import albumentations as A


class Config:
    # =========================================================
    # 1. Data
    # =========================================================
    # YAML 파일 경로 — 이것만 바꾸면 데이터셋 교체 완료
    DATA_YAML_PATH = os.path.join("data", "99_exp_total_dataset", "data.yaml")

    # =========================================================
    # 2. Model & Experiment
    # =========================================================
    # 사전 학습 모델 체크포인트
    # 후보: facebook/detr-resnet-50 | PekingU/rtdetr_r18vd | PekingU/rtdetr_v2_r34vd
    MODEL_CHECKPOINT = "PekingU/rtdetr_v2_r18vd"

    # 실험 결과 저장 경로: {BASE_SAVE_DIR}/{PROJECT_NAME}/{EXPERIMENT_NAME}{N}/
    BASE_SAVE_DIR = "./same_seed_runs"
    PROJECT_NAME = "rtdetr_r18"
    EXPERIMENT_NAME = "full-dataset_check"

    # 입력 이미지 크기 (증강 파이프라인의 RandomResizedCrop에서 사용)
    IMAGE_SIZE = 640

    # Backbone 가중치 동결 여부 (True면 backbone은 학습하지 않음)
    FREEZE_BACKBONE = False

    # =========================================================
    # 2-1. Loss Hyperparameters (RT-DETR v2 기본값)
    # =========================================================
    # True로 설정하면 매 에포크마다 개별 Loss 항목(VFL, L1 BBox, GIoU 등)의
    # 평균을 출력합니다. 디버깅/분석용이며, False면 기본 Trainer를 사용합니다.
    DEBUG_LOSS = False

    # 각 Loss 항목의 가중치 (클수록 해당 loss 비중 증가)
    WEIGHT_LOSS_VFL = 1.0       # Varifocal Loss 가중치 (default: 1.0)
    WEIGHT_LOSS_BBOX = 5.0      # L1 BBox Regression Loss 가중치 (default: 5.0)
    WEIGHT_LOSS_GIOU = 2.0      # GIoU Loss 가중치 (default: 2.0)

    # Focal Loss 파라미터
    FOCAL_LOSS_ALPHA = 0.75     # 양성/음성 클래스 균형 (높을수록 양성 가중) (default: 0.75)
    FOCAL_LOSS_GAMMA = 2.0      # 어려운 샘플 집중 정도 (높을수록 hard example 집중) (default: 2.0)

    # =========================================================
    # 2-2. Matcher (Hungarian) Cost Coefficients
    # =========================================================
    # bipartite matching 시 각 cost matrix의 가중치
    # 작은 객체의 spatial misalignment 패널티를 줄이기 위해 bbox/giou를 낮게 설정
    MATCHER_CLASS_COST = 2.0    # 분류 cost 가중치 (default: 2.0)
    MATCHER_BBOX_COST = 0.5    # L1 bbox cost 가중치 (default: 5.0) — 소형 객체 친화적으로 감소
    MATCHER_GIOU_COST = 0.5    # GIoU cost 가중치 (default: 2.0) — 소형 객체 친화적으로 감소

    # Matcher 디버그: 처음 N 스텝 동안 cost matrix 평균값 출력
    DEBUG_MATCHER_STEPS = 10

    # =========================================================
    # 3. Training Hyperparameters
    # =========================================================
    BATCH_SIZE = 16
    EPOCHS = 150
    # 옵티마이저: 'adamw_torch' | 'sgd' | 'adafactor'
    OPTIM = "adamw_torch"
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    # LR 스케줄러: 'linear' | 'cosine' | 'polynomial'
    LR_SCHEDULER_TYPE = "cosine"
    NUM_WORKERS = 12
    MAX_GRAD_NORM = 0.1

    # =========================================================
    # 4. Saving & Logging
    # =========================================================
    # 체크포인트를 최대 몇 개까지 유지할지 (오래된 것은 자동 삭제)
    SAVE_TOTAL_LIMIT = 1
    SEED = 42

    # =========================================================
    # 5. Early Stopping
    # =========================================================
    USE_EARLY_STOPPING = True
    # 성능 개선 없이 기다릴 최대 에포크 수
    EARLY_STOP_PATIENCE = 40
    # 개선으로 인정하는 최소 변화량 (0.0 = 어떤 개선이든 허용)
    EARLY_STOP_THRESHOLD = 0.0

    # =========================================================
    # 6. Evaluation & Metrics
    # =========================================================
    # mAP 계산 시 포함할 최소 confidence score
    # 낮게 설정할수록 더 많은 예측이 포함되어 AP 곡선이 정확해짐
    MAP_SCORE_THRESHOLD = 0.001

    # Precision / Recall / F1 계산 시 예측 박스의 최소 confidence score
    # 이보다 낮은 예측은 무시됨 (값이 높을수록 엄격한 평가)
    PR_SCORE_THRESHOLD = 0.25

    # TP/FP 판별에 사용하는 IoU 임계값 (표준: 0.5)
    # 이보다 높은 IoU를 가진 예측만 TP로 인정
    IOU_THRESHOLD = 0.5

    # Precision-Recall Curve 구성 시 포함할 최소 confidence score
    # 매우 낮게 설정하여 전체 곡선을 그림
    PR_CURVE_MIN_SCORE = 0.001

    # Confusion Matrix 생성 시 예측 박스의 최소 confidence score
    CONF_MATRIX_CONF_THRESHOLD = 0.25   # (=PR_SCORE_THRESHOLD) 권장

    # =========================================================
    # 7. Visualization
    # =========================================================
    # 학습 전 augmentation 확인 및 추론 결과 시각화에 사용할 샘플 수
    VIS_NUM_SAMPLES = 8

    # 추론 시각화에서 바운딩 박스를 표시할 최소 confidence score
    VIS_CONF_THRESHOLD = 0.4

    # =========================================================
    # Helpers
    # =========================================================
    @classmethod
    def get_output_dir(cls):
        """
        {BASE_SAVE_DIR}/{PROJECT_NAME}/{EXPERIMENT_NAME}{N} 형태로
        중복되지 않는 새 폴더 경로를 찾아 반환하고 생성합니다.
        """
        project_dir = os.path.join(cls.BASE_SAVE_DIR, cls.PROJECT_NAME)
        os.makedirs(project_dir, exist_ok=True)

        counter = 1
        while True:
            exp_dir_name = f"{cls.EXPERIMENT_NAME}{counter}"
            full_path = os.path.join(project_dir, exp_dir_name)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
                return full_path
            counter += 1

    @classmethod
    def get_train_transforms(cls):
        """
        Albumentations 증강 파이프라인.
        필요 없는 기법은 확률(p)을 0으로 하거나 주석 처리하면 됩니다.
        """
        img_size = cls.IMAGE_SIZE
        return A.Compose([
            # 1. 공간적 변환 (위치, 회전, 크기)
            A.RandomResizedCrop(
                size=(img_size, img_size),  # 잘라낸 후 다시 맞출 크기
                scale=(0.5, 1.0),     # 원본 이미지의 50% ~ 100% 면적을 랜덤 선택
                ratio=(0.75, 1.33),   # 가로세로 비율 유지 범위
                p=0.4
            ),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-10, 10),
                p=0.4
            ),

            # 2. 색상 및 조명 변환 - Color Distort
            A.ColorJitter(
                brightness=0.3, 
                contrast=0.3, 
                saturation=0.2, 
                hue=0.015, 
                p=0.5
            ),

            # 3. 화질 저하 및 가려짐
            A.MotionBlur(blur_limit=5, p=0.2),
            A.GaussNoise(std_range=(5.0 / 255, 15.0 / 255), p=0.1),
            A.CoarseDropout(
                num_holes_range=(2, 8),
                hole_height_range=(4, 16),
                hole_width_range=(4, 16),
                fill=0,
                p=0.2
            ),

        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], clip=True, min_visibility=0.2))

    @staticmethod
    def load_data_config(yaml_path=None):
        if yaml_path is None:
            yaml_path = Config.DATA_YAML_PATH

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        base_path = data.get('path', '')
        dataset_format = data.get('format', 'yolo')

        config = {
            'base_path': base_path,
            'train_path': data['train'],
            'val_path': data['val'],
            'test_path': data.get('test', None),
            'nc': data['nc'],
            'names': data['names'],
            'format': dataset_format,
            'image_dir': data.get('image_dir', 'images'),
        }

        config['train_full_path'] = os.path.join(base_path, config['train_path'])
        config['val_full_path'] = os.path.join(base_path, config['val_path'])
        if config['test_path']:
            config['test_full_path'] = os.path.join(base_path, config['test_path'])

        return config
