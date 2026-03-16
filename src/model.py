"""
모델 로딩 로직을 담당합니다. 나중에 Detr를 AutoModel로 바꾸거나
커스텀 모델을 넣을 때 여기만 수정하면 됩니다.
"""

from transformers import AutoModelForObjectDetection, AutoImageProcessor

# 프로세서만 가볍게 로드하는 함수 (데이터셋 초기화용)
def load_processor(checkpoint):
    return AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)

# 모델 로드 함수 (학습용) - Loss 파라미터를 Config에서 주입
def load_model(checkpoint, id2label, label2id, config=None):
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,   # 기존 가중치와 크기가 달라도 무시하고 새로 초기화
    )

    # Config가 주어지면 Loss 관련 파라미터를 모델 config에 반영
    if config is not None:
        model.config.weight_loss_vfl = config.WEIGHT_LOSS_VFL
        model.config.weight_loss_bbox = config.WEIGHT_LOSS_BBOX
        model.config.weight_loss_giou = config.WEIGHT_LOSS_GIOU
        model.config.focal_loss_alpha = config.FOCAL_LOSS_ALPHA
        model.config.focal_loss_gamma = config.FOCAL_LOSS_GAMMA

        # Matcher (Hungarian) cost coefficients
        model.config.matcher_class_cost = config.MATCHER_CLASS_COST
        model.config.matcher_bbox_cost = config.MATCHER_BBOX_COST
        model.config.matcher_giou_cost = config.MATCHER_GIOU_COST

    return model