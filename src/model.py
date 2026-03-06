"""
모델 로딩 로직을 담당합니다. 나중에 Detr를 AutoModel로 바꾸거나
커스텀 모델을 넣을 때 여기만 수정하면 됩니다.
"""

from transformers import AutoModelForObjectDetection, AutoImageProcessor

# 프로세서만 가볍게 로드하는 함수 (데이터셋 초기화용)
def load_processor(checkpoint):
    return AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)

# 모델 로드 함수 (학습용) - [수정됨] Processor 로딩 부분 제거
def load_model(checkpoint, id2label, label2id):
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,   # 기존 가중치와 크기가 달라도 무시하고 새로 초기화
    )
    return model