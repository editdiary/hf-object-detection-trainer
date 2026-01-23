"""
모델 로딩 로직을 담당합니다. 나중에 Detr를 AutoModel로 바꾸거나
커스텀 모델을 넣을 때 여기만 수정하면 됩니다.
"""

# TODO 이것도 Detr가 아니라 Auto로 변경해서 수정해야 함
#      (e.g., AutoModelForObjectDetection, AutoImageProcessor)
from transformers import DetrForObjectDetection, DetrImageProcessor

# 프로세서만 가볍게 로드하는 함수 (데이터셋 초기화용)
def load_processor(checkpoint):
    return DetrImageProcessor.from_pretrained(checkpoint)

# 모델 로드 함수 (학습용) - [수정됨] Processor 로딩 부분 제거
def load_model(checkpoint, id2label, label2id):
    model = DetrForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        #low_cpu_mem_usage=False, # 경고 방지용 옵션 유지
        #device_map=None
    )
    return model