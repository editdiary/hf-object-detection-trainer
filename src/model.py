"""
모델 로딩 로직을 담당합니다. 나중에 Detr를 AutoModel로 바꾸거나
커스텀 모델을 넣을 때 여기만 수정하면 됩니다.
"""

# TODO 이것도 Detr가 아니라 Auto로 변경해서 수정해야 함
#      (e.g., AutoModelForObjectDetection, AutoImageProcessor)
from transformers import DetrForObjectDetection, DetrImageProcessor

def load_model_and_processor(checkpoint, id2label, label2id):
    # 1. Processor 로드
    processor = DetrImageProcessor.from_pretrained(checkpoint)
    
    # 2. Model 로드
    model = DetrForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    return model, processor