"""
collate_fn이나 시각화 함수 등 보조 도구들을 모아둡니다.
"""

import torch

def get_collate_fn(processor):
    """
    Processor에 따라 collate_fn을 생성하여 반환하는 함수
    """
    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        batch_encoding = processor.pad(pixel_values, return_tensors="pt")
        
        return {
            "pixel_values": batch_encoding["pixel_values"],
            "pixel_mask": batch_encoding["pixel_mask"], 
            "labels": labels
        }
    return collate_fn

# TODO 여기 시각화 함수도 추가되어야 할 것 (e.g., PR_curve)