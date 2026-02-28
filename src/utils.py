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
        
        # 1. 현재 배치 내에서 가장 큰 이미지의 높이(H)와 너비(W) 찾기
        max_h = max(img.shape[1] for img in pixel_values)
        max_w = max(img.shape[2] for img in pixel_values)

        padded_pixel_values = []
        pixel_masks = []
        
        for img in pixel_values:
            c, h, w = img.shape
            
            # 2. 이미지 패딩 (가장 큰 크기에 맞춰 빈 공간을 0으로 채움)
            padded_img = torch.zeros((c, max_h, max_w), dtype=img.dtype)
            padded_img[:, :h, :w] = img
            padded_pixel_values.append(padded_img)
            
            # 3. 마스크 생성 (실제 이미지가 있는 곳은 1, 패딩된 빈 공간은 0)
            mask = torch.zeros((max_h, max_w), dtype=torch.long)
            mask[:h, :w] = 1
            pixel_masks.append(mask)
            
        return {
            "pixel_values": torch.stack(padded_pixel_values),
            "pixel_mask": torch.stack(pixel_masks),
            "labels": labels
        }
    
        # batch_encoding = processor.pad(pixel_values, return_tensors="pt")
        
        # return {
        #     "pixel_values": batch_encoding["pixel_values"],
        #     "pixel_mask": batch_encoding["pixel_mask"], 
        #     "labels": labels
        # }
    return collate_fn

# TODO 여기 시각화 함수도 추가되어야 할 것 (e.g., PR_curve)