"""
데이터셋 클래스만 깔끔하게 분리
"""

import os
import glob
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset


class CVATObjectDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, processor, transform=None):
        self.image_dir = image_dir
        self.processor = processor
        self.transform = transform
        
        # 클래스 정의 (필요시 config에서 가져오도록 수정 가능)
        self.label2id = {"ripe_chamoe": 0}
        self.id2label = {0: "ripe_chamoe"}
        
        self.samples = self._parse_xml(annotation_file)

    def _parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        samples = []
        
        for image_tag in root.findall('image'):
            file_name = image_tag.get('name')
            width = int(image_tag.get('width'))
            height = int(image_tag.get('height'))
            
            boxes = []
            class_labels = []
            
            for box in image_tag.findall('box'):
                label_name = box.get('label')
                if label_name not in self.label2id:
                    continue
                
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                
                if xbr > xtl and ybr > ytl:
                    boxes.append([xtl, ytl, xbr, ybr])
                    class_labels.append(self.label2id[label_name])
            
            samples.append({
                "file_name": file_name,
                "boxes": boxes,
                "class_labels": class_labels,
                "orig_size": (height, width)
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = os.path.join(self.image_dir, item['file_name'])
        image = Image.open(image_path).convert("RGB")

        image_np = np.array(image)  # (1) 이미지를 Numpy 배열로 변환

        # (2) CVAT 박스(pascal_voc)를 미리 COCO 포맷([x_min, y_min, w, h])으로 통일
        boxes = []
        labels = item['class_labels']
        for box in item['boxes']:
            x_min, y_min, x_max, y_max = box
            boxes.append([x_min, y_min, x_max-x_min, y_max-y_min])

        # (3) Albumentation Transform 적용
        if self.transform:
            transformed = self.transform(image=image_np, bboxes=boxes, category_ids=labels)
            image_np = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['category_ids']

        # (4) Hugging Face 형식으로 패키징
        formatted_annotations = []
        for box, label in zip(boxes, labels):
            x_min, y_min, w, h = box
            formatted_annotations.append({
                "id": idx, "image_id": idx, "category_id": label,
                "bbox": [x_min, y_min, w, h], "area": w * h, "iscrowd": 0
            })
        
        target = {'image_id': idx, 'annotations': formatted_annotations}

        # [Mod] (5) Numpy 배열을 그대로 processor에 전달
        encoding = self.processor(images=image_np, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return {"pixel_values": pixel_values, "labels": target}
    
class YoloObjectDetectionDataset(Dataset):
    def __init__(self, image_dir, class_names, processor, transform=None):
        """
        Args:
            image_dir (str): 이미지가 들어있는 폴더 경로 (예: .../train/images 또는 .../images/train)
            class_names (list): 클래스 이름 리스트 (from yaml)
            processor: Hugging Face Processor
        """
        self.image_dir = image_dir
        self.class_names = class_names
        self.processor = processor
        self.transform = transform

        # ID <-> Label 매핑
        self.label2id = {name: i for i, name in enumerate(class_names)}
        self.id2label = {i: name for i, name in enumerate(class_names)}
        
        # 이미지 파일 리스트 로드
        self.image_files = sorted(
            glob.glob(os.path.join(self.image_dir, "*.*"))
        )
        # 확장자가 이미지인 것만 필터링 (jpg, png, jpeg 등)
        self.image_files = [f for f in self.image_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]

        # 1. 라벨 파일 경로 추론 (YOLO Style)
        label_path = os.path.splitext(image_path)[0] + ".txt"   # 확장자 변경 (.jpg -> .txt)

        # 경로 치환 (images -> labels)
        if f"{os.sep}images{os.sep}" in label_path:
            label_path = label_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
        elif "images" in label_path: # 구분자가 명확하지 않은 경우 단순 치환 (차선책)
            label_path = label_path.replace("images", "labels")

        # 2. 이미지 로드
        image = Image.open(image_path).convert("RGB")
        width_orig, height_orig = image.size

        # [Mod] (1) 이미지를 Numpy 배열로 변환
        image_np = np.array(image)

        boxes = []
        labels = []
        # 3. 라벨 파일 찾기 (이미지와 같은 이름의 .txt)        
        # 라벨 파일이 존재하면 읽기 (없으면 Negative Sample)
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue
                class_id = int(parts[0])
                
                # YOLO Format: normalized center_x, center_y, w, h
                cx, cy, w, h = map(float, parts[1:5])
                
                # Un-normalize (0~1 -> 절대 픽셀 좌표)
                abs_w, abs_h = w * width_orig, h * height_orig
                abs_cx, abs_cy = cx * width_orig, cy * height_orig
                
                # Hugging Face Processor(COCO)가 원하는 포맷: [x_min, y_min, w, h]
                x_min, y_min = abs_cx - (abs_w / 2), abs_cy - (abs_h / 2)

                boxes.append([x_min, y_min, abs_w, abs_h])
                labels.append(class_id)
        
        # [Add] (2) Albumentations Transform 적용
        if self.transform:
            transformed = self.transform(image=image_np, bboxes=boxes, category_ids=labels)
            image_np = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['category_ids']
        
        # (3) Hugging Face 형식으로 패키징
        formatted_annotations = []
        for box, label in zip(boxes, labels):
            formatted_annotations.append({
                "id": idx, "image_id": idx, "category_id": label,
                "bbox": box, "area": box[2] * box[3], "iscrowd": 0
            })
        
        target = {'image_id': idx, 'annotations': formatted_annotations}

        # [Mod] (4) Numpy 배열을 그대로 processor에 전달
        encoding = self.processor(images=image_np, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return {"pixel_values": pixel_values, "labels": target}
    

# [추가] 데이터셋 생성을 전담하는 Factory 함수
def create_dataset(data_cfg, split, processor, transform=None):
    """
    Args:
        data_cfg (dict): config.py에서 로드한 데이터 설정 딕셔너리
        split (str): 'train', 'val', 'test' 중 하나
        processor: HF Processor
        transform: Albumentations transform
    """
    fmt = data_cfg['format'].lower()
    
    # 사용할 경로 선택
    if split == 'train':
        data_path = data_cfg['train_full_path']
    elif split == 'val':
        data_path = data_cfg['val_full_path']
    elif split == 'test':
        data_path = data_cfg.get('test_full_path')
        if data_path is None: return None
    
    # -------------------------------------------------------
    # 1. YOLO Format
    # -------------------------------------------------------
    if fmt == 'yolo':
        return YoloObjectDetectionDataset(
            image_dir=data_path, # YOLO는 여기에 이미지 폴더 경로가 들어옴
            class_names=data_cfg['names'],
            processor=processor,
            transform=transform
        )
    
    # -------------------------------------------------------
    # 2. CVAT XML Format
    # -------------------------------------------------------
    elif fmt == 'cvat':
        # CVAT는 이미지 폴더 경로가 따로 필요함 (data.yaml에 image_dir 키 활용)
        image_folder = os.path.join(data_cfg['base_path'], data_cfg.get('image_dir', 'images'))
        
        return CVATObjectDetectionDataset(
            image_dir=image_folder,
            annotation_file=data_path, # CVAT는 여기에 xml 파일 경로가 들어옴
            processor=processor,
            transform=transform
        )
    
    # -------------------------------------------------------
    # 3. (확장 가능) COCO Format 등
    # -------------------------------------------------------
    # elif fmt == 'coco':
    #     return CocoDetectionDataset(...)
    
    else:
        raise ValueError(f"Unknown dataset format: {fmt}")