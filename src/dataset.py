"""
데이터셋 클래스만 깔끔하게 분리
"""

import os
import glob
from PIL import Image
import xml.etree.ElementTree as ET

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

        # TODO 증강(Transform, albumentation) 적용 위치 (나중에 구현 시 여기에 추가)
        
        formatted_annotations = []
        for box, label in zip(item['boxes'], item['class_labels']):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            
            formatted_annotations.append({
                "id": idx,
                "image_id": idx,
                "category_id": label,
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0
            })
        
        target = {'image_id': idx, 'annotations': formatted_annotations}

        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
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
        # 논리: 이미지 경로 중 'images' 부분을 'labels'로 바꾸고, 확장자를 .txt로 변경
        # 예1: /data/train/images/01.jpg -> /data/train/labels/01.txt
        # 예2: /data/images/train/01.jpg -> /data/labels/train/01.txt

        # 확장자 변경 (.jpg -> .txt)
        label_path = os.path.splitext(image_path)[0] + ".txt"

        # 경로 치환 (images -> labels)
        # os.sep ('/' 또는 '\')를 붙여서 파일명 등에 포함된 'images' 단어 오인식을 방지
        # 예: 'images_01.jpg' 같은 파일명이 꼬이지 않게 함
        if f"{os.sep}images{os.sep}" in label_path:
            label_path = label_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
        elif "images" in label_path: # 구분자가 명확하지 않은 경우 단순 치환 (차선책)
            label_path = label_path.replace("images", "labels")

        # 2. 이미지 로드
        image = Image.open(image_path).convert("RGB")
        width_orig, height_orig = image.size

        formatted_annotations = []

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
                abs_cx = cx * width_orig
                abs_cy = cy * height_orig
                abs_w = w * width_orig
                abs_h = h * height_orig
                
                # Hugging Face Processor(COCO)가 원하는 포맷: [x_min, y_min, w, h]
                x_min = abs_cx - (abs_w / 2)
                y_min = abs_cy - (abs_h / 2)
                
                formatted_annotations.append({
                    "id": idx,
                    "image_id": idx,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, abs_w, abs_h], # COCO format
                    "area": abs_w * abs_h,
                    "iscrowd": 0
                })
        
        # 4. Processor 실행
        target = {'image_id': idx, 'annotations': formatted_annotations}
        
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
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