"""
데이터셋 클래스만 깔끔하게 분리
"""

import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image

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