"""
Folder-based inference with COCO JSON output (torchvision Faster R-CNN).

Runs a trained Faster R-CNN model on an arbitrary image folder
(no dataset labels required) and outputs results in COCO JSON format
compatible with pycocotools.

Usage:
    python inference_frcnn.py \
        --model_path runs/frcnn_r50fpnv2/.../best_model/model.pth \
        --image_dir /path/to/test/images \
        --output results.json \
        --conf_threshold 0.001
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Faster R-CNN object detection on an image folder → COCO JSON"
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to trained .pth weights")
    parser.add_argument("--image_dir", required=True,
                        help="Folder containing images (recursive search)")
    parser.add_argument("--output", default="predictions.json",
                        help="Output file path for COCO JSON results (default: predictions.json)")
    parser.add_argument("--conf_threshold", type=float, default=0.001,
                        help="Minimum confidence score (default: 0.001)")
    parser.add_argument("--device", default=None,
                        help="Device: 'cuda' or 'cpu' (default: auto-detect)")
    parser.add_argument("--image_size", type=int, default=640,
                        help="Model input size (default: 640)")
    parser.add_argument("--num_classes", type=int, default=1,
                        help="Number of object classes, excluding background (default: 1)")
    parser.add_argument("--class_names", default="ripe_chamoe",
                        help="Comma-separated class names (default: ripe_chamoe)")
    return parser.parse_args()


def collect_images(image_dir):
    """Recursively glob for common image extensions."""
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(
            glob.glob(os.path.join(image_dir, "**", ext), recursive=True)
        )
    image_files.sort()
    return image_files


def load_model(model_path, num_classes, image_size, device):
    """Rebuild Faster R-CNN architecture and load trained weights."""
    # num_classes +1 for background (torchvision convention)
    total_classes = num_classes + 1
    model = fasterrcnn_resnet50_fpn_v2(weights=None)

    # Replace box predictor (same as train_frcnn.py)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, total_classes)

    # Match image size to training config
    model.transform.min_size = (image_size,)
    model.transform.max_size = image_size

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(model, image_files, device, conf_threshold, class_names):
    """
    Run per-image inference and collect COCO-format results.

    Returns:
        coco_results: list of dicts with image_id, category_id, bbox [x,y,w,h], score
        coco_images:  list of dicts with id, file_name, width, height
    """
    coco_results = []
    coco_images = []
    ann_id = 1

    for image_id, img_path in enumerate(tqdm(image_files, desc="Running inference"), start=1):
        raw_img = Image.open(img_path).convert("RGB")
        W, H = raw_img.size

        coco_images.append({
            "id": image_id,
            "file_name": os.path.basename(img_path),
            "width": W,
            "height": H,
        })

        # Convert PIL → tensor [0,1] (torchvision FRCNN handles resize/normalize internally)
        img_tensor = torch.from_numpy(np.array(raw_img)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            outputs = model([img_tensor])[0]

        scores = outputs["scores"].cpu().tolist()
        labels = outputs["labels"].cpu().tolist()
        boxes = outputs["boxes"].cpu().tolist()  # [x1, y1, x2, y2]

        for score, label, (x1, y1, x2, y2) in zip(scores, labels, boxes):
            if score < conf_threshold:
                continue
            # Skip background predictions (label==0)
            if label == 0:
                continue
            # Reverse label +1 shift: torchvision label 1 → YOLO category_id 0
            category_id = label - 1
            # Convert xyxy → COCO [x, y, w, h]
            coco_results.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2 - x1, 2),
                    round(y2 - y1, 2),
                ],
                "score": round(score, 4),
            })
            ann_id += 1

    return coco_results, coco_images


def build_categories(class_names):
    """Build category list from class names (0-indexed, YOLO space)."""
    return [
        {"id": i, "name": name}
        for i, name in enumerate(class_names)
    ]


def write_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  → {path} ({len(data)} entries)")


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Parse class names
    class_names = [n.strip() for n in args.class_names.split(",")]
    assert len(class_names) == args.num_classes, (
        f"class_names count ({len(class_names)}) != num_classes ({args.num_classes})"
    )

    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, args.num_classes, args.image_size, device)

    # Collect images
    image_files = collect_images(args.image_dir)
    if not image_files:
        print(f"No images found in {args.image_dir}")
        return
    print(f"Found {len(image_files)} images in {args.image_dir}")

    # Inference
    t0 = time.time()
    coco_results, coco_images = run_inference(
        model, image_files, device, args.conf_threshold, class_names
    )
    elapsed = time.time() - t0
    print(f"Inference done: {len(coco_results)} detections in {elapsed:.1f}s "
          f"({len(image_files)/elapsed:.1f} img/s)")

    # Build categories
    coco_categories = build_categories(class_names)

    # Derive companion file paths from --output
    base, ext = os.path.splitext(args.output)
    images_path = f"{base}_images{ext}"
    categories_path = f"{base}_categories{ext}"

    # Write outputs
    print("Writing results:")
    write_json(coco_results, args.output)
    write_json(coco_images, images_path)
    write_json(coco_categories, categories_path)


if __name__ == "__main__":
    main()
