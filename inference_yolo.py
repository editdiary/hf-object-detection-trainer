"""
Folder-based inference with COCO JSON output (Ultralytics YOLO).

Runs a trained YOLO model on an arbitrary image folder
(no dataset labels required) and outputs results in COCO JSON format
compatible with pycocotools.

Usage:
    python inference_yolo.py \
        --model_path exp/runs/detect/runs/yolov11m/weights/best.pt \
        --image_dir /path/to/test/images \
        --output results.json \
        --conf_threshold 0.001
"""

import argparse
import glob
import json
import os
import time

import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on an image folder → COCO JSON"
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to YOLO weights (e.g., runs/.../best.pt)")
    parser.add_argument("--image_dir", required=True,
                        help="Folder containing images (recursive search)")
    parser.add_argument("--output", default="predictions.json",
                        help="Output file path for COCO JSON results (default: predictions.json)")
    parser.add_argument("--conf_threshold", type=float, default=0.001,
                        help="Minimum confidence score (default: 0.001)")
    parser.add_argument("--device", default=None,
                        help="Device: 'cuda' or 'cpu' (default: auto-detect)")
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


def run_inference(model, image_files, device, conf_threshold):
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

        result = model(img_path, conf=conf_threshold, device=device, verbose=False)[0]
        boxes = result.boxes

        xyxy = boxes.xyxy.cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        cls_ids = boxes.cls.cpu().tolist()

        for (x1, y1, x2, y2), score, cls_id in zip(xyxy, scores, cls_ids):
            coco_results.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(cls_id),
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


def build_categories(model):
    """Extract category list from model's names dict."""
    return [
        {"id": int(cat_id), "name": cat_name}
        for cat_id, cat_name in sorted(model.names.items(), key=lambda x: int(x[0]))
    ]


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  → {path} ({len(data)} entries)")


def main():
    args = parse_args()

    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from: {args.model_path}")
    model = YOLO(args.model_path)

    # Collect images
    image_files = collect_images(args.image_dir)
    if not image_files:
        print(f"No images found in {args.image_dir}")
        return
    print(f"Found {len(image_files)} images in {args.image_dir}")

    # Inference
    t0 = time.time()
    coco_results, coco_images = run_inference(
        model, image_files, device, args.conf_threshold
    )
    elapsed = time.time() - t0
    print(f"Inference done: {len(coco_results)} detections in {elapsed:.1f}s "
          f"({len(image_files)/elapsed:.1f} img/s)")

    # Build categories from model config
    coco_categories = build_categories(model)

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
