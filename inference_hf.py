"""
Folder-based inference with COCO JSON output.

Runs a trained HF object detection model on an arbitrary image folder
(no dataset labels required) and outputs results in COCO JSON format
compatible with pycocotools.

Usage:
    python inference_hf.py \
        --model_dir runs/rtdetr_v2_r34/repeat-exp_seed421/best_model \
        --image_dir /path/to/test/images \
        --output results.json \
        --conf_threshold 0.5
"""

import argparse
import glob
import json
import os
import time

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HF object detection on an image folder → COCO JSON"
    )
    parser.add_argument("--model_dir", required=True,
                        help="Path to saved model (e.g., runs/.../best_model)")
    parser.add_argument("--image_dir", required=True,
                        help="Folder containing images (recursive search)")
    parser.add_argument("--output", default="predictions.json",
                        help="Output file path for COCO JSON results (default: predictions.json)")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Minimum confidence score (default: 0.5)")
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


def run_inference(model, processor, image_files, device, conf_threshold):
    """
    Run per-image inference and collect COCO-format results.

    Returns:
        coco_results: list of dicts with image_id, category_id, bbox [x,y,w,h], score
        coco_images:  list of dicts with id, file_name, width, height
    """
    img_proc = getattr(processor, "image_processor", processor)

    coco_results = []
    coco_images = []
    ann_id = 1

    model.eval()
    for image_id, img_path in enumerate(tqdm(image_files, desc="Running inference"), start=1):
        raw_img = Image.open(img_path).convert("RGB")
        W, H = raw_img.size

        coco_images.append({
            "id": image_id,
            "file_name": os.path.basename(img_path),
            "width": W,
            "height": H,
        })

        inputs = processor(images=raw_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[H, W]], device=device)
        result = img_proc.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_threshold
        )[0]

        scores = result["scores"].tolist()
        labels = result["labels"].tolist()
        boxes = result["boxes"].tolist()  # [x1, y1, x2, y2]

        for score, label, (x1, y1, x2, y2) in zip(scores, labels, boxes):
            # Convert [x1, y1, x2, y2] → COCO [x, y, w, h]
            coco_results.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": label,
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
    """Extract category list from model's id2label config."""
    id2label = model.config.id2label
    return [
        {"id": int(cat_id), "name": cat_name}
        for cat_id, cat_name in sorted(id2label.items(), key=lambda x: int(x[0]))
    ]


def write_json(data, path):
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

    # Load model & processor
    print(f"Loading model from: {args.model_dir}")
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForObjectDetection.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # Collect images
    image_files = collect_images(args.image_dir)
    if not image_files:
        print(f"No images found in {args.image_dir}")
        return
    print(f"Found {len(image_files)} images in {args.image_dir}")

    # Inference
    t0 = time.time()
    coco_results, coco_images = run_inference(
        model, processor, image_files, device, args.conf_threshold
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
