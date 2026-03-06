import os
import csv
import json
import argparse

import torch
import torchvision.ops as ops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from src.config import Config
from src.dataset import create_dataset
from src.collate import get_collate_fn
from src.metrics import (get_model_probs, extract_scores_and_labels,
                         accepts_pixel_mask, compute_precision_recall_f1)
from src.visualization import generate_pr_curve, generate_confusion_matrix, visualize_inference_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Hugging Face Object Detection model.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the saved best_model directory")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--conf_threshold", type=float, default=Config.CONF_MATRIX_CONF_THRESHOLD,
                        help="Confidence threshold for Confusion Matrix")
    parser.add_argument("--iou_threshold", type=float, default=Config.IOU_THRESHOLD,
                        help="IoU threshold for Confusion Matrix matching")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save per-image visualizations and JSON prediction files")
    return parser.parse_args()


def _get_image_path(dataset, idx):
    """Resolve the original image file path for a given dataset index."""
    if hasattr(dataset, "image_files"):
        return dataset.image_files[idx]
    # CVAT dataset
    return os.path.join(dataset.image_dir, dataset.samples[idx]["file_name"])


def save_per_image_predictions(model, processor, dataset, device, output_dir,
                                id2label, conf_threshold):
    """
    Run inference on every image in *dataset*, then for each image save:
      - <stem>_pred.jpg  : original image with predicted boxes drawn
      - <stem>_pred.json : list of {label, score, box:[x1,y1,x2,y2]}
    Results are written to output_dir/predictions/.
    """
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    img_proc = getattr(processor, "image_processor", processor)

    model.eval()
    for idx in tqdm(range(len(dataset)), desc="Saving predictions"):
        img_path = _get_image_path(dataset, idx)
        raw_img = Image.open(img_path).convert("RGB")
        W, H = raw_img.size

        inputs = processor(images=raw_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[H, W]], device=device)
        result = img_proc.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_threshold
        )[0]

        scores = result["scores"].tolist()
        labels = result["labels"].tolist()
        boxes  = result["boxes"].tolist()

        # --- JSON ---
        predictions = [
            {"label": id2label[lbl], "score": round(sc, 4),
             "box": [round(v, 2) for v in box]}
            for sc, lbl, box in zip(scores, labels, boxes)
        ]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(pred_dir, f"{stem}_pred.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"image": img_path, "predictions": predictions}, f,
                      indent=2, ensure_ascii=False)

        # --- Visualization ---
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(raw_img)
        for sc, lbl, (x1, y1, x2, y2) in zip(scores, labels, boxes):
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{id2label[lbl]}: {sc:.2f}",
                    color="white", fontsize=8, weight="bold",
                    bbox=dict(facecolor="red", alpha=0.7, edgecolor="none", pad=1))
        ax.axis("off")
        plt.tight_layout(pad=0)
        vis_path = os.path.join(pred_dir, f"{stem}_pred.jpg")
        plt.savefig(vis_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"💾 Per-image predictions saved ({len(dataset)} images): {pred_dir}")


def _get_unique_output_dir(base_dir: str) -> str:
    """Return *base_dir* if it doesn't exist, otherwise append _1, _2, … until unique."""
    if not os.path.exists(base_dir):
        return base_dir
    version = 1
    while True:
        candidate = f"{base_dir}_{version}"
        if not os.path.exists(candidate):
            return candidate
        version += 1


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    print(f"✅ Loading model from: {args.model_dir}")

    data_cfg = Config.load_data_config()
    processor = AutoImageProcessor.from_pretrained(args.model_dir)

    eval_dataset = create_dataset(data_cfg, processor=processor, split=args.split)
    if eval_dataset is None:
        print(f"❌ '{args.split}' 데이터셋을 찾을 수 없습니다. data.yaml 구성을 확인하세요.")
        return

    id2label = eval_dataset.id2label
    label2id = eval_dataset.label2id

    model = AutoModelForObjectDetection.from_pretrained(
        args.model_dir,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)
    model.eval()

    collate_fn = get_collate_fn(processor)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    base_output_dir = os.path.join(os.path.dirname(os.path.normpath(args.model_dir)), f"eval_{args.split}")
    output_dir = _get_unique_output_dir(base_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    num_classes = len(id2label)

    print(f"🚀 Starting evaluation on '{args.split}' split ({len(eval_dataset)} images)...")
    _accepts_mask = accepts_pixel_mask(model)

    # Single-pass inference: cache predictions for mAP, PR curve, and confusion matrix
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="faster_coco_eval")
    metric.warn_on_many_detections = False
    cached_predictions = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)

        with torch.no_grad():
            if _accepts_mask:
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            else:
                outputs = model(pixel_values=pixel_values)

        probs = get_model_probs(outputs.logits, num_classes)

        preds_list = []
        targets_list = []

        for i in range(len(batch["labels"])):
            obj_scores, obj_labels = extract_scores_and_labels(probs[i], num_classes)

            t_boxes = ops.box_convert(
                batch["labels"][i]["boxes"].to(device), in_fmt="cxcywh", out_fmt="xyxy"
            )
            t_labels = batch["labels"][i]["class_labels"].to(device)
            targets_list.append({"boxes": t_boxes, "labels": t_labels})

            p_boxes = ops.box_convert(outputs.pred_boxes[i], in_fmt="cxcywh", out_fmt="xyxy")

            map_keep = obj_scores > Config.MAP_SCORE_THRESHOLD
            preds_list.append({
                "boxes": p_boxes[map_keep],
                "scores": obj_scores[map_keep],
                "labels": obj_labels[map_keep],
            })

            cached_predictions.append({
                "obj_scores": obj_scores.cpu(),
                "obj_labels": obj_labels.cpu(),
                "p_boxes_xyxy": p_boxes.cpu(),
                "t_boxes_xyxy": t_boxes.cpu(),
                "t_labels": t_labels.cpu(),
            })

        metric.update(preds_list, targets_list)

    results = metric.compute()

    precision, recall, f1 = compute_precision_recall_f1(
        cached_predictions,
        iou_threshold=0.5,
        conf_threshold=args.conf_threshold,
    )

    col_w = 22
    print("\n" + "=" * 50)
    print("📊 [Evaluation Results]")
    print("=" * 50)
    print(f"  {'Metric':<{col_w}} {'Value':>8}")
    print(f"  {'-'*col_w}  {'-'*8}")
    print(f"  {'mAP (IoU 0.50:0.95)':<{col_w}} {results['map'].item():>8.4f}")
    print(f"  {'mAP @ IoU=0.50':<{col_w}} {results['map_50'].item():>8.4f}")
    print(f"  {'mAP @ IoU=0.75':<{col_w}} {results['map_75'].item():>8.4f}")
    print(f"  {'-'*col_w}  {'-'*8}")
    print(f"  {'Precision (IoU=0.5)':<{col_w}} {precision:>8.4f}")
    print(f"  {'Recall    (IoU=0.5)':<{col_w}} {recall:>8.4f}")
    print(f"  {'F1-score  (IoU=0.5)':<{col_w}} {f1:>8.4f}")
    print("=" * 50)

    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["mAP_50_95",  round(results["map"].item(),    4)])
        writer.writerow(["mAP_50",     round(results["map_50"].item(), 4)])
        writer.writerow(["mAP_75",     round(results["map_75"].item(), 4)])
        writer.writerow(["precision",  round(precision, 4)])
        writer.writerow(["recall",     round(recall,    4)])
        writer.writerow(["f1",         round(f1,        4)])
    print(f"📄 Metrics saved to: {csv_path}")

    generate_pr_curve(cached_predictions, output_dir)
    generate_confusion_matrix(
        cached_predictions=cached_predictions,
        output_dir=output_dir,
        id2label=id2label,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
    )
    split_img_dir = data_cfg.get(f"{args.split}_full_path", data_cfg["val_full_path"])
    visualize_inference_samples(
        model=model,
        processor=processor,
        data_cfg=data_cfg,
        device=device,
        output_dir=output_dir,
        id2label=id2label,
        img_dir=split_img_dir,
    )

    if args.save_predictions:
        save_per_image_predictions(
            model=model,
            processor=processor,
            dataset=eval_dataset,
            device=device,
            output_dir=output_dir,
            id2label=id2label,
            conf_threshold=args.conf_threshold,
        )

    print(f"\n✅ 모든 평가 결과가 저장되었습니다: {output_dir}")


if __name__ == "__main__":
    main()
