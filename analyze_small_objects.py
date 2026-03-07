"""
YOLO Dataset Small Object Analyzer
Finds and visualizes bounding boxes with area ratio < 0.001 (< 0.1% of image area).
"""

import os
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Configuration ─────────────────────────────────────────────
DATASET_ROOT = Path("data/99_exp_dataset")
SPLITS = ["train", "val", "test"]
AREA_THRESHOLD = 0.00055          # strictly less than 0.05% (0.055% => 15x15 at 640x640)
MAX_IMAGES_TO_VISUALIZE = 20    # <-- change this to control how many images are shown
COLS = 2                        # grid columns in the output figure
# ──────────────────────────────────────────────────────────────


def parse_labels(label_dir: Path) -> list[dict]:
    """Return list of {image_path, boxes} for every label file that has at least
    one box whose area ratio (w*h) is strictly below AREA_THRESHOLD."""
    results = []
    image_dir = Path(str(label_dir).replace("labels", "images"))

    for txt_path in sorted(label_dir.glob("*.txt")):
        stem = txt_path.stem
        # find the matching image (jpg / png / jpeg)
        img_path = None
        for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
            candidate = image_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        small_boxes = []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                area_ratio = w * h
                if area_ratio < AREA_THRESHOLD:
                    small_boxes.append({
                        "cls_id": cls_id,
                        "cx": cx, "cy": cy,
                        "w": w, "h": h,
                        "area_ratio": area_ratio,
                    })

        if small_boxes:
            results.append({"image_path": img_path, "boxes": small_boxes})

    return results


def collect_all_small_objects() -> list[dict]:
    """Scan all splits and return merged list, sorted by smallest area first."""
    all_entries = []
    for split in SPLITS:
        label_dir = DATASET_ROOT / "labels" / split
        if not label_dir.exists():
            print(f"[skip] {label_dir} not found")
            continue
        entries = parse_labels(label_dir)
        print(f"[{split}] {len(entries)} images with small objects")
        all_entries.extend(entries)

    # sort: images with smallest bounding box first
    all_entries.sort(key=lambda e: min(b["area_ratio"] for b in e["boxes"]))
    return all_entries


def draw_boxes_on_image(img_bgr, boxes: list[dict]) -> "np.ndarray":
    """Draw bounding boxes on a BGR image (modifies a copy). Returns RGB."""
    img = img_bgr.copy()
    h_img, w_img = img.shape[:2]
    for box in boxes:
        x1 = int((box["cx"] - box["w"] / 2) * w_img)
        y1 = int((box["cy"] - box["h"] / 2) * h_img)
        x2 = int((box["cx"] + box["w"] / 2) * w_img)
        y2 = int((box["cy"] + box["h"] / 2) * h_img)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{box['area_ratio']*100:.4f}%"
        cv2.putText(img, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def visualize(entries: list[dict], max_images: int = MAX_IMAGES_TO_VISUALIZE):
    entries = entries[:max_images]
    n = len(entries)
    if n == 0:
        print("No images to visualize.")
        return

    rows = math.ceil(n / COLS)
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 5, rows * 4))
    axes = [axes] if rows == 1 and COLS == 1 else axes
    axes = [ax for row in (axes if hasattr(axes[0], '__iter__') else [axes])
            for ax in (row if hasattr(row, '__iter__') else [row])]

    for i, entry in enumerate(entries):
        img_bgr = cv2.imread(str(entry["image_path"]))
        if img_bgr is None:
            continue
        img_rgb = draw_boxes_on_image(img_bgr, entry["boxes"])
        ax = axes[i]
        ax.imshow(img_rgb)
        ax.axis("off")
        min_area = min(b["area_ratio"] for b in entry["boxes"]) * 100
        n_boxes = len(entry["boxes"])
        ax.set_title(
            f"{entry['image_path'].name}\n"
            f"{n_boxes} box(es) | min area: {min_area:.4f}%",
            fontsize=8,
        )

    # hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Images with bounding box area < {AREA_THRESHOLD*100:.1f}%  "
        f"(showing {n}/{len(entries_global)} total)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = "small_objects_visualization.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nSaved figure to: {out_path}")
    plt.show()


def print_summary(entries: list[dict]):
    total_boxes = sum(len(e["boxes"]) for e in entries)
    if not entries:
        print("No small objects found.")
        return

    all_areas = [b["area_ratio"] for e in entries for b in e["boxes"]]
    print("\n── Summary ──────────────────────────────────────────")
    print(f"  Images with small objects : {len(entries)}")
    print(f"  Total small bounding boxes: {total_boxes}")
    print(f"  Area ratio  min  : {min(all_areas)*100:.6f}%")
    print(f"  Area ratio  max  : {max(all_areas)*100:.6f}%  (threshold: {AREA_THRESHOLD*100:.1f}%)")
    print(f"  Area ratio  mean : {sum(all_areas)/len(all_areas)*100:.6f}%")
    print("─────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    entries_global = collect_all_small_objects()
    print_summary(entries_global)
    visualize(entries_global, max_images=MAX_IMAGES_TO_VISUALIZE)
