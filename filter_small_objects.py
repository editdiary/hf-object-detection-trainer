"""
YOLO Dataset Small Object Filter
Copies the dataset and removes bounding box lines with area ratio < AREA_THRESHOLD.
Original dataset is never modified.
"""

import shutil
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────
SRC_ROOT = Path("data/99_exp_dataset")
DST_ROOT = Path("data/99_exp_dataset_filtered")
SPLITS = ["train", "val", "test"]
AREA_THRESHOLD = 0.00055  # strictly less than 0.055%
# ──────────────────────────────────────────────────────────────


def copy_dataset():
    if DST_ROOT.exists():
        print(f"[warn] {DST_ROOT} already exists — removing before copy")
        shutil.rmtree(DST_ROOT)
    shutil.copytree(SRC_ROOT, DST_ROOT)
    print(f"[copy] {SRC_ROOT} -> {DST_ROOT}")


def filter_labels():
    total_files = 0
    total_removed = 0
    total_modified = 0

    for split in SPLITS:
        label_dir = DST_ROOT / "labels" / split
        if not label_dir.exists():
            print(f"[skip] {label_dir} not found")
            continue

        split_removed = 0
        split_modified = 0

        for txt_path in sorted(label_dir.glob("*.txt")):
            lines = txt_path.read_text().splitlines()
            kept = []
            removed = 0

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    kept.append(line)
                    continue
                w, h = float(parts[3]), float(parts[4])
                if w * h < AREA_THRESHOLD:
                    removed += 1
                else:
                    kept.append(line)

            total_files += 1
            if removed:
                # Write back — keep file even if now empty
                txt_path.write_text("\n".join(kept) + ("\n" if kept else ""))
                split_removed += removed
                split_modified += 1

        total_removed += split_removed
        total_modified += split_modified
        print(f"[{split}] {split_modified} files modified, {split_removed} boxes removed")

    print("\n── Summary ──────────────────────────────────────────")
    print(f"  Label files processed : {total_files}")
    print(f"  Files modified        : {total_modified}")
    print(f"  Boxes removed         : {total_removed}")
    print(f"  Area threshold        : < {AREA_THRESHOLD * 100:.3f}%")
    print("─────────────────────────────────────────────────────")


if __name__ == "__main__":
    copy_dataset()
    filter_labels()
