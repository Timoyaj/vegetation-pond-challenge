#!/usr/bin/env python3
"""
scripts/create_thumbnails.py

Move PNG files from repository root into `figures/` (if not already there) and
create thumbnail images prefixed with `thumb_`.

Usage:
    pip install pillow
    python scripts/create_thumbnails.py

This script is safe to run multiple times; it will skip files already in `figures/`.
"""

import os
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
SRC_DIR = ROOT
THUMB_MAX_WIDTH = 360

PNG_FILES = [
    "eda_distributions.png",
    "eda_correlation.png",
    "eda_class_balance.png",
    "eda_missingness.png",
    "eda_ndwi_temporal.png",
    "model_comparison.png",
    "confusion_matrices.png",
    "calibration_curve.png",
    "shap_importance.png",
    "test_prob_distribution.png",
]


def ensure_figures_dir():
    FIG_DIR.mkdir(exist_ok=True)


def move_if_needed(fname: str):
    src = SRC_DIR / fname
    dst = FIG_DIR / fname
    if not src.exists():
        # maybe already moved
        if dst.exists():
            print(f"{fname} already in figures/")
            return
        else:
            print(f"Warning: {fname} not found at repo root or in figures/")
            return
    # move file to figures/
    print(f"Moving {fname} -> figures/")
    shutil.move(str(src), str(dst))


def make_thumbnail(fname: str):
    img_path = FIG_DIR / fname
    if not img_path.exists():
        print(f"Skipping thumbnail for missing file: {fname}")
        return
    thumb_name = f"thumb_{fname}"
    thumb_path = FIG_DIR / thumb_name
    try:
        with Image.open(img_path) as im:
            # compute new size keeping aspect ratio
            w, h = im.size
            if w <= THUMB_MAX_WIDTH:
                # if already small, copy
                im.save(thumb_path)
                print(f"Copied small image as thumbnail: {thumb_name}")
                return
            new_h = int(h * (THUMB_MAX_WIDTH / w))
            im = im.resize((THUMB_MAX_WIDTH, new_h), Image.LANCZOS)
            im.save(thumb_path)
            print(f"Created thumbnail: {thumb_name}")
    except Exception as e:
        print(f"Failed to create thumbnail for {fname}: {e}")


def main():
    ensure_figures_dir()
    for fname in PNG_FILES:
        move_if_needed(fname)
    for fname in PNG_FILES:
        make_thumbnail(fname)
    print("Done. Thumbnails are in the figures/ directory. Commit the changes to include the figures/ files in the repo.")


if __name__ == "__main__":
    main()
