"""
Utility script to turn a small subset of the Xingu deforestation dataset
into ready-to-train PNG image/mask pairs.

Steps performed:
1. Combines Landsat bands (B4=red, B3=green, B2=blue) into RGB images.
2. Normalizes each band to 0-255 and handles nodata values.
3. Loads the supplied binary mask (.npy) for each scene.
4. Writes the processed images and masks into data/train|val|test directories.

Run from the project root:
    python scripts/prepare_xingu_subset.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "data"

# Simple split definition (feel free to adjust)
SCENE_SPLITS: Dict[str, List[str]] = {
    "train": ["AE-X01", "AE-X02", "AE-X03", "AE-X04", "AE-X05"],
    "val": ["AE-X06"],
    "test": ["AE-X07", "AE-X08"],
}

BANDS = {
    "red": 4,
    "green": 3,
    "blue": 2,
}

NODATA_THRESHOLD = -1e20  # Landsat nodata sentinel (~-3.4e38)


def load_band(scene: str, band: int) -> np.ndarray:
    band_path = RAW_DIR / f"{scene}_B{band}.tif"
    if not band_path.exists():
        raise FileNotFoundError(f"Missing band file: {band_path}")
    arr = np.array(Image.open(band_path), dtype=np.float32)
    arr[arr <= NODATA_THRESHOLD] = np.nan
    return arr


def normalize_band(band_arr: np.ndarray) -> np.ndarray:
    band_arr = np.nan_to_num(band_arr, nan=0.0)
    band_min = float(band_arr.min())
    band_max = float(band_arr.max())
    if band_max - band_min < 1e-6:
        return np.zeros_like(band_arr, dtype=np.uint8)
    scaled = (band_arr - band_min) / (band_max - band_min)
    scaled = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    return scaled


def load_mask(scene: str) -> np.ndarray:
    mask_id = scene.split("-")[1].lower()
    mask_path = RAW_DIR / f"truth_{mask_id}.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask array: {mask_path}")
    mask = np.load(mask_path)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)
    return mask * 255


def process_scene(scene: str, split: str) -> None:
    print(f"Processing {scene} -> {split}")
    bands = [normalize_band(load_band(scene, band)) for band in (BANDS["red"], BANDS["green"], BANDS["blue"])]
    rgb = np.stack(bands, axis=-1)

    mask = None
    if split in {"train", "val"}:
        mask = load_mask(scene)
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {rgb.shape[:2]} for {scene}")

    image_out_dir = OUTPUT_ROOT / split / "images"
    mask_out_dir = OUTPUT_ROOT / split / "masks"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    if split in {"train", "val"}:
        mask_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        mask_out_dir.mkdir(parents=True, exist_ok=True)  # keep structure consistent

    img = Image.fromarray(rgb)
    img.save(image_out_dir / f"{scene}.png")

    if mask is not None:
        mask_img = Image.fromarray(mask)
        mask_img.save(mask_out_dir / f"{scene}.png")


def main() -> None:
    for split, scenes in SCENE_SPLITS.items():
        for scene in scenes:
            process_scene(scene, split)


if __name__ == "__main__":
    main()


