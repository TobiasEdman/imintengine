"""
imint/training/dataset.py — PyTorch Dataset for LULC training

Loads cached Sentinel-2 + NMD tile pairs (.npz) and applies
normalisation and augmentation for Prithvi-based training.
"""
from __future__ import annotations

import os
import json
import random
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, WeightedRandomSampler
except ImportError:
    raise ImportError("PyTorch is required for training. Install with: pip install torch")

from .config import TrainingConfig


class LULCDataset(Dataset):
    """PyTorch Dataset for Sentinel-2 + NMD LULC training tiles.

    Each sample is a dict with:
        image: (6, 224, 224) float32 tensor, Prithvi-normalised
        label: (224, 224) int64 tensor, class indices
        metadata: dict with location, date, etc.

    Tile .npz files contain:
        image: (6, H, W) float32 reflectance [0, 1]
        label: (H, W) uint8 LULC class indices
        easting, northing: EPSG:3006 center
        date: ISO date string
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        config: TrainingConfig | None = None,
    ):
        self.config = config or TrainingConfig()
        self.split = split
        self.data_dir = Path(data_dir)
        self.tiles_dir = self.data_dir / "tiles"

        # Load split file
        split_file = self.data_dir / f"split_{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file) as f:
            self.tile_names = [line.strip() for line in f if line.strip()]

        # Prithvi normalisation constants (DN-scale)
        self.mean = np.array(self.config.prithvi_mean, dtype=np.float32).reshape(6, 1, 1)
        self.std = np.array(self.config.prithvi_std, dtype=np.float32).reshape(6, 1, 1)

        # Augmentation only for training
        self.augment = split == "train"

    def __len__(self) -> int:
        return len(self.tile_names)

    def __getitem__(self, idx: int) -> dict:
        tile_path = self.tiles_dir / self.tile_names[idx]
        data = np.load(tile_path, allow_pickle=True)

        image = data["image"].astype(np.float32)  # (6, H, W) reflectance [0,1]
        label = data["label"].astype(np.int64)     # (H, W) class indices

        # Normalise: reflectance → DN → Prithvi normalisation
        image = (image * 10000.0 - self.mean) / self.std

        # Augmentation
        if self.augment:
            image, label = self._augment(image, label)
        else:
            # Center crop to patch_pixels
            image, label = self._center_crop(image, label)

        return {
            "image": torch.from_numpy(image.copy()),
            "label": torch.from_numpy(label.copy()),
            "metadata": {
                "tile": self.tile_names[idx],
                "easting": int(data.get("easting", 0)),
                "northing": int(data.get("northing", 0)),
                "date": str(data.get("date", "")),
            },
        }

    def _augment(
        self, image: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation (training only).

        Augmentations:
            - Random crop from fetch_pixels to patch_pixels
            - Random horizontal/vertical flip
            - Random 90-degree rotation
        """
        cfg = self.config

        # Random crop
        if cfg.random_crop and image.shape[1] > cfg.patch_pixels:
            image, label = self._random_crop(image, label)
        else:
            image, label = self._center_crop(image, label)

        # Random flip
        if cfg.random_flip:
            if random.random() > 0.5:
                image = image[:, :, ::-1]  # Horizontal
                label = label[:, ::-1]
            if random.random() > 0.5:
                image = image[:, ::-1, :]  # Vertical
                label = label[::-1, :]

        # Random 90-degree rotation
        if cfg.random_rotation:
            k = random.randint(0, 3)
            if k > 0:
                image = np.rot90(image, k, axes=(1, 2))
                label = np.rot90(label, k, axes=(0, 1))

        return image, label

    def _random_crop(
        self, image: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Random crop from (6, H, W) to (6, patch, patch)."""
        p = self.config.patch_pixels
        _, h, w = image.shape
        y = random.randint(0, max(h - p, 0))
        x = random.randint(0, max(w - p, 0))
        return image[:, y:y + p, x:x + p], label[y:y + p, x:x + p]

    def _center_crop(
        self, image: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Center crop from (6, H, W) to (6, patch, patch)."""
        p = self.config.patch_pixels
        _, h, w = image.shape
        y = max((h - p) // 2, 0)
        x = max((w - p) // 2, 0)
        return image[:, y:y + p, x:x + p], label[y:y + p, x:x + p]


def build_weighted_sampler(
    dataset: LULCDataset,
    class_stats_path: str | None = None,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples rare classes.

    Args:
        dataset: The training dataset.
        class_stats_path: Path to class_stats.json with per-tile
            dominant class information.

    Returns:
        WeightedRandomSampler for DataLoader.
    """
    if class_stats_path and os.path.exists(class_stats_path):
        with open(class_stats_path) as f:
            stats = json.load(f)
        tile_weights = stats.get("tile_weights", {})
    else:
        # Uniform weights if no stats available
        tile_weights = {}

    weights = []
    for name in dataset.tile_names:
        w = tile_weights.get(name, 1.0)
        weights.append(w)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
