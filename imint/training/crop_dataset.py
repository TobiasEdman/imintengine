"""
imint/training/crop_dataset.py — PyTorch Dataset for Swedish crop classification

Loads LUCAS-SE crop tiles (.npz) produced by fetch_lucas_tiles.py and
prepares them for Prithvi fine-tuning. Multitemporal by design:
each tile is (18, 256, 256) = 3 timesteps × 6 Prithvi bands.

Integrates with TorchGeo transforms when available, falling back to
manual Prithvi normalisation otherwise.

Tile .npz format (from fetch_lucas_tiles.py):
    spectral:       (18, 256, 256) float32, reflectance [0, 1]
    label:          scalar uint8, crop class index (0-7)
    seasons_valid:  (3,) bool, which seasons have real data
    lat, lon:       float64, WGS84 coordinates
    point_id:       str, LUCAS point identifier
    bbox_3006:      (4,) int, EPSG:3006 bounding box

Usage:
    from imint.training.crop_dataset import CropDataset

    ds = CropDataset("data/crop_tiles", split="train", patch_size=224)
    sample = ds[0]
    # sample["image"]: (18, 224, 224) float32 normalised
    # sample["label"]: scalar int64, crop class
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, WeightedRandomSampler
except ImportError:
    raise ImportError(
        "PyTorch is required for training. Install with: pip install torch"
    )

from .crop_schema import NUM_CLASSES, CLASS_NAMES

# Prithvi-EO-2.0 normalisation constants (DN scale, per band)
# Bands: B02, B03, B04, B8A, B11, B12
PRITHVI_MEAN = np.array(
    [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32,
)
PRITHVI_STD = np.array(
    [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32,
)

# TorchGeo availability
try:
    from torchgeo.transforms import AugmentationSequential
    import kornia.augmentation as K
    HAS_TORCHGEO = True
except ImportError:
    HAS_TORCHGEO = False


def _check_torchgeo() -> bool:
    return HAS_TORCHGEO


class CropDataset(Dataset):
    """PyTorch Dataset for Swedish crop classification tiles.

    Loads multitemporal .npz tiles (18 bands = 3 timesteps × 6 Prithvi bands)
    and returns normalised image tensors with scalar crop labels.

    Args:
        data_dir: Directory containing .npz tiles (from fetch_lucas_tiles.py).
        split: "train" or "val". If split files exist, use them; otherwise
               auto-split based on val_fraction.
        patch_size: Output spatial size (default 224 for Prithvi).
        val_fraction: Fraction of tiles for validation (default 0.15).
        use_torchgeo_transforms: Use TorchGeo augmentations if available.
        seed: Random seed for reproducible splits.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        patch_size: int = 224,
        val_fraction: float = 0.15,
        use_torchgeo_transforms: bool = True,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.patch_size = patch_size

        # Prithvi normalisation: tiled for 3 timesteps
        mean_1t = PRITHVI_MEAN.reshape(6, 1, 1)
        std_1t = PRITHVI_STD.reshape(6, 1, 1)
        self.mean = np.tile(mean_1t, (3, 1, 1))  # (18, 1, 1)
        self.std = np.tile(std_1t, (3, 1, 1))     # (18, 1, 1)

        # Discover tiles
        all_tiles = sorted(self.data_dir.glob("*.npz"))
        all_tiles = [t for t in all_tiles if t.stem != "manifest"]

        if not all_tiles:
            raise FileNotFoundError(
                f"No .npz tiles found in {data_dir}"
            )

        # Split: use existing split files if present, else auto-split
        split_file = self.data_dir / f"split_{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                names = {line.strip() for line in f if line.strip()}
            self.tiles = [t for t in all_tiles if t.name in names]
        else:
            # Deterministic split
            rng = random.Random(seed)
            shuffled = list(all_tiles)
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_fraction))
            if split == "val":
                self.tiles = shuffled[:n_val]
            else:
                self.tiles = shuffled[n_val:]

        # TorchGeo augmentation pipeline (training only)
        self.augment = split == "train"
        self.use_torchgeo = use_torchgeo_transforms and HAS_TORCHGEO
        if self.augment and self.use_torchgeo:
            self.transforms = AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomRotation(degrees=90, p=0.3),
                data_keys=["image"],
            )
        else:
            self.transforms = None

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> dict:
        tile_path = self.tiles[idx]
        try:
            data = np.load(tile_path, allow_pickle=True)
        except Exception:
            alt = (idx + 1) % len(self.tiles)
            return self.__getitem__(alt)

        image = data["spectral"].astype(np.float32)  # (18, 256, 256)
        label = int(data["label"])
        seasons_valid = data.get("seasons_valid", np.ones(3, dtype=bool))

        # Replace zero-padded seasons with nearest valid
        image = self._replace_zero_frames(image, seasons_valid)

        # Normalise: reflectance [0,1] → DN → Prithvi normalisation
        image = (image * 10000.0 - self.mean) / self.std

        # Spatial crop
        if self.augment:
            image = self._random_crop(image)
        else:
            image = self._center_crop(image)

        image_tensor = torch.from_numpy(image.copy())

        # TorchGeo augmentations
        if self.transforms is not None:
            batch = {"image": image_tensor.unsqueeze(0)}
            batch = self.transforms(batch)
            image_tensor = batch["image"].squeeze(0)

        # Manual augmentations (fallback if no TorchGeo)
        if self.augment and not self.use_torchgeo:
            image_tensor = self._manual_augment(image_tensor)

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.int64),
            "seasons_valid": torch.from_numpy(
                seasons_valid.astype(np.float32),
            ),
            "metadata": {
                "tile": tile_path.name,
                "point_id": str(data.get("point_id", "")),
                "lat": float(data.get("lat", 0)),
                "lon": float(data.get("lon", 0)),
            },
        }

    @staticmethod
    def _replace_zero_frames(
        image: np.ndarray,
        seasons_valid: np.ndarray,
    ) -> np.ndarray:
        """Replace missing seasons with nearest valid season."""
        n_frames = 3
        n_bands = 6
        valid_indices = np.where(seasons_valid)[0]

        if len(valid_indices) == 0 or len(valid_indices) == n_frames:
            return image

        # Prefer summer (index 1) as replacement source
        preferred = [i for i in [1, 0, 2] if i in valid_indices]
        if not preferred:
            preferred = list(valid_indices)

        for t in range(n_frames):
            if not seasons_valid[t]:
                best = min(preferred, key=lambda c: abs(t - c))
                src = best * n_bands
                dst = t * n_bands
                image[dst:dst + n_bands] = image[src:src + n_bands]

        return image

    def _random_crop(self, image: np.ndarray) -> np.ndarray:
        """Random crop from (C, H, W) to (C, patch, patch)."""
        p = self.patch_size
        _, h, w = image.shape
        if h <= p and w <= p:
            return image
        y = random.randint(0, max(h - p, 0))
        x = random.randint(0, max(w - p, 0))
        return image[:, y:y + p, x:x + p]

    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        """Center crop from (C, H, W) to (C, patch, patch)."""
        p = self.patch_size
        _, h, w = image.shape
        if h <= p and w <= p:
            return image
        y = max((h - p) // 2, 0)
        x = max((w - p) // 2, 0)
        return image[:, y:y + p, x:x + p]

    @staticmethod
    def _manual_augment(image: torch.Tensor) -> torch.Tensor:
        """Simple augmentations without TorchGeo."""
        if random.random() > 0.5:
            image = torch.flip(image, [-1])  # Horizontal
        if random.random() > 0.5:
            image = torch.flip(image, [-2])  # Vertical
        k = random.randint(0, 3)
        if k > 0:
            image = torch.rot90(image, k, [-2, -1])
        return image


def build_crop_sampler(dataset: CropDataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler to handle class imbalance.

    Swedish crop distribution is highly skewed (ley/grass dominates).
    This sampler oversamples rare classes like potato and rapeseed.
    """
    # Count per class
    class_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    tile_labels = []
    for tile_path in dataset.tiles:
        try:
            data = np.load(tile_path, allow_pickle=True)
            label = int(data["label"])
        except Exception:
            label = 0
        tile_labels.append(label)
        class_counts[label] += 1

    # Inverse frequency weighting
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / class_counts
    class_weights /= class_weights.sum()

    # Per-sample weight
    sample_weights = [float(class_weights[lbl]) for lbl in tile_labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
