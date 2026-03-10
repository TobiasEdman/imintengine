"""
imint/training/dataset.py — PyTorch Dataset for LULC training

Loads cached Sentinel-2 + NMD tile pairs (.npz) and applies
normalisation and augmentation for Prithvi-based training.

Supports both single-date tiles (6, H, W) and multitemporal
tiles (T*6, H, W) with day-of-year temporal position encoding.

Optional auxiliary channels (each stored as (H, W) float32 in .npz):
  - height:     tree height in meters (Skogsstyrelsen trädhöjd)
  - volume:     timber volume in m³sk/ha (Skogliga grunddata)
  - basal_area: basal area in m²/ha (Skogliga grunddata grundyta)
  - diameter:   mean stem diameter in cm (Skogliga grunddata Dgv)
  - dem:        terrain elevation in meters (Copernicus DEM GLO-30)
  - vpp_sosd:   start of season day (HR-VPP phenology)
  - vpp_eosd:   end of season day (HR-VPP phenology)
  - vpp_length: season length in days (HR-VPP phenology)
  - vpp_maxv:   max Plant Phenology Index (HR-VPP)
  - vpp_minv:   min Plant Phenology Index (HR-VPP)
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
from .class_schema import _LUT_19_TO_10


class LULCDataset(Dataset):
    """PyTorch Dataset for Sentinel-2 + NMD LULC training tiles.

    Supports two tile formats:

    **Single-date** (default):
        image: (6, H, W) float32 reflectance [0, 1]
        label: (H, W) uint8 LULC class indices
        Returns: image (6, 224, 224), label (224, 224)

    **Multitemporal** (enable_multitemporal=True):
        image: (T*6, H, W) float32 reflectance [0, 1]
        temporal_mask: (T,) uint8, 1=valid 0=padded
        doy: (T,) int32, day-of-year per frame
        Returns: image (T*6, 224, 224), label (224, 224),
                 temporal_mask (T,), doy (T,)
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

        # Multitemporal settings
        self.multitemporal = self.config.enable_multitemporal
        self.n_frames = self.config.num_temporal_frames
        self.n_bands = len(self.config.prithvi_bands)  # 6

        # Build tiled mean/std for multitemporal normalisation
        # (T*6, 1, 1) — same normalisation applied to each frame
        if self.multitemporal:
            self.mean_mt = np.tile(self.mean, (self.n_frames, 1, 1))
            self.std_mt = np.tile(self.std, (self.n_frames, 1, 1))

        # Auxiliary channels: list of npz key names
        self.aux_channels: list[str] = []
        if self.config.enable_height_channel:
            self.aux_channels.append("height")
        if self.config.enable_volume_channel:
            self.aux_channels.append("volume")
        if self.config.enable_basal_area_channel:
            self.aux_channels.append("basal_area")
        if self.config.enable_diameter_channel:
            self.aux_channels.append("diameter")
        if self.config.enable_dem_channel:
            self.aux_channels.append("dem")
        if self.config.enable_vpp_channels:
            # VPP adds 5 channels stored as separate keys in .npz
            self.aux_channels.extend([
                "vpp_sosd", "vpp_eosd", "vpp_length",
                "vpp_maxv", "vpp_minv",
            ])

        # Augmentation only for training
        self.augment = split == "train"

    def __len__(self) -> int:
        return len(self.tile_names)

    def __getitem__(self, idx: int) -> dict:
        tile_path = self.tiles_dir / self.tile_names[idx]
        try:
            data = np.load(tile_path, allow_pickle=True)
        except Exception:
            # Corrupted tile — return a random valid tile instead
            alt = (idx + 1) % len(self.tile_names)
            return self.__getitem__(alt)

        image = data["image"].astype(np.float32)
        label = data["label"].astype(np.int64)

        # Load and stack auxiliary channels → (N, H, W) or None
        # Missing channels are zero-filled to ensure consistent batch keys
        aux_names: list[str] = []
        aux_arrays: list[np.ndarray] = []
        h, w = label.shape
        for ch_name in self.aux_channels:
            aux_names.append(ch_name)
            if ch_name in data:
                aux_arrays.append(data[ch_name].astype(np.float32))
            else:
                aux_arrays.append(np.zeros((h, w), dtype=np.float32))
        aux_stack = np.stack(aux_arrays) if aux_arrays else None  # (N,H,W)

        # Normalize aux channels (z-score using empirical mean/std)
        if aux_stack is not None and self.config.aux_norm:
            for i, ch_name in enumerate(aux_names):
                if ch_name in self.config.aux_norm:
                    mean, std = self.config.aux_norm[ch_name]
                    aux_stack[i] = (aux_stack[i] - mean) / max(std, 1e-6)

        # Detect tile format: multitemporal if 'multitemporal' key exists
        # or if image has more than 6 bands
        is_mt = bool(data.get("multitemporal", False))

        # Remap 19-class labels to grouped schema if configured
        if self.config.use_grouped_classes:
            label = _LUT_19_TO_10[np.clip(label, 0, 19)].astype(np.int64)

        if is_mt and self.multitemporal:
            # ── Multitemporal path ────────────────────────────────────
            n_frames = int(data.get("num_frames", self.n_frames))
            n_bands = int(data.get("num_bands", self.n_bands))
            temporal_mask = data.get("temporal_mask",
                                     np.ones(n_frames, dtype=np.uint8))
            doy = data.get("doy", np.zeros(n_frames, dtype=np.int32))

            # Normalise each frame: reflectance → DN → Prithvi norm
            # image is (T*6, H, W), mean_mt is (T*6, 1, 1)
            expected_c = n_frames * n_bands
            if image.shape[0] == expected_c:
                mean_mt = np.tile(self.mean, (n_frames, 1, 1))
                std_mt = np.tile(self.std, (n_frames, 1, 1))
                image = (image * 10000.0 - mean_mt) / std_mt
            else:
                # Fallback: normalise per-band assuming 6-band repeat
                image = (image * 10000.0 - self.mean) / self.std

            # Augmentation (spatial only — same transform for all frames)
            if self.augment:
                image, label, aux_stack = self._augment(
                    image, label, aux_stack)
            else:
                image, label, aux_stack = self._center_crop(
                    image, label, aux_stack)

            result = {
                "image": torch.from_numpy(image.copy()),
                "label": torch.from_numpy(label.copy()),
                "temporal_mask": torch.from_numpy(
                    temporal_mask.astype(np.float32)),
                "doy": torch.from_numpy(doy.astype(np.int64)),
                "metadata": {
                    "tile": self.tile_names[idx],
                    "easting": int(data.get("easting", 0)),
                    "northing": int(data.get("northing", 0)),
                    "dates": list(data.get("dates", [])),
                    "num_frames": n_frames,
                },
            }
            # Attach each auxiliary channel as (1, H, W) tensor
            if aux_stack is not None:
                for i, ch_name in enumerate(aux_names):
                    result[ch_name] = torch.from_numpy(
                        aux_stack[i:i+1].copy())  # (1, H, W)
            return result
        else:
            # ── Single-date path (backward compatible) ────────────────
            # Normalise: reflectance → DN → Prithvi normalisation
            if image.shape[0] == 6:
                image = (image * 10000.0 - self.mean) / self.std
            else:
                # Handle unexpected band count gracefully
                image = image * 10000.0

            # Augmentation
            if self.augment:
                image, label, aux_stack = self._augment(
                    image, label, aux_stack)
            else:
                image, label, aux_stack = self._center_crop(
                    image, label, aux_stack)

            result = {
                "image": torch.from_numpy(image.copy()),
                "label": torch.from_numpy(label.copy()),
                "metadata": {
                    "tile": self.tile_names[idx],
                    "easting": int(data.get("easting", 0)),
                    "northing": int(data.get("northing", 0)),
                    "date": str(data.get("date", "")),
                },
            }
            # Attach each auxiliary channel as (1, H, W) tensor
            if aux_stack is not None:
                for i, ch_name in enumerate(aux_names):
                    result[ch_name] = torch.from_numpy(
                        aux_stack[i:i+1].copy())  # (1, H, W)

            # If model expects multitemporal but tile is single-date,
            # replicate the single frame T times (with mask indicating
            # only the first frame is real)
            if self.multitemporal and not is_mt:
                img = result["image"]  # (6, H, W)
                # Repeat T times along channel axis
                img_mt = img.repeat(self.n_frames, 1, 1)  # (T*6, H, W)
                result["image"] = img_mt
                # Only first frame is real
                mask = torch.zeros(self.n_frames, dtype=torch.float32)
                mask[0] = 1.0
                result["temporal_mask"] = mask
                # Approximate DOY: use summer (day 182) for single-date
                doy = torch.zeros(self.n_frames, dtype=torch.int64)
                doy[0] = 182
                result["doy"] = doy

            return result

    def _augment(
        self, image: np.ndarray, label: np.ndarray,
        aux: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Apply data augmentation (training only).

        Works for both (C, H, W) and (T*C, H, W) since all ops
        are on axes 1,2 (spatial dimensions).

        Args:
            image: (C, H, W) or (T*C, H, W) float32
            label: (H, W) int64
            aux: optional (N, H, W) stacked auxiliary channels

        Augmentations:
            - Random crop from fetch_pixels to patch_pixels
            - Random horizontal/vertical flip
            - Random 90-degree rotation
        """
        cfg = self.config

        # Random crop
        if cfg.random_crop and image.shape[1] > cfg.patch_pixels:
            image, label, aux = self._random_crop(image, label, aux)
        else:
            image, label, aux = self._center_crop(image, label, aux)

        # Random flip
        if cfg.random_flip:
            if random.random() > 0.5:
                image = image[:, :, ::-1]  # Horizontal
                label = label[:, ::-1]
                if aux is not None:
                    aux = aux[:, :, ::-1]
            if random.random() > 0.5:
                image = image[:, ::-1, :]  # Vertical
                label = label[::-1, :]
                if aux is not None:
                    aux = aux[:, ::-1, :]

        # Random 90-degree rotation
        if cfg.random_rotation:
            k = random.randint(0, 3)
            if k > 0:
                image = np.rot90(image, k, axes=(1, 2))
                label = np.rot90(label, k, axes=(0, 1))
                if aux is not None:
                    aux = np.rot90(aux, k, axes=(1, 2))

        return image, label, aux

    def _random_crop(
        self, image: np.ndarray, label: np.ndarray,
        aux: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Random crop from (C, H, W) to (C, patch, patch)."""
        p = self.config.patch_pixels
        _, h, w = image.shape
        y = random.randint(0, max(h - p, 0))
        x = random.randint(0, max(w - p, 0))
        a_crop = aux[:, y:y + p, x:x + p] if aux is not None else None
        return image[:, y:y + p, x:x + p], label[y:y + p, x:x + p], a_crop

    def _center_crop(
        self, image: np.ndarray, label: np.ndarray,
        aux: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Center crop from (C, H, W) to (C, patch, patch)."""
        p = self.config.patch_pixels
        _, h, w = image.shape
        y = max((h - p) // 2, 0)
        x = max((w - p) // 2, 0)
        a_crop = aux[:, y:y + p, x:x + p] if aux is not None else None
        return image[:, y:y + p, x:x + p], label[y:y + p, x:x + p], a_crop


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
