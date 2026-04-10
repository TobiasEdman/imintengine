"""Pixel-context dataset for center-pixel classification.

Each sample is a (T×6, context_px, context_px) spectral context patch
centered on a labelled pixel.  The label is the unified 23-class integer
at the center pixel.

Frame layout (T=5 when frame_2016 present, T=4 otherwise):
    0 : 2016 summer (background anchor)
    1 : autumn year-1  (Sep-Oct from previous year)
    2 : growing-season frame 1 (VPP-guided spring/early)
    3 : growing-season frame 2 (VPP-guided peak-summer)
    4 : growing-season frame 3 (VPP-guided harvest)

If ``has_frame_2016 == 0`` or the tile lacks the key, T falls back to 4
and the caller is expected to build the model with matching ``num_frames``.

When ``enable_aux=True`` (default), ``__getitem__`` returns a 3-tuple
``(patch, aux_vec, label)`` where ``aux_vec`` is a ``(N_AUX,)`` float32
vector of center-pixel auxiliary values (height, volume, DEM, VPP, …)
normalized with the same log+z-score pipeline as UnifiedDataset.

Usage::

    from imint.training.pixel_dataset import PixelContextDataset

    tile_paths = list(Path("/data/unified_v2").glob("*.npz"))
    ds = PixelContextDataset(tile_paths, context_px=32, split="train")
    patch, aux, label = ds[0]   # patch: (30,32,32), aux: (11,), label: int64

    # Without AUX (legacy 2-tuple):
    ds_no_aux = PixelContextDataset(tile_paths, enable_aux=False)
    patch, label = ds_no_aux[0]
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from .unified_schema import NUM_UNIFIED_CLASSES
from .unified_dataset import (
    AUX_CHANNEL_NAMES,
    AUX_LOG_TRANSFORM,
    AUX_NORM,
)

# ── Constants ────────────────────────────────────────────────────────────

N_BANDS = 6
IGNORE_CLASS = 0   # background — excluded from training
N_AUX = len(AUX_CHANNEL_NAMES)  # 11 auxiliary channels


# ── Dataset ───────────────────────────────────────────────────────────────

class PixelContextDataset:
    """Center-pixel classification dataset.

    Draws stratified pixel samples from a collection of .npz tiles.
    At construction time, a sample index is built (tile_path, row, col,
    class) and stored in memory.  During iteration the patch is sliced
    on the fly from the tile's spectral array.

    Args:
        tile_paths: List of .npz file paths.
        context_px: Side length of the context window (must be even).
        split: One of ``"train"``, ``"val"``, ``"test"``.
        samples_per_tile: Number of pixel samples drawn per tile.
        use_frame_2016: Prepend ``frame_2016`` as T=0 if available.
        enable_aux: Return center-pixel AUX vector as second element
            of each item (default: True).  Set False for legacy
            2-tuple ``(patch, label)`` output.
        oversample_rare: Weight rare classes (id < 10 AND id > 10 but
            small frequency) with ``rare_weight`` vs 1.0 for common.
        rare_weight: Sampling multiplier for rare classes.
        transform: Optional callable applied to the patch tensor.
    """

    def __init__(
        self,
        tile_paths: Sequence[str | Path],
        *,
        context_px: int = 32,
        split: str = "train",
        samples_per_tile: int = 512,
        use_frame_2016: bool = True,
        enable_aux: bool = True,
        oversample_rare: bool = True,
        rare_weight: float = 3.0,
        boundary_weight: float = 2.0,
        boundary_px: int = 3,
        transform=None,
        seed: int = 42,
    ) -> None:
        assert context_px % 2 == 0, "context_px must be even"
        self.context_px = context_px
        self.half = context_px // 2
        self.split = split
        self.use_frame_2016 = use_frame_2016
        self.enable_aux = enable_aux
        self.transform = transform

        rng = np.random.default_rng(seed + {"train": 0, "val": 1, "test": 2}.get(split, 0))

        self._index: list[tuple[str, int, int, int]] = []  # (path, row, col, class)

        for path in tile_paths:
            path = str(path)
            try:
                data = np.load(path, allow_pickle=False)
            except Exception:
                continue

            label = data.get("label")
            if label is None:
                continue
            label = np.asarray(label).astype(np.int32)
            H, W = label.shape

            # Valid pixels: must have a known class and lie inside the border
            h = self.half
            valid_rows, valid_cols = np.where(
                (label[h:H - h, h:W - h] != IGNORE_CLASS)
            )
            valid_rows = valid_rows + h
            valid_cols = valid_cols + h

            if len(valid_rows) == 0:
                continue

            # Compute per-pixel sampling weights
            classes = label[valid_rows, valid_cols].astype(np.int32)
            weights = np.ones(len(classes), dtype=np.float64)

            if oversample_rare:
                # Count class frequencies in this tile
                counts = np.bincount(classes, minlength=NUM_UNIFIED_CLASSES)
                total = max(counts.sum(), 1)
                for i, cls in enumerate(classes):
                    if cls == IGNORE_CLASS or counts[cls] == 0:
                        weights[i] = 0.0
                    else:
                        freq = counts[cls] / total
                        if freq < 0.02:
                            weights[i] = rare_weight
                        elif freq < 0.05:
                            weights[i] = max(1.0, rare_weight * 0.5)

            if boundary_weight > 1.0:
                # Boundary pixels (within boundary_px of a class edge)
                boundary_mask = _compute_boundary_mask(label, boundary_px)
                on_boundary = boundary_mask[valid_rows, valid_cols]
                weights[on_boundary] *= boundary_weight

            # Normalise and cap total weight sum
            weights = np.clip(weights, 0.0, None)
            w_sum = weights.sum()
            if w_sum == 0:
                continue
            weights /= w_sum

            # Stratified draw
            n_draw = min(samples_per_tile, len(valid_rows))
            chosen = rng.choice(len(valid_rows), size=n_draw, replace=False, p=weights)

            for idx in chosen:
                r, c, cls = int(valid_rows[idx]), int(valid_cols[idx]), int(classes[idx])
                if cls == IGNORE_CLASS:
                    continue
                self._index.append((path, r, c, cls))

        # Shuffle (especially important for val/test where no augmentation)
        rng.shuffle(self._index)

    # ------------------------------------------------------------------
    # PyTorch Dataset protocol (also works without torch as plain iter)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        path, row, col, cls = self._index[idx]

        data = np.load(path, allow_pickle=False)
        patch = self._build_patch(data, row, col)

        if self.transform is not None:
            patch = self.transform(patch)

        if self.enable_aux:
            aux_vec = self._build_aux_vector(data, row, col)

        if _TORCH_AVAILABLE:
            import torch
            # torch.tensor() always allocates PyTorch-owned storage (a full copy).
            # torch.from_numpy() shares numpy memory, which DataLoader's pin_memory
            # collator cannot resize → "Trying to resize storage that is not resizable".
            patch_t = torch.tensor(patch, dtype=torch.float32)
            label_t = torch.tensor(cls, dtype=torch.long)
            if self.enable_aux:
                return patch_t, torch.tensor(aux_vec, dtype=torch.float32), label_t
            return patch_t, label_t

        if self.enable_aux:
            return patch, aux_vec, cls
        return patch, cls

    # ------------------------------------------------------------------
    # Patch construction
    # ------------------------------------------------------------------

    def _build_patch(
        self,
        data: np.lib.npyio.NpzFile,
        row: int,
        col: int,
    ) -> np.ndarray:
        """Extract a (T*6, context_px, context_px) spectral context patch."""
        h = self.half
        r0, r1 = row - h, row + h
        c0, c1 = col - h, col + h

        spectral = data.get("spectral")
        if spectral is None:
            spectral = data.get("image")
        spectral = np.asarray(spectral, dtype=np.float32)  # (T_base*6, H, W)

        patch_base = spectral[:, r0:r1, c0:c1]  # (T_base*6, ctx, ctx)

        if self.use_frame_2016:
            if int(data.get("has_frame_2016", 0)) == 1:
                frame_2016 = np.asarray(data["frame_2016"], dtype=np.float32)
                # frame_2016 shape: (6, H, W)
                patch_bg = frame_2016[:, r0:r1, c0:c1]  # (6, ctx, ctx)
                patch = np.concatenate([patch_bg, patch_base], axis=0)
            else:
                # Tile lacks a 2016 background frame — pad with zeros so that
                # all patches have the same (T*6, ctx, ctx) shape regardless
                # of which tiles have the frame.  This is required for
                # torch.stack() inside DataLoader workers (mixed batches of
                # tiles with and without frame_2016 would produce tensors of
                # different channel counts and crash collation).
                ctx_h = r1 - r0
                ctx_w = c1 - c0
                zeros = np.zeros((N_BANDS, ctx_h, ctx_w), dtype=np.float32)
                patch = np.concatenate([zeros, patch_base], axis=0)
        else:
            patch = patch_base

        return patch  # (T*6, context_px, context_px)

    def _build_aux_vector(
        self,
        data: "np.lib.npyio.NpzFile",
        row: int,
        col: int,
    ) -> np.ndarray:
        """Extract center-pixel auxiliary values as a (N_AUX,) float32 vector.

        Applies the same log(1+x) pre-transform and z-score normalization
        as ``UnifiedDataset._load_aux_channels()``.  Missing channels are
        zero-filled before normalization (→ yields the channel mean after
        subtraction, i.e. ≈ 0 in normalized space).

        Args:
            data: Loaded .npz tile.
            row: Center pixel row index.
            col: Center pixel column index.

        Returns:
            (N_AUX,) float32 normalized auxiliary vector.
        """
        aux_vec = np.zeros(N_AUX, dtype=np.float32)
        for i, ch_name in enumerate(AUX_CHANNEL_NAMES):
            if ch_name in data:
                arr = np.asarray(data[ch_name], dtype=np.float32)
                # Clamp to valid range in case of slight shape mismatch
                r = min(row, arr.shape[0] - 1)
                c = min(col, arr.shape[1] - 1)
                val = float(arr[r, c])
            else:
                val = 0.0  # missing channel — will normalize to ~0

            if ch_name in AUX_LOG_TRANSFORM:
                val = float(np.log1p(max(val, 0.0)))
            if ch_name in AUX_NORM:
                mean, std = AUX_NORM[ch_name]
                val = (val - mean) / max(std, 1e-6)

            aux_vec[i] = val
        return aux_vec


# ── Utilities ─────────────────────────────────────────────────────────────

def _compute_boundary_mask(label: np.ndarray, boundary_px: int) -> np.ndarray:
    """Return boolean mask True for pixels within boundary_px of a class edge."""
    from scipy.ndimage import binary_dilation, binary_erosion

    H, W = label.shape
    boundary = np.zeros((H, W), dtype=bool)
    struct = np.ones((3, 3), dtype=bool)

    # For each pair of adjacent labels, the boundary between them
    # Efficient: dilate each class mask and check if it touches another
    unique_classes = np.unique(label)
    for cls in unique_classes:
        if cls == IGNORE_CLASS:
            continue
        cls_mask = label == cls
        dilated = binary_dilation(cls_mask, structure=struct, iterations=boundary_px)
        boundary |= (dilated & ~cls_mask)

    return boundary


# ── If torch is available, expose as Dataset subclass ────────────────────

if _TORCH_AVAILABLE:
    from torch.utils.data import Dataset

    class PixelContextDatasetTorch(PixelContextDataset, Dataset):
        """PixelContextDataset exposed as a ``torch.utils.data.Dataset``."""
        pass

    # Alias for convenience
    PixelContextDataset = PixelContextDatasetTorch  # type: ignore[misc]
