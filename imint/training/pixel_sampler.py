"""Stratified pixel sampler for the pixel-context classifier.

Builds a balanced sample index across all training tiles so that:
- Each class occupies approximately 1/num_classes of the drawn samples.
- Rare classes (low tile frequency) are oversampled up to ``max_weight``.
- Boundary pixels (within ``boundary_px`` of a class edge) carry 2× weight.

The sampler pre-scans all tiles once at construction and stores the full
``(tile_path, row, col, class)`` index in memory (negligible RAM: each
entry is ~50 bytes, 6597 tiles × 512 samples/tile = 3.4 M entries ≈ 170 MB).

Usage::

    from imint.training.pixel_sampler import StratifiedPixelSampler

    sampler = StratifiedPixelSampler(tile_paths, samples_per_tile=512)
    print(sampler.class_distribution())
    # Attach to PixelContextDataset:
    from imint.training.pixel_dataset import PixelContextDataset
    ds = PixelContextDataset.from_sampler(tile_paths, sampler)
"""
from __future__ import annotations

import math
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np

from .unified_schema import NUM_UNIFIED_CLASSES, UNIFIED_CLASS_NAMES

IGNORE_CLASS = 0


class StratifiedPixelSampler:
    """Global stratified pixel sampler across all training tiles.

    Scans all tiles in parallel, collects valid pixels, and builds a
    class-balanced sample index.  Designed for use with ``PixelContextDataset``.

    Args:
        tile_paths: All training tile .npz paths.
        samples_per_tile: Target samples per tile (before rebalancing).
        target_class_frac: Desired fraction per class (default: uniform =
            1 / num_classes ≈ 4.35 %).
        max_weight: Maximum per-class oversampling multiplier.
        boundary_px: Pixels within this distance of a class boundary get
            ``boundary_weight`` × their normal weight.
        boundary_weight: Sampling multiplier at class boundaries.
        num_scan_workers: Threads used during tile scanning.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        tile_paths: Sequence[str | Path],
        *,
        samples_per_tile: int = 512,
        target_class_frac: float | None = None,
        max_weight: float = 5.0,
        boundary_px: int = 3,
        boundary_weight: float = 2.0,
        num_scan_workers: int = 32,
        seed: int = 42,
    ) -> None:
        self.samples_per_tile = samples_per_tile
        self.target_class_frac = target_class_frac or 1.0 / (NUM_UNIFIED_CLASSES - 1)
        self.max_weight = max_weight
        self.boundary_px = boundary_px
        self.boundary_weight = boundary_weight
        self.seed = seed

        # Index: list of (path_str, row, col, class_id)
        self._index: list[tuple[str, int, int, int]] = []
        self._class_counts: Counter = Counter()
        self._lock = threading.Lock()

        print(f"StratifiedPixelSampler: scanning {len(tile_paths):,} tiles "
              f"({num_scan_workers} workers) …")

        rng = np.random.default_rng(seed)
        tile_seeds = rng.integers(0, 2**31, size=len(tile_paths))

        with ThreadPoolExecutor(max_workers=num_scan_workers) as pool:
            futs = {
                pool.submit(self._scan_tile, str(p), int(s), samples_per_tile): p
                for p, s in zip(tile_paths, tile_seeds)
            }
            n_done = 0
            for fut in as_completed(futs):
                result = fut.result()
                if result:
                    with self._lock:
                        self._index.extend(result)
                        for _, _, _, cls in result:
                            self._class_counts[cls] += 1
                n_done += 1
                if n_done % 500 == 0:
                    print(f"  scanned {n_done:,}/{len(tile_paths):,} tiles …")

        rng.shuffle(self._index)
        print(f"  total samples: {len(self._index):,}")
        print(f"  classes represented: "
              f"{sum(1 for c, v in self._class_counts.items() if c != 0 and v > 0)}"
              f"/{NUM_UNIFIED_CLASSES - 1}")

    # ── Tile scanner ──────────────────────────────────────────────────

    def _scan_tile(
        self,
        path: str,
        tile_seed: int,
        samples_per_tile: int,
    ) -> list[tuple[str, int, int, int]] | None:
        try:
            data = np.load(path, allow_pickle=False)
        except Exception:
            return None

        label = data.get("label")
        if label is None:
            return None
        label = np.asarray(label).astype(np.int32)
        H, W = label.shape

        BORDER = 16  # keep 16px away from tile edge (context window half)
        valid_rows, valid_cols = np.where(
            label[BORDER:H - BORDER, BORDER:W - BORDER] != IGNORE_CLASS
        )
        valid_rows = valid_rows + BORDER
        valid_cols = valid_cols + BORDER

        if len(valid_rows) == 0:
            return None

        classes = label[valid_rows, valid_cols].astype(np.int32)

        # Per-pixel weights: class inverse-frequency within this tile
        counts = np.bincount(classes, minlength=NUM_UNIFIED_CLASSES).astype(np.float64)
        total = max(counts.sum(), 1.0)
        weights = np.ones(len(classes), dtype=np.float64)

        for i, cls in enumerate(classes):
            if cls == IGNORE_CLASS or counts[cls] == 0:
                weights[i] = 0.0
            else:
                freq = counts[cls] / total
                # Inverse-frequency capped at max_weight
                w = min(self.max_weight, (self.target_class_frac / freq))
                weights[i] = max(1.0, w)

        # Boundary bonus
        if self.boundary_px > 0 and self.boundary_weight > 1.0:
            bm = _boundary_mask(label, self.boundary_px)
            weights[bm[valid_rows, valid_cols]] *= self.boundary_weight

        # Normalise
        w_sum = weights.sum()
        if w_sum <= 0:
            return None
        weights /= w_sum

        n_draw = min(samples_per_tile, len(valid_rows))
        rng = np.random.default_rng(tile_seed)
        chosen = rng.choice(len(valid_rows), size=n_draw, replace=False, p=weights)

        result = []
        for idx in chosen:
            cls = int(classes[idx])
            if cls == IGNORE_CLASS:
                continue
            result.append((path, int(valid_rows[idx]), int(valid_cols[idx]), cls))

        return result

    # ── Public API ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[str, int, int, int]:
        return self._index[idx]

    def class_distribution(self) -> dict[str, float]:
        """Return per-class percentage of samples in the index."""
        total = max(sum(self._class_counts.values()), 1)
        return {
            UNIFIED_CLASS_NAMES[cls]: 100.0 * count / total
            for cls, count in sorted(self._class_counts.items())
            if cls != IGNORE_CLASS
        }

    def print_class_distribution(self) -> None:
        dist = self.class_distribution()
        print("Class distribution in sample index:")
        for name, pct in dist.items():
            bar = "█" * int(pct / 0.5)
            print(f"  {name:20s} {pct:5.1f}%  {bar}")


# ── Helpers ───────────────────────────────────────────────────────────────

def _boundary_mask(label: np.ndarray, boundary_px: int) -> np.ndarray:
    """Boolean mask True at pixels within boundary_px of a class boundary."""
    from scipy.ndimage import binary_dilation

    H, W = label.shape
    boundary = np.zeros((H, W), dtype=bool)
    struct = np.ones((3, 3), dtype=bool)

    unique = np.unique(label)
    for cls in unique:
        if cls == IGNORE_CLASS:
            continue
        cls_mask = label == cls
        dilated = binary_dilation(cls_mask, structure=struct, iterations=boundary_px)
        boundary |= (dilated & ~cls_mask)

    return boundary
