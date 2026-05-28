"""Baseline models to compare the ensemble against.

A multi-temporal Prithvi-EO-2.0 ensemble that beats nothing isn't a
win. This module owns four baselines that bound the difficulty of the
task from below:

    1. **trivial_majority**: per-class majority predictor (floor).
    2. **random_forest_bands**: scikit-learn RF on the 6 reflectance
       bands + DEM per pixel (no temporal context).
    3. **single_frame_prithvi**: same Prithvi-EO-2.0 backbone fine-tuned
       on just one frame per tile (slot 2 = peak summer typically).
       Tests whether multi-temporal actually helps.
    4. **prithvi_zero_shot**: Prithvi-EO-2.0 with the pre-trained head
       (no fine-tune). Floor for transfer learning.

Each baseline has a ``train(...)`` + ``predict(...)`` pair so the eval
loop can call them uniformly. Implementations are skeletons today.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class Baseline(ABC):
    """Uniform interface for any baseline model.

    Subclasses must implement ``train`` and ``predict``. State is held
    on the instance so the caller can reuse a trained baseline across
    evaluation splits.
    """

    name: str

    @abstractmethod
    def train(self, train_tiles: list[str], tiles_dir: str) -> None:
        ...

    @abstractmethod
    def predict(self, tile_name: str, tiles_dir: str) -> np.ndarray:
        """Return predicted class indices (H, W) for a single tile."""
        ...

    def save(self, path: Path) -> None:
        """Optional — for baselines that take non-trivial time to train."""
        raise NotImplementedError(f"{self.name}: save not implemented")


class TrivialMajority(Baseline):
    """Predicts the GLOBAL majority class for every pixel of every tile.

    Useless as a model but valuable as a floor: any segmentation model
    that doesn't beat this is just shifting class probabilities. We
    compute the mode across the entire training-split label population
    excluding ``ignore_index=0`` (background) so the baseline picks
    the most-common *foreground* class — typically a forest class on
    Swedish data.
    """

    name = "trivial_majority"

    def __init__(self, ignore_index: int = 0):
        self.ignore_index = ignore_index
        self.global_majority: int | None = None

    def train(self, train_tiles, tiles_dir):
        """Streaming bincount over training labels — never holds more
        than one tile's labels in memory at a time."""
        import os
        import numpy as np

        accumulator: np.ndarray | None = None
        scanned = 0
        for tile_name in train_tiles:
            path = os.path.join(tiles_dir, f"{tile_name}.npz")
            if not os.path.exists(path):
                continue
            try:
                with np.load(path, allow_pickle=True) as data:
                    if "label" not in data.files:
                        continue
                    label = np.asarray(data["label"]).astype(np.int64)
            except Exception:
                continue
            valid = label != self.ignore_index
            if not valid.any():
                continue
            counts = np.bincount(label[valid].ravel())
            if accumulator is None:
                accumulator = counts.astype(np.int64)
            elif counts.size > accumulator.size:
                counts[:accumulator.size] += accumulator
                accumulator = counts.astype(np.int64)
            else:
                accumulator[:counts.size] += counts
            scanned += 1

        if accumulator is None or accumulator.sum() == 0:
            raise RuntimeError(
                "TrivialMajority.train: no labelled pixels found in "
                f"{len(train_tiles)} train tiles (scanned {scanned})"
            )
        self.global_majority = int(accumulator.argmax())

    def predict(self, tile_name, tiles_dir):
        if self.global_majority is None:
            raise RuntimeError("TrivialMajority: call .train() first")
        import os
        import numpy as np
        path = os.path.join(tiles_dir, f"{tile_name}.npz")
        # Read label only to learn the output shape — labels were the
        # ground-truth ground-truth in training, but at prediction
        # time we only need ``label.shape``.
        with np.load(path, allow_pickle=True) as data:
            shape = data["label"].shape
        return np.full(shape, self.global_majority, dtype=np.int64)


class RandomForestBands(Baseline):
    """sklearn RandomForestClassifier on per-pixel band stack.

    Inputs: 6 Prithvi bands × 4 frames = 24 features + DEM = 25. No
    spatial context. Fast to train, hard to beat on majority classes,
    very poor on rare ones — that asymmetry is the point.
    """

    name = "random_forest_bands"

    def __init__(self, *, n_estimators: int = 100, max_depth: int = 20):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None

    def train(self, train_tiles, tiles_dir):
        raise NotImplementedError

    def predict(self, tile_name, tiles_dir):
        raise NotImplementedError


class SingleFramePrithvi(Baseline):
    """Prithvi-EO-2.0 fine-tuned on a single frame (slot 2 by default).

    The critical "is multi-temporal worth it?" comparison. Uses the same
    backbone + head architecture as the ensemble, just with
    ``num_temporal_frames=1``. Re-uses the project's training script
    with a one-frame config.
    """

    name = "single_frame_prithvi"

    def __init__(self, frame_idx: int = 2):
        self.frame_idx = frame_idx
        self.checkpoint_path: Path | None = None

    def train(self, train_tiles, tiles_dir):
        # TODO: invoke imint/training/train_unified.py with custom config.
        raise NotImplementedError

    def predict(self, tile_name, tiles_dir):
        raise NotImplementedError


class PrithviZeroShot(Baseline):
    """Pre-trained Prithvi-EO-2.0 with no fine-tuning.

    Establishes the transfer-learning floor — what does the backbone
    know about Sweden out of the box? Expected to be near-zero on our
    23-class schema (Prithvi has its own pretraining vocab) so we
    treat it as a feature-quality probe rather than a direct
    classifier: freeze backbone, fit a linear head on a tiny labelled
    set, measure how separable the embeddings are.
    """

    name = "prithvi_zero_shot"

    def __init__(self, num_probe_tiles: int = 50):
        self.num_probe_tiles = num_probe_tiles
        self.linear_head = None

    def train(self, train_tiles, tiles_dir):
        raise NotImplementedError

    def predict(self, tile_name, tiles_dir):
        raise NotImplementedError


# ── Registry for the eval orchestrator ──────────────────────────────────────


BASELINES: dict[str, type[Baseline]] = {
    "trivial_majority":  TrivialMajority,
    "random_forest":     RandomForestBands,
    "single_frame":      SingleFramePrithvi,
    "zero_shot":         PrithviZeroShot,
}
