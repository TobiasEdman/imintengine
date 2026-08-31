"""scripts/cache_predict.py — cache-backed predict_fn factory.

The bridge between Stage A (``infer_tiles.py`` — GPU, batched, writes a per-tile
prediction cache) and Stage B (``score_against_truth.py`` — CPU, reuses the
EXISTING scoring cores). A cache-backed ``predict_fn`` reads a tile's cached
``.npz`` (``pred``/``probs``/``fracs``) and returns arrays with EXACTLY the
contract the existing scorers call:

  * NFI / LUCAS-hard: ``predict_fn(tile_path) -> (class_map (H,W) int,
    probs (C,H,W) float32)``
  * LUCAS-fraction:   ``predict_fn(tile_path) -> (class_map, probs, fracs
    (4,H,W) float32)``

so ``score_against_nfi`` / ``score_against_lucas`` run unchanged on the cache.
The cache stores ``probs``/``fracs`` as float16 (Stage A) for size; here they
are widened back to float32 to match what ``run_inference`` /
``run_fraction_inference`` return (bit-parity is asserted to float16 tolerance
in ``tests/test_cached_validation_parity.py``).

The cache key is ``<cache-dir>/<ckpt_sha>/<tile_stem>.npz`` — the same layout
``infer_tiles.py`` writes, so a checkpoint's predictions are paid once and
scored by every truth source.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def cache_path_for(cache_dir: str | Path, ckpt_sha: str, tile_path: str | Path) -> Path:
    """Cache path for a tile: ``<cache-dir>/<ckpt_sha>/<tile_stem>.npz``.

    ``tile_path`` may be a full ``.../foo.npz`` path or a bare stem; only the
    stem is used so the key is dataset-location-independent.
    """
    stem = Path(str(tile_path)).stem
    return Path(cache_dir) / ckpt_sha / f"{stem}.npz"


def make_cached_predict_fn(
    cache_dir: str | Path, ckpt_sha: str, *, want_fracs: bool
):
    """Return a ``predict_fn(tile_path)`` reading the Stage-A prediction cache.

    Args:
        cache_dir: root of the prediction cache (``infer_tiles.py --cache-dir``).
        ckpt_sha: the 16-hex checkpoint digest naming the per-checkpoint subdir.
        want_fracs: if True, return the LUCAS-fraction 3-tuple
            ``(class_map, probs, fracs)``; else the NFI/LUCAS-hard 2-tuple
            ``(class_map, probs)``.

    Returns:
        A callable with the exact predict_fn contract the existing scorers use.
        Raises ``FileNotFoundError`` if a requested tile is not in the cache —
        never silently skips (a missing tile would otherwise corrupt the score).
    """
    cache_dir = Path(cache_dir)

    def predict_fn(tile_path):
        cpath = cache_path_for(cache_dir, ckpt_sha, tile_path)
        if not cpath.exists():
            raise FileNotFoundError(
                f"prediction cache miss for tile '{Path(str(tile_path)).stem}' "
                f"at {cpath} — run infer_tiles.py for checkpoint sha {ckpt_sha} "
                f"over this tile set first (never silently skipped)."
            )
        with np.load(cpath) as z:
            # argmax class map: stored uint8, widened to int64 to match
            # probs.argmax(0).astype(np.int64) from the direct path.
            class_map = z["pred"].astype(np.int64)
            probs = z["probs"].astype(np.float32)
            if want_fracs:
                fracs = z["fracs"].astype(np.float32)
                return class_map, probs, fracs
            return class_map, probs

    return predict_fn
