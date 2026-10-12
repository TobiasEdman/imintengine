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
``infer_tiles.py`` writes, with input and inference identity stored inside
each atomic NPZ. Legacy
entries without this metadata must be regenerated.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
import zlib
from pathlib import Path

import numpy as np

from imint._cache_identity import producer_revision, record_caller

record_caller()


def cache_path_for(cache_dir: str | Path, ckpt_sha: str, tile_path: str | Path) -> Path:
    """Cache path for a tile: ``<cache-dir>/<ckpt_sha>/<tile_stem>.npz``.

    ``tile_path`` may be a full ``.../foo.npz`` path or a bare stem; only the
    stem is used for layout compatibility. Entry metadata binds the full input
    path and bytes; this filename alone never authorizes reuse.
    """
    stem = Path(str(tile_path)).stem
    return Path(cache_dir) / ckpt_sha / f"{stem}.npz"


class CacheValidationError(ValueError):
    """The cache is stale, incomplete, or lacks authenticated provenance."""


def file_identity(path):
    """Hash a stable regular file using the inference loader's identity guards."""
    from scripts import inference_comparison as ic

    path = Path(path).absolute()
    fd, opened = ic._open_regular_nofollow(
        path, error_type=CacheValidationError, label="cache input")
    with os.fdopen(fd, "rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
        ic._assert_open_file_unchanged(
            stream.fileno(), path, opened,
            error_type=CacheValidationError, label="cache input")
    return {"path": str(path), "size": opened.st_size,
            "sha256": digest.hexdigest()}


def preprocessing_revision():
    """Identity of the captured and verified producer source/environment."""
    return producer_revision()


def atomic_write(path, write):
    """Publish a complete file with one replace; clean up failed writes."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_cache_entry(path, *, source, inference):
    """Validate provenance and every array before skip or scoring."""
    try:
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"].item()))
            if metadata != {"version": 1, "source": source, "inference": inference}:
                raise ValueError("input or inference identity mismatch")
            pred, probs, fracs = (z[k] for k in ("pred", "probs", "fracs"))
        if (pred.ndim != 2 or min(pred.shape) <= 0
                or max(pred.shape) > inference["img_size"]
                or pred.shape[0] != pred.shape[1]
                or probs.shape != (inference["num_classes"], *pred.shape)
                or fracs.shape != (4, *pred.shape)
                or pred.dtype != np.uint8 or probs.dtype != np.float16
                or fracs.dtype != np.float16
                or np.any(pred >= inference["num_classes"])
                or not np.isfinite(probs).all() or not np.isfinite(fracs).all()
                or np.any((probs < 0) | (probs > 1))
                or np.any((fracs < 0) | (fracs > 1))
                or not np.allclose(probs.sum(axis=0), 1, atol=0.002)):
            raise ValueError("invalid cache arrays")
        return pred, probs, fracs
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile, zlib.error) as exc:
        raise CacheValidationError(
            f"invalid prediction cache {path}: {exc}; rerun infer_tiles.py") from exc


def make_cached_predict_fn(
    cache_dir: str | Path, ckpt_sha: str, *, want_fracs: bool,
    img_size: int | None = None, num_classes: int | None = None,
):
    """Read verified Stage-A entries; legacy caches must be regenerated.

    Pin the manifest's inference identity when creating the predictor, then
    check each entry and current source bytes on every read. Scorers must pass
    their crop/class settings so incompatible scoring coordinates fail closed.
    """
    cache_dir = Path(cache_dir)
    try:
        manifest = json.loads((cache_dir / ckpt_sha / "MANIFEST.json").read_text())
        inference = manifest["inference"]
        if (inference["checkpoint"]["sha256"][:16] != ckpt_sha
                or inference["preprocessing"] != preprocessing_revision()
                or (img_size is not None and inference["img_size"] != img_size)
                or (num_classes is not None and inference["num_classes"] != num_classes)):
            raise ValueError("checkpoint, preprocessing or scoring configuration mismatch")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CacheValidationError(
            f"invalid prediction cache manifest: {exc}; rerun infer_tiles.py") from exc

    def predict_fn(tile_path):
        producer_revision(expected=inference["preprocessing"])
        cpath = cache_path_for(cache_dir, ckpt_sha, tile_path)
        if not cpath.exists():
            raise FileNotFoundError(f"prediction cache miss at {cpath}; run infer_tiles.py")
        pred, probs, fracs = read_cache_entry(
            cpath, source=file_identity(tile_path), inference=inference)
        result = (pred.astype(np.int64), probs.astype(np.float32))
        return (*result, fracs.astype(np.float32)) if want_fracs else result

    return predict_fn
