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
import contextlib
import zipfile
import zlib
from pathlib import Path


from imint._cache_identity import producer_revision
import numpy as np


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


def identity_differences(actual, expected, prefix=''):
    """Stable field names, without printing sensitive values or file contents."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        result = []
        for key in sorted(actual.keys() | expected.keys()):
            name = f'{prefix}.{key}' if prefix else key
            if key not in actual or key not in expected:
                result.append(name)
            else:
                result.extend(identity_differences(actual[key], expected[key], name))
        return result
    return [] if actual == expected else [prefix]


def contract_pin(inference):
    return hashlib.sha256(json.dumps(inference, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@contextlib.contextmanager
def cache_publication_lock(directory):
    """Serialize manifest decisions/publications, not inference or data loading."""
    import fcntl
    with (Path(directory) / '.publication.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read_cache_entry(path, *, source, inference):
    """Validate provenance and every array before skip or scoring."""
    try:
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"].item()))
            differences = identity_differences(metadata, {
                "version": 2, "source": source, "inference": inference})
            if differences:
                raise ValueError("identity mismatch fields: " + ", ".join(differences))
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
            f"invalid prediction cache {path}: {exc}; cache refused; "
            "inspect before explicitly authorizing infer_tiles.py --regenerate-invalid") from exc


def write_input_manifest(directory, identities):
    """Content-addressed receipt; NOT proof the underlying dataset is immutable."""
    payload = json.dumps({'version': 1, 'inputs': {i['path']: i for i in identities}},
                         sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(payload).hexdigest()
    path = Path(directory) / f'INPUTS-{digest}.json'
    atomic_write(path, lambda stream: stream.write(payload))
    return path, digest


def _immutable_inputs(path, digest, root, assume_immutable):
    if not all((path, digest, root, assume_immutable)):
        raise CacheValidationError('immutable mode requires input manifest, externally pinned sha256, '
                                   'snapshot root and explicit assume_immutable_inputs=True')
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise CacheValidationError('immutable input manifest sha256 mismatch')
    manifest = json.loads(payload)
    if manifest.get('version') != 1 or not isinstance(manifest.get('inputs'), dict):
        raise CacheValidationError('invalid immutable input manifest schema')
    root = Path(root).absolute()
    if root.resolve() != root:
        raise CacheValidationError('immutable input root must not traverse symlinks')
    return manifest['inputs'], root


def make_cached_predict_fn(
    cache_dir: str | Path, ckpt_sha: str, *, want_fracs: bool,
    img_size: int | None = None, num_classes: int | None = None,
    producer_pin: str | None = None,
    input_manifest=None, input_manifest_sha256=None,
    immutable_input_root=None, assume_immutable_inputs=False,
):
    """Read-only Stage B: recorded producer contract, independent reader runtime.

    Strict default authenticates tile bytes on EVERY read (no mtime memo).
    The optional cheap path requires an operator-guaranteed immutable snapshot
    and an externally supplied receipt digest. A receipt/stat is not immutability.
    """
    cache_dir = Path(cache_dir)
    try:
        manifest = json.loads((cache_dir / ckpt_sha / 'MANIFEST.json').read_text())
        inference = manifest['inference']
        if manifest.get('cache_version') != 2:
            raise ValueError('cache_version: legacy cache requires an explicit migration decision')
        if producer_pin is None or contract_pin(inference) != producer_pin:
            raise ValueError('producer_pin: supply the independently recorded Stage-A contract pin')
        if inference['checkpoint']['sha256'][:16] != ckpt_sha:
            raise ValueError('checkpoint.sha256 namespace mismatch')
        if img_size is not None and inference['img_size'] != img_size:
            raise ValueError('inference.img_size mismatch')
        if num_classes is not None and inference['num_classes'] != num_classes:
            raise ValueError('inference.num_classes mismatch')
        from imint._cache_identity import source_revision
        if source_revision(manifest['producer_sources']) != inference['preprocessing']:
            raise ValueError('producer_sources digest mismatch')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CacheValidationError(f'invalid prediction cache manifest: {exc}; no inference performed') from exc
    sealed = None
    if any((input_manifest, input_manifest_sha256, immutable_input_root, assume_immutable_inputs)):
        try:
            sealed = _immutable_inputs(input_manifest, input_manifest_sha256,
                                       immutable_input_root, assume_immutable_inputs)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CacheValidationError(f'immutable input contract refused: {exc}') from exc
    print('[cache_predict] input validation: ' +
          ('operator-guaranteed immutable snapshot (no tile payload hash)' if sealed else
           'strict SHA-256 of every source tile on every read'))

    def predict_fn(tile_path):
        cpath = cache_path_for(cache_dir, ckpt_sha, tile_path)
        if not cpath.exists():
            raise CacheValidationError(f'prediction cache miss at {cpath}; no inference performed')
        if sealed is None:
            source = file_identity(tile_path)
        else:
            inputs, root = sealed
            path = Path(tile_path).absolute()
            if not path.is_relative_to(root) or path.resolve() != path:
                raise CacheValidationError(f'input.path is outside immutable snapshot or symlinked: {path}')
            source = inputs.get(str(path))
            if source is None or source.get('path') != str(path):
                raise CacheValidationError(f'input.path absent from pinned immutable manifest: {path}')
            # Extra consistency checks, not a claim that stats authenticate bytes.
            from scripts import inference_comparison as ic
            fd, opened = ic._open_regular_nofollow(path, error_type=CacheValidationError,
                                                   label='immutable cache input')
            os.close(fd)
            if source.get('size') != opened.st_size:
                raise CacheValidationError(f'input.size differs from immutable manifest: {path}')
        pred, probs, fracs = read_cache_entry(cpath, source=source, inference=inference)
        result = (pred.astype(np.int64), probs.astype(np.float32))
        return (*result, fracs.astype(np.float32)) if want_fracs else result

    return predict_fn
