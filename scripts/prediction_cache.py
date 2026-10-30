"""Content-bound prediction caches shared by inference and CPU scoring.

MANIFEST.json fixes the complete input set and recipe. Each atomically published
tile has a JSON receipt binding the exact output bytes. Missing receipts are
incomplete work, never evidence for a successful cache hit.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import tempfile
from pathlib import Path

import numpy as np

from scripts import inference_comparison as inference

SCHEMA = "imint-prediction-cache-v2"
SHA256 = re.compile(r"[0-9a-f]{64}")


class CacheError(RuntimeError):
    """A cache is incomplete, changed, or incompatible with its contract."""


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def contract_key(contract: dict) -> str:
    return hashlib.sha256(canonical(contract)).hexdigest()


def file_identity(path: str | Path) -> dict:
    """Hash a stable regular file; reject links and concurrent replacement."""
    path = Path(path)
    fd, opened = inference._open_regular_nofollow(
        path, error_type=CacheError, label="cache input")
    with os.fdopen(fd, "rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
        inference._assert_open_file_unchanged(
            stream.fileno(), path, opened, error_type=CacheError,
            label="cache input")
    return {"sha256": digest.hexdigest(), "size_bytes": opened.st_size}



def _spatial_shape(data, family):
    from numpy.lib import format as fmt
    keys = (("tessera",) if family == "tessera" else
            ("spectral", "image") if family is not None else
            ("tessera", "spectral", "image"))
    for key in keys:
        if key in data.files:
            with data.zip.open(key + ".npy") as stream:
                version = fmt.read_magic(stream)
                if version == (1, 0):
                    shape, _fortran, dtype = fmt.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _fortran, dtype = fmt.read_array_header_2_0(stream)
                else:
                    shape, _fortran, dtype = fmt._read_array_header(stream, version)
            if dtype.hasobject or len(shape) < 2 or min(shape[-2:]) <= 0:
                raise CacheError("invalid spatial array header")
            return int(shape[-2]), int(shape[-1])
    raise CacheError(f"tile has no spatial array for family {family}")


def tile_metadata(path, identity, *, family, include_year=False):
    """Read geometry/year from the same authenticated bytes, with no pickle."""
    data, buffer = inference._load_npz_for_inference(
        path, expected_sha256=identity["sha256"],
        expected_size=identity["size_bytes"])
    try:
        height, width = _spatial_shape(data, family)
        metadata = {"height": height, "width": width}
        if include_year:
            from imint.training.nfi_colocate import tile_year
            metadata["year"] = tile_year(data)
        return metadata
    finally:
        data.close()
        buffer.close()


def source_identity(root: Path, git_sha: str) -> dict:
    """Hash actual repository code/config bytes, including uncommitted edits."""
    paths = sorted(p for folder in ("imint", "scripts")
                   for p in (root / folder).rglob("*")
                   if p.suffix in {".py", ".json", ".yaml", ".yml"}
                   and "__pycache__" not in p.parts and p.is_file())
    if not paths:
        raise CacheError("source tree is empty")
    return {"git_sha": git_sha, "files": {
        str(p.relative_to(root)): file_identity(p) for p in paths}}


def runtime_identity(torch, device) -> dict:
    """Record runtime versions and numerical/device settings affecting forwards.

    Package versions do not attest all external source bytes. The campaign must
    additionally pin its image and review external dependency provenance.
    """
    packages = []
    for dist in importlib.metadata.distributions():
        direct_url = dist.read_text("direct_url.json")
        packages.append([
            dist.metadata.get("Name", ""), dist.version,
            hashlib.sha256(direct_url.encode()).hexdigest() if direct_url else None,
        ])
    packages.sort(key=canonical)
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device_type": device.type,
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda"
                        else platform.machine()),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def _check_identity(identity):
    if (not isinstance(identity, dict)
            or not isinstance(identity.get("sha256"), str)
            or SHA256.fullmatch(identity["sha256"]) is None
            or type(identity.get("size_bytes")) is not int
            or identity["size_bytes"] <= 0):
        raise CacheError("expected positive size and full SHA-256")


def _check_stem(stem):
    if (not isinstance(stem, str) or stem in ("", ".", "..")
            or Path(stem).name != stem or "\\" in stem or "\x00" in stem):
        raise CacheError(f"invalid tile stem: {stem!r}")


def validate_contract(contract):
    try:
        if contract["schema"] != SCHEMA:
            raise CacheError("unsupported prediction cache schema")
        _check_identity(contract["checkpoint"])
        for name in ("source", "runtime", "model"):
            if not isinstance(contract[name], dict) or not contract[name]:
                raise CacheError(f"missing {name} identity")
        options = contract["options"]
        for name in ("img_size", "num_classes", "batch_size"):
            if type(options[name]) is not int or options[name] <= 0:
                raise CacheError(f"invalid {name}")
        if options["num_classes"] > 256:
            raise CacheError("uint8 cache supports at most 256 classes")
        if type(contract["model"]["fraction_head"]) is not bool:
            raise CacheError("fraction-head capability must be explicit")
        tiles = contract["tiles"]
        if not isinstance(tiles, dict) or not tiles:
            raise CacheError("empty tile inventory")
        for stem, entry in tiles.items():
            _check_stem(stem)
            _check_identity(entry["input"])
            if (type(entry["crop_size"]) is not int
                    or not 0 < entry["crop_size"] <= options["img_size"]):
                raise CacheError("invalid tile crop size")
    except (KeyError, TypeError) as exc:
        raise CacheError(f"incomplete prediction contract: {exc}") from exc


def _read_json(path):
    fd, opened = inference._open_regular_nofollow(
        path, error_type=CacheError, label="cache metadata")
    with os.fdopen(fd, "rb") as stream:
        data = stream.read()
        inference._assert_open_file_unchanged(
            stream.fileno(), path, opened, error_type=CacheError,
            label="cache metadata")
    try:
        return json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise CacheError(f"invalid cache metadata: {path}") from exc


def _publish_json(path: Path, value):
    fd, tmp = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def initialize(cache_root, contract):
    validate_contract(contract)
    key = contract_key(contract)
    directory = Path(cache_root) / key
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise CacheError("cache directory must not be a symlink")
    manifest = {"schema": SCHEMA, "cache_key": key, "contract": contract}
    fd = os.open(directory / ".manifest.lock",
                 os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / "MANIFEST.json"
        if path.exists() or path.is_symlink():
            if _read_json(path) != manifest:
                raise CacheError(f"existing metadata differs: {path}")
        else:
            _publish_json(path, manifest)
    return directory


def load_manifest(cache_root, key, *, require_complete=True):
    if not isinstance(key, str) or SHA256.fullmatch(key) is None:
        raise CacheError("cache key must be a full contract SHA-256; legacy caches are not reusable")
    directory = Path(cache_root) / key
    if directory.is_symlink():
        raise CacheError("cache directory must not be a symlink")
    manifest = _read_json(directory / "MANIFEST.json")
    try:
        contract = manifest["contract"]
        validate_contract(contract)
        if (manifest["schema"] != SCHEMA or manifest["cache_key"] != key
                or contract_key(contract) != key):
            raise CacheError("cache manifest identity mismatch")
    except (KeyError, TypeError) as exc:
        raise CacheError("missing cache contract") from exc
    if require_complete:
        # Publication completeness, with byte/content checks when each tile is
        # consumed. Never score a partial shard as a completed full cache.
        for stem in contract["tiles"]:
            _receipt(directory, contract, stem)
    return manifest


def _receipt(directory, contract, stem):
    _check_stem(stem)
    if stem not in contract["tiles"]:
        raise CacheError(f"tile {stem} is outside the frozen cache inventory")
    receipt = _read_json(directory / f"{stem}.json")
    if (receipt.get("schema") != SCHEMA
            or receipt.get("cache_key") != directory.name
            or receipt.get("input") != contract["tiles"][stem]["input"]
            or receipt.get("tile") != stem):
        raise CacheError(f"receipt identity mismatch for {stem}")
    _check_identity(receipt.get("output"))
    output = directory / f"{stem}.npz"
    if (output.is_symlink() or not output.is_file()
            or output.stat().st_size != receipt["output"]["size_bytes"]):
        raise CacheError(f"missing/changed cached output for {stem}")
    return receipt


def validate_arrays(arrays, contract, stem):
    crop = contract["tiles"][stem]["crop_size"]
    classes = contract["options"]["num_classes"]
    shapes = {"pred": (crop, crop), "probs": (classes, crop, crop)}
    if contract["model"]["fraction_head"]:
        shapes["fracs"] = (4, crop, crop)
    if set(arrays) != set(shapes):
        raise CacheError(f"wrong output members for {stem}")
    for name, shape in shapes.items():
        value = arrays[name]
        dtype = np.uint8 if name == "pred" else np.float16
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise CacheError(f"invalid {name} shape/dtype/finiteness for {stem}")
        if name != "pred" and ((value < 0).any() or (value > 1).any()):
            raise CacheError(f"out-of-range {name} for {stem}")
    pred, probs = arrays["pred"], arrays["probs"].astype(np.float32)
    if (pred >= classes).any() or not np.allclose(probs.sum(0), 1, atol=0.002, rtol=0):
        raise CacheError(f"invalid class/probability values for {stem}")
    # Float16 can introduce ties. The original float32 argmax must be ONE of
    # the stored maxima, not necessarily numpy's first maximum after rounding.
    chosen = np.take_along_axis(probs, pred[None].astype(np.int64), axis=0)[0]
    if not np.array_equal(chosen, probs.max(0)):
        raise CacheError(f"class map contradicts probabilities for {stem}")


def read_prediction(directory, contract, stem):
    directory = Path(directory)
    receipt = _receipt(directory, contract, stem)
    identity = receipt["output"]
    data, buffer = inference._load_npz_for_inference(
        directory / f"{stem}.npz", expected_sha256=identity["sha256"],
        expected_size=identity["size_bytes"])
    try:
        arrays = {name: data[name] for name in data.files}
    finally:
        data.close()
        buffer.close()
    validate_arrays(arrays, contract, stem)
    return arrays


def has_prediction(directory, contract, stem):
    directory = Path(directory)
    if not any((directory / f"{stem}.{ext}").exists()
               or (directory / f"{stem}.{ext}").is_symlink()
               for ext in ("npz", "json")):
        return False
    read_prediction(directory, contract, stem)
    return True


def publish_prediction(directory, contract, stem, arrays, produced_at):
    """Publish once under a per-tile lock; disjoint shards may run concurrently."""
    directory = Path(directory)
    _check_stem(stem)
    validate_arrays(arrays, contract, stem)
    lock_fd = os.open(directory / f".{stem}.lock",
                      os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "r+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if has_prediction(directory, contract, stem):
            return False
        fd, tmp = tempfile.mkstemp(prefix=".prediction-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            output_identity = file_identity(tmp)
            os.replace(tmp, directory / f"{stem}.npz")
            _publish_json(directory / f"{stem}.json", {
                "schema": SCHEMA, "cache_key": directory.name, "tile": stem,
                "input": contract["tiles"][stem]["input"],
                "output": output_identity, "produced_at": produced_at,
            })
        finally:
            Path(tmp).unlink(missing_ok=True)
    return True
