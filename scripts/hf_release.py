#!/usr/bin/env python3
"""Export a training checkpoint as a Hugging Face model release, and load one.

A release folder holds three generated files next to the hand-written model
card, LICENSE and NOTICE (``release/hf/<model>/``):

    model.safetensors  the checkpoint's state dict, keys and tensors verbatim
    config.json        the checkpoint config plus the input/output contract:
                       class names, fraction species, aux channel order and
                       normalization, and the embedding source. Metrics are
                       left out: results stay private until the paper.
    MANIFEST.json      sha256 + size of both files, the source checkpoint's
                       identity, and the exporter and training commits

``load_release`` verifies every file against MANIFEST.json and rebuilds the
model through ``build_model_from_checkpoint``, the same code that loads a
``best_model.pt``, so a release and its source checkpoint produce identical
outputs. Inputs are prepared as ``scripts/inference_comparison.py``
``_build_inference_inputs`` does at the training commit.

Usage:
    python scripts/hf_release.py export \\
        --checkpoint /data/checkpoints/im023/tessera_b28/best_model.pt \\
        --expected-sha256 <sha256> --expected-size <bytes> \\
        --model-id imint-tessera-b28 \\
        --training-git-sha <commit the model was trained at> \\
        --embedding geotessera-0.10.2 \\
        --out-dir release/hf/imint-tessera-b28
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.inference_comparison import (  # noqa: E402
    _load_checkpoint_for_inference,
    build_model_from_checkpoint,
    inference_aux_names,
)

FORMAT = "imint-hf-release/1"
WEIGHTS = "model.safetensors"
CONFIG = "config.json"
MANIFEST = "MANIFEST.json"
GENERATED = (WEIGHTS, CONFIG, MANIFEST)


class ReleaseIdentityError(ValueError):
    """A release file is missing or does not match MANIFEST.json."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict:
    return {"sha256": _sha256(path), "size": path.stat().st_size}


def _jsonable(value):
    """Checkpoint metrics are numpy scalars; JSON needs Python numbers."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _state_dict(ck: dict) -> dict:
    state = ck.get("model_state_dict", ck.get("state_dict"))
    if not state:
        raise ValueError("checkpoint has no state_dict")
    return state


def aux_contract(names: list[str]) -> list[dict]:
    """Order and normalization of the aux channels, as training applied them."""
    from imint.training.unified_dataset import (
        AUX_LOG_TRANSFORM, AUX_NAN_NODATA_CHANNELS, AUX_NORM,
        AUX_YYDDD_DATE_CHANNELS,
    )
    return [{"name": name,
             "mean": AUX_NORM[name][0],
             "std": AUX_NORM[name][1],
             "log1p": name in AUX_LOG_TRANSFORM,
             "yyddd_to_doy": name in AUX_YYDDD_DATE_CHANNELS,
             "nodata": "nan" if name in AUX_NAN_NODATA_CHANNELS else "zero"}
            for name in names]


def release_config(ck: dict, model, img_size: int, *, model_id: str,
                   embedding: str | None) -> dict:
    """config.json: the checkpoint config plus the input/output contract."""
    from imint.schema.unified_schema import UNIFIED_CLASS_NAMES
    from scripts.build_tradslag_fracs import SPECIES

    ck_cfg = dict(ck.get("config", {}))
    family = model.fm_spec.family
    if (family == "tessera") != (embedding is not None):
        raise ValueError("an embedding source is required for TESSERA models "
                         "and meaningless for the others")
    if model.num_classes > len(UNIFIED_CLASS_NAMES):
        raise ValueError(f"checkpoint has {model.num_classes} classes, the schema "
                         f"names {len(UNIFIED_CLASS_NAMES)}")
    has_frac = any("frac_head" in k for k in _state_dict(ck))
    aux_names = inference_aux_names(ck_cfg, model.n_aux_channels)
    return {
        "format": FORMAT,
        "model_id": model_id,
        "checkpoint_config": _jsonable(ck_cfg),
        "epoch": _jsonable(ck.get("epoch")),
        "classes": UNIFIED_CLASS_NAMES[:model.num_classes],
        "ignore_index": 0,
        "fraction_species": list(SPECIES) if has_frac else [],
        "input": {
            "family": family,
            "img_size": img_size,
            "num_temporal_frames": model.num_frames,
            "embedding": embedding,
            "aux": aux_contract(aux_names),
        },
    }


def export_release(checkpoint: str | Path, out_dir: str | Path, *,
                   expected_sha256: str, expected_size: int, model_id: str,
                   exporter_git_sha: str, training_git_sha: str,
                   embedding: str | None = None) -> dict:
    """Write model.safetensors, config.json and MANIFEST.json; return the manifest.

    The checkpoint is authenticated against ``expected_sha256`` and
    ``expected_size`` before it is deserialized. The model is built once
    from it, so a checkpoint the shared loader cannot rebuild never becomes
    a release, and the written weights are read back and compared tensor by
    tensor before the manifest is written.
    """
    import torch
    from safetensors.torch import load_file, save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = _load_checkpoint_for_inference(
        checkpoint, map_location="cpu",
        expected_sha256=expected_sha256, expected_size=expected_size)
    model, _, _, img_size = build_model_from_checkpoint(ck, "cpu")
    state = {k: v.contiguous() for k, v in _state_dict(ck).items()}

    save_file(state, out / WEIGHTS, metadata={"format": FORMAT, "model_id": model_id})
    written = load_file(out / WEIGHTS)
    if written.keys() != state.keys() or not all(
            written[k].dtype == v.dtype and torch.equal(written[k], v)
            for k, v in state.items()):
        raise ReleaseIdentityError(f"{out / WEIGHTS} does not read back as the checkpoint")

    config = release_config(ck, model, img_size, model_id=model_id, embedding=embedding)
    (out / CONFIG).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    manifest = {
        "format": FORMAT,
        "model_id": model_id,
        "produced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exporter_git_sha": exporter_git_sha,
        "training_git_sha": training_git_sha,
        "source_checkpoint": {"file": Path(checkpoint).name,
                              "sha256": expected_sha256, "size": expected_size},
        "files": {name: _identity(out / name) for name in (WEIGHTS, CONFIG)},
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_release(release_dir: str | Path, device="cpu"):
    """Verify a release folder against MANIFEST.json and rebuild its model.

    Returns ``(model, config)``. Any missing file, size or sha256 mismatch,
    or unknown format raises ``ReleaseIdentityError`` before a tensor is read.
    """
    from safetensors.torch import load_file

    root = Path(release_dir)
    try:
        manifest = json.loads((root / MANIFEST).read_text())
    except (OSError, ValueError) as exc:
        raise ReleaseIdentityError(f"cannot read {root / MANIFEST}: {exc}") from exc
    if manifest.get("format") != FORMAT:
        raise ReleaseIdentityError(f"unknown release format {manifest.get('format')!r}")
    if set(manifest.get("files", {})) != {WEIGHTS, CONFIG}:
        raise ReleaseIdentityError(f"MANIFEST.json must list exactly {WEIGHTS} and {CONFIG}")
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file():
            raise ReleaseIdentityError(f"missing release file {path}")
        if _identity(path) != expected:
            raise ReleaseIdentityError(f"{path} does not match MANIFEST.json")

    config = json.loads((root / CONFIG).read_text())
    ck = {"state_dict": load_file(root / WEIGHTS),
          "config": config["checkpoint_config"],
          "epoch": config["epoch"]}
    model, _, _, _ = build_model_from_checkpoint(
        ck, device, config["checkpoint_config"].get("backbone_name"),
        config["input"]["img_size"])
    return model, config


def _clean_head_sha() -> str:
    """The exporter's commit; refuses a dirty tree so the SHA describes the code."""
    root = Path(__file__).resolve().parents[1]
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                            cwd=root, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise SystemExit("refusing to export from a dirty tree:\n" + status)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="write a release folder from a checkpoint")
    exp.add_argument("--checkpoint", required=True)
    exp.add_argument("--expected-sha256", required=True)
    exp.add_argument("--expected-size", required=True, type=int)
    exp.add_argument("--model-id", required=True)
    exp.add_argument("--training-git-sha", required=True)
    exp.add_argument("--embedding", default=None,
                     help="embedding source the tiles record (TESSERA only), "
                          "e.g. geotessera-0.10.2")
    exp.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    manifest = export_release(
        args.checkpoint, args.out_dir,
        expected_sha256=args.expected_sha256, expected_size=args.expected_size,
        model_id=args.model_id, exporter_git_sha=_clean_head_sha(),
        training_git_sha=args.training_git_sha, embedding=args.embedding)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
