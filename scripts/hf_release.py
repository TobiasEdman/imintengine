#!/usr/bin/env python3
"""Export a training checkpoint as a Hugging Face model release, and load one.

A release folder holds three generated files, plus the model card and NOTICE
from ``release/hf/<model>/`` and the repository's LICENSE when ``--card-dir``
is given:

    model.safetensors  the checkpoint's state dict, keys and tensors verbatim
    config.json        the checkpoint config plus the input/output contract:
                       class names, fraction species, aux channel order and
                       normalization, and the embedding source. Metrics are
                       left out: results stay private until the paper.
    MANIFEST.json      sha256 + size of the weights and config (and of the
                       card, NOTICE and LICENSE), the source checkpoint's
                       identity, and the exporter and training commits

``load_release`` reads each file once, checks those bytes against
MANIFEST.json (and MANIFEST.json against an expected sha256 when given),
rebuilds the model through ``build_model_from_checkpoint`` (the code that
loads a ``best_model.pt``) and checks config.json against the rebuilt model.
A release and its source checkpoint therefore give identical outputs.

The CLI exports only from a clean tree whose HEAD is on origin/main, so the
card's ``git checkout <exporter_git_sha>`` works for anyone, and only when the
files that define the input contract are unchanged since the training commit.

Usage:
    python scripts/hf_release.py export \\
        --checkpoint /data/checkpoints/im023/tessera_b28/best_model.pt \\
        --expected-sha256 <sha256> --expected-size <bytes> \\
        --model-id imint-tessera-b28 \\
        --training-git-sha <commit the model was trained at> \\
        --embedding geotessera-0.10.2 \\
        --card-dir release/hf/imint-tessera-b28 \\
        --out-dir /tmp/hf-stage/imint-tessera-b28
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imint.schema.unified_schema import (  # noqa: E402
    NUM_CLASSES_BEFORE_BARRBLANDSKOG, NUM_UNIFIED_CLASSES, UNIFIED_CLASS_NAMES,
)
from scripts.inference_comparison import (  # noqa: E402
    _load_checkpoint_for_inference,
    build_model_from_checkpoint,
    inference_aux_names,
    model_has_frac_head,
)

FORMAT = "imint-hf-release/1"
WEIGHTS = "model.safetensors"
CONFIG = "config.json"
MANIFEST = "MANIFEST.json"
GENERATED = (WEIGHTS, CONFIG, MANIFEST)
CARD_FILES = ("README.md", "NOTICE")
CONFIG_KEYS = frozenset({"format", "model_id", "checkpoint_config", "epoch", "classes",
                         "ignore_index", "fraction_species", "input"})
CLASS_COUNTS = frozenset({NUM_CLASSES_BEFORE_BARRBLANDSKOG, NUM_UNIFIED_CLASSES})
# The files whose constants config.json describes: aux normalization, class
# names, fraction species. They must be unchanged since the training commit.
CONTRACT_FILES = ("imint/training/unified_dataset.py", "imint/schema/unified_schema.py",
                  "scripts/build_tradslag_fracs.py")
AUX_NORMALIZATION = (
    "Per channel, in order: if nan_to_mean, NaN becomes mean. If date_yyddd, "
    "values > 0 become value % 1000 (day of year) and all other values become "
    "mean; otherwise, if log1p, x becomes log1p(max(x, 0)). Then "
    "z = (x - mean) / max(std, 1e-6). Zero is an ordinary value except in "
    "date_yyddd channels."
)


class ReleaseIdentityError(ValueError):
    """A release file is missing, malformed, or does not match MANIFEST.json."""


def _digest(data: bytes) -> dict:
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def _jsonable(value):
    """Checkpoint configs can hold numpy scalars; JSON needs Python numbers."""
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
    """Order and normalization of the aux channels; see AUX_NORMALIZATION."""
    from imint.training.unified_dataset import (
        AUX_LOG_TRANSFORM, AUX_NAN_NODATA_CHANNELS, AUX_NORM,
        AUX_YYDDD_DATE_CHANNELS,
    )
    return [{"name": name,
             "mean": AUX_NORM[name][0],
             "std": AUX_NORM[name][1],
             "nan_to_mean": name in AUX_NAN_NODATA_CHANNELS,
             "date_yyddd": name in AUX_YYDDD_DATE_CHANNELS,
             "log1p": name in AUX_LOG_TRANSFORM}
            for name in names]


def release_config(ck: dict, model, img_size: int, *, model_id: str,
                   embedding: str | None) -> dict:
    """config.json: the checkpoint config plus the input/output contract."""
    from scripts.build_tradslag_fracs import SPECIES

    ck_cfg = dict(ck.get("config", {}))
    family = model.fm_spec.family
    if (family == "tessera") != (embedding is not None):
        raise ValueError("an embedding source is required for TESSERA models "
                         "and meaningless for the others")
    if model.num_classes not in CLASS_COUNTS:
        raise ValueError(f"checkpoint has {model.num_classes} classes; a release "
                         f"must have one of {sorted(CLASS_COUNTS)}")
    has_frac = model_has_frac_head(model)
    if has_frac and ck_cfg.get("num_tradslag", len(SPECIES)) != len(SPECIES):
        raise ValueError(f"fraction head has {ck_cfg['num_tradslag']} channels, "
                         f"the species list {len(SPECIES)}")
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
            "aux": aux_contract(inference_aux_names(ck_cfg, model.n_aux_channels)),
            "aux_normalization": AUX_NORMALIZATION,
        },
    }


def check_config_matches_model(config: dict, model, img_size: int) -> None:
    """config.json must describe the model that was actually built."""
    contract = config["input"]
    problems = []
    if len(config["classes"]) != model.num_classes:
        problems.append(f"{len(config['classes'])} class names for {model.num_classes} classes")
    if contract["family"] != model.fm_spec.family:
        problems.append(f"family {contract['family']!r} but the model is {model.fm_spec.family!r}")
    if contract["img_size"] != img_size:
        problems.append(f"img_size {contract['img_size']} but the model is built at {img_size}")
    aux = [a["name"] for a in contract["aux"]]
    if aux != inference_aux_names(model.ck_cfg, model.n_aux_channels):
        problems.append(f"aux channels {aux} do not match the checkpoint's")
    if bool(config["fraction_species"]) != model_has_frac_head(model):
        problems.append("fraction species do not match the fraction head")
    if problems:
        raise ReleaseIdentityError("config.json does not describe the model: "
                                   + "; ".join(problems))


def export_release(checkpoint: str | Path, out_dir: str | Path, *,
                   expected_sha256: str, expected_size: int, model_id: str,
                   exporter_git_sha: str, training_git_sha: str,
                   embedding: str | None = None,
                   card_dir: str | Path | None = None) -> dict:
    """Write the release folder; return the manifest.

    The checkpoint is authenticated against ``expected_sha256`` and
    ``expected_size`` before it is deserialized, the model is built once from
    it, and config.json is validated against that model before anything is
    written, so a refused export leaves no files. The written weights are read
    back and compared tensor by tensor before the manifest is written. With
    ``card_dir``, its README.md and NOTICE and the repository LICENSE are
    copied in and recorded in the manifest.
    """
    import torch
    from safetensors.torch import load, save

    ck = _load_checkpoint_for_inference(
        checkpoint, map_location="cpu",
        expected_sha256=expected_sha256, expected_size=expected_size)
    model, _, _, img_size = build_model_from_checkpoint(ck, "cpu")
    config = release_config(ck, model, img_size, model_id=model_id, embedding=embedding)
    check_config_matches_model(config, model, img_size)
    documents = {}
    if card_dir is not None:
        sources = [Path(card_dir) / name for name in CARD_FILES] + [ROOT / "LICENSE"]
        missing = [str(p) for p in sources if not p.is_file()]
        if missing:
            raise ValueError(f"missing release documents: {missing}")
        documents = {p.name: p.read_bytes() for p in sources}

    state = {k: v.contiguous() for k, v in _state_dict(ck).items()}
    weights = save(state, metadata={"format": FORMAT, "model_id": model_id})
    written = load(weights)
    if written.keys() != state.keys() or not all(
            written[k].dtype == v.dtype and torch.equal(written[k], v)
            for k, v in state.items()):
        raise ReleaseIdentityError("the serialized weights do not read back as the checkpoint")
    config_bytes = (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / WEIGHTS).write_bytes(weights)
    (out / CONFIG).write_bytes(config_bytes)
    for name, data in documents.items():
        (out / name).write_bytes(data)
    manifest = {
        "format": FORMAT,
        "model_id": model_id,
        "produced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exporter_git_sha": exporter_git_sha,
        "training_git_sha": training_git_sha,
        "source_checkpoint": {"file": Path(checkpoint).name,
                              "sha256": expected_sha256, "size": expected_size},
        "files": {WEIGHTS: _digest(weights), CONFIG: _digest(config_bytes)},
        "documents": {name: _digest(data) for name, data in documents.items()},
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _verified_bytes(root: Path, name: str, expected) -> bytes:
    """Read ``name`` once and check those exact bytes against the manifest entry."""
    if (not isinstance(expected, dict) or not isinstance(expected.get("sha256"), str)
            or type(expected.get("size")) is not int):
        raise ReleaseIdentityError(f"MANIFEST.json has a malformed entry for {name}")
    try:
        data = (root / name).read_bytes()
    except OSError as exc:
        raise ReleaseIdentityError(f"cannot read release file {root / name}: {exc}") from exc
    if _digest(data) != {"sha256": expected["sha256"], "size": expected["size"]}:
        raise ReleaseIdentityError(f"{root / name} does not match MANIFEST.json")
    return data


def load_release(release_dir: str | Path, device="cpu", *,
                 expected_manifest_sha256: str | None = None):
    """Verify a release folder and rebuild its model; return ``(model, config)``.

    Pass ``expected_manifest_sha256`` (printed by the export) to anchor trust:
    without it, the manifest only proves the files are consistent with each
    other. Every check raises ``ReleaseIdentityError`` before a tensor is
    deserialized, except the final comparison of config.json with the model.
    """
    from safetensors.torch import load

    root = Path(release_dir)
    try:
        manifest_bytes = (root / MANIFEST).read_bytes()
    except OSError as exc:
        raise ReleaseIdentityError(f"cannot read {root / MANIFEST}: {exc}") from exc
    if (expected_manifest_sha256 is not None
            and hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256):
        raise ReleaseIdentityError(f"{root / MANIFEST} is not the expected manifest")
    try:
        manifest = json.loads(manifest_bytes)
    except ValueError as exc:
        raise ReleaseIdentityError(f"cannot parse {root / MANIFEST}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ReleaseIdentityError(f"unknown release format in {root / MANIFEST}")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {WEIGHTS, CONFIG}:
        raise ReleaseIdentityError(f"MANIFEST.json must list exactly {WEIGHTS} and {CONFIG}")
    weights = _verified_bytes(root, WEIGHTS, files[WEIGHTS])
    config_bytes = _verified_bytes(root, CONFIG, files[CONFIG])

    try:
        config = json.loads(config_bytes)
    except ValueError as exc:
        raise ReleaseIdentityError(f"cannot parse {root / CONFIG}: {exc}") from exc
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ReleaseIdentityError(f"{root / CONFIG} must have exactly the keys {sorted(CONFIG_KEYS)}")
    if config["format"] != FORMAT or config["model_id"] != manifest.get("model_id"):
        raise ReleaseIdentityError(f"{root / CONFIG} names another format or model than the manifest")

    ck = {"state_dict": load(weights), "config": config["checkpoint_config"],
          "epoch": config["epoch"]}
    model, _, _, img_size = build_model_from_checkpoint(ck, device)
    check_config_matches_model(config, model, img_size)
    return model, config


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def export_guard(root: Path, training_git_sha: str) -> str:
    """Refuse an export the card could not honour; return the exporter's commit.

    The tree must be clean, HEAD must be on origin/main (the card tells users
    to check it out), the training commit must exist, and CONTRACT_FILES must
    be unchanged between the training commit and HEAD, because config.json is
    built from this checkout's constants, not the training commit's.
    """
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if status.returncode or status.stdout.strip():
        raise SystemExit("refusing to export from a dirty tree:\n" + status.stdout)
    if _git(root, "merge-base", "--is-ancestor", "HEAD", "origin/main").returncode:
        raise SystemExit("refusing to export: HEAD is not on origin/main, so the card's "
                         "checkout of the exporter commit would not work")
    if _git(root, "cat-file", "-e", f"{training_git_sha}^{{commit}}").returncode:
        raise SystemExit(f"refusing to export: training commit {training_git_sha} not found")
    drift = _git(root, "diff", "--name-only", training_git_sha, "HEAD", "--", *CONTRACT_FILES)
    if drift.returncode or drift.stdout.strip():
        raise SystemExit("refusing to export: the input contract changed since the training "
                         "commit:\n" + (drift.stdout or drift.stderr))
    return _git(root, "rev-parse", "HEAD").stdout.strip()


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
    exp.add_argument("--card-dir", required=True,
                     help="folder with the model card (README.md) and NOTICE")
    exp.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    exporter_git_sha = export_guard(ROOT, args.training_git_sha)
    manifest = export_release(
        args.checkpoint, args.out_dir,
        expected_sha256=args.expected_sha256, expected_size=args.expected_size,
        model_id=args.model_id, exporter_git_sha=exporter_git_sha,
        training_git_sha=args.training_git_sha, embedding=args.embedding,
        card_dir=args.card_dir)
    print(json.dumps(manifest, indent=2))
    manifest_sha = hashlib.sha256((Path(args.out_dir) / MANIFEST).read_bytes()).hexdigest()
    print(f"MANIFEST.json sha256: {manifest_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
