#!/usr/bin/env python3
"""Export a training checkpoint as a Hugging Face model release, and load one.

A release folder holds three generated files plus the model card, NOTICE and
LICENSE, taken from the exporter commit (``release/hf/<model_id>/`` and the
repository root). The documents are part of every release: the licence is
among what the manifest vouches for.

    model.safetensors  the checkpoint's state dict, keys and tensors verbatim
    config.json        the checkpoint config plus the input/output contract:
                       class names, fraction species, aux channel order and
                       normalization, and the embedding source. Metrics are
                       left out: results stay private until the paper.
    MANIFEST.json      sha256 + size of every other file, the source
                       checkpoint's identity, and the exporter and training
                       commits

``load_release`` needs the MANIFEST.json sha256 that was published on another
channel (``release/hf/<model>/RELEASES.md`` on GitHub): without that anchor a
consistent forgery (files changed and the manifest re-signed) cannot be told
apart from the release. It reads each file once and checks those bytes,
refuses files the manifest does not list, rebuilds the model through
``build_model_from_checkpoint`` (the code that loads a ``best_model.pt``) and
requires config.json to equal the config this checkout derives from the
rebuilt model. A release and its source checkpoint give identical outputs.

The CLI exports only from a clean tree whose HEAD is on origin/main, so the
card's ``git checkout <exporter_git_sha>`` works for anyone; only when the
training commit is a full sha that HEAD descends from; and only when the files
that define the input contract are unchanged since that commit. Untracked
files are the operator's responsibility (an untracked ``torch.py`` next to
the scripts would shadow the real package), and origin/main is the local
ref: fetch first.

Usage:
    git fetch origin main
    python scripts/hf_release.py export \\
        --checkpoint /data/checkpoints/im023/tessera_b28/best_model.pt \\
        --expected-sha256 <sha256> --expected-size <bytes> \\
        --model-id imint-tessera-b28 \\
        --training-git-sha <full sha the model was trained at> \\
        --embedding geotessera-0.10.2 \\
        --out-dir /tmp/hf-stage/imint-tessera-b28

The card and NOTICE come from ``release/hf/<model_id>/`` at HEAD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
    checkpoint_state_dict,
    inference_aux_names,
    model_has_frac_head,
)

FORMAT = "imint-hf-release/1"
WEIGHTS = "model.safetensors"
CONFIG = "config.json"
MANIFEST = "MANIFEST.json"
GENERATED = (WEIGHTS, CONFIG, MANIFEST)
DOCUMENTS = ("README.md", "NOTICE", "LICENSE")
CONFIG_KEYS = frozenset({"format", "model_id", "checkpoint_config", "epoch", "classes",
                         "ignore_index", "fraction_species", "input"})
MANIFEST_KEYS = frozenset({"format", "model_id", "produced_at", "exporter_git_sha",
                           "training_git_sha", "source_checkpoint", "files", "documents"})
CLASS_COUNTS = frozenset({NUM_CLASSES_BEFORE_BARRBLANDSKOG, NUM_UNIFIED_CLASSES})
FULL_SHA = re.compile(r"[0-9a-f]{40}")
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


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _safetensors_metadata(data: bytes) -> dict:
    """The ``__metadata__`` block of a serialized safetensors file."""
    if len(data) < 8:
        raise ReleaseIdentityError(f"{WEIGHTS} is too short to be safetensors")
    size = int.from_bytes(data[:8], "little")
    try:
        header = json.loads(data[8:8 + size])
    except ValueError as exc:
        raise ReleaseIdentityError(f"{WEIGHTS} has an unreadable header: {exc}") from exc
    return header.get("__metadata__", {}) if isinstance(header, dict) else {}


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
    """config.json: the checkpoint config plus the input/output contract.

    Everything but ``model_id``, ``embedding`` and the checkpoint config is
    derived from the built model and this checkout's constants, so a loader
    at the exporter commit can recompute it and compare.
    """
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


def export_release(checkpoint: str | Path, out_dir: str | Path, *,
                   expected_sha256: str, expected_size: int, model_id: str,
                   exporter_git_sha: str, training_git_sha: str,
                   documents: dict[str, bytes],
                   embedding: str | None = None) -> dict:
    """Write the release folder atomically; return the manifest.

    The checkpoint is authenticated against ``expected_sha256`` and
    ``expected_size`` before it is deserialized, the model is built once from
    it, and every argument is validated before anything is written. The
    folder is assembled in a temporary sibling and renamed into place, so a
    refused or failed export leaves nothing, and an existing ``out_dir`` is
    refused rather than mixed with. ``documents`` maps exactly
    ``DOCUMENTS`` to their bytes (the CLI reads them from the exporter commit).
    """
    import torch
    from safetensors.torch import load, save

    out = Path(out_dir)
    if out.exists():
        raise ValueError(f"{out} already exists; export into a new folder")
    for label, sha in (("exporter", exporter_git_sha), ("training", training_git_sha)):
        if not FULL_SHA.fullmatch(sha):
            raise ValueError(f"the {label} commit must be a full 40-hex sha, not {sha!r}")
    if set(documents) != set(DOCUMENTS):
        raise ValueError(f"documents must be exactly {DOCUMENTS}")

    ck = _load_checkpoint_for_inference(
        checkpoint, map_location="cpu",
        expected_sha256=expected_sha256, expected_size=expected_size)
    model, _, _, img_size = build_model_from_checkpoint(ck, "cpu")
    config = release_config(ck, model, img_size, model_id=model_id, embedding=embedding)

    state = {k: v.contiguous() for k, v in checkpoint_state_dict(ck).items()}
    if not state:
        raise ValueError("checkpoint has no state dict")
    weights = save(state, metadata={"format": FORMAT, "model_id": model_id})
    written = load(weights)
    if written.keys() != state.keys() or not all(
            written[k].dtype == v.dtype and torch.equal(written[k], v)
            for k, v in state.items()):
        raise ReleaseIdentityError("the serialized weights do not read back as the checkpoint")
    payload = {WEIGHTS: weights,
               CONFIG: (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode()}
    manifest = {
        "format": FORMAT,
        "model_id": model_id,
        "produced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exporter_git_sha": exporter_git_sha,
        "training_git_sha": training_git_sha,
        "source_checkpoint": {"file": Path(checkpoint).name,
                              "sha256": expected_sha256, "size": expected_size},
        "files": {name: _digest(data) for name, data in payload.items()},
        "documents": {name: _digest(data) for name, data in documents.items()},
    }
    payload.update(documents)
    payload[MANIFEST] = (json.dumps(manifest, indent=2) + "\n").encode()

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        for name, data in payload.items():
            (staging / name).write_bytes(data)
        os.rename(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _verified_bytes(root: Path, name: str, expected) -> bytes:
    """Read ``name`` once and check those exact bytes against the manifest entry."""
    if (not isinstance(expected, dict) or set(expected) != {"sha256", "size"}
            or not isinstance(expected["sha256"], str) or type(expected["size"]) is not int):
        raise ReleaseIdentityError(f"MANIFEST.json has a malformed entry for {name}")
    try:
        data = (root / name).read_bytes()
    except OSError as exc:
        raise ReleaseIdentityError(f"cannot read release file {root / name}: {exc}") from exc
    if _digest(data) != expected:
        raise ReleaseIdentityError(f"{root / name} does not match MANIFEST.json")
    return data


def _read_manifest(root: Path, expected_sha256: str | None, trust_unanchored: bool) -> dict:
    if expected_sha256 is None and not trust_unanchored:
        raise ReleaseIdentityError(
            "pass expected_manifest_sha256 (published in release/hf/<model>/RELEASES.md "
            "on GitHub); without it a re-signed forgery loads like the release")
    try:
        data = (root / MANIFEST).read_bytes()
    except OSError as exc:
        raise ReleaseIdentityError(f"cannot read {root / MANIFEST}: {exc}") from exc
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ReleaseIdentityError(f"{root / MANIFEST} is not the expected manifest")
    try:
        manifest = json.loads(data)
    except ValueError as exc:
        raise ReleaseIdentityError(f"cannot parse {root / MANIFEST}: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ReleaseIdentityError(f"MANIFEST.json must have exactly the keys {sorted(MANIFEST_KEYS)}")
    if manifest["format"] != FORMAT or not isinstance(manifest["model_id"], str):
        raise ReleaseIdentityError("MANIFEST.json names an unknown format or no model")
    if not all(isinstance(manifest[k], str) and FULL_SHA.fullmatch(manifest[k])
               for k in ("exporter_git_sha", "training_git_sha")):
        raise ReleaseIdentityError("MANIFEST.json commits must be full 40-hex shas")
    files, documents = manifest["files"], manifest["documents"]
    if not isinstance(files, dict) or set(files) != {WEIGHTS, CONFIG}:
        raise ReleaseIdentityError(f"MANIFEST.json must list exactly {WEIGHTS} and {CONFIG}")
    if not isinstance(documents, dict) or set(documents) != set(DOCUMENTS):
        raise ReleaseIdentityError(f"MANIFEST.json documents must be exactly {DOCUMENTS}")
    return manifest


def load_release(release_dir: str | Path, device="cpu", *,
                 expected_manifest_sha256: str | None,
                 trust_unanchored: bool = False):
    """Verify a release folder and rebuild its model; return ``(model, config)``.

    ``expected_manifest_sha256`` is required; ``trust_unanchored=True`` with
    ``None`` skips the anchor and is meant for tests. Every check raises
    ``ReleaseIdentityError`` before a tensor is deserialized, except the final
    comparison of config.json with the config derived from the rebuilt model.
    Hidden entries (``.gitattributes``, a download cache) are ignored; any
    other file the manifest does not list is refused.
    """
    from safetensors.torch import load

    root = Path(release_dir)
    manifest = _read_manifest(root, expected_manifest_sha256, trust_unanchored)
    listed = {MANIFEST, *manifest["files"], *manifest["documents"]}
    unlisted = sorted(p.name for p in root.iterdir()
                      if not p.name.startswith(".") and p.name not in listed)
    if unlisted:
        raise ReleaseIdentityError(f"{root} holds files MANIFEST.json does not list: {unlisted}")
    for name, expected in manifest["documents"].items():
        _verified_bytes(root, name, expected)
    weights = _verified_bytes(root, WEIGHTS, manifest["files"][WEIGHTS])
    config_bytes = _verified_bytes(root, CONFIG, manifest["files"][CONFIG])

    if _safetensors_metadata(weights) != {"format": FORMAT, "model_id": manifest["model_id"]}:
        raise ReleaseIdentityError(f"{WEIGHTS} metadata names another format or model")
    try:
        config = json.loads(config_bytes)
    except ValueError as exc:
        raise ReleaseIdentityError(f"cannot parse {root / CONFIG}: {exc}") from exc
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ReleaseIdentityError(f"{root / CONFIG} must have exactly the keys {sorted(CONFIG_KEYS)}")
    if config["format"] != FORMAT or config["model_id"] != manifest["model_id"]:
        raise ReleaseIdentityError(f"{root / CONFIG} names another format or model than the manifest")
    if not isinstance(config["checkpoint_config"], dict) or not isinstance(config["input"], dict):
        raise ReleaseIdentityError(f"{root / CONFIG} has a malformed checkpoint_config or input")

    ck = {"state_dict": load(weights), "config": config["checkpoint_config"],
          "epoch": config["epoch"]}
    # A forged checkpoint_config can fail inside the build; every refusal is the same error.
    # The build runs on the CPU so a device failure (CUDA out of memory is a RuntimeError,
    # like a strict-load shape mismatch) is not reported as an identity refusal.
    try:
        model, _, _, img_size = build_model_from_checkpoint(ck, "cpu")
        expected = release_config(ck, model, img_size, model_id=config["model_id"],
                                  embedding=config["input"].get("embedding"))
    except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as exc:
        raise ReleaseIdentityError(f"config.json does not describe the model: {exc}") from exc
    if _canonical(config) != _canonical(expected):
        raise ReleaseIdentityError("config.json does not describe the model this "
                                   "checkout builds from the weights")
    return model.to(device), config


def _git(root: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=text)


def export_guard(root: Path, training_git_sha: str) -> str:
    """Refuse an export the card could not honour; return the exporter's commit.

    The training commit must be a full sha of a commit HEAD descends from, the
    tracked tree must be clean, HEAD must be on origin/main (the card tells
    users to check it out), and CONTRACT_FILES must be unchanged between the
    training commit and HEAD, because config.json is built from this
    checkout's constants, not the training commit's.
    """
    if not FULL_SHA.fullmatch(training_git_sha):
        raise SystemExit(f"refusing to export: --training-git-sha must be a full 40-hex sha, "
                         f"not {training_git_sha!r}")
    resolved = _git(root, "rev-parse", "--verify", "--quiet", f"{training_git_sha}^{{commit}}")
    if resolved.returncode or resolved.stdout.strip() != training_git_sha:
        raise SystemExit(f"refusing to export: training commit {training_git_sha} not found")
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if status.returncode or status.stdout.strip():
        raise SystemExit("refusing to export from a dirty tree:\n" + status.stdout)
    if _git(root, "merge-base", "--is-ancestor", "HEAD", "origin/main").returncode:
        raise SystemExit("refusing to export: HEAD is not on origin/main, so the card's "
                         "checkout of the exporter commit would not work")
    if _git(root, "merge-base", "--is-ancestor", training_git_sha, "HEAD").returncode:
        raise SystemExit(f"refusing to export: HEAD does not descend from the training "
                         f"commit {training_git_sha}")
    drift = _git(root, "diff", "--name-only", training_git_sha, "HEAD", "--", *CONTRACT_FILES)
    if drift.returncode or drift.stdout.strip():
        raise SystemExit("refusing to export: the input contract changed since the training "
                         "commit:\n" + (drift.stdout or drift.stderr))
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def documents_at_head(root: Path, model_id: str) -> dict[str, bytes]:
    """README.md and NOTICE from ``release/hf/<model_id>/`` and the root LICENSE, as committed at HEAD.

    The card folder follows from the model id, so one model's card cannot
    ship under another model's manifest.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model_id):
        raise SystemExit(f"refusing to export: {model_id!r} is not a plain model id")
    paths = {name: f"release/hf/{model_id}/{name}" for name in DOCUMENTS if name != "LICENSE"}
    paths["LICENSE"] = "LICENSE"
    documents = {}
    for name, path in paths.items():
        shown = _git(root, "show", f"HEAD:{path}", text=False)
        if shown.returncode:
            raise SystemExit(f"refusing to export: {path} is not committed at HEAD")
        documents[name] = shown.stdout
    return documents


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
    exp.add_argument("--out-dir", required=True, help="a new folder; must not exist")
    args = parser.parse_args(argv)

    exporter_git_sha = export_guard(ROOT, args.training_git_sha)
    manifest = export_release(
        args.checkpoint, args.out_dir,
        expected_sha256=args.expected_sha256, expected_size=args.expected_size,
        model_id=args.model_id, exporter_git_sha=exporter_git_sha,
        training_git_sha=args.training_git_sha, embedding=args.embedding,
        documents=documents_at_head(ROOT, args.model_id))
    print(json.dumps(manifest, indent=2))
    manifest_sha = hashlib.sha256((Path(args.out_dir) / MANIFEST).read_bytes()).hexdigest()
    print(f"MANIFEST.json sha256: {manifest_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
