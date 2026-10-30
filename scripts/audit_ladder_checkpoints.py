#!/usr/bin/env python3
"""E00 checkpoint integrity audit; no model construction or evaluation.

A full inventory is mandatory. Each selected cell runs in a fresh CPU process.
Exit 0 means all requested checks passed, 1 means failures or missing evidence.
Results do not establish training provenance, strict model load or inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

from scripts.capture_crop_distill_evidence import _load_json_bytes, _read_regular_file
from scripts.crop_distill_protocol import CROP_MODELS
from scripts.inference_comparison import _load_checkpoint_for_inference

FOUR_FRAME_MODELS = {"prithvi300m4f", "prithvi600m"}
CELLS = {f"{model}_r{rung}" for model in CROP_MODELS for rung in range(1, 5)}
DIRECT_FIELDS = (
    "num_classes", "decoder_type", "decoder_channels", "feature_indices",
    "dropout", "enable_multitemporal", "enable_tradslag_head", "num_tradslag",
    "img_size", "aux_fusion", "enable_temporal_pooling", "enable_multilevel_aux",
    "era5_mode",
)
AUX_FLAGS = (
    "enable_height_channel", "enable_volume_channel", "enable_basal_area_channel",
    "enable_diameter_channel", "enable_dem_channel", "enable_vpp_channels",
    "enable_markfukt_channel", "enable_delta_sar_channels", "enable_era5_channels",
)
IMAGE = ("ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:"
         "25919e5cc0bc5ccba770b80f5906d0721cf47a779e3b4591a07cf5c5f3dc3599")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_inventory(path: Path) -> tuple[dict, bytes]:
    raw = _read_regular_file(path, "inventory", limit=1024 * 1024)
    rows = {}
    for line in raw.splitlines():
        row = _load_json_bytes(line, "inventory row")
        cell = row.get("cell")
        if cell not in CELLS or cell in rows:
            raise ValueError(f"unknown or duplicate cell: {cell!r}")
        cp = row.get("checkpoint", {})
        for value in (cp.get("sha256"), row.get("training_log_sha256")):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"invalid SHA256 for {cell}")
        size = cp.get("size_bytes")
        if type(size) is not int or not 0 < size <= 4 * 1024**3:
            raise ValueError(f"invalid checkpoint size for {cell}")
        if row.get("identity_status") != "hashed_stable":
            raise ValueError(f"unverified identity for {cell}")
        rows[cell] = row
    if rows.keys() != CELLS:
        raise ValueError("inventory must contain all 28 unique ladder cells")
    return rows, raw


def inspect_payload(payload: dict, log: dict, cell: str) -> dict:
    import torch
    from imint.fm.registry import MODEL_CONFIGS, LEGACY_BACKBONE_ALIAS, resolve_backbone_name
    from imint.training.config import TrainingConfig

    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("expected best-checkpoint payload with config")
    raw_state = payload.get("state_dict")
    if not isinstance(raw_state, dict) or not raw_state:
        raise ValueError("expected nonempty best-checkpoint state_dict")
    cfg, recorded = payload["config"], log["config"]
    errors, missing = [], []

    def compare(label, actual, expected):
        # Tuple/list representation differs between torch and JSON serialization.
        if isinstance(actual, tuple):
            actual = list(actual)
        if isinstance(expected, tuple):
            expected = list(expected)
        if actual != expected:
            errors.append(label)

    def field(key, expected):
        if key not in cfg:
            missing.append(f"checkpoint.config.{key}")
        else:
            compare(f"config.{key}", cfg[key], expected)

    for key in DIRECT_FIELDS:
        if key not in recorded:
            missing.append(f"log.config.{key}")
        else:
            field(key, recorded[key])
    if "epoch" not in payload or "best_epoch" not in log:
        missing.append("best_epoch")
    else:
        compare("best_epoch", payload["epoch"], log["best_epoch"])

    model, rung_text = cell.rsplit("_r", 1)
    rung = int(rung_text)
    protocol = CROP_MODELS[model]
    field("num_classes", 23 if rung == 1 else 28)
    field("img_size", protocol.img_size)
    for label, config in (("checkpoint", cfg), ("log", recorded)):
        name, legacy = config.get("backbone_name"), config.get("backbone")
        if name is None and legacy not in LEGACY_BACKBONE_ALIAS and legacy not in MODEL_CONFIGS:
            missing.append(f"{label}.backbone")
        else:
            compare(f"{label}.backbone", resolve_backbone_name(name, legacy), protocol.backbone)
    spec = MODEL_CONFIGS[protocol.backbone]
    field("patch_size", spec.patch_size)
    if type(recorded.get("enable_multitemporal")) is bool and "num_temporal_frames" in recorded:
        frames = recorded["num_temporal_frames"] if recorded["enable_multitemporal"] else 1
        field("num_temporal_frames", frames)
        compare("arm.num_temporal_frames", frames, 4 if model in FOUR_FRAME_MODELS else 1)
    else:
        frames = None
        missing.append("log.temporal_config")
    if all(type(recorded.get(key)) is bool for key in AUX_FLAGS):
        aux = list(TrainingConfig.enabled_aux_names.fget(SimpleNamespace(**recorded)))
        field("enabled_aux_names", aux)
        field("n_aux_channels", len(aux))
    else:
        aux = None
        missing.append("log.aux_flags")

    state, dtypes, nonfinite, count, nbytes = {}, {}, [], 0, 0
    for key, value in raw_state.items():
        if not isinstance(key, str) or not key.startswith("model.") or not key[6:]:
            raise ValueError("invalid state_dict key")
        key = key[6:]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError(f"unsupported state tensor: {key}")
        state[key] = value
        count += value.numel()
        nbytes += value.numel() * value.element_size()
        dtype = str(value.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            nonfinite.append(key)
    if nonfinite:
        errors.append("nonfinite_weights")

    def head(key, outputs):
        weight, bias = state.get(key + ".weight"), state.get(key + ".bias")
        if weight is None or weight.ndim != 4 or weight.shape[0] != outputs:
            errors.append(f"shape.{key}.weight")
        if bias is None or tuple(bias.shape) != (outputs,):
            errors.append(f"shape.{key}.bias")

    classifier = ("head.head.2" if spec.family == "prithvi" else
                  "classifier" if spec.family == "tessera" else "decoder_head.head.head.2")
    head(classifier, 23 if rung == 1 else 28)
    prefix = "" if spec.family in ("prithvi", "tessera") else "decoder_head."
    frac = prefix + "frac_head"
    # Head construction and fraction supervision are separate. Historical
    # R1-R3 jobs often enabled the head but supplied no fraction targets.
    enabled = cfg.get("enable_tradslag_head")
    if type(enabled) is not bool:
        missing.append("checkpoint.config.enable_tradslag_head")
    elif enabled:
        field("num_tradslag", 4)
        head(frac, 4)
    elif any(key.startswith(frac + ".") for key in state):
        errors.append("unexpected_fraction_head")
    if rung == 4 and enabled is False:
        errors.append("protocol.r4_fraction_head_disabled")
    if aux is not None and aux:
        if spec.family == "tessera":
            aux_key = "aux_proj.0.weight"
        elif spec.family == "prithvi" and type(cfg.get("enable_multilevel_aux")) is not bool:
            aux_key = None
            missing.append("checkpoint.config.enable_multilevel_aux")
        elif spec.family == "prithvi" and cfg["enable_multilevel_aux"] is False:
            aux_key = "aux_encoder.net.0.conv.weight"
        else:
            aux_key = prefix + "lidar_branch.net.0.conv.weight"
        weight = state.get(aux_key)
        if aux_key is not None and (weight is None or weight.ndim != 4 or weight.shape[1] != len(aux)):
            errors.append("shape.aux_input")
    if spec.family == "prithvi" and frames is not None:
        position = state.get("encoder.pos_embed")
        expected = (1, 1 + frames * (protocol.img_size // spec.patch_size)**2, spec.embed_dim)
        if position is None or tuple(position.shape) != expected:
            errors.append("shape.encoder.pos_embed")
    return {
        "status": "failed" if errors else "incomplete" if missing else "passed",
        "errors": sorted(set(errors)), "missing_evidence": sorted(set(missing)),
        "tensor_count": len(state), "elements": count, "tensor_bytes": nbytes,
        "dtypes": dtypes, "nonfinite_count": len(nonfinite), "nonfinite_keys": nonfinite[:20],
        "fraction_head_enabled": enabled if type(enabled) is bool else None,
        "fraction_head_weight_shape": (
            list(state[frac + ".weight"].shape) if frac + ".weight" in state else None
        ),
        "fraction_head_bias_shape": (
            list(state[frac + ".bias"].shape) if frac + ".bias" in state else None
        ),
        "not_verified": ["strict_model_load", "forward_pass", "preprocessing",
                         "training_source_provenance", "data_leakage", "model_quality"],
    }


def audit_cell(root: Path, row: dict) -> dict:
    import torch
    torch.set_num_threads(4)
    started = time.monotonic()
    directory = root / row["cell"]
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("checkpoint cell must be a real directory")
    raw_log = _read_regular_file(directory / "training_log.json", "training log", limit=8 * 1024**2)
    if digest(raw_log) != row["training_log_sha256"]:
        raise ValueError("training log SHA256 mismatch")
    log = _load_json_bytes(raw_log, "training log")
    cp = row["checkpoint"]
    payload = _load_checkpoint_for_inference(
        directory / "best_model.pt", map_location="cpu",
        expected_sha256=cp["sha256"], expected_size=cp["size_bytes"],
    )
    result = inspect_payload(payload, log, row["cell"])
    result.update({
        "checkpoint_sha256": cp["sha256"], "training_log_sha256": digest(raw_log),
        "elapsed_s": round(time.monotonic() - started, 3),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "python": platform.python_version(), "torch": torch.__version__,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/cephfs/checkpoints/ladder"))
    parser.add_argument("--cell", choices=sorted(CELLS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    rows, raw = load_inventory(args.inventory)
    if args.worker:
        if not args.cell:
            parser.error("--worker requires --cell")
        try:
            result = audit_cell(args.root, rows[args.cell])
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, allow_nan=False))
        return 0
    if args.output is None:
        parser.error("--output is required")
    cells = [args.cell] if args.cell else sorted(rows)
    status = 0
    with args.output.open("x", encoding="utf-8") as output:
        for cell in cells:
            try:
                child = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker",
                     "--inventory", str(args.inventory.resolve()), "--root", str(args.root.resolve()),
                     "--cell", cell],
                    capture_output=True, text=True, timeout=180, check=True,
                )
                result = json.loads(child.stdout)
            except (subprocess.SubprocessError, ValueError) as exc:
                result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:4000]}
            result.update({
                "schema": "imint-checkpoint-audit-v1", "cell": cell,
                "inventory_sha256": digest(raw),
                "auditor_sha256": digest(Path(__file__).read_bytes()),
                "loader_sha256": digest(Path(sys.modules[_load_checkpoint_for_inference.__module__].__file__).read_bytes()),
                "pod_uid": os.environ.get("POD_UID"), "image": os.environ.get("AUDIT_IMAGE"),
            })
            line = json.dumps(result, sort_keys=True, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            os.fsync(output.fileno())
            print(line, flush=True)
            status |= result["status"] != "passed"
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
