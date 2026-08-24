#!/usr/bin/env python3
"""Create and validate immutable provenance for the ERA5 600M smoke test."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any


RUN_SCHEMA = "era5-smoke-run-v2"
BASE_COMPLETION_SCHEMA = "era5-smoke-base-completion-v1"
COMPLETION_SCHEMA = "era5-smoke-run-completion-v2"
CODE_BUNDLE_SCHEMA = "era5-smoke-code-bundle-v1"
ERA5_COVERAGE_SCHEMA = "era5-smoke-sidecar-bundle-v1"
FOUNDATION_CHECKPOINT_NAME = "Prithvi_EO_V2_600M_TL.pt"
FOUNDATION_CHECKPOINT_SIZE_BYTES = 2_638_217_218
FOUNDATION_CHECKPOINT_SHA256 = (
    "7b92c53b0204a76bb775bd8930f045e"
    "05776251caa8c83f7367ed0b75b594702"
)
TERMINAL_STATUSES = {"completed"}
CONTROL_AUX_IDENTITY = {
    "schema": "era5-smoke-neutral-control-v1",
    "ordered_channels": [
        "era5_t2m_mean", "era5_tp_sum", "era5_swvl1_mean",
        "era5_ssrd_sum", "era5_gdd",
    ],
    "normalized_values": [0.0, 0.0, 0.0, 0.0, 0.0],
}
CROP_CLASS_INDICES = {
    "vete": 11,
    "korn": 12,
    "havre": 13,
    "oljeväxter": 14,
    "slåttervall": 15,
    "bete": 16,
    "potatis": 17,
    "sockerbetor": 18,
    "trindsäd": 19,
    "råg": 20,
    "majs": 21,
}


def sha256_file(path: Path) -> str:
    """Hash a file without loading a checkpoint into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_foundation_checkpoint(path: Path) -> dict[str, Any]:
    """Verify the exact published IBM/NASA Prithvi-EO-2.0 600M-TL file."""
    try:
        file_stat = path.stat()
    except OSError as exc:
        raise ValueError(
            f"Foundation checkpoint is not readable: {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"Foundation checkpoint is not a regular file: {path}")
    if file_stat.st_size <= 0:
        raise ValueError(f"Foundation checkpoint is empty: {path}")
    if file_stat.st_size != FOUNDATION_CHECKPOINT_SIZE_BYTES:
        raise ValueError(
            "Foundation checkpoint size mismatch: "
            f"expected {FOUNDATION_CHECKPOINT_SIZE_BYTES}, got {file_stat.st_size}"
        )
    try:
        actual_sha256 = sha256_file(path)
    except OSError as exc:
        raise ValueError(
            f"Foundation checkpoint is not readable: {path}: {exc}"
        ) from exc
    if actual_sha256 != FOUNDATION_CHECKPOINT_SHA256:
        raise ValueError(
            "Foundation checkpoint SHA256 mismatch: "
            f"expected {FOUNDATION_CHECKPOINT_SHA256}, got {actual_sha256}"
        )
    return {
        "name": FOUNDATION_CHECKPOINT_NAME,
        "size_bytes": FOUNDATION_CHECKPOINT_SIZE_BYTES,
        "sha256": FOUNDATION_CHECKPOINT_SHA256,
    }


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def code_bundle_sha256(files: dict[str, bytes]) -> str:
    """Hash logical filenames and contents in a stable order."""
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[name])
        digest.update(b"\0")
    return digest.hexdigest()


def cohort_bundle_sha256(cohort_dir: Path) -> str:
    """Bind cohort metadata and both exact split lists into one identity."""
    names = ("manifest.json", "split_train.txt", "split_val.txt")
    missing = [name for name in names if not (cohort_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete cohort bundle: {missing}")
    return code_bundle_sha256({
        name: (cohort_dir / name).read_bytes() for name in names
    })


def environment_bundle_sha256(environment_dir: Path) -> str:
    """Hash the stable runtime facts that must match between both arms."""
    names = ("gpu_identity.txt", "pip_freeze.txt", "python_version.txt")
    missing = [name for name in names if not (environment_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete environment bundle: {missing}")
    return code_bundle_sha256({
        name: (environment_dir / name).read_bytes() for name in names
    })


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def write_once_or_verify(path: Path, value: dict[str, Any]) -> None:
    """Create immutable JSON, accepting an exactly identical retry."""
    if path.exists():
        if load_json(path) != value:
            raise ValueError(f"Refusing to overwrite mismatched provenance: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.create")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with tmp.open("x") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if load_json(path) != value:
                raise ValueError(
                    f"Refusing to overwrite mismatched provenance: {path}"
                )
    finally:
        tmp.unlink(missing_ok=True)
    if load_json(path) != value:
        raise ValueError(f"Provenance verification failed after write: {path}")


def lock_file(args: argparse.Namespace) -> None:
    """Publish one immutable environment artifact, or verify an exact retry."""
    if args.require_existing and not args.target.is_file():
        raise FileNotFoundError(f"Required immutable artifact is missing: {args.target}")
    if args.target.exists():
        if sha256_file(args.source) != sha256_file(args.target):
            raise ValueError(f"Immutable artifact mismatch: {args.target}")
        return
    args.target.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.target.with_name(f".{args.target.name}.{os.getpid()}.tmp")
    try:
        with args.source.open("rb") as source, tmp.open("xb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(tmp, args.target)
        except FileExistsError:
            if sha256_file(args.source) != sha256_file(args.target):
                raise ValueError(f"Immutable artifact mismatch: {args.target}")
    finally:
        tmp.unlink(missing_ok=True)
    if sha256_file(args.source) != sha256_file(args.target):
        raise ValueError(f"Immutable artifact verification failed: {args.target}")


def verify_code_bundle(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("schema") != CODE_BUNDLE_SCHEMA:
        raise ValueError(f"Unexpected code bundle schema in {manifest_path}")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("Code bundle manifest has no files")

    contents: dict[str, bytes] = {}
    for name, expected_hash in expected_files.items():
        path = manifest_path.parent / name
        if not path.is_file():
            raise FileNotFoundError(f"Code bundle file is missing: {path}")
        contents[name] = path.read_bytes()
        actual_hash = hashlib.sha256(contents[name]).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Code bundle file hash mismatch: {name}")

    actual_bundle_hash = code_bundle_sha256(contents)
    if actual_bundle_hash != manifest.get("bundle_sha256"):
        raise ValueError("Code bundle aggregate hash mismatch")
    return manifest


def verify_code_command(args: argparse.Namespace) -> None:
    print(json.dumps(
        verify_code_bundle(args.code_manifest), indent=2, sort_keys=True,
    ))


def verify_foundation_command(args: argparse.Namespace) -> None:
    print(json.dumps(
        verify_foundation_checkpoint(args.foundation_checkpoint),
        indent=2,
        sort_keys=True,
    ))


def normalized_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove attempt path and the one deliberately arm-specific field."""
    normalized = dict(config)
    normalized.pop("checkpoint_dir", None)
    normalized.pop("era5_mode", None)
    normalized.pop("era5_dir", None)
    return normalized


def training_exposure_sha256(
    log: dict[str, Any],
    *,
    num_classes: int,
    crop_names: list[str],
    thresholds: dict[str, int],
) -> str:
    """Validate realized random-crop exposure and return its identity."""
    evidence: list[dict[str, Any]] = []
    aggregate_pixels = [0] * num_classes
    aggregate_tiles = [set() for _ in range(num_classes)]
    for epoch in log.get("epochs", []):
        digest = epoch.get("train_exposure_sha256")
        support = epoch.get("train_target_support")
        class_tiles = epoch.get("train_class_tiles")
        if (
            not isinstance(digest, str) or len(digest) != 64
            or not isinstance(support, list) or len(support) != num_classes
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in support
            )
            or not isinstance(class_tiles, dict)
        ):
            raise ValueError("Training log lacks a valid exposure ledger")
        normalized_tiles: dict[str, list[str]] = {}
        for raw_index, names in class_tiles.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise ValueError("Exposure ledger has an invalid class index") from exc
            if (
                index < 0 or index >= num_classes
                or not isinstance(names, list)
                or len(names) != len(set(names))
                or any(not isinstance(name, str) or not name for name in names)
            ):
                raise ValueError("Exposure ledger has invalid class-tile support")
            normalized_tiles[str(index)] = sorted(names)
            aggregate_tiles[index].update(names)
        for index, value in enumerate(support):
            aggregate_pixels[index] += value
        evidence.append({
            "epoch": epoch.get("epoch"),
            "train_exposure_sha256": digest,
            "train_target_support": support,
            "train_class_tiles": normalized_tiles,
        })

    min_pixels = int(thresholds["min_train_pixels"])
    min_tiles = int(thresholds["min_train_tiles"])
    for name in crop_names:
        index = CROP_CLASS_INDICES[name]
        if (
            aggregate_pixels[index] < min_pixels
            or len(aggregate_tiles[index]) < min_tiles
        ):
            raise ValueError(
                f"Realized training exposure is insufficient for {name}: "
                f"pixels={aggregate_pixels[index]}, tiles={len(aggregate_tiles[index])}"
            )
    return canonical_sha256(evidence)


def build_run_manifest(args: argparse.Namespace) -> dict[str, Any]:
    code = verify_code_bundle(args.code_manifest)
    if not args.run_id.endswith(code["bundle_sha256"][:12]):
        raise ValueError(
            "run_id must end with the first 12 characters of code_sha256"
        )
    cohort = load_json(args.cohort_manifest)
    if cohort.get("schema") != "era5-smoke-cohort-v4":
        raise ValueError("Unexpected cohort manifest schema")
    crop_classes = cohort.get("val_supported_crop_classes")
    if not isinstance(crop_classes, list) or not crop_classes:
        raise ValueError("Cohort has no fixed validation crop-class support")
    val_counts = cohort.get("label_support", {}).get("val", {}).get(
        "pixel_counts", {}
    )
    val_target_support = [
        int(val_counts.get(str(index), 0)) for index in range(args.num_classes)
    ]
    if sum(val_target_support) <= 0:
        raise ValueError("Cohort has no exact validation target support")

    seeds = [int(seed) for seed in args.seeds.split(",")]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be a non-empty unique list")
    base_completion = load_json(args.base_completion)
    if base_completion.get("schema") != BASE_COMPLETION_SCHEMA:
        raise ValueError("Unexpected base completion schema")
    initial_checkpoint_sha256 = sha256_file(args.initial_checkpoint)
    if base_completion.get("checkpoint_sha256") != initial_checkpoint_sha256:
        raise ValueError("Base completion does not bind the initial checkpoint")
    foundation_identity = verify_foundation_checkpoint(args.foundation_checkpoint)
    expected_invariants = {
        "base_git_sha": args.base_git_sha,
        "code_sha256": code["bundle_sha256"],
        "cohort_sha256": cohort_bundle_sha256(args.cohort_manifest.parent),
        "environment_sha256": environment_bundle_sha256(args.environment_dir),
        "foundation_checkpoint_sha256": foundation_identity["sha256"],
        "foundation_checkpoint_size_bytes": foundation_identity["size_bytes"],
    }
    for key, expected in expected_invariants.items():
        if base_completion.get(key) != expected:
            raise ValueError(f"Base completion {key} mismatch")
    return {
        "schema": RUN_SCHEMA,
        "run_id": args.run_id,
        "base_git_sha": args.base_git_sha,
        "code_sha256": code["bundle_sha256"],
        "cohort_sha256": expected_invariants["cohort_sha256"],
        "environment_sha256": expected_invariants["environment_sha256"],
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "base_completion_sha256": sha256_file(args.base_completion),
        "base_expected_epochs": int(base_completion["expected_epochs"]),
        "base_seed": int(base_completion["seed"]),
        "foundation_checkpoint_sha256": expected_invariants[
            "foundation_checkpoint_sha256"
        ],
        "foundation_checkpoint_size_bytes": expected_invariants[
            "foundation_checkpoint_size_bytes"
        ],
        "container_image": args.container_image,
        "seeds": seeds,
        "expected_arms": ["control", "treatment"],
        "expected_epochs": args.expected_epochs,
        "verdict_epoch": args.expected_epochs,
        "num_classes": args.num_classes,
        "val_tile_count": int(cohort["counts"]["val"]),
        "model_patch_px": int(cohort["model_patch_px"]),
        "val_supported_crop_classes": crop_classes,
        "crop_support_thresholds": cohort["crop_support_thresholds"],
        "val_target_support": val_target_support,
        "thresholds": {
            "verify_median_delta_miou": 0.005,
            "verify_median_delta_crop_macro_iou": 0.01,
            "verify_positive_pairs": 2,
        },
    }


def init_run(args: argparse.Namespace) -> None:
    expected = build_run_manifest(args)
    path = args.root / "run_manifest.json"
    if args.require_existing and not path.is_file():
        raise FileNotFoundError(f"Control run manifest is missing: {path}")
    write_once_or_verify(path, expected)
    print(json.dumps(expected, indent=2, sort_keys=True))


def _validate_epoch_log(
    log: dict[str, Any],
    *,
    expected_epochs: int,
    require_final_confusion: bool,
) -> list[int]:
    if log.get("status") not in TERMINAL_STATUSES:
        raise ValueError(f"Training log is not terminal: {log.get('status')!r}")
    epochs = log.get("epochs")
    if not isinstance(epochs, list) or len(epochs) != expected_epochs:
        raise ValueError(
            f"Expected {expected_epochs} completed epochs, got "
            f"{len(epochs) if isinstance(epochs, list) else 'invalid'}"
        )
    epoch_numbers = [item.get("epoch") for item in epochs]
    if epoch_numbers != list(range(1, expected_epochs + 1)):
        raise ValueError(f"Epoch sequence is incomplete: {epoch_numbers}")
    if require_final_confusion:
        matrix = epochs[-1].get("confusion_matrix")
        if not isinstance(matrix, list) or not matrix:
            raise ValueError("Fixed final verdict epoch lacks a confusion matrix")
    return epoch_numbers


def _validate_fixed_checkpoint_storage(run_dir: Path, config: dict) -> None:
    if config.get("fixed_checkpoint_only") is not True:
        raise ValueError("Smoke run did not enable fixed_checkpoint_only")
    redundant = [
        name for name in ("best_model.pt", "last_checkpoint.pt")
        if (run_dir / name).exists()
    ]
    if redundant:
        raise ValueError(
            f"Fixed-checkpoint smoke produced redundant snapshots: {redundant}"
        )


def _base_completion_payload(args: argparse.Namespace) -> dict[str, Any]:
    code = verify_code_bundle(args.code_manifest)
    log_path = args.base_dir / "training_log.json"
    checkpoint_path = args.base_dir / f"epoch_{args.expected_epochs:03d}.pt"
    log = load_json(log_path)
    epoch_numbers = _validate_epoch_log(
        log,
        expected_epochs=args.expected_epochs,
        require_final_confusion=True,
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Fixed base checkpoint is missing: {checkpoint_path}")
    config = log.get("config")
    if not isinstance(config, dict):
        raise ValueError("Base training log config is missing")
    _validate_fixed_checkpoint_storage(args.base_dir, config)
    expected_config = {
        "seed": args.seed,
        "deterministic": True,
        "deterministic_warn_only": False,
        "epochs": args.expected_epochs,
        "num_classes": args.num_classes,
        "backbone_name": "prithvi_600m",
        "enable_multitemporal": True,
        "num_temporal_frames": 4,
        "enable_era5_channels": True,
        "era5_mode": "control",
        "freeze_spectral": False,
        "unfreeze_backbone_layers": 0,
        "log_confusion_every_epoch": True,
        "log_training_exposure": True,
        "fixed_checkpoint_only": True,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Base training config mismatch: {mismatches}")
    cohort_manifest = load_json(args.cohort_dir / "manifest.json")
    exposure_sha256 = training_exposure_sha256(
        log,
        num_classes=args.num_classes,
        crop_names=cohort_manifest["val_supported_crop_classes"],
        thresholds=cohort_manifest["crop_support_thresholds"],
    )
    foundation_identity = verify_foundation_checkpoint(args.foundation_checkpoint)
    return {
        "schema": BASE_COMPLETION_SCHEMA,
        "base_git_sha": args.base_git_sha,
        "code_sha256": code["bundle_sha256"],
        "cohort_sha256": cohort_bundle_sha256(args.cohort_dir),
        "environment_sha256": environment_bundle_sha256(args.environment_dir),
        "foundation_checkpoint_sha256": foundation_identity["sha256"],
        "foundation_checkpoint_size_bytes": foundation_identity["size_bytes"],
        "seed": args.seed,
        "expected_epochs": args.expected_epochs,
        "completed_epochs": epoch_numbers,
        "config_sha256": canonical_sha256(config),
        "training_log_sha256": sha256_file(log_path),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_exposure_sha256": exposure_sha256,
    }


def finalize_base(args: argparse.Namespace) -> None:
    payload = _base_completion_payload(args)
    write_once_or_verify(args.base_dir / "completion.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def validate_base(args: argparse.Namespace) -> None:
    expected = _base_completion_payload(args)
    actual = load_json(args.base_dir / "completion.json")
    if actual != expected:
        raise ValueError("Base completion provenance mismatch")
    print(json.dumps(actual, indent=2, sort_keys=True))


def _coverage_identity(path: Path) -> tuple[str, str]:
    coverage = load_json(path)
    if coverage.get("schema") != ERA5_COVERAGE_SCHEMA:
        raise ValueError("Unexpected ERA5 coverage schema")
    if (
        coverage.get("requested") != coverage.get("valid")
        or not isinstance(coverage.get("requested"), int)
        or coverage["requested"] <= 0
        or coverage.get("atmosphere_cells", {}).get("overlap") != []
    ):
        raise ValueError("ERA5 coverage is incomplete or spatially overlapping")
    bundle_hash = coverage.get("sidecar_bundle_sha256")
    if not isinstance(bundle_hash, str) or len(bundle_hash) != 64:
        raise ValueError("ERA5 coverage lacks a sidecar bundle hash")
    return sha256_file(path), bundle_hash


def _completion_payload(
    seed_dir: Path,
    run_manifest: dict[str, Any],
    *,
    arm: str,
    seed: int,
    era5_coverage: Path | None = None,
) -> dict[str, Any]:
    if arm not in run_manifest["expected_arms"]:
        raise ValueError(f"Unexpected arm: {arm}")
    if seed not in run_manifest["seeds"]:
        raise ValueError(f"Unexpected seed: {seed}")

    log_path = seed_dir / "training_log.json"
    verdict_epoch = int(run_manifest["verdict_epoch"])
    checkpoint_path = seed_dir / f"epoch_{verdict_epoch:03d}.pt"
    log = load_json(log_path)
    expected_epochs = int(run_manifest["expected_epochs"])
    epoch_numbers = _validate_epoch_log(
        log,
        expected_epochs=expected_epochs,
        require_final_confusion=True,
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Fixed verdict checkpoint is missing: {checkpoint_path}")

    config = log.get("config")
    if not isinstance(config, dict):
        raise ValueError("Training log config is missing")
    _validate_fixed_checkpoint_storage(seed_dir, config)
    expected_config = {
        "seed": seed,
        "deterministic": True,
        "deterministic_warn_only": False,
        "epochs": expected_epochs,
        "num_classes": int(run_manifest["num_classes"]),
        "backbone_name": "prithvi_600m",
        "enable_multitemporal": True,
        "num_temporal_frames": 4,
        "enable_era5_channels": True,
        "era5_mode": arm,
        "freeze_spectral": True,
        "strict_checkpoint_loading": True,
        "log_confusion_every_epoch": True,
        "log_training_exposure": True,
        "fixed_checkpoint_only": True,
        "enable_collapse_rewind": False,
        "unfreeze_backbone_layers": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Training config mismatch: {mismatches}")

    exposure_sha256 = training_exposure_sha256(
        log,
        num_classes=int(run_manifest["num_classes"]),
        crop_names=run_manifest["val_supported_crop_classes"],
        thresholds=run_manifest["crop_support_thresholds"],
    )

    if arm == "treatment":
        if era5_coverage is None:
            raise ValueError("Treatment completion requires sealed ERA5 coverage")
        coverage_sha256, aux_bundle_sha256 = _coverage_identity(era5_coverage)
    else:
        if era5_coverage is not None:
            raise ValueError("Control completion must not reference treatment coverage")
        coverage_sha256 = None
        aux_bundle_sha256 = canonical_sha256(CONTROL_AUX_IDENTITY)
    return {
        "schema": COMPLETION_SCHEMA,
        "run_id": run_manifest["run_id"],
        "arm": arm,
        "seed": seed,
        "base_git_sha": run_manifest["base_git_sha"],
        "code_sha256": run_manifest["code_sha256"],
        "cohort_sha256": run_manifest["cohort_sha256"],
        "environment_sha256": run_manifest["environment_sha256"],
        "base_completion_sha256": run_manifest["base_completion_sha256"],
        "base_checkpoint_sha256": run_manifest["initial_checkpoint_sha256"],
        "era5_mode": arm,
        "era5_coverage_sha256": coverage_sha256,
        "era5_aux_bundle_sha256": aux_bundle_sha256,
        "config_sha256": canonical_sha256(normalized_training_config(config)),
        "training_log_sha256": sha256_file(log_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "verdict_epoch": verdict_epoch,
        "completed_epochs": epoch_numbers,
        "training_exposure_sha256": exposure_sha256,
    }


def finalize_seed(args: argparse.Namespace) -> None:
    run_manifest = load_json(args.run_manifest)
    if run_manifest.get("schema") != RUN_SCHEMA:
        raise ValueError("Unexpected run manifest schema")
    payload = _completion_payload(
        args.seed_dir, run_manifest, arm=args.arm, seed=args.seed,
        era5_coverage=getattr(args, "era5_coverage", None),
    )
    write_once_or_verify(args.seed_dir / "completion.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def validate_seed(args: argparse.Namespace) -> None:
    run_manifest = load_json(args.run_manifest)
    expected = _completion_payload(
        args.seed_dir, run_manifest, arm=args.arm, seed=args.seed,
        era5_coverage=getattr(args, "era5_coverage", None),
    )
    actual = load_json(args.seed_dir / "completion.json")
    if actual != expected:
        raise ValueError(f"Completion provenance mismatch: {args.seed_dir}")
    print(json.dumps(actual, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init-run")
    init.add_argument("--root", required=True, type=Path)
    init.add_argument("--run-id", required=True)
    init.add_argument("--base-git-sha", required=True)
    init.add_argument("--code-manifest", required=True, type=Path)
    init.add_argument("--cohort-manifest", required=True, type=Path)
    init.add_argument("--environment-dir", required=True, type=Path)
    init.add_argument("--initial-checkpoint", required=True, type=Path)
    init.add_argument("--base-completion", required=True, type=Path)
    init.add_argument("--foundation-checkpoint", required=True, type=Path)
    init.add_argument("--container-image", required=True)
    init.add_argument("--seeds", default="41,42,43")
    init.add_argument("--expected-epochs", type=int, default=5)
    init.add_argument("--num-classes", type=int, default=23)
    init.add_argument("--require-existing", action="store_true")
    init.set_defaults(func=init_run)

    for name, func in (("finalize-seed", finalize_seed),
                       ("validate-seed", validate_seed)):
        sub = commands.add_parser(name)
        sub.add_argument("--seed-dir", required=True, type=Path)
        sub.add_argument("--run-manifest", required=True, type=Path)
        sub.add_argument("--arm", required=True, choices=("control", "treatment"))
        sub.add_argument("--seed", required=True, type=int)
        sub.add_argument("--era5-coverage", type=Path)
        sub.set_defaults(func=func)

    for name, func in (
        ("finalize-base", finalize_base),
        ("validate-base", validate_base),
    ):
        sub = commands.add_parser(name)
        sub.add_argument("--base-dir", required=True, type=Path)
        sub.add_argument("--base-git-sha", required=True)
        sub.add_argument("--code-manifest", required=True, type=Path)
        sub.add_argument("--cohort-dir", required=True, type=Path)
        sub.add_argument("--environment-dir", required=True, type=Path)
        sub.add_argument("--foundation-checkpoint", required=True, type=Path)
        sub.add_argument("--seed", type=int, default=20260821)
        sub.add_argument("--expected-epochs", type=int, default=10)
        sub.add_argument("--num-classes", type=int, default=23)
        sub.set_defaults(func=func)

    lock = commands.add_parser("lock-file")
    lock.add_argument("--source", required=True, type=Path)
    lock.add_argument("--target", required=True, type=Path)
    lock.add_argument("--require-existing", action="store_true")
    lock.set_defaults(func=lock_file)

    verify_code = commands.add_parser("verify-code-bundle")
    verify_code.add_argument("--code-manifest", required=True, type=Path)
    verify_code.set_defaults(func=verify_code_command)

    verify_foundation = commands.add_parser("verify-foundation")
    verify_foundation.add_argument(
        "--foundation-checkpoint", required=True, type=Path,
    )
    verify_foundation.set_defaults(func=verify_foundation_command)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
