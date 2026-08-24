#!/usr/bin/env python3
"""Validate sealed runs and produce the paired ERA5 smoke-test verdict."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any


COMPLETION_SCHEMA = "era5-smoke-run-completion-v2"
RUN_MANIFEST_SCHEMA = "era5-smoke-run-v2"
VERDICT_SCHEMA = "era5-prithvi600m-smoke-v3"
ERA5_COVERAGE_SCHEMA = "era5-smoke-sidecar-bundle-v1"
TERMINAL_STATUSES = frozenset({"completed"})
CROP_CLASS_NAMES = (
    "vete", "korn", "havre", "oljeväxter", "slåttervall", "bete",
    "potatis", "sockerbetor", "trindsäd", "råg", "majs",
)
CROP_CLASS_INDICES = tuple(range(11, 22))
CONTROL_AUX_IDENTITY = {
    "schema": "era5-smoke-neutral-control-v1",
    "ordered_channels": [
        "era5_t2m_mean", "era5_tp_sum", "era5_swvl1_mean",
        "era5_ssrd_sum", "era5_gdd",
    ],
    "normalized_values": [0.0, 0.0, 0.0, 0.0, 0.0],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def cohort_bundle_sha256(cohort_dir: Path) -> str:
    """Hash the three logical files that define a persisted cohort."""
    digest = hashlib.sha256()
    for name in ("manifest.json", "split_train.txt", "split_val.txt"):
        path = cohort_dir / name
        if not path.is_file():
            raise ValueError(f"Missing cohort artifact: {path}")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def environment_bundle_sha256(environment_dir: Path) -> str:
    """Hash stable package, Python, and GPU identities for arm parity."""
    digest = hashlib.sha256()
    for name in ("gpu_identity.txt", "pip_freeze.txt", "python_version.txt"):
        path = environment_dir / name
        if not path.is_file():
            raise ValueError(f"Missing environment artifact: {path}")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _run_dir(root: Path, arm: str, seed: int) -> Path:
    return root / f"{arm}_seed{seed}"


def _validate_terminal_log(path: Path, *, arm: str, seed: int) -> dict:
    log = load_json(path)
    if log.get("status") not in TERMINAL_STATUSES:
        raise ValueError(
            f"Run {arm}/{seed} is not terminal: status={log.get('status')!r}"
        )
    if not isinstance(log.get("epochs"), list) or not log["epochs"]:
        raise ValueError(f"Run {arm}/{seed} has no completed epochs")
    config = log.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Run {arm}/{seed} has no serialized config")
    if config.get("seed") != seed:
        raise ValueError(
            f"Run {arm}/{seed} contains seed={config.get('seed')!r}"
        )
    if config.get("deterministic") is not True:
        raise ValueError(f"Run {arm}/{seed} was not deterministic")
    if config.get("deterministic_warn_only") is not False:
        raise ValueError(f"Run {arm}/{seed} did not enforce strict determinism")
    if config.get("log_training_exposure") is not True:
        raise ValueError(f"Run {arm}/{seed} lacks the training-exposure ledger")
    if config.get("enable_era5_channels") is not True:
        raise ValueError(f"Run {arm}/{seed} did not use the paired ERA5 architecture")
    if config.get("era5_mode") != arm:
        raise ValueError(
            f"Run {arm}/{seed} attests era5_mode={config.get('era5_mode')!r}"
        )
    required_protocol = {
        "backbone_name": "prithvi_600m",
        "enable_multitemporal": True,
        "num_temporal_frames": 4,
        "freeze_spectral": True,
        "strict_checkpoint_loading": True,
        "log_confusion_every_epoch": True,
        "fixed_checkpoint_only": True,
        "enable_collapse_rewind": False,
        "unfreeze_backbone_layers": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in required_protocol.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Run {arm}/{seed} protocol mismatch: {mismatches}")
    expected_dir = f"{arm}_seed{seed}"
    if Path(str(config.get("checkpoint_dir", ""))).name != expected_dir:
        raise ValueError(
            f"Run {arm}/{seed} checkpoint_dir does not identify {expected_dir}"
        )
    return log


def _normalized_config(config: dict) -> dict:
    """Match finalizer: exclude path and the deliberate arm treatment."""
    return {
        key: value for key, value in config.items()
        if key not in {"checkpoint_dir", "era5_mode", "era5_dir"}
    }


def _training_exposure_sha256(log: dict) -> str:
    evidence = []
    for epoch in log.get("epochs", []):
        evidence.append({
            "epoch": epoch.get("epoch"),
            "train_exposure_sha256": epoch.get("train_exposure_sha256"),
            "train_target_support": epoch.get("train_target_support"),
            "train_class_tiles": epoch.get("train_class_tiles"),
        })
    return json_sha256(evidence)


def _load_sealed_run(
    *,
    root: Path,
    run_id: str,
    arm: str,
    seed: int,
    verdict_epoch: int,
) -> dict:
    run_dir = _run_dir(root, arm, seed)
    log_path = run_dir / "training_log.json"
    checkpoint_path = run_dir / f"epoch_{verdict_epoch:03d}.pt"
    completion_path = run_dir / "completion.json"
    completion = load_json(completion_path)
    identity = {
        "schema": COMPLETION_SCHEMA,
        "run_id": run_id,
        "arm": arm,
        "seed": seed,
    }
    for key, expected in identity.items():
        if completion.get(key) != expected:
            raise ValueError(
                f"Completion identity mismatch in {completion_path}: "
                f"{key}={completion.get(key)!r}, expected {expected!r}"
            )

    log = _validate_terminal_log(log_path, arm=arm, seed=seed)
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise ValueError(f"Missing fixed verdict checkpoint: {checkpoint_path}")
    expected_hashes = {
        "training_log_sha256": file_sha256(log_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_sha256": json_sha256(_normalized_config(log["config"])),
    }
    for key, expected in expected_hashes.items():
        if completion.get(key) != expected:
            raise ValueError(
                f"Stale or mismatched {key} for {arm}/{seed}: "
                f"sealed={completion.get(key)!r}, current={expected!r}"
            )
    for key in (
        "cohort_sha256", "code_sha256", "base_git_sha", "environment_sha256",
        "base_completion_sha256", "base_checkpoint_sha256",
        "era5_aux_bundle_sha256", "training_exposure_sha256",
    ):
        if not isinstance(completion.get(key), str) or not completion[key]:
            raise ValueError(f"Completion for {arm}/{seed} lacks {key}")
    if completion.get("era5_mode") != arm:
        raise ValueError(f"Completion for {arm}/{seed} has wrong ERA5 mode")
    if completion.get("verdict_epoch") != verdict_epoch:
        raise ValueError(f"Completion for {arm}/{seed} has wrong verdict epoch")
    if completion.get("checkpoint_name") != checkpoint_path.name:
        raise ValueError(f"Completion for {arm}/{seed} has wrong checkpoint name")
    if completion["training_exposure_sha256"] != _training_exposure_sha256(log):
        raise ValueError(f"Training exposure ledger is stale for {arm}/{seed}")
    return {"log": log, "completion": completion}


def _fixed_epoch(
    log: dict,
    *,
    arm: str,
    seed: int,
    verdict_epoch: int,
    num_classes: int,
) -> dict:
    candidates = [
        item for item in log["epochs"] if item.get("epoch") == verdict_epoch
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Run {arm}/{seed} lacks unique verdict epoch {verdict_epoch}"
        )
    epoch = candidates[0]
    matrix = epoch.get("confusion_matrix")
    if not isinstance(matrix, list) or not matrix:
        raise ValueError(
            f"Fixed epoch for {arm}/{seed} lacks a confusion matrix; "
            "fixed-support metrics cannot be computed"
        )
    if len(matrix) != num_classes or any(
        not isinstance(row, list) or len(row) != num_classes
        for row in matrix
    ):
        raise ValueError(f"Invalid confusion matrix for {arm}/{seed}")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for row in matrix for value in row
    ):
        raise ValueError(f"Invalid confusion-matrix counts for {arm}/{seed}")
    return epoch


def _target_support(matrix: list[list[int]]) -> tuple[int, ...]:
    return tuple(sum(int(value) for value in row) for row in matrix)


def _iou(matrix: list[list[int]], class_index: int) -> float:
    true_positive = int(matrix[class_index][class_index])
    false_positive = sum(int(row[class_index]) for row in matrix) - true_positive
    false_negative = sum(int(value) for value in matrix[class_index]) - true_positive
    denominator = true_positive + false_positive + false_negative
    if denominator <= 0:
        raise ValueError(f"Class {class_index} has no IoU denominator")
    return true_positive / denominator


def _load_run_manifest(root: Path, *, run_id: str, seeds: list[int]) -> dict:
    path = root / "run_manifest.json"
    manifest = load_json(path)
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        raise ValueError(f"Unexpected run-manifest schema in {path}")
    if manifest.get("run_id") != run_id:
        raise ValueError(
            f"Requested run_id={run_id!r} does not match manifest "
            f"run_id={manifest.get('run_id')!r}"
        )
    if manifest.get("seeds") != seeds:
        raise ValueError(
            f"Requested seeds={seeds!r} do not match manifest seeds={manifest.get('seeds')!r}"
        )
    expected_arms = manifest.get("expected_arms")
    if (not isinstance(expected_arms, list)
            or sorted(expected_arms) != ["control", "treatment"]):
        raise ValueError("Run manifest must require exactly control and treatment arms")
    required_strings = (
        "base_git_sha", "code_sha256", "cohort_sha256", "environment_sha256",
        "initial_checkpoint_sha256", "foundation_checkpoint_sha256",
        "base_completion_sha256", "container_image",
    )
    for key in required_strings:
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"Run manifest lacks {key}")
    if not re.fullmatch(
        r"ghcr\.io/tobiasedman/imint-era5-smoke@sha256:[0-9a-f]{64}",
        manifest["container_image"],
    ):
        raise ValueError("Run manifest container image is not digest-pinned")
    if not run_id.endswith(manifest["code_sha256"][:12]):
        raise ValueError("run_id is not bound to code_sha256[:12]")
    if (not isinstance(manifest.get("expected_epochs"), int)
            or isinstance(manifest["expected_epochs"], bool)
            or manifest["expected_epochs"] <= 0):
        raise ValueError("Run manifest expected_epochs must be a positive integer")
    if manifest.get("verdict_epoch") != manifest["expected_epochs"]:
        raise ValueError("Run manifest must score the preregistered final epoch")
    if (
        not isinstance(manifest.get("base_expected_epochs"), int)
        or manifest["base_expected_epochs"] <= 0
        or not isinstance(manifest.get("base_seed"), int)
    ):
        raise ValueError("Run manifest lacks the fixed common-base protocol")
    if (not isinstance(manifest.get("num_classes"), int)
            or isinstance(manifest["num_classes"], bool)
            or manifest["num_classes"] <= 1):
        raise ValueError("Run manifest num_classes must be an integer greater than one")
    if (
        not isinstance(manifest.get("val_tile_count"), int)
        or manifest["val_tile_count"] <= 0
        or manifest.get("model_patch_px") != 504
    ):
        raise ValueError("Run manifest lacks exact validation geometry")
    crop_names = manifest.get("val_supported_crop_classes")
    if (not isinstance(crop_names, list) or not crop_names
            or any(name not in CROP_CLASS_NAMES for name in crop_names)
            or len(crop_names) != len(set(crop_names))):
        raise ValueError("Run manifest has invalid val_supported_crop_classes")
    crop_support_thresholds = manifest.get("crop_support_thresholds")
    if (
        not isinstance(crop_support_thresholds, dict)
        or set(crop_support_thresholds) != {
            "min_train_tiles", "min_val_tiles",
            "min_train_pixels", "min_val_pixels",
        }
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in crop_support_thresholds.values()
        )
    ):
        raise ValueError("Run manifest has invalid crop-support thresholds")
    val_target_support = manifest.get("val_target_support")
    if (
        not isinstance(val_target_support, list)
        or len(val_target_support) != manifest["num_classes"]
        or any(not isinstance(value, int) or value < 0 for value in val_target_support)
        or sum(val_target_support) <= 0
    ):
        raise ValueError("Run manifest has invalid exact validation support")
    expected_val_pixels = manifest["val_tile_count"] * manifest["model_patch_px"] ** 2
    if sum(val_target_support) != expected_val_pixels:
        raise ValueError("Run manifest validation support has the wrong pixel total")
    thresholds = manifest.get("thresholds")
    required_thresholds = {
        "verify_median_delta_miou", "verify_median_delta_crop_macro_iou",
        "verify_positive_pairs",
    }
    if not isinstance(thresholds, dict) or set(thresholds) != required_thresholds:
        raise ValueError("Run manifest has invalid thresholds")
    if (not all(isinstance(thresholds[key], (int, float))
                and not isinstance(thresholds[key], bool)
                for key in required_thresholds)
            or not isinstance(thresholds["verify_positive_pairs"], int)
            or thresholds["verify_positive_pairs"] < 1):
        raise ValueError("Run manifest thresholds must be numeric")
    actual_cohort_digest = cohort_bundle_sha256(root / "cohort")
    if manifest["cohort_sha256"] != actual_cohort_digest:
        raise ValueError("Run manifest cohort_sha256 does not match root/cohort")
    cohort_manifest = load_json(root / "cohort" / "manifest.json")
    if cohort_manifest.get("schema") != "era5-smoke-cohort-v4":
        raise ValueError("Unexpected sealed cohort-manifest schema")
    if (
        cohort_manifest.get("counts", {}).get("val")
        != manifest["val_tile_count"]
        or cohort_manifest.get("model_patch_px") != manifest["model_patch_px"]
        or cohort_manifest.get("val_supported_crop_classes")
        != manifest["val_supported_crop_classes"]
    ):
        raise ValueError("Run manifest validation contract differs from cohort")
    cohort_val_counts = cohort_manifest.get("label_support", {}).get(
        "val", {}
    ).get("pixel_counts", {})
    if not isinstance(cohort_val_counts, dict):
        raise ValueError("Sealed cohort lacks exact validation pixel counts")
    cohort_target_support = [
        int(cohort_val_counts.get(str(index), 0))
        for index in range(manifest["num_classes"])
    ]
    if cohort_target_support != val_target_support:
        raise ValueError("Run manifest validation support differs from cohort")
    if cohort_manifest.get("crop_support_thresholds") != crop_support_thresholds:
        raise ValueError("Run manifest crop thresholds differ from cohort")
    actual_environment_digest = environment_bundle_sha256(root / "environment")
    if manifest["environment_sha256"] != actual_environment_digest:
        raise ValueError(
            "Run manifest environment_sha256 does not match root/environment"
        )
    base_completion_path = root / "base_model" / "completion.json"
    if manifest["base_completion_sha256"] != file_sha256(base_completion_path):
        raise ValueError("Run manifest base completion hash is stale")
    base_completion = load_json(base_completion_path)
    if (
        not isinstance(base_completion.get("training_exposure_sha256"), str)
        or len(base_completion["training_exposure_sha256"]) != 64
    ):
        raise ValueError("Common base lacks a sealed training-exposure ledger")
    base_log_path = root / "base_model" / "training_log.json"
    base_log = load_json(base_log_path)
    if (
        base_completion.get("training_log_sha256") != file_sha256(base_log_path)
        or base_completion["training_exposure_sha256"]
        != _training_exposure_sha256(base_log)
        or base_log.get("status") != "completed"
        or [epoch.get("epoch") for epoch in base_log.get("epochs", [])]
        != list(range(1, manifest["base_expected_epochs"] + 1))
    ):
        raise ValueError("Common-base training evidence is stale or incomplete")
    base_checkpoint = root / "base_model" / base_completion.get("checkpoint_name", "")
    if (
        base_completion.get("checkpoint_sha256")
        != manifest["initial_checkpoint_sha256"]
        or file_sha256(base_checkpoint) != manifest["initial_checkpoint_sha256"]
    ):
        raise ValueError("Run manifest common-base checkpoint is stale")
    return manifest


def _sidecar_bundle_sha256(era5_dir: Path, names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        path = era5_dir / name
        if not path.is_file():
            raise ValueError(f"Missing sealed ERA5 sidecar: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_era5_coverage(root: Path) -> tuple[str, str]:
    era5_dir = root / "era5"
    path = era5_dir / "coverage.json"
    coverage = load_json(path)
    if coverage.get("schema") != ERA5_COVERAGE_SCHEMA:
        raise ValueError("Unexpected ERA5 coverage schema")
    tiles = coverage.get("tiles")
    if (
        not isinstance(tiles, list)
        or len(tiles) != coverage.get("requested")
        or coverage.get("requested") != coverage.get("valid")
        or coverage.get("atmosphere_cells", {}).get("overlap") != []
    ):
        raise ValueError("ERA5 coverage is incomplete or overlapping")
    names = [item.get("name") for item in tiles]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("ERA5 coverage contains an invalid tile identity")
    if len(names) != len(set(names)):
        raise ValueError("ERA5 coverage contains duplicate tile identities")
    for item in tiles:
        sidecar = era5_dir / item["name"]
        if file_sha256(sidecar) != item.get("sidecar_sha256"):
            raise ValueError(f"ERA5 sidecar hash mismatch: {sidecar}")
    bundle_hash = _sidecar_bundle_sha256(era5_dir, names)
    if bundle_hash != coverage.get("sidecar_bundle_sha256"):
        raise ValueError("ERA5 sidecar bundle hash mismatch")
    return file_sha256(path), bundle_hash


def analyze(*, root: Path, run_id: str, seeds: list[int]) -> dict:
    """Validate all six sealed runs and calculate fixed-support metrics."""
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("The smoke verdict requires exactly three unique seeds")
    manifest = _load_run_manifest(root, run_id=run_id, seeds=seeds)
    verdict_epoch = int(manifest["verdict_epoch"])
    coverage_sha256, treatment_aux_bundle_sha256 = _validate_era5_coverage(root)
    control_aux_bundle_sha256 = json_sha256(CONTROL_AUX_IDENTITY)

    loaded: dict[tuple[str, int], dict] = {}
    for seed in seeds:
        for arm in ("control", "treatment"):
            loaded[(arm, seed)] = _load_sealed_run(
                root=root, run_id=run_id, arm=arm, seed=seed,
                verdict_epoch=verdict_epoch,
            )

    # Code, cohort, base commit, and run identity are invariants across all six.
    invariant_keys = (
        "cohort_sha256", "code_sha256", "base_git_sha", "environment_sha256",
        "base_completion_sha256",
    )
    for key in invariant_keys:
        expected = manifest[key]
        for (arm, seed), run in loaded.items():
            if run["completion"][key] != expected:
                raise ValueError(f"{key} differs for {arm}/{seed}")
    for (arm, seed), run in loaded.items():
        completion = run["completion"]
        if completion["base_checkpoint_sha256"] != manifest["initial_checkpoint_sha256"]:
            raise ValueError(f"Base checkpoint differs for {arm}/{seed}")
        expected_aux_bundle = (
            control_aux_bundle_sha256
            if arm == "control" else treatment_aux_bundle_sha256
        )
        expected_coverage = None if arm == "control" else coverage_sha256
        if (
            completion["era5_aux_bundle_sha256"] != expected_aux_bundle
            or completion.get("era5_coverage_sha256") != expected_coverage
        ):
            raise ValueError(f"ERA5 evidence differs for {arm}/{seed}")

    # Config includes seed, so parity is asserted within each seed pair only.
    for seed in seeds:
        control = loaded[("control", seed)]
        treatment = loaded[("treatment", seed)]
        if control["completion"]["config_sha256"] != treatment["completion"]["config_sha256"]:
            raise ValueError(f"Config hash differs between arms for seed {seed}")
        if _normalized_config(control["log"]["config"]) != _normalized_config(
            treatment["log"]["config"]
        ):
            raise ValueError(f"Training config differs between arms for seed {seed}")
        if (
            control["completion"]["training_exposure_sha256"]
            != treatment["completion"]["training_exposure_sha256"]
        ):
            raise ValueError(
                f"Realized training exposure differs between arms for seed {seed}"
            )
        for arm, run in (("control", control), ("treatment", treatment)):
            config = run["log"]["config"]
            if config.get("num_classes") != manifest["num_classes"]:
                raise ValueError(f"num_classes mismatch for {arm}/{seed}")
            epoch_numbers = [epoch.get("epoch") for epoch in run["log"]["epochs"]]
            expected_epoch_numbers = list(range(1, manifest["expected_epochs"] + 1))
            if epoch_numbers != expected_epoch_numbers:
                raise ValueError(
                    f"Run {arm}/{seed} is incomplete: epochs={epoch_numbers!r}, "
                    f"expected={expected_epoch_numbers!r}"
                )

    epochs = {
        key: _fixed_epoch(
            run["log"], arm=key[0], seed=key[1],
            verdict_epoch=verdict_epoch,
            num_classes=manifest["num_classes"],
        )
        for key, run in loaded.items()
    }
    support_vectors = {
        key: _target_support(epoch["confusion_matrix"])
        for key, epoch in epochs.items()
    }
    reference_support = next(iter(support_vectors.values()))
    if any(support != reference_support for support in support_vectors.values()):
        raise ValueError("Validation target support differs across arms/seeds")
    if reference_support != tuple(manifest["val_target_support"]):
        raise ValueError(
            "Confusion-matrix target support differs from the sealed cohort"
        )
    supported_classes = tuple(
        index for index, count in enumerate(reference_support)
        if index != 0 and count > 0
    )
    crop_index_by_name = {
        name: index for name, index in zip(CROP_CLASS_NAMES, CROP_CLASS_INDICES)
    }
    supported_crops = tuple(
        crop_index_by_name[name]
        for name in manifest["val_supported_crop_classes"]
    )
    if not supported_classes:
        raise ValueError("Validation cohort has no supported non-background classes")
    if not supported_crops:
        raise ValueError("Validation cohort has no supported crop classes")
    if any(
        index >= len(reference_support) or reference_support[index] <= 0
        for index in supported_crops
    ):
        raise ValueError(
            "A pre-registered crop class lacks fixed validation target support"
        )
    supported_crop_names = list(manifest["val_supported_crop_classes"])

    pairs = []
    for seed in seeds:
        metrics = {}
        for arm in ("control", "treatment"):
            epoch = epochs[(arm, seed)]
            matrix = epoch["confusion_matrix"]
            metrics[arm] = {
                "epoch": epoch["epoch"],
                "miou": statistics.fmean(_iou(matrix, i) for i in supported_classes),
                "crop_macro_iou": statistics.fmean(_iou(matrix, i) for i in supported_crops),
            }
        pairs.append({
            "seed": seed,
            "control": metrics["control"],
            "treatment": metrics["treatment"],
            "delta_miou": metrics["treatment"]["miou"] - metrics["control"]["miou"],
            "delta_crop_macro_iou": (
                metrics["treatment"]["crop_macro_iou"]
                - metrics["control"]["crop_macro_iou"]
            ),
        })

    median_miou = statistics.median(pair["delta_miou"] for pair in pairs)
    median_crop = statistics.median(pair["delta_crop_macro_iou"] for pair in pairs)
    positive = sum(
        pair["delta_miou"] > 0 and pair["delta_crop_macro_iou"] > 0
        for pair in pairs
    )
    thresholds = manifest["thresholds"]
    if (median_miou >= thresholds["verify_median_delta_miou"]
            and median_crop >= thresholds["verify_median_delta_crop_macro_iou"]
            and positive >= thresholds["verify_positive_pairs"]):
        verdict = "passes_smoke_threshold"
        reason = "ERA5 clears the pre-registered overall and crop-IoU thresholds."
    elif median_miou <= 0 and median_crop <= 0 and positive <= 1:
        verdict = "fails_smoke_threshold"
        reason = "ERA5 does not improve either paired median metric."
    else:
        verdict = "inconclusive"
        reason = "Effect is mixed or below the pre-registered smoke thresholds."
    return {
        "schema": VERDICT_SCHEMA,
        "run_id": run_id,
        "seeds": seeds,
        **{key: manifest[key] for key in invariant_keys},
        "run_manifest_sha256": file_sha256(root / "run_manifest.json"),
        "supported_class_indices": list(supported_classes),
        "supported_crop_indices": list(supported_crops),
        "supported_crop_names": supported_crop_names,
        "pairs": pairs,
        "median_delta_miou": median_miou,
        "median_delta_crop_macro_iou": median_crop,
        "positive_pairs": positive,
        "verdict": verdict,
        "reason": reason,
        "thresholds": thresholds,
        "claim_scope": (
            "Preregistered three-seed decision for the fixed five-epoch "
            "AUX-only adaptation protocol; not a population-level "
            "significance or equivalence claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",")]
    result = analyze(root=args.root, run_id=args.run_id, seeds=seeds)
    output = args.output or args.root / "verdict.json"
    atomic_write_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
