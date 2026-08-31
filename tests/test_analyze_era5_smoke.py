"""Reproducibility gates for the sealed ERA5 A/B verdict."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_era5_smoke import (
    COMPLETION_SCHEMA,
    CONTROL_AUX_IDENTITY,
    ERA5_COVERAGE_SCHEMA,
    RUN_MANIFEST_SCHEMA,
    _normalized_config,
    _sidecar_bundle_sha256,
    _training_exposure_sha256,
    analyze,
    cohort_bundle_sha256,
    environment_bundle_sha256,
    file_sha256,
    json_sha256,
)


SEEDS = [41, 42, 43]
CODE_SHA = "a" * 64
RUN_ID = f"era5-20260821-{CODE_SHA[:12]}"
MODEL_PIXELS = 504 * 504
TARGET_SUPPORT = [0] * 23
TARGET_SUPPORT[1] = 100_000
TARGET_SUPPORT[11] = 80_000
TARGET_SUPPORT[12] = MODEL_PIXELS - TARGET_SUPPORT[1] - TARGET_SUPPORT[11]


def _matrix(*, control: bool) -> list[list[int]]:
    matrix = [[0] * 23 for _ in range(23)]
    scores = (50_000, 40_000, 30_000) if control else (60_000, 55_000, 50_000)
    for class_index, correct in zip((1, 11, 12), scores):
        matrix[class_index][class_index] = correct
        matrix[class_index][0] = TARGET_SUPPORT[class_index] - correct
    # Class 13 has no target support. Only treatment predicts it, which must
    # not change the fixed ground-truth-supported macro denominator.
    if not control:
        matrix[1][13] = 5
        matrix[1][0] -= 5
    return matrix


def _write_run(
    root: Path,
    arm: str,
    seed: int,
    cohort_digest: str,
    environment_digest: str,
    base_completion_digest: str,
    base_checkpoint_digest: str,
    coverage_digest: str,
    treatment_bundle_digest: str,
) -> None:
    run_dir = root / f"{arm}_seed{seed}"
    run_dir.mkdir(parents=True)
    config = {
        "seed": seed,
        "deterministic": True,
        "deterministic_warn_only": False,
        "enable_era5_channels": True,
        "era5_mode": arm,
        "era5_dir": str(root / "era5") if arm == "treatment" else None,
        "backbone_name": "prithvi_600m",
        "enable_multitemporal": True,
        "num_temporal_frames": 4,
        "freeze_spectral": True,
        "strict_checkpoint_loading": True,
        "log_confusion_every_epoch": True,
        "log_training_exposure": True,
        "fixed_checkpoint_only": True,
        "enable_collapse_rewind": False,
        "unfreeze_backbone_layers": 0,
        "checkpoint_dir": str(run_dir),
        "num_classes": 23,
        "weight_decay": 0.35,
    }
    log = {
        "status": "completed",
        "config": config,
        "epochs": [{
            "epoch": 1,
            "val_miou": 0.4 if arm == "control" else 0.5,
            "confusion_matrix": _matrix(control=arm == "control"),
            "train_exposure_sha256": f"{seed % 10}" * 64,
            "train_target_support": [
                0, *([0] * 10), 2048, 2048, *([0] * 10),
            ],
            "train_class_tiles": {
                "11": ["train-a.npz", "train-b.npz"],
                "12": ["train-a.npz", "train-b.npz"],
            },
        }],
    }
    log_path = run_dir / "training_log.json"
    log_path.write_text(json.dumps(log))
    checkpoint_path = run_dir / "epoch_001.pt"
    checkpoint_path.write_bytes(f"checkpoint:{arm}:{seed}".encode())
    normalized_config = _normalized_config(config)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "run_id": RUN_ID,
        "arm": arm,
        "seed": seed,
        "cohort_sha256": cohort_digest,
        "code_sha256": CODE_SHA,
        "base_git_sha": "deadbeef",
        "environment_sha256": environment_digest,
        "base_completion_sha256": base_completion_digest,
        "base_checkpoint_sha256": base_checkpoint_digest,
        "era5_mode": arm,
        "era5_coverage_sha256": coverage_digest if arm == "treatment" else None,
        "era5_aux_bundle_sha256": (
            treatment_bundle_digest
            if arm == "treatment" else json_sha256(CONTROL_AUX_IDENTITY)
        ),
        "config_sha256": json_sha256(normalized_config),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "training_log_sha256": file_sha256(log_path),
        "checkpoint_name": checkpoint_path.name,
        "verdict_epoch": 1,
        "training_exposure_sha256": _training_exposure_sha256(log),
    }
    (run_dir / "completion.json").write_text(json.dumps(completion))


@pytest.fixture
def sealed_root(tmp_path: Path) -> Path:
    cohort_dir = tmp_path / "cohort"
    cohort_dir.mkdir()
    cohort_manifest = {
        "schema": "era5-smoke-cohort-v4",
        "counts": {"train": 1, "val": 1},
        "model_patch_px": 504,
        "val_supported_crop_classes": ["vete", "korn"],
        "crop_support_thresholds": {
            "min_train_tiles": 2,
            "min_val_tiles": 2,
            "min_train_pixels": 1024,
            "min_val_pixels": 1024,
        },
        "label_support": {
            "val": {
                "pixel_counts": {
                    str(index): count
                    for index, count in enumerate(TARGET_SUPPORT)
                    if count
                },
            },
        },
    }
    (cohort_dir / "manifest.json").write_text(json.dumps(cohort_manifest))
    (cohort_dir / "split_train.txt").write_text("train.npz\n")
    (cohort_dir / "split_val.txt").write_text("val.npz\n")
    cohort_digest = cohort_bundle_sha256(cohort_dir)
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "gpu_identity.txt").write_text("NVIDIA H100, 570.0\n")
    (environment_dir / "pip_freeze.txt").write_text("torch==2.6.0\n")
    (environment_dir / "python_version.txt").write_text("Python 3.12.0\n")
    environment_digest = environment_bundle_sha256(environment_dir)
    base_dir = tmp_path / "base_model"
    base_dir.mkdir()
    base_checkpoint = base_dir / "epoch_010.pt"
    base_checkpoint.write_bytes(b"fixed common base")
    base_checkpoint_digest = file_sha256(base_checkpoint)
    base_log = {
        "status": "completed",
        "epochs": [
            {
                "epoch": epoch,
                "train_exposure_sha256": f"{epoch % 10}" * 64,
                "train_target_support": [
                    0, *([0] * 10), 2048, 2048, *([0] * 10),
                ],
                "train_class_tiles": {
                    "11": ["train-a.npz", "train-b.npz"],
                    "12": ["train-a.npz", "train-b.npz"],
                },
            }
            for epoch in range(1, 11)
        ],
    }
    base_log_path = base_dir / "training_log.json"
    base_log_path.write_text(json.dumps(base_log))
    base_completion = {
        "checkpoint_name": base_checkpoint.name,
        "checkpoint_sha256": base_checkpoint_digest,
        "training_log_sha256": file_sha256(base_log_path),
        "training_exposure_sha256": _training_exposure_sha256(base_log),
    }
    base_completion_path = base_dir / "completion.json"
    base_completion_path.write_text(json.dumps(base_completion))
    base_completion_digest = file_sha256(base_completion_path)

    era5_dir = tmp_path / "era5"
    era5_dir.mkdir()
    for name, payload in (("train.npz", b"weather-train"),
                          ("val.npz", b"weather-val")):
        (era5_dir / name).write_bytes(payload)
    treatment_bundle_digest = _sidecar_bundle_sha256(
        era5_dir, ["train.npz", "val.npz"],
    )
    coverage = {
        "schema": ERA5_COVERAGE_SCHEMA,
        "requested": 2,
        "valid": 2,
        "atmosphere_cells": {"train": ["a"], "val": ["b"], "overlap": []},
        "sidecar_bundle_sha256": treatment_bundle_digest,
        "tiles": [
            {"name": name, "sidecar_sha256": file_sha256(era5_dir / name)}
            for name in ("train.npz", "val.npz")
        ],
    }
    coverage_path = era5_dir / "coverage.json"
    coverage_path.write_text(json.dumps(coverage))
    coverage_digest = file_sha256(coverage_path)
    run_manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "run_id": RUN_ID,
        "base_git_sha": "deadbeef",
        "code_sha256": CODE_SHA,
        "cohort_sha256": cohort_digest,
        "environment_sha256": environment_digest,
        "initial_checkpoint_sha256": base_checkpoint_digest,
        "base_completion_sha256": base_completion_digest,
        "base_expected_epochs": 10,
        "base_seed": 20260821,
        "foundation_checkpoint_sha256": "c" * 64,
        "container_image": (
            "ghcr.io/tobiasedman/imint-era5-smoke@sha256:" + "d" * 64
        ),
        "seeds": SEEDS,
        "expected_arms": ["control", "treatment"],
        "expected_epochs": 1,
        "verdict_epoch": 1,
        "num_classes": 23,
        "val_tile_count": 1,
        "model_patch_px": 504,
        "val_supported_crop_classes": ["vete", "korn"],
        "crop_support_thresholds": {
            "min_train_tiles": 2,
            "min_val_tiles": 2,
            "min_train_pixels": 1024,
            "min_val_pixels": 1024,
        },
        "val_target_support": TARGET_SUPPORT,
        "thresholds": {
            "verify_median_delta_miou": 0.005,
            "verify_median_delta_crop_macro_iou": 0.01,
            "verify_positive_pairs": 2,
        },
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(run_manifest))
    for seed in SEEDS:
        for arm in ("control", "treatment"):
            _write_run(
                tmp_path, arm, seed, cohort_digest, environment_digest,
                base_completion_digest, base_checkpoint_digest,
                coverage_digest, treatment_bundle_digest,
            )
    return tmp_path


def _rewrite_completion_hashes(run_dir: Path) -> None:
    log_path = run_dir / "training_log.json"
    log = json.loads(log_path.read_text())
    completion_path = run_dir / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["training_log_sha256"] = file_sha256(log_path)
    completion["config_sha256"] = json_sha256(
        _normalized_config(log["config"])
    )
    completion["training_exposure_sha256"] = _training_exposure_sha256(log)
    completion_path.write_text(json.dumps(completion))


def test_analyze_uses_one_fixed_ground_truth_support(sealed_root: Path):
    result = analyze(root=sealed_root, run_id=RUN_ID, seeds=SEEDS)

    assert result["supported_class_indices"] == [1, 11, 12]
    assert result["supported_crop_indices"] == [11, 12]
    assert result["supported_crop_names"] == ["vete", "korn"]
    assert len(result["pairs"]) == 3
    assert all(pair["delta_crop_macro_iou"] > 0 for pair in result["pairs"])


@pytest.mark.parametrize("failure", ["running", "tampered_checkpoint", "wrong_arm"])
def test_analyze_rejects_unsealed_or_mismatched_runs(
    sealed_root: Path, failure: str,
):
    run_dir = sealed_root / "control_seed41"
    if failure == "running":
        log_path = run_dir / "training_log.json"
        log = json.loads(log_path.read_text())
        log["status"] = "running"
        log_path.write_text(json.dumps(log))
    elif failure == "tampered_checkpoint":
        (run_dir / "epoch_001.pt").write_bytes(b"changed after sealing")
    else:
        completion_path = run_dir / "completion.json"
        completion = json.loads(completion_path.read_text())
        completion["arm"] = "treatment"
        completion_path.write_text(json.dumps(completion))

    with pytest.raises(ValueError):
        analyze(root=sealed_root, run_id=RUN_ID, seeds=SEEDS)


def test_analyze_rejects_code_or_base_revision_mismatch(sealed_root: Path):
    completion_path = sealed_root / "treatment_seed42" / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["base_git_sha"] = "cafebabe"
    completion_path.write_text(json.dumps(completion))

    with pytest.raises(ValueError, match="base_git_sha differs"):
        analyze(root=sealed_root, run_id=RUN_ID, seeds=SEEDS)


def test_analyze_rejects_config_mismatch_even_when_resealed(sealed_root: Path):
    run_dir = sealed_root / "treatment_seed42"
    log_path = run_dir / "training_log.json"
    log = json.loads(log_path.read_text())
    log["config"]["weight_decay"] = 0.1
    log_path.write_text(json.dumps(log))
    _rewrite_completion_hashes(run_dir)

    with pytest.raises(ValueError, match="Config hash differs"):
        analyze(root=sealed_root, run_id=RUN_ID, seeds=SEEDS)


def test_analyze_rejects_changed_validation_support(sealed_root: Path):
    run_dir = sealed_root / "treatment_seed43"
    log_path = run_dir / "training_log.json"
    log = json.loads(log_path.read_text())
    log["epochs"][0]["confusion_matrix"][11][0] += 1
    log_path.write_text(json.dumps(log))
    _rewrite_completion_hashes(run_dir)

    with pytest.raises(ValueError, match="target support differs"):
        analyze(root=sealed_root, run_id=RUN_ID, seeds=SEEDS)


def test_analyze_rejects_different_realized_training_exposure(sealed_root: Path):
    run_dir = sealed_root / "treatment_seed41"
    log_path = run_dir / "training_log.json"
    log = json.loads(log_path.read_text())
    log["epochs"][0]["train_exposure_sha256"] = "f" * 64
    log_path.write_text(json.dumps(log))
    _rewrite_completion_hashes(run_dir)

    with pytest.raises(ValueError, match="Realized training exposure differs"):
        analyze(root=sealed_root, run_id=RUN_ID, seeds=SEEDS)
