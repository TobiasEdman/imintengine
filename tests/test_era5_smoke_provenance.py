"""Regression tests for immutable ERA5 smoke-test provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

import scripts.era5_smoke_provenance as provenance
from scripts.era5_smoke_provenance import (
    CODE_BUNDLE_SCHEMA,
    _base_completion_payload,
    _completion_payload,
    build_run_manifest,
    cohort_bundle_sha256,
    code_bundle_sha256,
    environment_bundle_sha256,
    validate_seed,
    verify_foundation_checkpoint,
    write_once_or_verify,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    seed_dir = tmp_path / "control_seed41"
    seed_dir.mkdir()
    checkpoint = seed_dir / "epoch_002.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = {
        "seed": 41,
        "deterministic": True,
        "deterministic_warn_only": False,
        "epochs": 2,
        "num_classes": 23,
        "backbone_name": "prithvi_600m",
        "enable_multitemporal": True,
        "num_temporal_frames": 4,
        "enable_era5_channels": True,
        "era5_mode": "control",
        "freeze_spectral": True,
        "strict_checkpoint_loading": True,
        "log_confusion_every_epoch": True,
        "log_training_exposure": True,
        "fixed_checkpoint_only": True,
        "enable_collapse_rewind": False,
        "unfreeze_backbone_layers": 0,
        "checkpoint_dir": "/attempt-specific/path",
    }
    _write_json(seed_dir / "training_log.json", {
        "status": "completed",
        "config": config,
        "best_epoch": 2,
        "epochs": [
            {
                "epoch": 1,
                "train_exposure_sha256": "1" * 64,
                "train_target_support": [0] * 11 + [2048, 2048] + [0] * 10,
                "train_class_tiles": {
                    "11": ["a.npz", "b.npz"],
                    "12": ["a.npz", "b.npz"],
                },
            },
            {
                "epoch": 2,
                "confusion_matrix": [[1]],
                "train_exposure_sha256": "2" * 64,
                "train_target_support": [0] * 11 + [2048, 2048] + [0] * 10,
                "train_class_tiles": {
                    "11": ["a.npz", "b.npz"],
                    "12": ["a.npz", "b.npz"],
                },
            },
        ],
    })
    run = {
        "schema": "era5-smoke-run-v2",
        "run_id": "smoke-test",
        "expected_arms": ["control", "treatment"],
        "seeds": [41],
        "expected_epochs": 2,
        "verdict_epoch": 2,
        "num_classes": 23,
        "base_git_sha": "a" * 40,
        "code_sha256": "b" * 64,
        "cohort_sha256": "c" * 64,
        "environment_sha256": "d" * 64,
        "base_completion_sha256": "e" * 64,
        "initial_checkpoint_sha256": "f" * 64,
        "val_supported_crop_classes": ["vete", "korn"],
        "crop_support_thresholds": {
            "min_train_tiles": 2,
            "min_val_tiles": 2,
            "min_train_pixels": 1024,
            "min_val_pixels": 1024,
        },
    }
    return seed_dir, run


def test_code_bundle_hash_binds_names_and_contents(tmp_path):
    files = {"a.py": b"one", "b.py": b"two"}
    digest = code_bundle_sha256(files)
    assert digest == code_bundle_sha256(dict(reversed(list(files.items()))))
    assert digest != code_bundle_sha256({"renamed.py": b"one", "b.py": b"two"})

    manifest = {
        "schema": CODE_BUNDLE_SCHEMA,
        "bundle_sha256": digest,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    assert len(manifest["bundle_sha256"]) == 64


def test_cohort_hash_binds_manifest_and_both_split_files(tmp_path):
    (tmp_path / "manifest.json").write_text("{}\n")
    (tmp_path / "split_train.txt").write_text("a.npz\n")
    (tmp_path / "split_val.txt").write_text("b.npz\n")
    digest = cohort_bundle_sha256(tmp_path)
    (tmp_path / "split_val.txt").write_text("c.npz\n")
    assert cohort_bundle_sha256(tmp_path) != digest


def test_environment_hash_requires_all_stable_facts(tmp_path):
    (tmp_path / "gpu_identity.txt").write_text("H100, 570.0\n")
    (tmp_path / "pip_freeze.txt").write_text("torch==2.6.0\n")
    with pytest.raises(FileNotFoundError, match="python_version"):
        environment_bundle_sha256(tmp_path)
    (tmp_path / "python_version.txt").write_text("Python 3.12.0\n")
    assert len(environment_bundle_sha256(tmp_path)) == 64


def _set_test_foundation_identity(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> None:
    monkeypatch.setattr(
        provenance, "FOUNDATION_CHECKPOINT_SIZE_BYTES", len(payload),
    )
    monkeypatch.setattr(
        provenance,
        "FOUNDATION_CHECKPOINT_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )


def test_published_foundation_identity_constants_are_exact():
    assert provenance.FOUNDATION_CHECKPOINT_SIZE_BYTES == 2_638_217_218
    assert provenance.FOUNDATION_CHECKPOINT_SHA256 == (
        "7b92c53b0204a76bb775bd8930f045e"
        "05776251caa8c83f7367ed0b75b594702"
    )


def test_foundation_checkpoint_requires_exact_published_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b"official-prithvi-600m-tl-fixture"
    _set_test_foundation_identity(monkeypatch, payload)
    checkpoint = tmp_path / provenance.FOUNDATION_CHECKPOINT_NAME
    checkpoint.write_bytes(payload)

    identity = verify_foundation_checkpoint(checkpoint)

    assert identity == {
        "name": "Prithvi_EO_V2_600M_TL.pt",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_foundation_checkpoint_rejects_wrong_size_before_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b"official-prithvi-600m-tl-fixture"
    _set_test_foundation_identity(monkeypatch, payload)
    checkpoint = tmp_path / provenance.FOUNDATION_CHECKPOINT_NAME
    checkpoint.write_bytes(payload[:-1])

    with pytest.raises(ValueError, match="size mismatch"):
        verify_foundation_checkpoint(checkpoint)


def test_foundation_checkpoint_rejects_same_size_wrong_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b"official-prithvi-600m-tl-fixture"
    _set_test_foundation_identity(monkeypatch, payload)
    checkpoint = tmp_path / provenance.FOUNDATION_CHECKPOINT_NAME
    checkpoint.write_bytes(b"x" * len(payload))

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_foundation_checkpoint(checkpoint)


def test_foundation_checkpoint_rejects_non_regular_file(tmp_path: Path):
    with pytest.raises(ValueError, match="not a regular file"):
        verify_foundation_checkpoint(tmp_path)


def test_foundation_checkpoint_rejects_unreadable_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b"official-prithvi-600m-tl-fixture"
    _set_test_foundation_identity(monkeypatch, payload)
    checkpoint = tmp_path / provenance.FOUNDATION_CHECKPOINT_NAME
    checkpoint.write_bytes(payload)

    def _unreadable(_path: Path) -> str:
        raise PermissionError("denied")

    monkeypatch.setattr(provenance, "sha256_file", _unreadable)
    with pytest.raises(ValueError, match="not readable"):
        verify_foundation_checkpoint(checkpoint)


def _write_code_bundle(directory: Path) -> Path:
    code_file = directory / "code.py"
    code_file.write_bytes(b"code")
    files = {"code.py": code_file.read_bytes()}
    manifest = {
        "schema": CODE_BUNDLE_SCHEMA,
        "bundle_sha256": code_bundle_sha256(files),
        "files": {
            name: hashlib.sha256(contents).hexdigest()
            for name, contents in files.items()
        },
    }
    manifest_path = directory / "bundle_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _write_cohort(directory: Path) -> Path:
    directory.mkdir()
    manifest = {
        "schema": "era5-smoke-cohort-v4",
        "counts": {"train": 1, "val": 1},
        "model_patch_px": 504,
        "val_supported_crop_classes": ["vete"],
        "crop_support_thresholds": {
            "min_train_tiles": 1,
            "min_val_tiles": 1,
            "min_train_pixels": 1,
            "min_val_pixels": 1,
        },
        "label_support": {"val": {"pixel_counts": {"11": 1}}},
    }
    manifest_path = directory / "manifest.json"
    _write_json(manifest_path, manifest)
    (directory / "split_train.txt").write_text("train.npz\n")
    (directory / "split_val.txt").write_text("val.npz\n")
    return manifest_path


def _write_environment(directory: Path) -> None:
    directory.mkdir()
    (directory / "gpu_identity.txt").write_text("H100\n")
    (directory / "pip_freeze.txt").write_text("torch==2.6.0\n")
    (directory / "python_version.txt").write_text("Python 3.12.0\n")


def test_base_and_run_provenance_bind_expected_foundation_size_and_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b"official-prithvi-600m-tl-fixture"
    _set_test_foundation_identity(monkeypatch, payload)
    foundation = tmp_path / provenance.FOUNDATION_CHECKPOINT_NAME
    foundation.write_bytes(payload)
    code_manifest = _write_code_bundle(tmp_path)
    cohort_dir = tmp_path / "cohort"
    cohort_manifest = _write_cohort(cohort_dir)
    environment_dir = tmp_path / "environment"
    _write_environment(environment_dir)

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "epoch_001.pt").write_bytes(b"base-checkpoint")
    base_config = {
        "seed": 20260821,
        "deterministic": True,
        "deterministic_warn_only": False,
        "epochs": 1,
        "num_classes": 23,
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
    _write_json(base_dir / "training_log.json", {
        "status": "completed",
        "config": base_config,
        "epochs": [{
            "epoch": 1,
            "confusion_matrix": [[1]],
            "train_exposure_sha256": "a" * 64,
            "train_target_support": [0] * 11 + [1] + [0] * 11,
            "train_class_tiles": {"11": ["train.npz"]},
        }],
    })
    base_args = argparse.Namespace(
        base_dir=base_dir,
        base_git_sha="a" * 40,
        code_manifest=code_manifest,
        cohort_dir=cohort_dir,
        environment_dir=environment_dir,
        foundation_checkpoint=foundation,
        seed=20260821,
        expected_epochs=1,
        num_classes=23,
    )
    base_completion = _base_completion_payload(base_args)
    assert base_completion["foundation_checkpoint_size_bytes"] == len(payload)
    assert base_completion["foundation_checkpoint_sha256"] == hashlib.sha256(
        payload,
    ).hexdigest()
    base_completion_path = base_dir / "completion.json"
    _write_json(base_completion_path, base_completion)

    run_args = argparse.Namespace(
        run_id=f"era5-smoke-{code_bundle_sha256({'code.py': b'code'})[:12]}",
        base_git_sha="a" * 40,
        code_manifest=code_manifest,
        cohort_manifest=cohort_manifest,
        environment_dir=environment_dir,
        initial_checkpoint=base_dir / "epoch_001.pt",
        base_completion=base_completion_path,
        foundation_checkpoint=foundation,
        container_image="ghcr.io/example/image@sha256:" + "b" * 64,
        seeds="41,42,43",
        expected_epochs=5,
        num_classes=23,
    )
    run_manifest = build_run_manifest(run_args)
    assert run_manifest["foundation_checkpoint_size_bytes"] == len(payload)
    assert run_manifest["foundation_checkpoint_sha256"] == hashlib.sha256(
        payload,
    ).hexdigest()

    base_completion["foundation_checkpoint_size_bytes"] += 1
    _write_json(base_completion_path, base_completion)
    with pytest.raises(
        ValueError,
        match="foundation_checkpoint_size_bytes mismatch",
    ):
        build_run_manifest(run_args)


def test_runner_authenticates_foundation_before_model_construction():
    runner = Path("scripts/run_era5_smoke_arm.sh").read_text()
    assert (
        'FOUNDATION_CHECKPOINT="/checkpoints/model_cache/Prithvi_EO_V2_600M_TL.pt"'
        in runner
    )
    fetch = runner.index("fetch_foundation.py")
    verification = runner.index("verify-foundation")
    assert fetch < verification
    assert verification < runner.index("ln -sf \"$FOUNDATION_CHECKPOINT\"")
    assert verification < runner.index("python scripts/train_unified.py")


def test_completion_rejects_running_or_incomplete_training(tmp_path):
    seed_dir, run = _fixture(tmp_path)
    log_path = seed_dir / "training_log.json"
    log = json.loads(log_path.read_text())
    log["status"] = "running"
    _write_json(log_path, log)
    with pytest.raises(ValueError, match="not terminal"):
        _completion_payload(seed_dir, run, arm="control", seed=41)

    log["status"] = "completed"
    log["epochs"] = [{"epoch": 1}]
    _write_json(log_path, log)
    with pytest.raises(ValueError, match="Expected 2 completed epochs"):
        _completion_payload(seed_dir, run, arm="control", seed=41)


def test_completion_detects_checkpoint_tampering(tmp_path):
    seed_dir, run = _fixture(tmp_path)
    completion = _completion_payload(seed_dir, run, arm="control", seed=41)
    _write_json(seed_dir / "completion.json", completion)
    run_path = tmp_path / "run_manifest.json"
    _write_json(run_path, run)

    (seed_dir / "epoch_002.pt").write_bytes(b"changed")
    args = argparse.Namespace(
        seed_dir=seed_dir, run_manifest=run_path, arm="control", seed=41,
    )
    with pytest.raises(ValueError, match="Completion provenance mismatch"):
        validate_seed(args)


@pytest.mark.parametrize("name", ["best_model.pt", "last_checkpoint.pt"])
def test_completion_rejects_redundant_large_smoke_snapshots(tmp_path, name):
    seed_dir, run = _fixture(tmp_path)
    (seed_dir / name).write_bytes(b"redundant")
    with pytest.raises(ValueError, match="redundant snapshots"):
        _completion_payload(seed_dir, run, arm="control", seed=41)


def test_write_once_refuses_provenance_relabel(tmp_path):
    path = tmp_path / "run_manifest.json"
    write_once_or_verify(path, {"run_id": "first"})
    write_once_or_verify(path, {"run_id": "first"})
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        write_once_or_verify(path, {"run_id": "second"})
