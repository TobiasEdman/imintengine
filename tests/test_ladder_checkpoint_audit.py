"""Disk round trips and negative cases for E00 checkpoint auditing."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts import audit_ladder_checkpoints as audit


def fixture_payload(cell="tessera_r2"):
    cfg = {
        "backbone_name": "tessera_v1", "num_classes": 28, "decoder_type": "upernet",
        "decoder_channels": 256, "feature_indices": [2, 5, 8, 11], "dropout": 0.1,
        "enable_multitemporal": False, "enable_tradslag_head": False,
        "num_tradslag": 4, "img_size": 504, "aux_fusion": "gated",
        "enable_temporal_pooling": False, "enable_multilevel_aux": False,
        "era5_mode": "none", "patch_size": 1, "num_temporal_frames": 1,
        "n_aux_channels": 0, "enabled_aux_names": [],
    }
    log = {"config": {**cfg, **dict.fromkeys(audit.AUX_FLAGS, False)}, "best_epoch": 28}
    payload = {
        "config": cfg, "epoch": 28, "metrics": {"per_class_iou": {"0": float("nan")}},
        "state_dict": {"model.classifier.weight": torch.ones(28, 128, 1, 1),
                       "model.classifier.bias": torch.zeros(28),
                       "model.counter": torch.tensor(3)},
    }
    return payload, log


def write_inputs(tmp_path):
    payload, log = fixture_payload()
    cell = tmp_path / "tessera_r2"
    cell.mkdir()
    checkpoint = cell / "best_model.pt"
    torch.save(payload, checkpoint)
    log_path = cell / "training_log.json"
    log_path.write_text(json.dumps(log))
    row = {
        "cell": cell.name, "identity_status": "hashed_stable",
        "checkpoint": {"sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       "size_bytes": checkpoint.stat().st_size},
        "training_log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
    }
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text("".join(json.dumps({**row, "cell": name}) + "\n" for name in sorted(audit.CELLS)))
    return row, inventory


def test_real_safe_loader_and_cpu_process_disk_round_trip(tmp_path):
    row, inventory = write_inputs(tmp_path)
    result = audit.audit_cell(tmp_path, row)
    assert result["status"] == "passed"
    assert result["tensor_count"] == 3
    assert result["elements"] == 28 * 128 + 28 + 1
    assert result["nonfinite_count"] == 0
    output = tmp_path / "result.jsonl"
    run = subprocess.run(
        [sys.executable, "-m", "scripts.audit_ladder_checkpoints",
         "--inventory", str(inventory), "--root", str(tmp_path),
         "--cell", "tessera_r2", "--output", str(output)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        # Allow the audit's 180s worker deadline plus cold interpreter imports.
        capture_output=True, text=True, timeout=240,
    )
    assert run.returncode == 0, run.stderr + run.stdout
    record = json.loads(output.read_text())
    assert record["status"] == "passed"
    assert record["cell"] == "tessera_r2"
    assert record["inventory_sha256"] == hashlib.sha256(inventory.read_bytes()).hexdigest()
    assert record["tensor_bytes"] == (28 * 128 + 28) * 4 + 8


@pytest.mark.parametrize("target", ["checkpoint", "log"])
def test_changed_bytes_rejected(tmp_path, target):
    row, _ = write_inputs(tmp_path)
    path = tmp_path / row["cell"] / ("best_model.pt" if target == "checkpoint" else "training_log.json")
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(Exception, match="mismatch"):
        audit.audit_cell(tmp_path, row)


def test_missing_metadata_never_filled_from_current_defaults(monkeypatch):
    payload, log = fixture_payload()
    del payload["config"]["img_size"]
    del log["config"]["enable_vpp_channels"]
    monkeypatch.setenv("IMINT_IMG_SIZE", "504")
    result = audit.inspect_payload(payload, log, "tessera_r2")
    assert result["status"] == "incomplete"
    assert "checkpoint.config.img_size" in result["missing_evidence"]
    assert "log.aux_flags" in result["missing_evidence"]


@pytest.mark.parametrize("change,error", [
    ("nan", "nonfinite_weights"), ("inf", "nonfinite_weights"),
    ("classes", "shape.classifier.weight"), ("epoch", "best_epoch"),
    ("frames", "config.num_temporal_frames"), ("fraction", "unexpected_fraction_head"),
    ("backbone", "checkpoint.backbone"),
])
def test_corruption_and_mismatch(change, error):
    payload, log = fixture_payload()
    if change in ("nan", "inf"):
        payload["state_dict"]["model.classifier.weight"][0, 0, 0, 0] = float(change)
    elif change == "classes":
        payload["state_dict"]["model.classifier.weight"] = torch.ones(23, 128, 1, 1)
    elif change == "epoch":
        payload["epoch"] = 30
    elif change == "frames":
        payload["config"]["num_temporal_frames"] = 4
    elif change == "fraction":
        payload["state_dict"]["model.frac_head.weight"] = torch.ones(4, 128, 1, 1)
    elif change == "backbone":
        payload["config"]["backbone_name"] = "prithvi_300m"
    result = audit.inspect_payload(payload, log, "tessera_r2")
    assert result["status"] == "failed"
    assert error in result["errors"]


def test_legacy_backbone_missing_does_not_fall_back():
    payload, log = fixture_payload()
    del payload["config"]["backbone_name"]
    result = audit.inspect_payload(payload, log, "tessera_r2")
    assert result["status"] == "incomplete"
    assert "checkpoint.backbone" in result["missing_evidence"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "path", "hash", "size"])
def test_invalid_inventory_fails(tmp_path, mutation):
    _, inventory = write_inputs(tmp_path)
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation == "path":
        rows[0]["cell"] = "../tessera_r2"
    elif mutation == "hash":
        rows[0]["checkpoint"]["sha256"] = "123"
    elif mutation == "size":
        rows[0]["checkpoint"]["size_bytes"] = True
    inventory.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        audit.load_inventory(inventory)


def test_symlink_checkpoint_rejected(tmp_path):
    row, _ = write_inputs(tmp_path)
    path = tmp_path / row["cell"] / "best_model.pt"
    moved = path.with_suffix(".real")
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(Exception):
        audit.audit_cell(tmp_path, row)


def test_wrong_payload_format():
    _, log = fixture_payload()
    with pytest.raises(ValueError, match="state_dict"):
        audit.inspect_payload({"config": {}, "model_state_dict": {"x": torch.ones(1)}}, log, "tessera_r2")


def test_fraction_head_and_aux_shape_contract():
    payload, log = fixture_payload()
    for cfg in (payload["config"], log["config"]):
        cfg["enable_tradslag_head"] = True
    log["config"]["enable_height_channel"] = True
    payload["config"]["n_aux_channels"] = 1
    payload["config"]["enabled_aux_names"] = ["height"]
    payload["state_dict"].update({
        "model.frac_head.weight": torch.ones(4, 128, 1, 1),
        "model.frac_head.bias": torch.zeros(4),
        "model.aux_proj.0.weight": torch.ones(32, 1, 1, 1),
    })
    assert audit.inspect_payload(payload, log, "tessera_r4")["status"] == "passed"
    payload["state_dict"]["model.frac_head.bias"] = torch.zeros(3)
    payload["state_dict"]["model.aux_proj.0.weight"] = torch.ones(32, 2, 1, 1)
    result = audit.inspect_payload(payload, log, "tessera_r4")
    assert result["status"] == "failed"
    assert set(result["errors"]) == {"shape.frac_head.bias", "shape.aux_input"}


def test_temporal_prithvi_positions_use_effective_frames():
    payload, log = fixture_payload()
    for cfg in (payload["config"], log["config"]):
        cfg.update(backbone_name="prithvi_300m", img_size=496, patch_size=16)
    # Dormant config may say 4, but disabled temporal mode was saved as 1.
    log["config"]["num_temporal_frames"] = 4
    payload["state_dict"] = {
        "model.head.head.2.weight": torch.ones(28, 256, 1, 1),
        "model.head.head.2.bias": torch.zeros(28),
        "model.encoder.pos_embed": torch.zeros(1, 962, 1024),
    }
    assert audit.inspect_payload(payload, log, "prithvi300m_r2")["status"] == "passed"
    payload["state_dict"]["model.encoder.pos_embed"] = torch.zeros(1, 3845, 1024)
    assert "shape.encoder.pos_embed" in audit.inspect_payload(payload, log, "prithvi300m_r2")["errors"]


def test_failed_worker_is_persisted_and_exits_nonzero(tmp_path):
    _, inventory = write_inputs(tmp_path)
    output = tmp_path / "failed.jsonl"
    run = subprocess.run(
        [sys.executable, "-m", "scripts.audit_ladder_checkpoints",
         "--inventory", str(inventory), "--root", str(tmp_path),
         "--cell", "clay_r1", "--output", str(output)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        # Allow the audit's 180s worker deadline plus cold interpreter imports.
        capture_output=True, text=True, timeout=240,
    )
    assert run.returncode == 1
    assert json.loads(output.read_text())["status"] == "failed"


def test_job_is_bounded_readonly_and_pins_exact_inputs(tmp_path):
    from scripts.prepare_checkpoint_audit_job import build_job
    _, inventory = write_inputs(tmp_path)
    first = build_job(inventory)
    assert first == build_job(inventory)
    config, job = first["items"]
    assert config["immutable"] is True
    assert config["data"]["inventory.jsonl"] == inventory.read_text()
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert job["spec"]["activeDeadlineSeconds"] == 1800
    assert job["spec"]["backoffLimit"] == 0
    assert container["resources"]["requests"] == container["resources"]["limits"]
    assert "nvidia.com/gpu" not in container["resources"]["limits"]
    assert container["image"] == audit.IMAGE
    pvc_mount = next(x for x in container["volumeMounts"] if x["name"] == "checkpoints")
    assert pvc_mount["readOnly"] is True
    assert not pod["automountServiceAccountToken"]
    inventory.write_text(inventory.read_text() + "\n")
    # Empty inventory rows are rejected, not silently ignored.
    with pytest.raises(ValueError):
        build_job(inventory)


def test_one_frame_cannot_pass_as_four_frame_arm():
    payload, log = fixture_payload()
    for cfg in (payload["config"], log["config"]):
        cfg.update(backbone_name="prithvi_300m", img_size=496, patch_size=16)
    payload["state_dict"] = {
        "model.head.head.2.weight": torch.ones(28, 256, 1, 1),
        "model.head.head.2.bias": torch.zeros(28),
        "model.encoder.pos_embed": torch.zeros(1, 962, 1024),
    }
    result = audit.inspect_payload(payload, log, "prithvi300m4f_r2")
    assert result["status"] == "failed"
    assert "arm.num_temporal_frames" in result["errors"]


def test_prithvi_single_level_aux_is_supported():
    payload, log = fixture_payload()
    for cfg in (payload["config"], log["config"]):
        cfg.update(backbone_name="prithvi_300m", img_size=496, patch_size=16)
    log["config"]["enable_height_channel"] = True
    payload["config"].update(enabled_aux_names=["height"], n_aux_channels=1)
    payload["state_dict"] = {
        "model.head.head.2.weight": torch.ones(28, 256, 1, 1),
        "model.head.head.2.bias": torch.zeros(28),
        "model.encoder.pos_embed": torch.zeros(1, 962, 1024),
        "model.aux_encoder.net.0.conv.weight": torch.ones(32, 1, 3, 3),
    }
    assert audit.inspect_payload(payload, log, "prithvi300m_r2")["status"] == "passed"
    del payload["config"]["enable_multilevel_aux"]
    result = audit.inspect_payload(payload, log, "prithvi300m_r2")
    assert result["status"] == "incomplete"
    assert not result["errors"]
    assert "checkpoint.config.enable_multilevel_aux" in result["missing_evidence"]


def test_projected_configmap_symlink_is_rejected(tmp_path):
    _, inventory = write_inputs(tmp_path)
    projected = tmp_path / "projected.jsonl"
    projected.symlink_to(inventory)
    with pytest.raises(Exception):
        audit.load_inventory(projected)
    from scripts.prepare_checkpoint_audit_job import build_job
    container = build_job(inventory)["items"][1]["spec"]["template"]["spec"]["containers"][0]
    inputs = [mount for mount in container["volumeMounts"] if mount["name"] == "inputs"]
    assert {mount["subPath"] for mount in inputs} == {"inventory.jsonl", "audit_ladder_checkpoints.py"}
    assert all(mount["readOnly"] for mount in inputs)
    assert all(mount["mountPath"] == "/audit/" + mount["subPath"] for mount in inputs)


@pytest.mark.parametrize("rung", [1, 2, 3])
def test_enabled_fraction_head_allowed_before_r4(rung):
    payload, log = fixture_payload()
    classes = 23 if rung == 1 else 28
    for config in (payload["config"], log["config"]):
        config.update(enable_tradslag_head=True, num_classes=classes)
    payload["state_dict"] = {
        "model.classifier.weight": torch.ones(classes, 128, 1, 1),
        "model.classifier.bias": torch.zeros(classes),
        "model.frac_head.weight": torch.ones(4, 128, 1, 1),
        "model.frac_head.bias": torch.zeros(4),
    }
    result = audit.inspect_payload(payload, log, f"tessera_r{rung}")
    assert result["status"] == "passed"
    assert result["fraction_head_enabled"] is True
    assert result["fraction_head_weight_shape"] == [4, 128, 1, 1]
    payload["state_dict"]["model.frac_head.bias"] = torch.zeros(3)
    assert "shape.frac_head.bias" in audit.inspect_payload(payload, log, f"tessera_r{rung}")["errors"]


def test_r4_without_enabled_fraction_head_is_protocol_failure():
    payload, log = fixture_payload()
    result = audit.inspect_payload(payload, log, "tessera_r4")
    assert result["status"] == "failed"
    assert result["errors"] == ["protocol.r4_fraction_head_disabled"]
    assert result["fraction_head_enabled"] is False
    assert result["fraction_head_weight_shape"] is None


def test_fraction_flag_checkpoint_log_mismatch_still_fails():
    payload, log = fixture_payload()
    log["config"]["enable_tradslag_head"] = True
    result = audit.inspect_payload(payload, log, "tessera_r2")
    assert result["status"] == "failed"
    assert result["errors"] == ["config.enable_tradslag_head"]


def test_missing_fraction_flag_is_missing_evidence():
    payload, log = fixture_payload()
    del payload["config"]["enable_tradslag_head"]
    result = audit.inspect_payload(payload, log, "tessera_r2")
    assert result["status"] == "incomplete"
    assert "checkpoint.config.enable_tradslag_head" in result["missing_evidence"]
