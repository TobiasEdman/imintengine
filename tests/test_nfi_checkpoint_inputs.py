"""CPU metadata extraction and strict stored-input audit; no model construction."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]
from nfi_checkpoint_inputs import checkpoint_inputs
from scripts.inference_comparison import CheckpointIdentityError
from inference_comparison import checkpoint_aux_count, inference_aux_names
from prepare_nfi_holdout import file_identity, audit_auxiliary_inputs, tile_readiness


def test_meta_checkpoint_is_authenticated_and_weight_width_wins(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    torch.save({"config": {"n_aux_channels": 11, "enabled_aux_names": ["dem"]},
                "model_state_dict": {"lidar_branch.net.0.conv.weight": torch.ones(2, 1, 3, 3)}}, path)
    import scripts.inference_comparison as inference
    monkeypatch.setattr(inference, "load_model", lambda *a, **k: pytest.fail("no model construction"))
    cells = {"clay_r1": {"checkpoint": file_identity(path)}}
    result = checkpoint_inputs(cells)["cells"]["clay_r1"]
    assert result["enabled_aux_names"] == ["dem"]
    assert result["count_source"] == "state_dict" and result["n_aux_channels"] == 1
    cells["clay_r1"]["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(CheckpointIdentityError, match="SHA256|sha256"):
        checkpoint_inputs(cells)


def test_contradictory_checkpoint_widths_fail():
    state = {"a.lidar_branch.net.0.conv.weight": torch.ones(2, 1, 3, 3),
             "b.lidar_branch.net.0.conv.weight": torch.ones(2, 2, 3, 3)}
    with pytest.raises(ValueError, match="contradictory"):
        checkpoint_aux_count({}, state)


@pytest.mark.parametrize("names", [["dem", "dem"], ["invented"], "dem"])
def test_bad_names_fail(names):
    with pytest.raises(ValueError, match="invalid"):
        inference_aux_names({"enabled_aux_names": names})


def test_count_mismatch_and_era5_fail(tmp_path):
    with pytest.raises(ValueError, match="aux conv takes"):
        inference_aux_names({"enabled_aux_names": ["dem"]}, 2)
    path = tmp_path / "era5.pt"
    torch.save({"config": {"n_aux_channels": 1, "enabled_aux_names": ["era5_gdd"]}}, path)
    with pytest.raises(ValueError, match="ERA5"):
        checkpoint_inputs({"clay_r1": {"checkpoint": file_identity(path)}})


def audit(tmp_path, data, names=("dem",), cell="clay_r1"):
    path = tmp_path / "tile.npz"
    np.savez(path, **data)
    contract = {"cells": {cell: {"enabled_aux_names": list(names)}},
                "computed_channels": ["delta_vv", "delta_vh"],
                "nan_nodata_channels": ["markfukt"]}
    return audit_auxiliary_inputs(path, {"height": 8, "width": 8}, contract)


def test_aux_zeros_valid_missing_and_malformed_excluded(tmp_path):
    assert audit(tmp_path, {"dem": np.zeros((8, 8))})[1] == []
    assert audit(tmp_path, {})[1] == ["dem:missing"]
    assert audit(tmp_path, {"dem": np.zeros((4, 4))})[1] == ["dem:invalid_shape_or_dtype"]
    assert audit(tmp_path, {"dem": np.full((8, 8), np.inf)})[1] == ["dem:nonfinite"]


def test_markfukt_partial_nodata_recorded_all_nodata_excluded(tmp_path):
    arr = np.zeros((8, 8)); arr[0] = np.nan
    result, failures = audit(tmp_path, {"markfukt": arr}, ["markfukt"])
    assert failures == [] and result["nodata_fraction"]["markfukt"] == 1 / 8
    assert audit(tmp_path, {"markfukt": np.full((8, 8), np.nan)}, ["markfukt"])[1]


def test_delta_baseline_policy_and_optical_padding(tmp_path):
    result, failures = audit(tmp_path, {}, ["delta_vv"])
    assert not failures and result["gaps"]["s1_vv_vh_2016"] == "native_neutral_delta_no_baseline"
    assert audit(tmp_path, {"s1_vv_vh_2016": np.zeros((2, 8, 8))}, ["delta_vv"])[1]
    arr = np.zeros((4, 8, 8)); arr[0, 0, 0] = np.nan; arr[1:] = 1
    result, failures = audit(tmp_path, {"dem": np.zeros((8, 8)), "b01": arr}, cell="croma_r1")
    assert not failures
    assert result["gaps"]["b01"] == {"native_zero_padded_frames": [0]}
    assert result["gaps"]["b09"] == "native_optical_zero_padding"


@pytest.mark.parametrize("key,shape", [("tessera", (128,4,4)), ("s1_vv_vh", (2,4,4)),
                                       ("b08", (8,8)), ("rededge", (3,8,8))])
def test_mandatory_arrays_must_share_grid_and_frame_count(tmp_path, key, shape):
    path = tmp_path / "tile.npz"
    data = {"spectral": np.ones((24,8,8)), "tessera": np.ones((128,8,8)),
            "s1_vv_vh": np.ones((2,8,8)), "b08": np.ones((4,8,8)),
            "rededge": np.ones((12,8,8)), "has_tessera": 1, "has_s1": 1,
            "s1_enrich_v": 4, "tessera_source": "geotessera-0.10.2", "year": 2024}
    data[key] = np.ones(shape)
    np.savez(path, **data)
    assert tile_readiness(path)[1] == "mandatory_input_shape:" + key


def test_zero_aux_checkpoint_has_empty_contract(tmp_path):
    path = tmp_path / "no_aux.pt"
    torch.save({"config": {"n_aux_channels": 0}}, path)
    result = checkpoint_inputs({"tessera_r1": {"checkpoint": file_identity(path)}})
    contract = result["cells"]["tessera_r1"]
    assert contract["n_aux_channels"] == 0
    assert contract["enabled_aux_names"] == []


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_invalid_aux_count_rejected(count):
    with pytest.raises(ValueError, match="nonnegative"):
        checkpoint_aux_count({"n_aux_channels": count}, {})
