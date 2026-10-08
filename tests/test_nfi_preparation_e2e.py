"""Synthetic CPU preparation through main, including real checkpoint meta loads."""
import io
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from scripts import prepare_nfi_holdout as prep
from imint.eval.fieldtruth import load_frozen_holdout, sha256_file
from tests.nfi_freeze_fixtures import complete_manifest


def head_bytes(**overrides):
    stream = io.BytesIO()
    np.savez(stream, seed=np.int64(42), n_train_plots=np.int64(1),
             n_features=np.int64(1), **overrides)
    return stream.getvalue()


def test_head_metadata_requires_split_and_width_agreement():
    frame = pd.DataFrame({"f000": [1.]})
    split = {"seed": 42, "n_train_plots": 1}
    assert prep.teacher_head_metadata(head_bytes(), split, frame)["n_features"] == 1
    for key in ("seed", "n_train_plots"):
        with pytest.raises(ValueError, match=key):
            prep.teacher_head_metadata(head_bytes(), {**split, key: 2}, frame)
    with pytest.raises(ValueError, match="feature width"):
        prep.teacher_head_metadata(head_bytes(), split, frame.assign(f001=1.))


def test_sidecar_inventory_checks_non_teacher_training_tiles(tmp_path):
    np.savez(tmp_path / "train.npz", head_sha="a" * 16)
    np.savez(tmp_path / "other.npz", head_sha="b" * 16)
    with pytest.raises(ValueError, match="stale"):
        prep.teacher_sidecar_inventory(tmp_path, {"train"}, "a" * 16, set())
    np.savez(tmp_path / "other.npz", head_sha="a" * 16)
    result = prep.teacher_sidecar_inventory(tmp_path, {"train", "other"}, "a" * 16, set())
    assert result["verified_tiles"] == 2
    with pytest.raises(ValueError, match="missing teacher"):
        prep.teacher_sidecar_inventory(tmp_path, {"missing"}, "a" * 16, set())
    with pytest.raises(ValueError, match="campaign"):
        prep.teacher_sidecar_inventory(tmp_path, {"train"}, "a" * 16, {"other"})
    np.savez(tmp_path / "other.npz", label=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="stale"):
        prep.teacher_sidecar_inventory(tmp_path, {"train"}, "a" * 16, set())


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    directories = {n: tmp_path / n for n in ("cohort-dir", "staging-dir", "teacher-root", "distill-root", "checkpoint-root")}
    for p in directories.values(): p.mkdir()
    tile = dict(spectral=np.ones((24,8,8)), tessera=np.ones((128,8,8)),
                s1_vv_vh=np.ones((2,8,8)), b08=np.ones((4,8,8)), rededge=np.ones((12,8,8)),
                has_tessera=1, tessera_source="geotessera-0.10.2", has_s1=1, s1_enrich_v=4,
                dem=np.zeros((8,8)), year=2024, easting=500000., northing=6500000.,
                doy=np.array([280,150,180,210]))
    for name in ("train", "oldtest"):
        np.savez(directories["cohort-dir"] / (name + ".npz"), **tile)
    stream = io.BytesIO(); np.savez(stream, **tile)
    for i in range(475):
        (directories["staging-dir"] / f"campaign{i:03}.npz").write_bytes(stream.getvalue())
    index = pd.DataFrame({"TractID": [1, 2, 3, 1], "PlotID": [1]*4, "Year": [2024]*4,
                          "tile_name": ["train", "oldtest", "campaign000", "campaign001"],
                          "row": [4]*4, "col": [4]*4, "VolPine": [100.]*4,
                          "VolContorta": [0.]*4, "VolSpruce": [0.]*4,
                          "VolBirch": [0.]*4, "VolOtherDec": [0.]*4,
                          "Easting": [500000.]*4, "Northing": [6500000.]*4})
    values = dict(directories)
    for name, frame in (("plot-index", index), ("teacher-index", index.iloc[:2])):
        values[name] = tmp_path / (name + ".parquet"); frame.to_parquet(values[name])
    split = {"seed": 42, "train_tiles": ["train"], "test_tiles": ["oldtest"], "n_train_plots": 1, "n_test_plots": 1}
    for model in prep.DISTILL:
        head = directories["teacher-root"] / f"{model}_r2_head.npz"; head.write_bytes(head_bytes())
        (directories["teacher-root"] / f"{model}_r2_split.json").write_text(json.dumps(split))
        index.iloc[:2].drop(columns="Year").assign(f000=1.).to_parquet(directories["teacher-root"] / f"{model}_r2_plot_features.parquet")
        sidecars = directories["distill-root"] / f"{model}_r2"; sidecars.mkdir()
        for name in ("train", "oldtest"): np.savez(sidecars / (name + ".npz"), head_sha=sha256_file(head)[:16])
        for rung in range(1, 5):
            checkpoint = directories["checkpoint-root"] / f"{model}_r{rung}" / "best_model.pt"
            checkpoint.parent.mkdir()
            torch.save({"config": {"n_aux_channels": 1, "enabled_aux_names": ["dem"]},
                        "model_state_dict": {"lidar_branch.net.0.conv.weight": torch.zeros(2,1,3,3)}}, checkpoint)
    values["promotion-job-json"] = tmp_path / "job.json"
    values["promotion-job-json"].write_text(json.dumps({"metadata": {"name": "tessera-promote-v2"}, "status": {"conditions": [{"type": "Complete", "status": "True"}]}}))
    values["promotion-report"] = tmp_path / "report.json"
    values["promotion-report"].write_text(json.dumps({"source": "geotessera-0.10.2", "states": {"promoted": 2}, "verify": {"has_v1": 2, "stamped": 2, "has_v2_left": 0}}))
    values["nmd2023"] = tmp_path / "baseline.tif"; values["nmd2023"].write_bytes(b"not sampled during preparation")
    values["out-dir"] = tmp_path / "freeze"
    runtime = complete_manifest({})["preparation_runtime"]
    monkeypatch.setattr(prep, "preparation_runtime", lambda *a: runtime)
    # Only the sealed-image verifier is a fixture. The real CPU child verifies
    # and parses all 28 checkpoints, and main reads/writes the real NPZ/parquets.
    argv = ["prepare"]
    for name, value in values.items(): argv.extend(["--" + name, str(value)])
    argv.extend(["--runtime-image", runtime["image"]["ref"], "--source-git-sha", "a"*40,
                 "--runtime-manifest", "/fixture/runtime.json"])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    return values


def test_main_freezes_unique_conservative_holdout_and_all_28_contracts(prepared_inputs):
    prep.main()
    path = prepared_inputs["out-dir"] / "manifest.json"
    holdout, manifest = load_frozen_holdout(path, sha256_file(path))
    assert holdout.TractID.tolist() == [3]
    assert manifest["recorded_training_plot_years"] == 1
    assert manifest["excluded_teacher_feature_plot_years"] == 2
    assert manifest["campaign"]["training_tiles"] == 0
    assert len(manifest["auxiliary_audit"]) == 477
    assert len(manifest["cells"]) == 28
    assert all(c["input_contract"]["enabled_aux_names"] == ["dem"] for c in manifest["cells"].values())
    assert all(p["sidecars"]["verified_tiles"] == 2 for p in manifest["teacher_provenance"].values())
    with pytest.raises(ValueError, match="never overwrite"):
        prep.main()


def test_main_active_promotion_cannot_create_output(prepared_inputs):
    path = prepared_inputs["promotion-job-json"]
    job = json.loads(path.read_text()); job["status"]["active"] = 1; path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="terminal"):
        prep.main()
    assert not prepared_inputs["out-dir"].exists()


def test_main_rejects_parsed_input_replaced_after_read(prepared_inputs, monkeypatch):
    original = prep.teacher_training_set
    path = prepared_inputs["plot-index"]
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        frame = pd.read_parquet(path).assign(VolPine=99.)
        frame.to_parquet(path)
        return result
    monkeypatch.setattr(prep, "teacher_training_set", mutate)
    with pytest.raises(ValueError, match="parsed input changed"):
        prep.main()
    assert not prepared_inputs["out-dir"].exists()


def test_campaign_population_is_selected_before_freeze(prepared_inputs, monkeypatch):
    index_path = prepared_inputs["plot-index"]
    frame = pd.read_parquet(index_path)
    extra = frame.iloc[[0]].assign(TractID=999)
    pd.concat([frame, extra]).to_parquet(index_path)
    monkeypatch.setattr(sys, "argv", sys.argv + ["--population", "campaign"])
    prep.main()
    path = prepared_inputs["out-dir"] / "manifest.json"
    held, manifest = load_frozen_holdout(path, sha256_file(path))
    assert held.TractID.tolist() == [3]
    assert held.tile_role.tolist() == ["campaign"]
    assert manifest["protocol"]["primary_population"] == "campaign"
