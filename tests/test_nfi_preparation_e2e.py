"""Synthetic CPU preparation through main, including real checkpoint meta loads."""
import io
import json
import os
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


def test_sidecar_inventory_allows_only_missing_files_when_asked(tmp_path):
    """allow_missing (support reports) records a shortfall; an empty expected
    set, evaluation-tile sidecars and stale stamps stay fatal."""
    np.savez(tmp_path / "train.npz", head_sha="a" * 16)
    result = prep.teacher_sidecar_inventory(tmp_path, {"train", "gone"}, "a" * 16, set(), allow_missing=True)
    assert (result["expected_tiles"], result["verified_tiles"], result["missing_tiles"]) == (2, 1, 1)
    with pytest.raises(ValueError, match="missing teacher"):
        prep.teacher_sidecar_inventory(tmp_path, set(), "a" * 16, set(), allow_missing=True)
    with pytest.raises(ValueError, match="campaign"):
        prep.teacher_sidecar_inventory(tmp_path, {"train"}, "a" * 16, {"train"}, allow_missing=True)
    np.savez(tmp_path / "other.npz", head_sha="b" * 16)
    with pytest.raises(ValueError, match="stale"):
        prep.teacher_sidecar_inventory(tmp_path, {"train"}, "a" * 16, set(), allow_missing=True)


def _readiness_base():
    return dict(spectral=np.ones((24, 8, 8)), tessera=np.ones((128, 8, 8)), has_tessera=1,
                tessera_source="geotessera-0.10.2")


@pytest.mark.parametrize("arrays, message", [
    (None, "unreadable tile"),
    ({k: v for k, v in _readiness_base().items() if k != "tessera"}, "positive has_tessera without an array"),
    (dict(_readiness_base(), tessera=np.zeros((128, 8, 8))), "positive has_tessera on empty/invalid array"),
    (dict(_readiness_base(), tessera_source="geotessera-0.9"), "unpromoted Tessera source"),
    (dict(_readiness_base(), spectral=np.full((24, 8, 8), np.nan), s1_vv_vh=np.ones((2, 8, 8)),
          b08=np.ones((4, 8, 8)), rededge=np.ones((12, 8, 8)), has_s1=1, s1_enrich_v=4,
          s1_source="pc-rtc-gamma0"), "invalid spectral array"),
])
def test_every_readiness_raise_becomes_a_reason_without_its_path(tmp_path, arrays, message):
    path = tmp_path / "nfi_tile_42.npz"
    if arrays is None:
        path.write_bytes(b"not a zip")
    else:
        np.savez(path, **arrays)
    with pytest.raises(ValueError) as caught:
        prep.tile_readiness(path)
    reason = prep.readiness_error_reason(caught.value)
    assert reason == "readiness_error:" + message
    assert "nfi_tile_42" not in reason and str(tmp_path) not in reason


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    directories = {n: tmp_path / n for n in ("cohort-dir", "staging-dir", "teacher-root", "distill-root", "checkpoint-root")}
    for p in directories.values(): p.mkdir()
    tile = dict(spectral=np.ones((24,8,8)), tessera=np.ones((128,8,8)),
                s1_vv_vh=np.ones((2,8,8)), b08=np.ones((4,8,8)), rededge=np.ones((12,8,8)),
                has_tessera=1, tessera_source="geotessera-0.10.2", has_s1=1, s1_enrich_v=4, s1_source="pc-rtc-gamma0",
                dem=np.zeros((8,8)), easting=500000., northing=6500000.,
                dates=np.array(["2023-10-07","2024-05-29","2024-06-28","2024-07-28"]),
                doy=np.array([280,150,180,210]))
    for name in ("train", "oldtest"):
        np.savez(directories["cohort-dir"] / (name + ".npz"), **dict(tile, easting=400000.))
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
    import rasterio
    from rasterio.transform import from_origin
    values["nmd2023"] = tmp_path / "baseline.tif"
    with rasterio.open(values["nmd2023"], "w", driver="GTiff", width=4, height=4,
                       count=1, dtype="uint16", crs="EPSG:3006", nodata=0,
                       transform=from_origin(499980, 6500020, 10, 10)) as dst:
        dst.write(np.full((1,4,4), 111, dtype=np.uint16))
    values["out-dir"] = tmp_path / "freeze"
    runtime = complete_manifest({})["preparation_runtime"]
    monkeypatch.setattr(prep, "preparation_runtime", lambda *a: runtime)
    # Only the sealed-image verifier is a fixture. The real CPU child verifies
    # and parses all 28 checkpoints, and main reads/writes the real NPZ/parquets.
    argv = ["prepare", "--population", "all"]
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


def test_default_balance_requires_years_before_any_runtime_or_data_read(prepared_inputs, monkeypatch):
    argv = list(sys.argv)
    i = argv.index("--population"); del argv[i:i+2]
    monkeypatch.setattr(sys, "argv", argv)
    def forbidden(*args):
        raise AssertionError("must reject incomplete scope before accessing runtime")
    monkeypatch.setattr(prep, "preparation_runtime", forbidden)
    with pytest.raises(ValueError, match="balanced year"):
        prep.main()
    assert not prepared_inputs["out-dir"].exists()


@pytest.mark.parametrize("missing_nmd", [False, True])
def test_balanced_population_freezes_exact_equal_counts(prepared_inputs, monkeypatch, missing_nmd):
    index_path = prepared_inputs["plot-index"]
    frame = pd.read_parquet(index_path)
    additions = frame.iloc[[2,2,2]].copy()
    additions["TractID"] = [4,5,6]
    additions["Year"] = [2023,2023,2024]
    additions["tile_name"] = ["campaign002", "campaign002", "campaign000"]
    if missing_nmd:
        additions.loc[additions.TractID.isin([4, 6]), "Easting"] = 600000.
    pd.concat([frame, additions]).to_parquet(index_path, index=False)
    tile = prepared_inputs["staging-dir"] / "campaign002.npz"
    with np.load(tile, allow_pickle=False) as z:
        data = dict(z)
    data["dates"] = np.array(["2022-10-07", "2023-05-29", "2023-06-28", "2023-07-28"])
    np.savez(tile, **data)
    monkeypatch.setattr(sys, "argv", sys.argv + ["--population", "balanced",
                       "--inventory-years", "2023", "2024", "--observations-per-year", "1"])
    prep.main()
    path = prepared_inputs["out-dir"] / "manifest.json"
    held, manifest = load_frozen_holdout(path, sha256_file(path))
    assert held.Year.value_counts().to_dict() == {2023:1, 2024:1}
    assert not set(held.TractID) & {1,2}
    assert manifest["protocol"]["year_selection"]["years"] == [2023, 2024]
    assert manifest["selection"]["observations"] == 2
    assert manifest["selection"]["pre_balance_observations"] == (2 if missing_nmd else 4)
    assert manifest["selection"]["excluded_by_year_quota"] == (0 if missing_nmd else 2)
    assert manifest["selection"]["excluded_for_nmd_coverage"] == (2 if missing_nmd else 0)
    if missing_nmd:
        assert held.TractID.tolist() == [3, 5]
    assert manifest["selection"]["tile_role_support"] == {"campaign": 2}


def test_multiple_evaluation_roots_keep_campaign_invariant_and_balance(prepared_inputs, monkeypatch):
    extra = prepared_inputs["out-dir"].parent / "evaluation2019"
    extra.mkdir()
    with np.load(prepared_inputs["staging-dir"] / "campaign000.npz", allow_pickle=False) as tile:
        data = dict(tile)
    data["dates"] = np.array(["2018-10-07", "2019-05-29", "2019-06-28", "2019-07-28"])
    np.savez(extra / "new2019.npz", **data)
    index_path = prepared_inputs["plot-index"]
    frame = pd.read_parquet(index_path)
    new = frame.iloc[[2]].assign(TractID=7, Year=2019, tile_name="new2019")
    pd.concat([frame, new]).to_parquet(index_path, index=False)
    monkeypatch.setattr(sys, "argv", sys.argv + ["--evaluation-root", f"nfi2019:2019:{extra}",
                       "--population", "balanced", "--inventory-years", "2019", "2024",
                       "--observations-per-year", "1"])
    prep.main()
    path = prepared_inputs["out-dir"] / "manifest.json"
    held, manifest = load_frozen_holdout(path, sha256_file(path))
    assert held.Year.value_counts().to_dict() == {2019: 1, 2024: 1}
    assert set(held.tile_role) == {"campaign", "nfi2019"}
    assert manifest["campaign"]["tiles"] == 475
    assert manifest["campaign"]["training_tiles"] == 0
    assert [r["tiles"] for r in manifest["evaluation_roots"]] == [475, 1]


def test_evaluation_root_name_collision_is_rejected_early(prepared_inputs, monkeypatch):
    monkeypatch.setattr(sys, "argv", sys.argv + ["--evaluation-root", "holdout_copy:any:" + str(prepared_inputs["cohort-dir"])])
    with pytest.raises(ValueError, match="globally unique tile names"):
        prep.main()
    assert not prepared_inputs["out-dir"].exists()


def _report_only_argv(argv):
    """Drop --out-dir; report-only must never be given one."""
    argv = list(argv)
    i = argv.index("--out-dir"); del argv[i:i+2]
    return argv + ["--report-only"]


def _support_report(text):
    begin, end = "=== SUPPORT_REPORT_BEGIN ===\n", "\n=== SUPPORT_REPORT_END ==="
    return json.loads(text[text.index(begin) + len(begin):text.index(end)])


def _files(root):
    """Every path and its mtime, directories included."""
    return {p: p.stat().st_mtime_ns for p in root.rglob("*")}


def _add_evaluation_root(prepared_inputs, tiles):
    """A topup:2024 root of campaign000 variants, one indexed plot-year each."""
    root = prepared_inputs["out-dir"].parent / "topup2024"
    root.mkdir()
    with np.load(prepared_inputs["staging-dir"] / "campaign000.npz", allow_pickle=False) as tile:
        base = dict(tile)
    frame = pd.read_parquet(prepared_inputs["plot-index"])
    rows = []
    for tract, (name, change) in enumerate(tiles.items(), start=5):
        np.savez(root / f"{name}.npz", **change(dict(base)))
        rows.append(frame.iloc[[2]].assign(TractID=tract, tile_name=name))
    pd.concat([frame, *rows]).to_parquet(prepared_inputs["plot-index"], index=False)
    return root


def _with(**arrays):
    return lambda data: {**data, **arrays}


def _without(*names, **arrays):
    return lambda data: {**{k: v for k, v in data.items() if k not in names}, **arrays}


BASELINE_2016 = dict(has_frame_2016=1, frame_2016=np.ones((6, 8, 8)), s1_vv_vh_2016=np.ones((2, 8, 8)))


def test_report_only_prints_every_variant_and_writes_nothing(prepared_inputs, monkeypatch, capsys):
    root = _add_evaluation_root(prepared_inputs, {
        "full": _with(**BASELINE_2016),
        "nodem": _without("dem"),   # dem is zero-filled by training, never mean-filled
    })
    # A second plot-year on "full", outside the NMD raster: dropped by coverage.
    frame = pd.read_parquet(prepared_inputs["plot-index"])
    outside = frame.loc[frame.tile_name == "full"].assign(TractID=99, Easting=600000.)
    pd.concat([frame, outside]).to_parquet(prepared_inputs["plot-index"], index=False)
    monkeypatch.setattr(sys, "argv", _report_only_argv(sys.argv) + ["--evaluation-root", f"topup:2024:{root}"])
    before = _files(prepared_inputs["out-dir"].parent)
    prep.main()
    assert _files(prepared_inputs["out-dir"].parent) == before
    report = _support_report(capsys.readouterr().out)
    v = report["variants"]
    assert v["freeze"]["year_support"] == {"2024": 2}
    assert v["freeze"]["role_year_support"] == {"campaign": {"2024": 1}, "topup": {"2024": 1}}
    assert v["freeze"]["frame_2016_by_year"] == {"2024": {"present": 1, "absent": 1}}
    # Selection counts are after NMD coverage, as in the freeze manifest.
    assert v["freeze"]["selection"]["observations"] == 2
    assert v["freeze"]["selection"]["pre_nmd_coverage_observations"] == 3
    assert v["freeze"]["selection"]["tile_role_support"] == {"campaign": 1, "topup": 1}
    assert v["absent_aux_mean_fill"]["year_support"] == {"2024": 2}
    assert v["freeze_with_frame_2016"]["role_year_support"] == {"topup": {"2024": 1}}
    assert report["excluded_evaluation_tiles"] == {"topup": {"auxiliary_gaps:dem:missing": 1}}
    assert [r["tiles"] for r in report["evaluation_roots"]] == [475, 2]
    # Aggregates only: no tile name or plot identity leaves the data environment.
    text = json.dumps(report)
    assert not any(name in text for name in ("campaign000", "full.npz", "nodem", "TractID"))


def test_report_only_mean_fills_only_absent_nan_nodata_channels(prepared_inputs, monkeypatch, capsys):
    """height is a NaN-nodata channel: absent, training fills it with its mean.
    dem is not: absent, training zero-fills it, so the tile stays out."""
    for checkpoint in prepared_inputs["checkpoint-root"].glob("*/best_model.pt"):
        torch.save({"config": {"n_aux_channels": 2, "enabled_aux_names": ["dem", "height"]},
                    "model_state_dict": {"lidar_branch.net.0.conv.weight": torch.zeros(2, 2, 3, 3)}}, checkpoint)
    height = np.ones((8, 8))
    with np.load(prepared_inputs["staging-dir"] / "campaign000.npz", allow_pickle=False) as tile:
        np.savez(prepared_inputs["staging-dir"] / "campaign000.npz", **dict(tile), height=height)
    # The base tile (campaign000) now carries height.
    root = _add_evaluation_root(prepared_inputs, {
        "full": _with(**BASELINE_2016),
        "flag0": _with(**dict(BASELINE_2016, has_frame_2016=0)),
        "s1zero": _with(**dict(BASELINE_2016, s1_vv_vh_2016=np.zeros((2, 8, 8)))),
        "noheight": _without("height"),
        "nodem": _without("dem"),
        "nodem_noheight": _without("dem", "height"),
        "badheight": _with(height=np.ones((4, 4))),
    })
    monkeypatch.setattr(sys, "argv", _report_only_argv(sys.argv) + ["--evaluation-root", f"topup:2024:{root}"])
    prep.main()
    report = _support_report(capsys.readouterr().out)
    v = report["variants"]
    assert v["freeze"]["role_year_support"] == {"campaign": {"2024": 1}, "topup": {"2024": 3}}
    assert v["freeze"]["frame_2016_by_year"] == {"2024": {"present": 1, "absent": 3}}
    assert v["absent_aux_mean_fill"]["role_year_support"] == {"campaign": {"2024": 1}, "topup": {"2024": 4}}
    # The 474 other campaign tiles lack height: mean-filled, not admitted by the freeze.
    assert v["freeze"]["admitted_tiles_by_role"]["campaign"] == 1
    assert v["absent_aux_mean_fill"]["admitted_tiles_by_role"]["campaign"] == 475
    assert v["freeze_with_frame_2016"]["role_year_support"] == {"topup": {"2024": 1}}
    assert report["excluded_evaluation_tiles"]["topup"] == {
        "auxiliary_gaps:height:missing": 1, "auxiliary_gaps:dem:missing": 1,
        "auxiliary_gaps:dem:missing,height:missing": 1, "auxiliary_gaps:height:invalid_shape_or_dtype": 1}


def _unclean_promotion(prepared_inputs, **states):
    """Shaped like the real report, extra keys included: one cohort tile had no
    Tessera v2 to promote."""
    prepared_inputs["promotion-report"].write_text(json.dumps({
        "source": "geotessera-0.10.2", "states": {"promoted": 1, "no_v2": 1, **states},
        "verify": {"has_v1": 1, "stamped": 1, "has_v2_left": 0},
        "campaign_stamped": 475, "old_preserved_as": "tessera_v1 / tessera_v1_year",
        "recovered_from_zero": [], "still_empty_in_v2": ["train.npz"]}))


def test_report_only_records_a_failed_promotion_gate(prepared_inputs, monkeypatch, capsys):
    _unclean_promotion(prepared_inputs)
    monkeypatch.setattr(sys, "argv", _report_only_argv(sys.argv))
    prep.main()
    report = _support_report(capsys.readouterr().out)
    assert report["promotion_gate"] == {
        "passed": False, "reason": "promotion report lacks complete, clean v2 coverage",
        "cohort_tiles": 2, "states": {"promoted": 1, "no_v2": 1},
        "verify": {"has_v1": 1, "stamped": 1, "has_v2_left": 0}}
    assert report["variants"]["freeze"]["observations"] == 1


def test_the_freeze_still_refuses_a_failed_promotion_gate(prepared_inputs):
    _unclean_promotion(prepared_inputs)
    with pytest.raises(ValueError, match="lacks complete, clean v2 coverage"):
        prep.main()
    assert not prepared_inputs["out-dir"].exists()


def test_the_gate_never_records_a_non_integer_counter(prepared_inputs, monkeypatch, capsys):
    _unclean_promotion(prepared_inputs, unreadable=["train.npz"])
    monkeypatch.setattr(sys, "argv", _report_only_argv(sys.argv))
    prep.main()
    gate = _support_report(capsys.readouterr().out)["promotion_gate"]
    assert gate["reason"] == "promotion counters must be nonnegative integers"
    assert gate["states"] == {"promoted": 1, "no_v2": 1}
    assert "train" not in json.dumps(gate)


def test_report_only_records_a_passed_promotion_gate(prepared_inputs, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", _report_only_argv(sys.argv))
    prep.main()
    assert _support_report(capsys.readouterr().out)["promotion_gate"] == {
        "passed": True, "cohort_tiles": 2, "states": {"promoted": 2},
        "verify": {"has_v1": 2, "stamped": 2, "has_v2_left": 0}}


def _drop_sidecar(prepared_inputs, model="clay", tile="oldtest"):
    (prepared_inputs["distill-root"] / f"{model}_r2" / f"{tile}.npz").unlink()


def test_report_only_records_missing_teacher_sidecars(prepared_inputs, monkeypatch, capsys):
    _drop_sidecar(prepared_inputs)
    monkeypatch.setattr(sys, "argv", _report_only_argv(sys.argv))
    prep.main()
    report = _support_report(capsys.readouterr().out)
    assert report["sidecar_inventory"]["clay"] == {"expected_tiles": 2, "verified_tiles": 1, "missing_tiles": 1}
    assert all(v["missing_tiles"] == 0 for m, v in report["sidecar_inventory"].items() if m != "clay")
    assert report["variants"]["freeze"]["observations"] == 1


def test_the_freeze_still_refuses_missing_teacher_sidecars(prepared_inputs):
    _drop_sidecar(prepared_inputs)
    with pytest.raises(ValueError, match="missing teacher sidecars"):
        prep.main()
    assert not prepared_inputs["out-dir"].exists()


@pytest.mark.parametrize("report_only", [False, True])
def test_a_teacher_sidecar_on_an_evaluation_tile_stops_every_mode(prepared_inputs, monkeypatch, report_only):
    """Exposure: a sidecar on an evaluation tile is never recorded and skipped."""
    sidecars = prepared_inputs["distill-root"] / "clay_r2"
    with np.load(sidecars / "train.npz", allow_pickle=False) as train:
        np.savez(sidecars / "campaign000.npz", **dict(train))
    if report_only:
        monkeypatch.setattr(sys, "argv", _report_only_argv(sys.argv))
    with pytest.raises(ValueError, match="evaluation-only campaign has distilled training sidecars"):
        prep.main()


def _flag_empty_tessera(prepared_inputs, tile="campaign002"):
    path = prepared_inputs["staging-dir"] / f"{tile}.npz"
    with np.load(path, allow_pickle=False) as data:
        data = dict(data)
    np.savez(path, **dict(data, tessera=np.zeros_like(data["tessera"]), has_tessera=1))


def test_report_only_records_a_readiness_error_as_an_exclusion(prepared_inputs, monkeypatch, capsys):
    _flag_empty_tessera(prepared_inputs)
    monkeypatch.setattr(sys, "argv", _report_only_argv(sys.argv))
    prep.main()
    report = _support_report(capsys.readouterr().out)
    assert report["excluded_evaluation_tiles"]["campaign"] == {
        "readiness_error:positive has_tessera on empty/invalid array": 1}
    assert report["variants"]["freeze"]["observations"] == 1
    assert "campaign002" not in json.dumps(report)


def test_report_only_still_raises_a_readiness_error_on_a_cohort_tile(prepared_inputs, monkeypatch):
    """The decision covered evaluation tiles; a cohort tile that fails stays fatal."""
    path = prepared_inputs["cohort-dir"] / "train.npz"
    with np.load(path, allow_pickle=False) as data:
        data = dict(data)
    np.savez(path, **dict(data, tessera=np.zeros_like(data["tessera"]), has_tessera=1))
    monkeypatch.setattr(sys, "argv", _report_only_argv(sys.argv))
    with pytest.raises(ValueError, match="positive has_tessera on empty/invalid array"):
        prep.main()


def test_report_only_does_not_swallow_a_non_value_readiness_error(prepared_inputs, monkeypatch):
    original = prep.tile_readiness

    def broken(path):
        if path.stem == "campaign002":
            raise TypeError("unexpected")
        return original(path)
    monkeypatch.setattr(prep, "tile_readiness", broken)
    monkeypatch.setattr(sys, "argv", _report_only_argv(sys.argv))
    with pytest.raises(TypeError, match="unexpected"):
        prep.main()


def test_the_freeze_still_raises_a_readiness_error(prepared_inputs):
    _flag_empty_tessera(prepared_inputs)
    with pytest.raises(ValueError, match="positive has_tessera on empty/invalid array"):
        prep.main()
    assert not prepared_inputs["out-dir"].exists()


def test_report_only_reports_an_empty_variant_instead_of_failing(prepared_inputs, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", _report_only_argv(sys.argv))
    prep.main()
    v = _support_report(capsys.readouterr().out)["variants"]
    assert v["freeze"]["observations"] == 1
    assert v["freeze_with_frame_2016"] == {"observations": 0, "admitted_tiles_by_role": {},
                                           "empty_support": "no eligible observations with common input support"}


def test_report_only_does_not_swallow_other_selection_errors(prepared_inputs, monkeypatch, capsys):
    def mismatch(*args, **kwargs):
        raise ValueError("NFI observation year differs from spectral year: x")
    monkeypatch.setattr(sys, "argv", _report_only_argv(sys.argv))
    monkeypatch.setattr(prep, "select_holdout", mismatch)
    with pytest.raises(ValueError, match="observation year differs"):
        prep.main()
    assert "SUPPORT_REPORT_BEGIN" not in capsys.readouterr().out


def test_report_only_refuses_a_tile_changed_during_the_report(prepared_inputs, monkeypatch, capsys):
    root = _add_evaluation_root(prepared_inputs, {"full": _with(**BASELINE_2016)})
    monkeypatch.setattr(sys, "argv", _report_only_argv(sys.argv) + ["--evaluation-root", f"topup:2024:{root}"])
    original = prep.frame_2016_on_bar

    def read_then_touch(path):
        result = original(path)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return result
    monkeypatch.setattr(prep, "frame_2016_on_bar", read_then_touch)
    with pytest.raises(ValueError, match="input changed during holdout preparation"):
        prep.main()
    assert "SUPPORT_REPORT_BEGIN" not in capsys.readouterr().out


@pytest.mark.parametrize("failures, admitted", [
    (["height:missing"], True),
    (["height:missing", "markfukt:missing"], True),
    (["dem:missing"], False),
    (["dem:missing", "height:missing"], False),
    (["height:missing", "dem:invalid_shape_or_dtype"], False),
    (["height:nonfinite"], False),
    ([], False),
])
def test_mean_fill_admits_only_absent_nan_nodata_channels(failures, admitted):
    assert prep.mean_filled_absent_only(failures, {"height", "markfukt"}) is admitted


@pytest.mark.parametrize("arrays, on_bar", [
    (BASELINE_2016, True),
    (dict(BASELINE_2016, has_frame_2016=0), False),
    (dict(BASELINE_2016, s1_vv_vh_2016=np.zeros((2, 8, 8))), False),
    (dict(BASELINE_2016, frame_2016=np.zeros((6, 8, 8))), False),
    ({k: v for k, v in BASELINE_2016.items() if k != "frame_2016"}, False),
    ({k: v for k, v in BASELINE_2016.items() if k != "s1_vv_vh_2016"}, False),
])
def test_frame_2016_on_bar_mirrors_the_census(tmp_path, arrays, on_bar):
    path = tmp_path / "t.npz"
    np.savez(path, **arrays)
    assert prep.frame_2016_on_bar(path) is on_bar


@pytest.mark.parametrize("extra, message", [
    (["--population", "balanced", "--inventory-years", "2023", "2024", "--observations-per-year", "1"],
     "requires --population all"),
    (["--out-dir", "/never/created"], "requires --population all and no --out-dir"),
])
def test_report_only_refuses_quotas_and_outputs_before_any_read(prepared_inputs, monkeypatch, extra, message):
    monkeypatch.setattr(sys, "argv", _report_only_argv(sys.argv) + extra)
    monkeypatch.setattr(prep, "preparation_runtime", lambda *a: pytest.fail("runtime read before validation"))
    with pytest.raises(ValueError, match=message):
        prep.main()


def test_freeze_still_requires_an_out_dir(prepared_inputs, monkeypatch):
    argv = list(sys.argv); i = argv.index("--out-dir"); del argv[i:i+2]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="--out-dir is required"):
        prep.main()
