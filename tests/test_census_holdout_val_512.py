"""The holdout census must count what the plan asks for, and leak no identity."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import census_holdout_val_512 as census  # noqa: E402

FIELDTRUTH_COMMIT = "ba31235c6bb5f4c2fa9d5c71929926a0023d062b"
MODELS = ("clay", "croma", "prithvi300m", "prithvi300m4f", "prithvi600m", "terramind", "tessera")
SIZE = 16  # spectral side in the fixtures; the census checks 24 bands and squareness, not 512


@pytest.fixture(scope="module")
def fieldtruth_file(tmp_path_factory) -> Path:
    """Codex's identity module at the sizing's process commit, or skip when the object is absent."""
    proc = subprocess.run(["git", "-C", str(REPO), "show", f"{FIELDTRUTH_COMMIT}:imint/eval/fieldtruth.py"],
                          capture_output=True)
    if proc.returncode != 0:
        pytest.skip("fieldtruth.py at the task-0103 process commit is not in this clone")
    path = tmp_path_factory.mktemp("ft") / "fieldtruth.py"
    path.write_bytes(proc.stdout)
    assert census.sha256_path(path) == census.FIELDTRUTH_SHA256
    return path


def _bbox(cell_x: int, cell_y: int) -> dict:
    w, s = cell_x * 5120, cell_y * 5120
    return {"west": w, "south": s, "east": w + 5120, "north": s + 5120}


def _tile(path: Path, bbox: dict, year: int, *, dates=None, tessera_zero=False, frame=True,
          stamped=False, vpp=True, mode=0o644) -> None:
    arrays = {
        "spectral": np.full((24, SIZE, SIZE), 1200, dtype=np.uint16),
        "label": np.ones((SIZE, SIZE), dtype=np.uint8),
        "bbox_3006": np.array([bbox["west"], bbox["south"], bbox["east"], bbox["north"]]),
        "tile_size_px": np.array(512),
        "dates": np.array(dates or [f"{year - 1}-09-10", f"{year}-06-01", f"{year}-07-01", f"{year}-08-01"]),
        "temporal_mask": np.array([1, 1, 1, 1], dtype=np.uint8),
        "s1_vv_vh": np.ones((2, SIZE, SIZE), dtype=np.float32), "has_s1": np.array(1),
        "s1_vv_vh_2016": np.ones((2, SIZE, SIZE), dtype=np.float32) if frame else np.zeros((2, SIZE, SIZE), np.float32),
        "tessera": np.zeros((2, SIZE, SIZE), np.float16) if tessera_zero else np.ones((2, SIZE, SIZE), np.float16),
        "has_tessera": np.array(1), "tessera_year": np.array(year),
    }
    for key in census.AUX_KEYS:
        arrays[key] = np.full((SIZE, SIZE), 3.0, dtype=np.float32)
    if frame:
        arrays["frame_2016"] = np.full((6, SIZE, SIZE), 900, dtype=np.uint16)
        arrays["has_frame_2016"] = np.array(1)
        arrays["frame_2016_year"] = np.array(2016)
    if stamped:
        arrays["tessera_source"] = np.array("geotessera-0.10.2")
    if vpp:
        arrays["vpp_year"] = np.array(year)
    np.savez(path, **arrays)
    os.chmod(path, mode)


def _plots(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["TractID", "PlotID", "Year", "Easting", "Northing"])
    for col in ("VolPine", "VolContorta", "VolSpruce", "VolBirch", "VolOtherDec"):
        frame[col] = 10.0
    frame["LandUseClass"] = 1
    return frame


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for sub in ("holdout_val_512", "audits", "nfi", "distill/heads", "unified_v2_512",
                "unified_v2_512_nfi2024_staging", "ladder_inference", "checkpoints/ladder/clay_r2"):
        (root / sub).mkdir(parents=True)
    a, b, c = _bbox(100, 1250), _bbox(110, 1250), _bbox(120, 1250)
    hold = root / "holdout_val_512"
    _tile(hold / "holdoutval_100_1250_2019.npz", a, 2019)                       # complete, unstamped
    _tile(hold / "holdoutval_110_1250_2022.npz", b, 2022, tessera_zero=True,     # lying flag, no frame,
          frame=False, stamped=True, vpp=False, mode=0o600)                      # not world-readable
    _tile(hold / "holdoutval_120_1250_2019.npz", c, 2019, stamped=True,          # complete, promoted stamp,
          dates=["2018-09-10", "2020-06-01", "2020-07-01", "2020-08-01"])        # dates contradict year below
    with np.load(hold / "holdoutval_120_1250_2019.npz") as d:
        arrays = dict(d)
    arrays["year"] = np.array(2019)
    np.savez(hold / "holdoutval_120_1250_2019.npz", **arrays)
    manifest = [{"name": "holdoutval_100_1250_2019", "bbox_3006": a, "year": 2019, "source": "lulc"},
                {"name": "holdoutval_110_1250_2022", "bbox_3006": b, "year": 2022, "source": "lulc"},
                {"name": "holdoutval_120_1250_2019", "bbox_3006": c, "year": 2019, "source": "lulc"},
                {"name": "holdoutval_130_1250_2023", "bbox_3006": _bbox(130, 1250), "year": 2023, "source": "lulc"}]
    (root / "audits/holdout_val_phaseA.json").write_text(json.dumps(manifest))

    def inside(bbox, dx=2560, dy=2560):
        return bbox["west"] + dx, bbox["north"] - dy

    plots = _plots([
        (770001, 1, 2019, *inside(a)),            # eligible on A
        (770002, 1, 2019, *inside(a, dx=40)),     # on A, inside the 80 m border -> outside the crop
        (770003, 1, 2019, *inside(a, dy=1000)),   # on A, in a teacher pool -> excluded
        (770004, 1, 2020, *inside(a)),            # wrong year for A -> not co-located
        (770005, 1, 2022, *inside(b)),            # eligible on B, which is not world-readable
        (770006, 1, 2019, *inside(c)),            # on C, invalid volume (set below)
        (770007, 1, 2019, *inside(c, dy=1000)),   # eligible on C, already in index v4
    ])
    plots.loc[plots["TractID"] == 770006, "VolPine"] = -1.0
    plots.to_parquet(root / "nfi/nfi_plots.parquet", index=False)
    pinned = pd.DataFrame({"tile_name": ["tile_1_1"], "TractID": [770003], "PlotID": [1], "Year": [2019],
                           "row": [100], "col": [100]})
    pinned.to_parquet(root / "nfi/nfi_index_unified_v2_512.parquet", index=False)
    v4 = pd.concat([pinned, pd.DataFrame({"tile_name": ["tile_2_2"], "TractID": [770007], "PlotID": [1],
                                          "Year": [2019], "row": [200], "col": [200]})])
    v4.to_parquet(root / "nfi/nfi_index_v4_2026-09-30.parquet", index=False)
    for model in MODELS:
        pd.DataFrame({"tile_name": ["tile_1_1"], "TractID": [770003], "PlotID": [1], "f0": [0.5]}).to_parquet(
            root / f"distill/heads/{model}_r2_plot_features.parquet", index=False)
        (root / f"distill/heads/{model}_r2_split.json").write_text(json.dumps(
            {"train_tiles": ["tile_1_1"], "test_tiles": ["tile_2_2"]}))
    # training cohort: one same-year tile overlapping A, one other-year tile overlapping C
    np.savez(root / "unified_v2_512/tile_514560_6402560.npz",
             bbox_3006=np.array([a["west"] + 2560, a["south"] + 2560, a["east"] + 2560, a["north"] + 2560]),
             year=np.array(2019), spectral=np.zeros((2, 2, 2)))
    np.savez(root / "unified_v2_512/tile_616960_6402560.npz",
             bbox_3006=np.array([c["west"] + 2560, c["south"] + 2560, c["east"] + 2560, c["north"] + 2560]),
             year=np.array(2021), spectral=np.zeros((2, 2, 2)))
    (root / "unified_v2_512/train.txt").write_text("tile_514560_6402560\n")
    (root / "unified_v2_512/val.txt").write_text("tile_616960_6402560\n")
    (root / "ladder_inference/tiles.json").write_text(json.dumps({"tiles": ["holdoutval_100_1250_2019"]}))
    (root / "checkpoints/ladder/clay_r2/training_log.json").write_text(json.dumps({"epochs": []}))
    _tile(root / "unified_v2_512_nfi2024_staging/nfi2024_1_1.npz", _bbox(1, 1), 2024, stamped=True)
    return root


def _run(root: Path, fieldtruth_file: Path, tmp_path: Path, extra: list[str] | None = None) -> dict:
    out = tmp_path / "census.json"
    report = census.main(["--data-root", str(root), "--fieldtruth-file", str(fieldtruth_file),
                          "--out-json", str(out), *(extra or [])])
    assert json.loads(out.read_text()) == report
    return report


def test_census_counts_files_keys_flags_and_modalities(data_root, fieldtruth_file, tmp_path):
    report = _run(data_root, fieldtruth_file, tmp_path)
    files = report["files"]
    assert files["present"] == 3 and files["readable"] == 3 and files["unreadable"] == []
    assert files["not_world_readable"] == 1 and files["not_world_readable_by_year"] == {"2022": 1}
    assert files["planned_but_missing_by_year"] == {"2023": 1}
    c = report["census"]
    assert c["flag_on_empty_array"] == {"tessera": 1}
    assert c["all_zero"]["tessera"] == {"present": 3, "all_zero": 1}
    assert c["all_zero"]["s1_vv_vh_2016"]["all_zero"] == 1
    assert set(c["keys_behind_nfi2024_bar"]) >= {"frame_2016", "has_frame_2016", "vpp_year", "frame_2016_year"}
    assert c["spectral_ok"] == 3 and c["bbox_on_grid"] == 3 and c["bbox_matches_manifest"] == 3
    m = report["modalities"]
    assert m["tiles_by_year"] == {"2019": 2, "2022": 1}
    assert m["year_conflict_strict"] == 1 and m["year_matches_manifest"] == {"2019": 2, "2022": 1}
    assert m["tessera_source"] == {"unstamped": 1, "geotessera-0.10.2": 2}
    assert m["frame_2016_usable"] == {"2019": 2}
    assert m["bar_modalities"] == {"2019": 2}
    assert m["bar_full_promoted_tessera"] == {"2019": 1}


def test_eligibility_uses_identity_crop_exclusion_and_separates_restricted_files(
        data_root, fieldtruth_file, tmp_path):
    report = _run(data_root, fieldtruth_file, tmp_path)
    assert report["teacher_feature_union"] == 1
    e = report["eligibility_by_year"]
    y19, y22, y20 = e["2019"], e["2022"], e["2020"]
    assert y19["plot_years_on_tiles"] == 5 and y19["in_common_crop"] == 4
    assert y19["excluded_teacher_pool"] == 1 and y19["invalid_values"] == 1
    assert y19["eligible"] == 2 == y19["eligible_world_readable"] and y19["eligible_restricted_files"] == 0
    assert y19["eligible_also_in_v4_crop"] == 1 and y19["eligible_new_vs_v4"] == 1
    assert y19["eligible_on_bar_modalities"] == 2 and y19["eligible_on_bar_full"] == 1
    assert y19["gap_to_target_all_files"] == 998 and y19["gap_to_target_bar_full"] == 999
    assert y22["eligible"] == 1 and y22["eligible_world_readable"] == 0 and y22["eligible_restricted_files"] == 1
    assert y22["eligible_on_bar_modalities"] == 0
    assert y20["plot_years_on_tiles"] == 0 and y20["gap_to_target_all_files"] == 1000


def test_never_trained_on_passes_and_reports_footprint_and_visualisation(data_root, fieldtruth_file, tmp_path):
    p = _run(data_root, fieldtruth_file, tmp_path)["provenance"]
    assert p["used_for_training_or_selection"] is False
    assert p["names_in_training_cohort_dir"] == 0 and p["split_files_naming_holdout_tiles"] == []
    assert [Path(x).name for x in p["visualisation_files_naming_holdout_tiles"]] == ["tiles.json"]
    assert set(p["teacher_feature_tables_naming_holdout_tiles"]) == set(MODELS)
    assert not any(p["teacher_feature_tables_naming_holdout_tiles"].values())
    assert p["plot_indexes_naming_holdout_tiles"] == {"pinned_index": 0, "v4_index": 0}
    scanned = {Path(f["path"]).name for f in p["split_files_scanned"]}
    assert {"train.txt", "val.txt", "clay_r2_split.json", "tiles.json"} <= scanned
    fp = p["footprint"]
    assert fp["cohort_tiles_with_geometry"] == 2
    assert fp["holdout_tiles_overlapping_same_year_training_tile"] == {"2019": 1}
    assert fp["holdout_tiles_overlapping_any_year_training_tile"] == {"2019": 2}
    assert [Path(h["path"]).name for h in p["text_files_mentioning_holdout"]] == ["holdout_val_phaseA.json", "tiles.json"]


def test_a_holdout_name_in_a_split_file_fails_the_check(data_root, fieldtruth_file, tmp_path):
    (data_root / "unified_v2_512/val.txt").write_text("tile_616960_6402560\nholdoutval_120_1250_2019\n")
    p = _run(data_root, fieldtruth_file, tmp_path)["provenance"]
    assert p["used_for_training_or_selection"] is True
    assert [Path(x).name for x in p["split_files_naming_holdout_tiles"]] == ["val.txt"]


def test_report_carries_no_plot_identity_or_coordinate(data_root, fieldtruth_file, tmp_path):
    text = json.dumps(_run(data_root, fieldtruth_file, tmp_path))
    for ident in ("770001", "770005", "770007", "Easting", "Northing", "_bbox", "_meta"):
        assert ident not in text


def test_sizing_cross_check_requires_identical_inputs_and_union(data_root, fieldtruth_file, tmp_path):
    sizing = {"inputs": {"/cephfs/nfi/nfi_plots.parquet": {"sha256": census.sha256_path(data_root / "nfi/nfi_plots.parquet")}},
              "teacher_feature_union": 1, "legacy_metadata_issues": {"metadata_PermissionError": 1},
              "legacy_existing_files": 3}
    path = tmp_path / "sizing.json"
    path.write_text(json.dumps(sizing))
    report = _run(data_root, fieldtruth_file, tmp_path, ["--expected-sizing", str(path)])
    assert report["sizing_cross_check"]["input_hash_mismatches"] == []
    sizing["inputs"]["/cephfs/nfi/nfi_plots.parquet"]["sha256"] = "0" * 64
    path.write_text(json.dumps(sizing))
    with pytest.raises(SystemExit, match="sizing cross-check failed"):
        _run(data_root, fieldtruth_file, tmp_path, ["--expected-sizing", str(path)])


def test_wrong_fieldtruth_version_is_refused(tmp_path):
    bogus = tmp_path / "fieldtruth.py"
    bogus.write_text("NFI_KEY = []\n")
    with pytest.raises(SystemExit, match="refusing a different identity"):
        census.load_fieldtruth(bogus)


def test_out_json_under_the_data_root_is_refused(data_root, fieldtruth_file):
    with pytest.raises(SystemExit, match="must not be under the data root"):
        census.main(["--data-root", str(data_root), "--fieldtruth-file", str(fieldtruth_file),
                     "--out-json", str(data_root / "census.json")])
