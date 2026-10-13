"""The top-up planner must exclude, cover and place by the decided rules, and leak no identity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

import derive_missing_vpp_mgrs  # noqa: E402
from scripts import plan_nfi_topup_manifest as plan  # noqa: E402


@pytest.fixture(autouse=True)
def fake_mgrs(monkeypatch):
    """The mgrs wheel is a pod dependency, not a test one: one cell per 100 km of easting."""
    monkeypatch.setattr(derive_missing_vpp_mgrs, "mgrs_tiles_for_bbox",
                        lambda bbox: {f"33V{int(bbox['west'] // 100_000):02d}"})
from tests.test_census_holdout_val_512 import MODELS, _bbox, _plots, _tile, fieldtruth_file  # noqa: E402,F401


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for sub in ("holdout_eval_512", "nfi", "distill/heads", "unified_v2_512", "vpp_wekeo", "manifests"):
        (root / sub).mkdir(parents=True)
    kept = _bbox(100, 1250)                      # evaluation-copy tile, 2019
    _tile(root / "holdout_eval_512/holdoutval_100_1250_2019.npz", kept, 2019, stamped=True)
    train = _bbox(140, 1250)                     # same-year (2019) training tile
    np.savez(root / "unified_v2_512/tile_a.npz",
             bbox_3006=np.array([train["west"], train["south"], train["east"], train["north"]]),
             year=np.array(2019), spectral=np.zeros((2, 2, 2)))
    other = _bbox(160, 1250)                     # other-year (2021) training tile
    np.savez(root / "unified_v2_512/tile_b.npz",
             bbox_3006=np.array([other["west"], other["south"], other["east"], other["north"]]),
             year=np.array(2021), spectral=np.zeros((2, 2, 2)))

    def at(bbox, dx, dy):
        return bbox["west"] + dx, bbox["north"] - dy

    free_a, free_b, free_c = _bbox(120, 1250), _bbox(121, 1250), _bbox(180, 1250)
    plots = _plots([
        (880001, 1, 2019, *at(kept, 2560, 2560)),    # covered by the evaluation copy
        (880002, 1, 2019, *at(kept, 2560, 2000)),    # in a teacher pool
        (880003, 1, 2019, *at(train, 2560, 2560)),   # inside a same-year training tile: infeasible
        (880004, 1, 2019, *at(free_a, 2560, 2560)),  # free cell -> candidate tile 1
        (880005, 1, 2019, *at(free_a, 2600, 2600)),  # same cell
        (880006, 1, 2019, *at(free_b, 100, 2560)),   # neighbour free cell, own tile
        (880007, 1, 2019, *at(other, 2560, 2560)),   # on an other-year training tile: allowed
        (880008, 1, 2019, *at(free_c, 2560, 2560)),  # far free cell
        (880009, 1, 2019, *at(_bbox(139, 1250), 5000, 2560)),  # cell next to the training tile; centred tile overlaps it -> dropped
    ])
    plots.loc[plots["TractID"] == 880008, "VolSpruce"] = -5.0  # invalid -> never a candidate
    plots.to_parquet(root / "nfi/nfi_plots.parquet", index=False)
    pinned = pd.DataFrame({"tile_name": ["tile_x"], "TractID": [880002], "PlotID": [1], "Year": [2019],
                           "row": [100], "col": [100]})
    pinned.to_parquet(root / "nfi/nfi_index_unified_v2_512.parquet", index=False)
    for model in MODELS:
        pd.DataFrame({"tile_name": ["tile_x"], "TractID": [880002], "PlotID": [1], "f0": [0.5]}).to_parquet(
            root / f"distill/heads/{model}_r2_plot_features.parquet", index=False)
    return root


def _run(root: Path, fieldtruth_file: Path, tmp_path: Path, target: int) -> dict:
    out = tmp_path / "plan"
    summary = plan.main(["--data-root", str(root), "--fieldtruth-file", str(fieldtruth_file),
                         "--years", "2019", "--target", str(target), "--out-dir", str(out),
                         "--summary-json", str(tmp_path / "summary.json")])
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    return summary


def test_planner_applies_every_rule_and_closes_the_gap(data_root, fieldtruth_file, tmp_path):
    s = _run(data_root, fieldtruth_file, tmp_path, target=4)["by_year"][0]
    assert s["covered_on_evaluation_copy"] == 1 and s["need"] == 3
    assert s["excluded_teacher_pool"] == 1 and s["already_covered"] == 1
    assert s["invalid_values"] == 1
    assert s["inside_same_year_training_footprint"] == 1
    assert s["candidates"] == 5                       # 880004/5/6/7/9
    # 880004+880005 share a tile; 880006 own tile; 880007 on the 2021 tile (allowed);
    # 880009's centred tile reaches into the 2019 training tile -> dropped
    assert s["candidate_tiles"] == 4 and s["candidate_tiles_dropped_same_year_overlap"] == 1
    assert s["feasible_tiles"] == 3 and s["feasible_plot_years"] == 4
    assert s["chosen_plot_years"] >= 3 and s["gap_after_plan"] == 0
    manifest = json.loads((tmp_path / "plan/nfi2019_topup_512.json").read_text())
    assert len(manifest) == s["chosen_tiles"]
    for t in manifest:
        assert set(t) == {"name", "cell", "bbox_3006", "year", "source", "n_plots"}
        assert t["bbox_3006"]["east"] - t["bbox_3006"]["west"] == 5120
        assert all(v % 10 == 0 for v in t["bbox_3006"].values())
    gaps = json.loads((tmp_path / "plan/nfi2019_topup_vpp_gaps.json").read_text())
    assert gaps["year"] == 2019 and gaps["missing"] == gaps["needed"] and len(gaps["needed"]) >= 1


def test_no_gap_means_an_empty_manifest(data_root, fieldtruth_file, tmp_path):
    s = _run(data_root, fieldtruth_file, tmp_path, target=1)["by_year"][0]
    assert s["need"] == 0 and s["chosen_tiles"] == 0 and s["gap_after_plan"] == 0
    assert json.loads((tmp_path / "plan/nfi2019_topup_512.json").read_text()) == []


def test_selection_is_prespecified_by_identity_hash(data_root, fieldtruth_file, tmp_path):
    s = _run(data_root, fieldtruth_file, tmp_path, target=2)["by_year"][0]   # need 1 -> one tile
    manifest = json.loads((tmp_path / "plan/nfi2019_topup_512.json").read_text())
    assert s["chosen_tiles"] == 1 and len(manifest) == 1
    # the chosen tile is the one whose best plot ranks first under the frozen seed
    ranks = {"880004_880005": min(plan.rank_key(880004, 1, 2019), plan.rank_key(880005, 1, 2019)),
             "880006": plan.rank_key(880006, 1, 2019), "880007": plan.rank_key(880007, 1, 2019)}
    best = min(ranks, key=ranks.get)
    assert manifest[0]["n_plots"] == (2 if best == "880004_880005" else 1)


def test_missing_cohort_refuses_rather_than_planning_without_the_footprint_rule(data_root, fieldtruth_file, tmp_path):
    import shutil
    shutil.rmtree(data_root / "unified_v2_512")
    with pytest.raises(SystemExit, match="cohort directory .* is missing"):
        plan.main(["--data-root", str(data_root), "--fieldtruth-file", str(fieldtruth_file),
                   "--years", "2019", "--target", "4", "--out-dir", str(tmp_path / "plan"),
                   "--summary-json", str(tmp_path / "s.json")])
    assert not (tmp_path / "plan").exists() and not (tmp_path / "s.json").exists()


def test_outputs_carry_no_plot_identity_or_coordinate(data_root, fieldtruth_file, tmp_path):
    _run(data_root, fieldtruth_file, tmp_path, target=4)
    text = (tmp_path / "summary.json").read_text() + (tmp_path / "plan/nfi2019_topup_512.json").read_text()
    for ident in ("880004", "880005", "880006", "880007", "Easting", "Northing", "_keys", "_rank"):
        assert ident not in text


# ---- cells without an HR-VPP product (decision 2026-10-13: replan with substitutes) ----

def _cell(bbox: dict) -> str:
    """A valid MGRS-shaped id per kilometre of easting, distinct for every fixture tile."""
    return "33VW" + "ABCDEFGHJKLMNPQRSTUV"[int(bbox["west"] // 1000) % 20]


@pytest.fixture
def named_cells(monkeypatch):
    monkeypatch.setattr(derive_missing_vpp_mgrs, "mgrs_tiles_for_bbox", lambda bbox: {_cell(bbox)})


def _plan(root, fieldtruth_file, out, target, *no_vpp):
    args = ["--data-root", str(root), "--fieldtruth-file", str(fieldtruth_file), "--years", "2019",
            "--target", str(target), "--out-dir", str(out / "plan"), "--summary-json", str(out / "s.json")]
    if no_vpp:
        args += ["--no-vpp", *no_vpp]
    s = plan.main(args)["by_year"][0]
    return s, json.loads((out / "plan/nfi2019_topup_512.json").read_text()), \
        json.loads((out / "plan/nfi2019_topup_vpp_gaps.json").read_text())


def test_a_tile_in_a_cell_without_vpp_is_replaced_by_the_next_in_rank(data_root, fieldtruth_file, tmp_path, named_cells):
    s0, m0, _ = _plan(data_root, fieldtruth_file, tmp_path / "a", 2)          # need 1 -> the best tile
    assert s0["chosen_tiles"] == 1 and s0["candidate_tiles_dropped_no_vpp_product"] == 0
    blocked = _cell(m0[0]["bbox_3006"])
    s1, m1, gaps = _plan(data_root, fieldtruth_file, tmp_path / "b", 2, blocked)
    assert s1["no_vpp_cells"] == [blocked]
    assert s1["candidate_tiles_dropped_no_vpp_product"] == 1
    assert s1["plots_on_no_vpp_tiles"] == m0[0]["n_plots"]
    assert s1["feasible_tiles"] == s0["feasible_tiles"] - 1
    assert s1["gap_after_plan"] == 0 and len(m1) == 1 and m1[0]["name"] != m0[0]["name"]
    assert blocked not in gaps["needed"] and blocked not in gaps["missing"]
    # the substitute is the best-ranked tile among the rest, i.e. the next in the first run's order
    s2, m2, _ = _plan(data_root, fieldtruth_file, tmp_path / "c", 5)          # need 4 -> every feasible tile
    order = [t["name"] for t in m2]
    assert m1[0]["name"] in order and m0[0]["name"] in order


def test_a_year_scoped_cell_equals_an_all_years_cell_for_that_year(data_root, fieldtruth_file, tmp_path, named_cells):
    _, m0, _ = _plan(data_root, fieldtruth_file, tmp_path / "a", 2)
    blocked = _cell(m0[0]["bbox_3006"])
    s_all, m_all, _ = _plan(data_root, fieldtruth_file, tmp_path / "b", 2, blocked)
    s_year, m_year, _ = _plan(data_root, fieldtruth_file, tmp_path / "c", 2, f"2019:{blocked}")
    assert m_all == m_year and s_all["no_vpp_cells"] == s_year["no_vpp_cells"] == [blocked]


def test_blocking_every_feasible_cell_reports_the_gap_rather_than_choosing_a_blocked_tile(
        data_root, fieldtruth_file, tmp_path, named_cells):
    _, m_all, _ = _plan(data_root, fieldtruth_file, tmp_path / "a", 5)       # every feasible tile
    cells = sorted({_cell(t["bbox_3006"]) for t in m_all})
    s, m, gaps = _plan(data_root, fieldtruth_file, tmp_path / "b", 5, *cells)
    assert m == [] and gaps["needed"] == [] and s["feasible_tiles"] == 0
    assert s["gap_after_plan"] == s["need"] == 4


@pytest.mark.parametrize("token, message", [
    ("33vwq", "is not CELL or YEAR:CELL"),
    ("33VWQ1", "is not CELL or YEAR:CELL"),
    (":33UVC", "is not CELL or YEAR:CELL"),        # empty year is a typo, not "every year"
    ("2018:", "is not CELL or YEAR:CELL"),
    ("2018::33UVC", "is not CELL or YEAR:CELL"),
    ("18:33UVC", "is not CELL or YEAR:CELL"),
    ("x:33VWQ", "is not CELL or YEAR:CELL"),
    ("2018-33UVC", "is not CELL or YEAR:CELL"),
    (" 33UVC", "is not CELL or YEAR:CELL"),
    ("00UVC", "is not an MGRS 100 km square"),      # zone 0
    ("61UVC", "is not an MGRS 100 km square"),      # zone 61
    ("33IVC", "is not an MGRS 100 km square"),      # band I
    ("33YVC", "is not an MGRS 100 km square"),      # band Y
    ("34WCI", "is not an MGRS 100 km square"),      # row I
    ("34WCW", "is not an MGRS 100 km square"),      # row beyond V
    ("33UCC", "is not an MGRS 100 km square"),      # column C belongs to zones 1, 4, ... not 33
    ("34WOA", "is not an MGRS 100 km square"),      # column O
    ("2021:33VWQ", "names a year that is not planned"),
])
def test_a_malformed_or_unplanned_token_refuses_before_any_output(data_root, fieldtruth_file, tmp_path, token, message):
    with pytest.raises(SystemExit, match=message):
        _plan(data_root, fieldtruth_file, tmp_path, 4, token)
    assert not (tmp_path / "plan").exists() and not (tmp_path / "s.json").exists()


@pytest.mark.parametrize("cell", ["33UVC", "34WCA", "34WDR", "35WLP", "34WCT", "32VPL",
                                  "01CAA", "60XZV", "31NHV", "02DJA", "03ESA"])
def test_real_mgrs_cells_are_accepted(cell):
    assert plan.parse_no_vpp([cell, f"2019:{cell}"], [2019]) == {2019: frozenset({cell})}


def test_parse_no_vpp_scopes_tokens_by_year():
    got = plan.parse_no_vpp(["33UVC", "2018:34WCA"], [2018, 2021])
    assert got == {2018: frozenset({"33UVC", "34WCA"}), 2021: frozenset({"33UVC"})}
    assert plan.parse_no_vpp([], [2019]) == {2019: frozenset()}
