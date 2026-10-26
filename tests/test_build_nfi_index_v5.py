"""Index v5: v4's rows over several roots, strict where the freeze is strict."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import build_nfi_index_v5 as v5

E, N = 500_000, 6_500_000          # tile centre (SWEREF99 TM)
HALF = 512 * 10 // 2


def _tile(path, year, *, easting=E, northing=N):
    np.savez(path, easting=float(easting), northing=float(northing), year=year,
             bbox_3006=np.array([easting - HALF, northing - HALF, easting + HALF, northing + HALF]),
             spectral=np.zeros((1, 2, 2), np.float32))


@pytest.fixture
def layout(tmp_path):
    nfi = tmp_path / "nfi"; nfi.mkdir()
    pd.DataFrame({"TractID": [1, 2, 3, 4], "PlotID": [1, 1, 1, 1], "Year": [2023, 2024, 2024, 2024],
                  "Easting": [E, E, E + 100, 900_000], "Northing": [N, N, N + 100, N],
                  "VolPine": [10.0, 20.0, 30.0, 40.0]}).to_parquet(nfi / "nfi_plots.parquet")
    cohort, copy = tmp_path / "cohort", tmp_path / "copy"
    cohort.mkdir(); copy.mkdir()
    _tile(cohort / "c2023.npz", 2023)
    _tile(copy / "h2024.npz", 2024)
    _tile(copy / "h2024.tmp.npz", 2024)     # a writer's temp file is never a member
    return {"nfi": nfi, "cohort": cohort, "copy": copy, "out": tmp_path / "out" / "v5.parquet"}


def _argv(layout, *roots):
    argv = ["--nfi-dir", str(layout["nfi"]), "--out", str(layout["out"])]
    for role, key in roots:
        argv += ["--root", f"{role}:{layout[key]}"]
    return argv


def test_indexes_every_root_year_exact_with_an_aggregate_sidecar(layout):
    record = v5.main(_argv(layout, ("cohort", "cohort"), ("holdout_copy", "copy")))
    index = pd.read_parquet(layout["out"])
    # 2023 plot on the 2023 tile only; both on-tile 2024 plots on the 2024 tile; the far plot nowhere.
    assert sorted(zip(index.tile_name, index.TractID)) == [("c2023", 1), ("h2024", 2), ("h2024", 3)]
    assert list(index.columns[:2]) == ["tile_name", "tile_path"]       # v4's order
    assert list(index.columns[-3:]) == ["row", "col", "tier"]
    assert [(r["role"], r["tiles"], r["rows"]) for r in record["roots"]] == [("cohort", 1, 1), ("holdout_copy", 1, 2)]
    sidecar = json.loads(layout["out"].with_suffix(".json").read_text())
    assert sidecar["parquet"]["sha256"] == v5.sha256_file(layout["out"])
    assert sidecar["observations"] == 3 and sidecar["rows_by_year"] == {"2023": 1, "2024": 2}
    assert "TractID" not in json.dumps(sidecar)


def test_rows_equal_the_v4_builder_on_the_same_root(layout, monkeypatch):
    from scripts import nfi_tile_coverage as v4
    v4_out = layout["out"].parent / "v4.parquet"
    monkeypatch.setattr(sys, "argv", ["v4", "--tile-dir", str(layout["copy"]), "--infer-size",
                                      "--nfi-dir", str(layout["nfi"]), "--out", str(v4_out)])
    v4.main()
    v5.main(_argv(layout, ("holdout_copy", "copy")))
    pd.testing.assert_frame_equal(pd.read_parquet(v4_out), pd.read_parquet(layout["out"]))


def test_refuses_to_overwrite(layout):
    v5.main(_argv(layout, ("holdout_copy", "copy")))
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        v5.main(_argv(layout, ("holdout_copy", "copy")))


def test_an_output_that_appears_during_the_build_is_never_overwritten(layout, monkeypatch):
    original = v5.build

    def build_then_race(*args):
        result = original(*args)
        layout["out"].parent.mkdir(parents=True, exist_ok=True)
        layout["out"].write_bytes(b"another writer")
        return result
    monkeypatch.setattr(v5, "build", build_then_race)
    with pytest.raises(FileExistsError):
        v5.main(_argv(layout, ("holdout_copy", "copy")))
    assert layout["out"].read_bytes() == b"another writer"
    assert not layout["out"].with_suffix(".json").exists()


def test_a_tile_name_in_two_roots_is_refused(layout):
    _tile(layout["cohort"] / "h2024.npz", 2024)
    with pytest.raises(SystemExit, match="h2024 is in roots cohort and holdout_copy"):
        v5.main(_argv(layout, ("cohort", "cohort"), ("holdout_copy", "copy")))
    assert not layout["out"].exists()


def test_an_unreadable_tile_fails_instead_of_being_skipped(layout):
    (layout["copy"] / "broken.npz").write_bytes(b"not a zip")
    with pytest.raises(Exception):
        v5.main(_argv(layout, ("holdout_copy", "copy")))
    assert not layout["out"].exists()


@pytest.mark.parametrize("bbox", [
    [E, N, E - 10, N + 10],            # east before west
    [E, N, E + 10, N - 10],            # north below south
    [E, N, E + 5125, N + 5125],        # extent not a whole number of 10 m pixels
    [E, N, E + 10, np.nan],
])
def test_an_invalid_bbox_fails(layout, bbox):
    np.savez(layout["copy"] / "odd.npz", easting=float(E), northing=float(N), year=2024,
             bbox_3006=np.array(bbox, dtype=float))
    with pytest.raises(SystemExit, match="invalid bbox_3006"):
        v5.main(_argv(layout, ("holdout_copy", "copy")))
    assert not layout["out"].exists()


def test_a_tile_without_bbox_uses_its_stored_size_or_fails(layout):
    np.savez(layout["copy"] / "sized.npz", easting=float(E), northing=float(N), year=2024, tile_size_px=512)
    v5.main(_argv(layout, ("holdout_copy", "copy")))
    index = pd.read_parquet(layout["out"])
    assert sorted(index.loc[index.tile_name == "sized", "TractID"]) == [2, 3]
    np.savez(layout["cohort"] / "bare.npz", easting=float(E), northing=float(N), year=2023)
    layout["out"] = layout["out"].with_name("again.parquet")
    with pytest.raises(SystemExit, match="neither bbox_3006 nor tile_size_px"):
        v5.main(_argv(layout, ("cohort", "cohort")))


def test_a_duplicated_plot_year_fails(layout):
    plots = pd.read_parquet(layout["nfi"] / "nfi_plots.parquet")
    pd.concat([plots, plots.iloc[[1]]]).to_parquet(layout["nfi"] / "nfi_plots.parquet")
    with pytest.raises(SystemExit, match="occurs twice on one tile"):
        v5.main(_argv(layout, ("holdout_copy", "copy")))
    assert not layout["out"].exists()


def test_an_empty_index_keeps_v4_column_order(layout, tmp_path):
    empty = tmp_path / "empty"; empty.mkdir()
    _tile(empty / "far.npz", 2024, easting=800_000)
    v5.main(_argv({**layout, "empty": empty}, ("holdout_copy", "empty")))
    index = pd.read_parquet(layout["out"])
    assert index.empty
    assert list(index.columns[:2]) == ["tile_name", "tile_path"]
    assert list(index.columns[-3:]) == ["row", "col", "tier"]
