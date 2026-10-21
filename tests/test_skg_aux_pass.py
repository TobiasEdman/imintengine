"""skg_aux_pass writes the five mirror channels into tiles, records gaps and provenance, never zero-fills."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")

from imint.training import skg_open  # noqa: E402
from scripts import skg_aux_pass as ap  # noqa: E402
from tests.skg_mirror_fixture import write_raster  # noqa: E402

W, S, E, N = 500_000, 6_250_000, 500_640, 6_250_640       # one 64 x 64 tile on the lattice
PX = 64
GRUND = {"volym": 300, "grundyta": 25, "medeldiameter": 22}


def _mirror(root: Path, *, height_staged: bool = True, markfukt_value: int = 40) -> None:
    files, derived = {}, {}

    def add(rel, data, res, nodata, source, **extra):
        write_raster(root / rel, data, W - 100, N + 100, res, nodata=nodata)
        files[source] = {"sha256": "s" * 64, "index_modified": "4/1/2026 10:00 AM"}
        derived[rel] = {"from": source, "sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest(), **extra}

    for var, v in GRUND.items():
        data = np.full((PX + 20, PX + 20), v, dtype=np.int16)
        add(f"grunddata/omdrev2/{var}_01.tif", data, 10.0, -1, f"sks_{var}_omdrev2_01.zip")
        write_raster(root / f"grunddata/{var}_omdrev2.vrt", data, W - 100, N + 100, 10.0, nodata=-1)
    mf = np.full(((PX + 20) * 5, (PX + 20) * 5), markfukt_value, dtype=np.uint16)
    add("markfukt/markfukt_01.tif", mf, 2.0, 65535, "sksSLUMarkfuktighet01.zip")
    write_raster(root / "markfukt/markfukt.vrt", mf, W - 100, N + 100, 2.0, nodata=65535)
    th = np.full((PX + 20, PX + 20), 183, dtype=np.int16)
    add("tradhojd/tradhojd_10m_01.tif", th, 10.0, -1, "sksTradhojd01.zip", statistic="p90")
    write_raster(root / "tradhojd/tradhojd_10m.vrt", th, W - 100, N + 100, 10.0, nodata=-1)
    vrt = {"tradhojd/tradhojd_10m.vrt": {"sources": ["tradhojd/tradhojd_10m_01.tif"] if height_staged else []}}
    (root / "MANIFEST.json").write_text(json.dumps({"files": files, "derived": derived, "vrt": vrt}))


def _tile(path: Path, **extra) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, spectral=np.ones((6, PX, PX), dtype=np.float32),
                        easting=np.array(W + PX * 5), northing=np.array(S + PX * 5),
                        aux_complete=np.array(0), aux_missing=np.array(list(ap.AUX)),
                        aux_gap_reason=np.array("forestry_markfukt_unavailable__pending_IM-022"), **extra)
    return path


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv(skg_open.ENV_MIRROR_DIR, str(tmp_path / "mirror"))
    monkeypatch.setenv("IMINT_GIT_SHA", "abc")
    return tmp_path


def _load(p: Path) -> dict:
    with np.load(p, allow_pickle=True) as z:
        return dict(z)


def _run(env, *extra) -> int:
    return ap.main(["--data-dir", str(env / "tiles"), "--report", str(env / "report.json"), "--workers", "1", *extra])


def test_all_five_are_written_and_the_tile_is_unmarked(env, capsys):
    _mirror(env / "mirror")
    t = _tile(env / "tiles/nfi2018_a.npz")
    assert _run(env) == 0
    d = _load(t)
    assert int(d["aux_complete"]) == 1 and "aux_missing" not in d and "aux_gap_reason" not in d
    assert float(d["volume"][10, 10]) == 300 and float(d["basal_area"][0, 0]) == 25
    assert float(d["diameter"][5, 5]) == 22 and float(d["height"][3, 3]) == pytest.approx(18.3)
    assert float(d["markfukt"][7, 7]) == pytest.approx(0.40)
    assert all(d[k].shape == (PX, PX) and d[k].dtype == np.float32 for k in ap.AUX)
    (prov,) = json.loads(str(d["aux_provenance"]))
    assert prov["written"] == list(ap.AUX) and prov["gaps"] == {} and prov["git_sha"] == "abc"
    assert prov["counties"] == ["01"]
    assert prov["grunddata_omdrev_pixels"]["volym"] == {"3": 0, "2": PX * PX, "1": 0, "0": 0}
    assert set(prov["rasters"]) == {"grunddata/omdrev2/volym_01.tif", "grunddata/omdrev2/grundyta_01.tif",
                                    "grunddata/omdrev2/medeldiameter_01.tif", "markfukt/markfukt_01.tif",
                                    "tradhojd/tradhojd_10m_01.tif"}
    th = prov["rasters"]["tradhojd/tradhojd_10m_01.tif"]
    assert th["source"] == "sksTradhojd01.zip" and th["statistic"] == "p90" and th["published"].startswith("4/1/2026")
    assert prov["mirror_manifest_sha256"] == hashlib.sha256((env / "mirror/MANIFEST.json").read_bytes()).hexdigest()
    np.testing.assert_array_equal(d["spectral"], np.ones((6, PX, PX), dtype=np.float32))     # nothing else touched
    report = json.loads((env / "report.json").read_text())
    assert report["counts"] == {"completed": 1} and report["tiles"]["nfi2018_a"]["counties"] == ["01"]
    assert str(W) not in capsys.readouterr().out                                              # counts only in the log


def test_an_unstaged_height_county_is_a_recorded_gap_not_a_zero_array(env):
    _mirror(env / "mirror", height_staged=False)
    t = _tile(env / "tiles/nfi2018_a.npz")
    assert _run(env) == 0
    d = _load(t)
    assert "height" not in d and int(d["aux_complete"]) == 0
    assert list(d["aux_missing"]) == ["height"]
    reason = str(d["aux_gap_reason"])
    assert reason.startswith("skg_open_mirror_gap__verified_") and "height: county ['01'] not staged" in reason
    assert str(W) not in reason
    assert all(k in d for k in ("volume", "basal_area", "diameter", "markfukt"))       # the other four still land


def test_a_window_without_valid_markfukt_is_a_gap(env):
    _mirror(env / "mirror", markfukt_value=65535)                    # nodata over the whole window
    t = _tile(env / "tiles/nfi2018_a.npz", markfukt=np.full((PX, PX), np.nan, dtype=np.float32))
    _run(env)
    d = _load(t)
    assert "markfukt" not in d and list(d["aux_missing"]) == ["markfukt"]
    assert "markfukt: no valid pixel in the mirror window" in str(d["aux_gap_reason"])


def test_a_channel_with_data_is_kept_and_an_empty_one_is_reread(env):
    _mirror(env / "mirror")
    kept = np.full((PX, PX), 123.0, dtype=np.float32)
    t = _tile(env / "tiles/nfi2018_a.npz", volume=kept, diameter=np.zeros((PX, PX), dtype=np.float32))
    _run(env)
    d = _load(t)
    np.testing.assert_array_equal(d["volume"], kept)
    assert float(d["diameter"][0, 0]) == 22
    (prov,) = json.loads(str(d["aux_provenance"]))
    assert "volume" not in prov["written"] and "diameter" in prov["written"]
    assert prov["kept_equal_to_mirror"] == {"volume": False}            # another version than the mirror's


def test_a_kept_channel_equal_to_the_mirror_is_recorded_as_such(env, capsys):
    _mirror(env / "mirror")
    same = np.full((PX, PX), 300.0, dtype=np.float32)                   # what the mirror holds for volume
    t = _tile(env / "tiles/nfi2018_a.npz", volume=same)
    _run(env)
    (prov,) = json.loads(str(_load(t)["aux_provenance"]))
    assert prov["kept_equal_to_mirror"] == {"volume": True}
    assert "kept, equal to the mirror volume: True: 1 tiles" in capsys.readouterr().out


def test_a_second_run_leaves_a_complete_tile_byte_identical(env):
    _mirror(env / "mirror")
    t = _tile(env / "tiles/nfi2018_a.npz")
    _run(env)
    before = t.read_bytes()
    assert _run(env) == 0
    assert t.read_bytes() == before
    assert json.loads((env / "report.json").read_text())["counts"] == {"complete": 1}


def test_a_staged_county_later_completes_a_marked_tile(env):
    _mirror(env / "mirror", height_staged=False)
    t = _tile(env / "tiles/nfi2018_a.npz")
    _run(env)
    _mirror(env / "mirror")
    _run(env)
    d = _load(t)
    assert int(d["aux_complete"]) == 1 and "aux_gap_reason" not in d
    first, second = json.loads(str(d["aux_provenance"]))
    assert first["gaps"] == {"height": "county ['01'] not staged in the mirror"} and second["written"] == ["height"]


def test_an_error_leaves_the_tile_untouched_and_fails_the_run(env, capsys):
    _mirror(env / "mirror")
    bad = env / "tiles/no_location.npz"
    bad.parent.mkdir(parents=True)
    np.savez_compressed(bad, spectral=np.ones((6, PX, PX), dtype=np.float32), aux_complete=np.array(0))
    before = bad.read_bytes()
    assert _run(env) == 1
    assert bad.read_bytes() == before
    assert "ERROR RuntimeError: 1 tiles (untouched)" in capsys.readouterr().out


def test_dry_run_writes_neither_tiles_nor_report(env):
    _mirror(env / "mirror")
    t = _tile(env / "tiles/nfi2018_a.npz")
    before = t.read_bytes()
    assert _run(env, "--dry-run") == 0
    assert t.read_bytes() == before and not (env / "report.json").exists()


def test_an_empty_root_refuses(env):
    with pytest.raises(SystemExit, match="REFUSING"):
        _run(env)


def test_a_recorded_upstream_gap_is_named_as_such(env):
    _mirror(env / "mirror")
    m = json.loads((env / "mirror/MANIFEST.json").read_text())
    m["derived"]["tradhojd/tradhojd_10m_01.tif"]["gaps"] = [[W + 100, S + 100, W + 200, S + 200]]
    (env / "mirror/MANIFEST.json").write_text(json.dumps(m))
    t = _tile(env / "tiles/nfi2018_a.npz")
    assert _run(env) == 0
    d = _load(t)
    assert "height" not in d and list(d["aux_missing"]) == ["height"]
    assert "height: county ['01'] has a recorded upstream gap in the window" in str(d["aux_gap_reason"])
