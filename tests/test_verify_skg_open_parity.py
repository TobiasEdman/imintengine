"""The parity report compares cache arrays with the mirror-backed fetchers."""
from __future__ import annotations

import json

import numpy as np

from imint.training import skg_open
from scripts import verify_skg_open_parity as vp
from tests.skg_mirror_fixture import write_raster

W, S, E, N, PX = 500_000, 6_250_000, 500_640, 6_250_640, 64


def test_compare_reports_exactness_and_bias():
    ref = np.array([[0, 5], [10, 20]], dtype=np.float32)
    assert vp.compare(ref, ref) == {"exact": 1.0, "exact_where_both_valid": 1.0, "mae": 0.0,
                                    "bias": 0.0, "ref_nonzero": 0.75, "cand_nonzero": 0.75}
    out = vp.compare(ref, np.array([[0, 5], [10, 24]], dtype=np.float32))
    assert out["exact"] == 0.75 and out["bias"] == -1.0 and out["mae"] == 1.0


def test_report_runs_the_real_fetchers_against_a_cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv(skg_open.ENV_MIRROR_DIR, str(tmp_path / "mirror"))
    vol = np.full((PX, PX), 42, dtype=np.int16)
    write_raster(tmp_path / "mirror" / "grunddata" / "volym_omdrev2.vrt", vol, W, N, 10.0, nodata=-1)
    root = tmp_path / "root"
    (root / "cache" / "volume").mkdir(parents=True)
    ref = vol.astype(np.float32)
    ref[0, 0] = 40                                                    # one pixel differs
    np.save(root / "cache" / "volume" / f"volume_{W}_{S}_{E}_{N}.npy", ref)
    out = tmp_path / "report.json"
    assert vp.main(["--roots", str(root), "--channels", "volume", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    s = rep["per_root"][str(root)]["volume"]["summary"]
    assert s["n"] == 1 and s["errors"] == 0 and s["exact"] == round(1 - 1 / PX**2, 4)
    assert rep["per_root"][str(root)]["volume"]["tiles"][0]["bbox"] == [W, S, E, N]
