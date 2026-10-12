"""Offline tests of mpc_s1.fetch_s1_scene (the sigma0 path) after the STAC-helper refactor.

The MPC catalogue and signer are faked; the measurement and calibration files
are a synthetic GCP-referenced GRD and a constant-LUT calibration XML on disk,
read through the real window read, LUT interpolation and sigma0 computation.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from imint.training import mpc_s1, s1_shared
from tests.s1_synthetic import synthetic_grd

_CAL_XML = """<calibration><calibrationVectorList count="2">
<calibrationVector><line>0</line><pixel count="3">0 1200 2400</pixel>
<sigmaNought count="3">500.0 500.0 500.0</sigmaNought></calibrationVector>
<calibrationVector><line>1800</line><pixel count="3">0 1200 2400</pixel>
<sigmaNought count="3">500.0 500.0 500.0</sigmaNought></calibrationVector>
</calibrationVectorList></calibration>"""


def _sigma0_setup(tmp_path, monkeypatch, *, zero_data: bool = False) -> dict:
    """Local GRD + calibration XML behind a faked MPC catalogue and signer."""
    import rasterio
    from pyproj import Transformer

    safe = tmp_path / "S1A_TEST.SAFE"
    vv = safe / "measurement" / "iw-vv.tiff"
    header = synthetic_grd(vv)
    if zero_data:
        with rasterio.open(vv, "r+") as dst:
            dst.write(np.zeros((header["height"], header["width"]), dtype=np.uint16), 1)
    (safe / "measurement" / "iw-vh.tiff").write_bytes(vv.read_bytes())
    cal_dir = safe / "annotation" / "calibration"
    cal_dir.mkdir(parents=True)
    for pol in ("vv", "vh"):
        (cal_dir / f"calibration-iw-{pol}.xml").write_text(_CAL_XML)

    e, n = Transformer.from_crs(4326, 3006, always_xy=True).transform(11.93, 57.875)
    req = (e - 320.0, n - 320.0, e + 320.0, n + 320.0)
    item = SimpleNamespace(id=f"S1A_IW_GRDH_1SDV_TEST_{tmp_path.name}",
                           properties={"sar:instrument_mode": "IW", "sar:product_type": "GRD",
                                       "sat:orbit_state": "descending"},
                           bbox=[11.6, 57.7, 12.2, 58.0])

    def asset(path: Path) -> SimpleNamespace:
        return SimpleNamespace(href=f"{path}?sig=token")

    signed = SimpleNamespace(id=item.id, assets={
        "vv": asset(vv), "vh": asset(safe / "measurement" / "iw-vh.tiff"),
        "schema-calibration-vv": asset(cal_dir / "calibration-iw-vv.xml"),
        "schema-calibration-vh": asset(cal_dir / "calibration-iw-vh.xml")})
    calls: list[dict] = []
    reads: list[str] = []
    real_read_window = s1_shared.read_window

    def read_window(uri, *args, **kwargs):
        reads.append(uri)
        return real_read_window(uri.removeprefix("/vsicurl/").split("?")[0], *args, **kwargs)

    def search(datetime_range, **kwargs):
        calls.append({"interval": datetime_range, **kwargs})
        return [item]

    monkeypatch.setattr(mpc_s1, "search_grd_items", search)
    monkeypatch.setattr(mpc_s1, "sign_item", lambda it: signed if it is item else None)
    monkeypatch.setattr(mpc_s1, "_http_get", lambda url: Path(url.split("?")[0]).read_bytes())
    monkeypatch.setattr(s1_shared, "read_window", read_window)
    monkeypatch.setattr(s1_shared, "_lut_cache", {})
    return {"req": req, "vv": vv, "calls": calls, "reads": reads, "read_window": real_read_window}


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_mpc_fetch_s1_scene_sigma0_path_offline(tmp_path, monkeypatch):
    from rasterio.warp import transform_bounds

    env = _sigma0_setup(tmp_path, monkeypatch)
    req = env["req"]
    sar, orbit = mpc_s1.fetch_s1_scene(*req, "2023-06-15", size_px=64, nodata_threshold=None)
    assert orbit == "DESCENDING"
    (call,) = env["calls"]
    assert call == {"interval": "2023-06-15T00:00:00Z/2023-06-16T00:00:00Z",
                    "bbox_4326": transform_bounds("EPSG:3006", "EPSG:4326", *req, densify_pts=21)}
    assert env["reads"] == [f"/vsicurl/{env['vv']}?sig=token",
                            f"/vsicurl/{env['vv'].with_name('iw-vh.tiff')}?sig=token"]
    dn, _ = env["read_window"](env["vv"], *req, 3006, 64, 64)
    expected = np.where(dn == 0, 0.0, dn.astype(np.float32) ** 2 / np.float32(500.0) ** 2)
    assert sar.shape == (2, 64, 64) and sar.dtype == np.float32
    assert (dn > 0).all()
    np.testing.assert_array_equal(sar[0], expected.astype(np.float32))
    np.testing.assert_array_equal(sar[1], sar[0])
    sar_db, _ = mpc_s1.fetch_s1_scene(*req, "2023-06-15", size_px=64, nodata_threshold=None,
                                      output_db=True)
    np.testing.assert_array_equal(sar_db, s1_shared.to_db(sar))


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_mpc_fetch_s1_scene_failure_modes(tmp_path, monkeypatch):
    env = _sigma0_setup(tmp_path, monkeypatch, zero_data=True)
    req = env["req"]
    # All-nodata window: rejected by the default 10 % nodata threshold.
    assert mpc_s1.fetch_s1_scene(*req, "2023-06-15", size_px=64) is None

    def raise_(exc):
        def f(*args, **kwargs):
            raise exc
        return f

    monkeypatch.setattr(mpc_s1, "search_grd_items", raise_(RuntimeError("STAC down")))
    assert mpc_s1.fetch_s1_scene(*req, "2023-06-15", size_px=64) is None
    monkeypatch.setattr(mpc_s1, "search_grd_items", raise_(ImportError("no pystac_client")))
    with pytest.raises(ImportError, match="pystac_client"):
        mpc_s1.fetch_s1_scene(*req, "2023-06-15", size_px=64)
    monkeypatch.setattr(mpc_s1, "search_grd_items", lambda *a, **k: [])
    assert mpc_s1.fetch_s1_scene(*req, "2023-06-15", size_px=64) is None
