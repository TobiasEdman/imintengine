"""s1_shared.read_window after the GCP-referenced sigma0 path was retired (task 0113).

The old test here pinned sigma0 from ``mpc_s1.fetch_s1_scene`` through the real
window read over a GCP-referenced GRD. That path is retired: its tie points sit
at ellipsoid height (the sea lands 41-56 m toward the satellite) and the
calibration LUT was sampled at warped-grid offsets. These tests pin that a
GCP-referenced source is refused, and that a projected source (the RTC COGs
``imint.training.pc_s1_rtc`` reads) reads exactly as before the retirement:
the pinned numbers below were recorded with the pre-retirement implementation.
"""
from __future__ import annotations

import numpy as np
import pytest

from imint.data import mpc_s1, s1_shared
from tests.s1_synthetic import synthetic_grd


def _projected(path, crs, x0, y0, data, nodata):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0],
                       count=1, dtype=data.dtype.name, crs=crs,
                       transform=from_origin(x0, y0, 10, 10), nodata=nodata) as dst:
        dst.write(data, 1)
    return path


def _window(win) -> tuple[float, float, float, float]:
    return (float(win.col_off), float(win.row_off), float(win.width), float(win.height))


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_gcp_referenced_grd_is_refused(tmp_path):
    grd = tmp_path / "S1A_TEST.SAFE" / "measurement" / "iw-vv.tiff"
    synthetic_grd(grd)
    with pytest.raises(RuntimeError, match="task 0113") as exc:
        s1_shared.read_window(grd, 364000, 6421000, 364640, 6421640, 3006, 64, 64)
    assert s1_shared.GCP_SIGMA0_RETIRED in str(exc.value)
    assert "pc_s1_rtc" in str(exc.value)


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_unreferenced_source_still_raises(tmp_path):
    import rasterio

    path = tmp_path / "plain.tif"
    with rasterio.open(path, "w", driver="GTiff", width=8, height=8, count=1,
                       dtype="uint16") as dst:
        dst.write(np.ones((8, 8), dtype=np.uint16), 1)
    with pytest.raises(RuntimeError, match="no projected CRS and no GCP CRS"):
        s1_shared.read_window(path, 0, 0, 80, 80, 3006, 8, 8)


def test_retired_sigma0_fetchers_are_gone():
    """No module can still produce sigma0 from a GCP-referenced GRD."""
    import importlib

    from imint.training import cdse_s1_stac

    assert not hasattr(s1_shared, "interp_lut")
    assert not hasattr(mpc_s1, "fetch_s1_scene")
    assert not hasattr(cdse_s1_stac, "fetch_s1_scene")
    for gone in ("imint.training.aws_s1", "imint.training.s1_fetch"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(gone)
    # The vessel chain's STAC helpers stay.
    assert callable(mpc_s1.search_grd_items) and callable(mpc_s1.sign_item)


def test_cdse_calibrated_scene_refuses_before_download(monkeypatch):
    from imint.training import cdse_s1_stac

    def no_download(item):
        raise AssertionError("must refuse before downloading the product")

    monkeypatch.setattr(cdse_s1_stac, "_ensure_product_cached", no_download)
    with pytest.raises(RuntimeError, match="task 0113"):
        cdse_s1_stac._read_calibrated_scene(object(), 0, 0, 640, 640, 3006, 64, 64,
                                            output_db=True)


def test_projected_same_crs_read_is_an_exact_copy(tmp_path):
    data = (np.arange(400 * 400, dtype=np.uint32) % 60000 + 1).astype(np.uint16).reshape(400, 400)
    path = _projected(tmp_path / "sweref.tif", "EPSG:3006", 380000, 6173000, data, 0)
    dn, win = s1_shared.read_window(path, 380500, 6170000, 381140, 6170640, 3006, 64, 64)
    assert _window(win) == (50.0, 236.0, 64.0, 64.0)
    assert dn.dtype == np.float32 and dn.shape == (64, 64)
    np.testing.assert_array_equal(dn, data[236:300, 50:114].astype(np.float32))


def test_projected_read_off_the_raster_fills_zero(tmp_path):
    """Projected reads stay boundless: the part of the window off the raster is 0."""
    data = np.full((400, 400), 7, dtype=np.uint16)
    path = _projected(tmp_path / "sweref.tif", "EPSG:3006", 380000, 6173000, data, 0)
    dn, win = s1_shared.read_window(path, 383500, 6171000, 384780, 6172280, 3006, 128, 128)
    assert _window(win) == (350.0, 72.0, 128.0, 128.0)
    assert (dn[:, :50] == 7).all() and (dn[:, 50:] == 0).all()


def _rtc_like(tmp_path):
    """Float32 UTM 32 raster with an RTC-style -32768 nodata strip along the top."""
    r, c = np.mgrid[0:600, 0:600]
    a = (0.1 + 1e-3 * r + 5e-4 * c + 0.05 * np.sin(c / 7.0) * np.cos(r / 11.0)).astype(np.float32)
    a[:20, :] = -32768
    return _projected(tmp_path / "rtc.tif", "EPSG:32632", 657000, 6403000, a, -32768)


# (bbox in EPSG:3006, window, sum, {pixel: value}) recorded with the pre-retirement read_window.
_RTC_PINS = [
    ((302270, 6401337, 302910, 6401977),
     (265.27621927138534, 265.34812356729526, 69.39783480241022, 69.39783479226753),
     2249.9437180757523,
     {(0, 0): 0.50357586145401, (10, 20): 0.5118050575256348,
      (32, 32): 0.5748937726020813, (63, 63): 0.588545560836792}),
    ((302270, 6404157, 302910, 6404797),
     (240.3368806792132, -15.501118818297982, 69.39919943884888, 69.39919942850247),
     -37748163.615686074,
     {(0, 0): 0.0, (10, 20): 0.0, (32, 32): 0.24534662067890167,
      (63, 63): 0.30763116478919983}),
]


@pytest.mark.parametrize("bbox, window, total, pixels", _RTC_PINS)
def test_projected_rtc_like_read_is_unchanged(tmp_path, bbox, window, total, pixels):
    dn, win = s1_shared.read_window(_rtc_like(tmp_path), *bbox, 3006, 64, 64)
    np.testing.assert_allclose(_window(win), window, rtol=0, atol=1e-6)
    assert dn.dtype == np.float32 and dn.shape == (64, 64)
    np.testing.assert_allclose(float(dn.astype(np.float64).sum()), total, rtol=1e-9)
    for ij, value in pixels.items():
        assert float(dn[ij]) == pytest.approx(value, abs=1e-6)
