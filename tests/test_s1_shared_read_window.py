"""s1_shared.read_window after the GCP-referenced sigma0 path was retired (task 0113).

The old test here pinned sigma0 from ``mpc_s1.fetch_s1_scene`` through the real
window read over a GCP-referenced GRD. That path is retired: its tie points sit
at ellipsoid height (the sea is expected 41-56 m toward the satellite) and the
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

    for name in ("interp_lut", "load_cal_grid", "compute_sigma0", "pick_calibration_urls"):
        assert not hasattr(s1_shared, name), name
    for name in ("_ensure_product_cached", "_download", "clear_cache"):
        assert not hasattr(cdse_s1_stac, name), name
    assert not hasattr(mpc_s1, "fetch_s1_scene")
    assert not hasattr(cdse_s1_stac, "fetch_s1_scene")
    for gone in ("imint.training.aws_s1", "imint.training.s1_fetch"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(gone)
    # The vessel chain's STAC helpers stay.
    assert callable(mpc_s1.search_grd_items) and callable(mpc_s1.sign_item)


def test_cdse_calibrated_scene_refuses():
    from imint.training import cdse_s1_stac

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


def _pre_retirement_read_window(path, west, south, east, north, src_epsg, h_px, w_px):
    """The projected branch of ``s1_shared.read_window`` as it was on main
    f5fefde, before task 0113 (verbatim but for the removed GCP branch).

    Run on the same machine as the code under test, it is a portable
    baseline: pixel values recorded on one rasterio/GDAL build do not hold
    on another (CI's rasterio 1.5.2 resamples the nodata edge differently
    from a local 1.4.4), but old and new code on one build must agree
    exactly.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds

    with rasterio.open(str(path)) as ds:
        dst_bounds = transform_bounds(f"EPSG:{src_epsg}", ds.crs, west, south, east, north,
                                      densify_pts=21)
        window = window_from_bounds(*dst_bounds, transform=ds.transform)
        dn = ds.read(1, window=window, out_shape=(h_px, w_px), resampling=Resampling.bilinear,
                     boundless=True, fill_value=0).astype(np.float32)
    return dn, window


# Inside the raster, and across its -32768 nodata strip.
_RTC_BBOXES = [(302270, 6401337, 302910, 6401977), (302270, 6404157, 302910, 6404797)]


@pytest.mark.parametrize("bbox", _RTC_BBOXES)
def test_projected_rtc_like_read_is_unchanged(tmp_path, bbox):
    path = _rtc_like(tmp_path)
    dn, win = s1_shared.read_window(path, *bbox, 3006, 64, 64)
    ref, ref_win = _pre_retirement_read_window(path, *bbox, 3006, 64, 64)
    assert dn.dtype == np.float32 and dn.shape == (64, 64)
    assert _window(win) == _window(ref_win)
    np.testing.assert_array_equal(dn, ref)
