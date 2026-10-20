"""imint.geoid: EGM96 undulation from the shipped grid, checked against PROJ."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from imint import geoid


def _proj_vgridshift(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """PROJ's own bilinear read of the same grid file (independent implementation)."""
    from pyproj import Transformer

    t = Transformer.from_pipeline(
        "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad "
        f"+step +proj=vgridshift +grids={geoid.EGM96_GRID} +multiplier=1 "
        "+step +proj=unitconvert +xy_in=rad +xy_out=deg")
    _, _, n = t.transform(lon, lat, np.zeros_like(lon))
    return np.asarray(n)


def test_undulation_matches_proj_vgridshift():
    rng = np.random.default_rng(0)
    lon = np.r_[rng.uniform(-180, 180, 500), 10.5, 11.7, 179.9, -179.99, 0.0]
    lat = np.r_[rng.uniform(-89.9, 89.9, 500), 57.55, 58.05, 10.0, -10.0, 89.99]
    np.testing.assert_allclose(geoid.egm96_undulation(lon, lat), _proj_vgridshift(lon, lat),
                               rtol=0, atol=1e-6)


def test_undulation_over_skagen_is_the_sea_surface_height():
    # The diagnosis: the sea surface off Bohuslan lies 36-38 m above WGS84.
    n = geoid.egm96_undulation([10.5, 11.7, 11.0], [57.55, 58.05, 57.8])
    assert ((n > 35.0) & (n < 39.0)).all()
    assert np.ndim(geoid.egm96_undulation(11.0, 57.8)) == 0


def test_grid_is_the_pinned_file():
    assert geoid.egm96_grid_sha256() == hashlib.sha256(geoid.EGM96_GRID.read_bytes()).hexdigest()
    assert geoid.egm96_grid_sha256() == (
        "db493027562c9b004d7220fa881f5603adada4e1c5029b933fa7de4547b0e78d")


@pytest.mark.parametrize("lon, lat", [(11.0, 90.5), (11.0, -91.0), (np.nan, 57.0),
                                      (11.0, np.inf)])
def test_invalid_coordinates_fail_loud(lon, lat):
    with pytest.raises(ValueError):
        geoid.egm96_undulation(lon, lat)


def test_missing_grid_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(geoid, "EGM96_GRID", tmp_path / "absent.tif")
    geoid._egm96.cache_clear()
    geoid.egm96_grid_sha256.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="EGM96 geoid grid missing"):
            geoid.egm96_undulation(11.0, 57.8)
        with pytest.raises(FileNotFoundError):
            geoid.egm96_grid_sha256()
    finally:
        monkeypatch.undo()
        geoid._egm96.cache_clear()
        geoid.egm96_grid_sha256.cache_clear()
