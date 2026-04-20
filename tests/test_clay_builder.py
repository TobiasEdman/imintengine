"""Tests for imint.fm.loaders.clay.build_s2_clay_tensor.

These don't require the Clay pretrained weights — they verify the
band-order mapping from our on-disk tensors (spectral + b08 + rededge)
into Clay's expected 10-band stack.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from imint.fm.loaders.clay import (
    CLAY_S2_BAND_META,
    CLAY_S2_BAND_ORDER,
    build_s2_clay_tensor,
    get_clay_norm,
    get_clay_wavelengths,
)


def _make_inputs(h: int = 8, w: int = 8):
    """Synthesize predictable inputs: each band is filled with its
    unique 'signature' value so the builder's band ordering is verifiable."""
    spectral = np.stack(
        [np.full((h, w), 100.0, dtype=np.float32),  # B02 blue
         np.full((h, w), 200.0, dtype=np.float32),  # B03 green
         np.full((h, w), 300.0, dtype=np.float32),  # B04 red
         np.full((h, w), 800.0, dtype=np.float32),  # B8A nir08
         np.full((h, w), 1100.0, dtype=np.float32), # B11 swir16
         np.full((h, w), 1200.0, dtype=np.float32), # B12 swir22
         ], axis=0,
    )
    b08 = np.full((h, w), 700.0, dtype=np.float32)  # B08 nir
    rededge = np.stack([
        np.full((h, w), 500.0, dtype=np.float32),   # B05 rededge1
        np.full((h, w), 600.0, dtype=np.float32),   # B06 rededge2
        np.full((h, w), 650.0, dtype=np.float32),   # B07 rededge3
    ], axis=0)
    return spectral, b08, rededge


class TestBuildS2ClayTensor:
    def test_default_ordering_is_clay_spec(self):
        spectral, b08, rededge = _make_inputs()
        t = build_s2_clay_tensor(spectral, b08, rededge)
        # Expected in CLAY_S2_BAND_ORDER:
        # blue(100) green(200) red(300) rededge1(500) rededge2(600)
        # rededge3(650) nir(700) nir08(800) swir16(1100) swir22(1200)
        expected_first_pixel = [100, 200, 300, 500, 600, 650, 700, 800, 1100, 1200]
        actual_first_pixel = [float(t[i, 0, 0]) for i in range(10)]
        assert actual_first_pixel == expected_first_pixel

    def test_output_shape(self):
        spectral, b08, rededge = _make_inputs(h=16, w=16)
        t = build_s2_clay_tensor(spectral, b08, rededge)
        assert t.shape == (10, 16, 16)

    def test_numpy_input_returns_numpy(self):
        spectral, b08, rededge = _make_inputs()
        t = build_s2_clay_tensor(spectral, b08, rededge)
        assert isinstance(t, np.ndarray)

    def test_torch_input_returns_torch(self):
        spectral_np, b08_np, rededge_np = _make_inputs()
        t = build_s2_clay_tensor(
            torch.from_numpy(spectral_np),
            torch.from_numpy(b08_np),
            torch.from_numpy(rededge_np),
        )
        assert isinstance(t, torch.Tensor)
        assert t.shape == (10, 8, 8)

    def test_missing_rededge_raises_clear_error(self):
        spectral, b08, _ = _make_inputs()
        with pytest.raises(KeyError, match="rededge"):
            build_s2_clay_tensor(spectral, b08, rededge=None)

    def test_subset_bands_no_rededge(self):
        """If the caller restricts to non-rededge bands, missing rededge
        tensor is fine."""
        spectral, b08, _ = _make_inputs()
        subset = ("blue", "green", "red", "nir", "nir08", "swir16", "swir22")
        t = build_s2_clay_tensor(spectral, b08, rededge=None, bands=subset)
        assert t.shape == (7, 8, 8)

    def test_unknown_band_raises(self):
        spectral, b08, rededge = _make_inputs()
        with pytest.raises(KeyError, match="Unknown"):
            build_s2_clay_tensor(
                spectral, b08, rededge,
                bands=("blue", "not_a_band"),
            )


class TestClayWavelengthsAndNorm:
    def test_wavelengths_shape_matches_default(self):
        wl = get_clay_wavelengths()
        assert wl.shape == (10,)
        # Sanity: values in µm, 0.4-2.5 range
        assert wl.min() > 0.3
        assert wl.max() < 3.0

    def test_wavelengths_in_expected_order(self):
        wl = get_clay_wavelengths()
        # Should equal the metadata values in CLAY_S2_BAND_ORDER
        expected = [CLAY_S2_BAND_META[b]["wavelength"] for b in CLAY_S2_BAND_ORDER]
        assert torch.allclose(wl, torch.tensor(expected, dtype=torch.float32))

    def test_norm_shapes(self):
        mean, std = get_clay_norm()
        assert mean.shape == (10,)
        assert std.shape == (10,)
        assert (std > 0).all()

    def test_norm_subset(self):
        subset = ("blue", "green", "red")
        mean, std = get_clay_norm(bands=subset)
        assert mean.shape == (3,)
        # First entry is blue's mean 1105
        assert float(mean[0]) == 1105.0
