"""Tests for the WaterQualityAnalyzer package.

Covers the deterministic units (classical indices, water masking) with
hand-computed expected values, the orchestrator with synthetic input,
and the soft-import / skip-and-warn behaviour of the AI backends.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from imint.analyzers.water_quality import WaterQualityAnalyzer
from imint.analyzers.water_quality.c2rcc_wrapper import C2RCCUnavailable
from imint.analyzers.water_quality.classical_indices import compute_mci, compute_ndci
from imint.analyzers.water_quality.mdn_inference import MDNUnavailable, MDN_BANDS
from imint.analyzers.water_quality.water_mask import (
    build_water_mask,
    compute_mndwi,
    water_mask_from_scl,
)


# ---------------------------------------------------------------------- #
#  classical_indices
# ---------------------------------------------------------------------- #


def test_ndci_known_values():
    red = np.array([[0.10, 0.05]], dtype=np.float32)
    re1 = np.array([[0.15, 0.10]], dtype=np.float32)
    out = compute_ndci(red, re1)
    expected = np.array(
        [[(0.15 - 0.10) / (0.15 + 0.10), (0.10 - 0.05) / (0.10 + 0.05)]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected, atol=1e-5)
    assert out.dtype == np.float32


def test_ndci_zero_division_safe():
    red = np.zeros((1, 1), dtype=np.float32)
    re1 = np.zeros((1, 1), dtype=np.float32)
    out = compute_ndci(red, re1)
    assert np.isfinite(out).all(), "_EPS must keep result finite when both bands zero"


def test_mci_flat_spectrum_is_zero():
    flat = np.full((3, 3), 0.05, dtype=np.float32)
    out = compute_mci(flat, flat, flat)
    np.testing.assert_allclose(out, 0.0, atol=1e-7)


def test_mci_red_edge_peak_is_positive():
    red = np.array([[0.05]], dtype=np.float32)
    re1 = np.array([[0.12]], dtype=np.float32)
    re2 = np.array([[0.07]], dtype=np.float32)
    out = compute_mci(red, re1, re2)
    expected = 0.12 - (0.05 + 0.389 * (0.07 - 0.05))
    np.testing.assert_allclose(out, expected, atol=1e-5)
    assert out.item() > 0


# ---------------------------------------------------------------------- #
#  water_mask
# ---------------------------------------------------------------------- #


def test_water_mask_from_scl_excludes_clouds_and_shadows():
    scl = np.array([[6, 6, 8, 3], [6, 9, 6, 10]], dtype=np.uint8)
    expected = np.array([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=bool)
    np.testing.assert_array_equal(water_mask_from_scl(scl), expected)


def test_mndwi_threshold_water_is_positive():
    green = np.array([[0.10]], dtype=np.float32)
    swir = np.array([[0.05]], dtype=np.float32)
    assert compute_mndwi(green, swir).item() > 0


def test_build_water_mask_prefers_scl():
    bands = {
        "B03": np.full((2, 2), 0.10, dtype=np.float32),
        "B11": np.full((2, 2), 0.05, dtype=np.float32),
    }
    scl = np.full((2, 2), 6, dtype=np.uint8)
    mask, method = build_water_mask(bands, scl=scl)
    assert method == "scl"
    assert mask.all()


def test_build_water_mask_falls_back_to_mndwi():
    bands = {
        "B03": np.array([[0.10, 0.05]], dtype=np.float32),
        "B11": np.array([[0.05, 0.10]], dtype=np.float32),
    }
    mask, method = build_water_mask(bands, scl=None)
    assert method == "mndwi"
    np.testing.assert_array_equal(mask, np.array([[True, False]]))


def test_build_water_mask_raises_when_nothing_usable():
    with pytest.raises(ValueError):
        build_water_mask({}, scl=None)


# ---------------------------------------------------------------------- #
#  mdn_inference & c2rcc_wrapper — soft-import / failure paths
# ---------------------------------------------------------------------- #


def test_mdn_unavailable_when_repo_missing(monkeypatch):
    """When the upstream MDN package can't be imported, raise MDNUnavailable.

    The wrapper has a fallback that probes ``~/code`` for a developer-cloned
    MDN — this test masks the upstream package via ``sys.modules`` so the
    import fails regardless of what's on disk, and verifies the wrapper
    surfaces that as :class:`MDNUnavailable` rather than letting an
    ``ImportError`` propagate.
    """
    from imint.analyzers.water_quality.mdn_inference import _import_mdn

    # Block any future `from MDN import image_estimates` by poisoning the cache
    monkeypatch.setitem(sys.modules, "MDN", None)
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(MDNUnavailable):
            _import_mdn(mdn_repo_path=td)


def test_mdn_unavailable_when_band_missing():
    """run_mdn must raise MDNUnavailable if any of the 7 MSI bands is absent."""
    from imint.analyzers.water_quality.mdn_inference import run_mdn

    # Drop one required band (B07)
    bands = {b: np.zeros((4, 4), dtype=np.float32) for b in MDN_BANDS if b != "B07"}
    mask = np.ones((4, 4), dtype=bool)
    with pytest.raises(MDNUnavailable, match="B07"):
        run_mdn(bands, mask)


def test_mdn_unavailable_when_no_water_pixels():
    """An empty water mask must short-circuit before any TF import."""
    from imint.analyzers.water_quality.mdn_inference import run_mdn

    bands = {b: np.zeros((4, 4), dtype=np.float32) for b in MDN_BANDS}
    empty_mask = np.zeros((4, 4), dtype=bool)
    with pytest.raises(MDNUnavailable, match="no water pixels"):
        run_mdn(bands, empty_mask)


def test_c2rcc_unavailable_when_acolite_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "acolite", None)
    from imint.analyzers.water_quality.c2rcc_wrapper import _import_acolite

    with pytest.raises(C2RCCUnavailable):
        _import_acolite()


def test_c2rcc_unavailable_when_band_missing():
    from imint.analyzers.water_quality.c2rcc_wrapper import C2RCC_BANDS, _check_bands

    bands = {b: np.zeros((4, 4), dtype=np.float32) for b in C2RCC_BANDS[:-1]}
    with pytest.raises(C2RCCUnavailable):
        _check_bands(bands)


# ---------------------------------------------------------------------- #
#  WaterQualityAnalyzer — orchestrator end-to-end
# ---------------------------------------------------------------------- #


def _synthetic_tile(H: int = 8, W: int = 8) -> tuple[dict, np.ndarray, np.ndarray]:
    bands = {
        b: np.full((H, W), 0.05, dtype=np.float32)
        for b in ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
    }
    # Inject a chlorophyll-like signature in the middle: B05 high, B04/B06 low
    bands["B05"][2:6, 2:6] = 0.12
    bands["B06"][2:6, 2:6] = 0.07
    bands["B04"][2:6, 2:6] = 0.05
    scl = np.full((H, W), 6, dtype=np.uint8)
    rgb = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1)
    return bands, scl, rgb


def test_orchestrator_skips_unavailable_methods_gracefully():
    bands, scl, rgb = _synthetic_tile()
    az = WaterQualityAnalyzer(config={"methods": {
        "mdn": {"enabled": True, "weights_url": None},
        "c2rcc": {"enabled": True},
    }})

    with tempfile.TemporaryDirectory() as td:
        result = az.run(
            rgb=rgb, bands=bands, scl=scl, geo=None,
            date="2026-04-08", output_dir=td,
        )

    assert result.success
    assert result.error is None
    # Classical indices always succeed; AI backends skipped (no torch, no acolite, no weights)
    assert "ndci" in result.outputs
    assert "mci" in result.outputs
    assert result.metadata["methods_succeeded"]["ndci"] is True
    assert result.metadata["methods_succeeded"]["mdn"] is False
    assert result.metadata["methods_succeeded"]["c2rcc"] is False


def test_orchestrator_returns_failure_when_all_methods_fail():
    """No bands at all → all methods skipped → success=False."""
    az = WaterQualityAnalyzer(config={})
    H, W = 4, 4
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    scl = np.full((H, W), 6, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as td:
        result = az.run(
            rgb=rgb, bands={}, scl=scl, geo=None,
            date="2026-04-08", output_dir=td,
        )
    assert not result.success


def test_orchestrator_writes_summary_json():
    bands, scl, rgb = _synthetic_tile()
    az = WaterQualityAnalyzer(config={})

    with tempfile.TemporaryDirectory() as td:
        result = az.run(
            rgb=rgb, bands=bands, scl=scl, geo=None,
            date="2026-04-08", output_dir=td,
        )
        summary_path = Path(td) / "water_quality" / "2026-04-08" / "summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)

    assert summary["mask_method"] == "scl"
    assert summary["n_water_pixels"] == 64
    assert "ndci" in summary["stats"]
    assert "mci" in summary["stats"]
    assert summary["stats"]["ndci"]["max"] == pytest.approx(0.4117647, abs=1e-4)


def test_orchestrator_classical_index_values_match_manual():
    bands, scl, rgb = _synthetic_tile()
    az = WaterQualityAnalyzer(config={})

    with tempfile.TemporaryDirectory() as td:
        result = az.run(rgb=rgb, bands=bands, scl=scl, geo=None,
                        date="d", output_dir=td)

    # Inside the patch: NDCI = (0.12-0.05)/(0.12+0.05)
    expected_inside = (0.12 - 0.05) / (0.12 + 0.05)
    ndci = result.outputs["ndci"]
    np.testing.assert_allclose(
        ndci[3, 3], expected_inside, atol=1e-4,
        err_msg="NDCI inside the chlorophyll patch must match the band-ratio formula",
    )
    # Outside the patch: B05 = B04 = 0.05 → NDCI = 0
    np.testing.assert_allclose(ndci[0, 0], 0.0, atol=1e-6)


def test_orchestrator_uses_mndwi_when_scl_absent():
    bands, _, rgb = _synthetic_tile()
    az = WaterQualityAnalyzer(config={})

    with tempfile.TemporaryDirectory() as td:
        result = az.run(rgb=rgb, bands=bands, scl=None, geo=None,
                        date="d", output_dir=td)

    assert result.success
    assert result.metadata["mask_method"] == "mndwi"
