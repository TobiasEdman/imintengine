"""ΔVV/ΔVH SAR-change aux channels (2016 clearcut anchor via aux path).

The 2016 SAR clearcut anchor enters training through the AUX path (the
pretrained SAR encoders are locked to 2-ch input, so the anchor can't be a
third SAR frame). Two aux channels, `delta_vv` / `delta_vh`, are computed at
read time as the dB difference between the growing-season γ⁰ composite
(`s1_vv_vh`) and the 2016 γ⁰ composite (`s1_vv_vh_2016`):

    ΔX = 10·log10(season_X) − 10·log10(2016_X)

A fresh clearcut collapses volume scattering → a sharp NEGATIVE Δ. This suite
covers the pure compute (`compute_delta_sar`) and the dataset wiring
(`_load_aux_channels` via `config.enabled_aux_names` opt-in):

  * correct sign/magnitude for a simulated harvest (season << 2016),
  * zeros when the 2016 composite is absent (~10% of tiles),
  * NaN/zero-nodata safety (no −inf, no NaN leaking to the model),
  * the opt-in wiring appends delta_vv/delta_vh LAST and normalizes them.
"""
from __future__ import annotations

import numpy as np
import pytest

from imint.training.config import TrainingConfig
from imint.training.unified_dataset import (
    AUX_NORM,
    UnifiedDataset,
    compute_delta_sar,
    normalize_aux_channel,
)

H = W = 32
N_CLASSES = 23


# ── db helper: linear γ⁰ that renders to a target dB ──────────────────────

def _lin_from_db(db: float) -> float:
    """Linear γ⁰ whose 10·log10 equals ``db``."""
    return float(10.0 ** (db / 10.0))


# ── compute_delta_sar: pure math ──────────────────────────────────────────

def test_delta_sign_and_magnitude_for_harvest():
    """Season << 2016 (backscatter drop) → sharp negative Δ of the right size."""
    # 2016 forest: VV ≈ −8 dB, VH ≈ −14 dB (typical standing spruce γ⁰).
    # Season (post-harvest): VV ≈ −14 dB, VH ≈ −22 dB → Δ = −6 / −8 dB.
    baseline = np.stack([
        np.full((H, W), _lin_from_db(-8.0), dtype=np.float32),
        np.full((H, W), _lin_from_db(-14.0), dtype=np.float32),
    ])
    season = np.stack([
        np.full((H, W), _lin_from_db(-14.0), dtype=np.float32),
        np.full((H, W), _lin_from_db(-22.0), dtype=np.float32),
    ])
    data = {"s1_vv_vh": season, "s1_vv_vh_2016": baseline}
    delta = compute_delta_sar(data)
    assert delta is not None and delta.shape == (2, H, W)
    assert np.allclose(delta[0], -6.0, atol=1e-3)   # ΔVV
    assert np.allclose(delta[1], -8.0, atol=1e-3)   # ΔVH
    assert (delta < 0).all(), "harvest must be a negative Δ"


def test_delta_zero_for_no_change():
    """Identical season and 2016 composites → Δ == 0 (neutral)."""
    comp = np.stack([
        np.full((H, W), _lin_from_db(-9.0), dtype=np.float32),
        np.full((H, W), _lin_from_db(-15.0), dtype=np.float32),
    ])
    delta = compute_delta_sar({"s1_vv_vh": comp, "s1_vv_vh_2016": comp.copy()})
    assert np.allclose(delta, 0.0, atol=1e-4)


def test_delta_none_when_2016_absent():
    """Missing 2016 composite → None (caller emits zeros = 'no change')."""
    season = np.random.rand(2, H, W).astype(np.float32) * 0.2
    assert compute_delta_sar({"s1_vv_vh": season}) is None
    assert compute_delta_sar({"s1_vv_vh_2016": season}) is None
    assert compute_delta_sar({}) is None


def test_delta_nan_and_zero_nodata_safe():
    """Zero/NaN nodata pixels never produce −inf or NaN."""
    season = np.full((2, H, W), _lin_from_db(-12.0), dtype=np.float32)
    baseline = np.full((2, H, W), _lin_from_db(-10.0), dtype=np.float32)
    # Inject nodata: zeros (linear γ⁰ nodata) + NaN in both composites.
    season[:, :4, :] = 0.0
    baseline[:, 4:8, :] = np.nan
    season[0, 8, 8] = np.nan
    delta = compute_delta_sar({"s1_vv_vh": season, "s1_vv_vh_2016": baseline})
    assert delta is not None
    assert np.isfinite(delta).all(), "no −inf / NaN may reach the model"


def test_delta_wrong_shape_returns_none():
    """A mis-shaped composite (not (2,H,W)) → None rather than a crash."""
    bad = np.random.rand(4, H, W).astype(np.float32)   # per-frame v1 layout
    good = np.random.rand(2, H, W).astype(np.float32)
    assert compute_delta_sar({"s1_vv_vh": bad, "s1_vv_vh_2016": good}) is None


# ── normalization constants ───────────────────────────────────────────────

def test_delta_norm_constants_centred_on_zero():
    """A zero Δ normalizes to exactly 0; a −8 dB harvest to z ≈ −2."""
    assert AUX_NORM["delta_vv"] == (0.0, 4.0)
    assert AUX_NORM["delta_vh"] == (0.0, 4.0)
    assert normalize_aux_channel("delta_vv", 0.0) == pytest.approx(0.0)
    assert normalize_aux_channel("delta_vh", -8.0) == pytest.approx(-2.0)
    # Ordinary phenological jitter (±2 dB) stays inside ±0.5 z.
    assert abs(normalize_aux_channel("delta_vv", 2.0)) <= 0.5


# ── config opt-in ─────────────────────────────────────────────────────────

def test_config_appends_delta_last():
    """enable_delta_sar_channels adds delta_vv/delta_vh AFTER all prior aux."""
    cfg = TrainingConfig(
        enable_height_channel=True, enable_volume_channel=True,
        enable_basal_area_channel=True, enable_diameter_channel=True,
        enable_dem_channel=True, enable_vpp_channels=True,
        enable_markfukt_channel=True, enable_delta_sar_channels=True,
    )
    names = cfg.enabled_aux_names
    assert names[-2:] == ("delta_vv", "delta_vh")
    assert names.index("markfukt") < names.index("delta_vv")

    # Off by default → no delta channels, prior ordering untouched.
    cfg_off = TrainingConfig(
        enable_height_channel=True, enable_vpp_channels=True,
    )
    assert "delta_vv" not in cfg_off.enabled_aux_names


# ── dataset wiring (real _load_aux_channels path) ─────────────────────────

def _write_tile(path, *, with_2016: bool, harvest: bool = False):
    """Minimal tile the dataset can read, with/without the 2016 composite."""
    season = np.full((2, H, W), _lin_from_db(-14.0 if harvest else -9.0),
                     dtype=np.float32)
    data = dict(
        spectral=(np.random.rand(24, H, W) * 0.4).astype(np.float32),
        label=np.random.randint(0, N_CLASSES, (H, W)).astype(np.int64),
        doy=np.array([260, 130, 190, 220], dtype=np.float32),
        year=np.int32(2022),
        easting=np.float32(500000.0), northing=np.float32(6500000.0),
        s1_vv_vh=season, s1_enrich_v=np.int32(3),
        s1_orbit=np.bytes_("DESCENDING"), s1_source=np.bytes_("pc-rtc-gamma0"),
    )
    if with_2016:
        data["s1_vv_vh_2016"] = np.full(
            (2, H, W), _lin_from_db(-8.0), dtype=np.float32,
        )
        data["has_frame_2016"] = np.int32(1)
    np.savez_compressed(str(path), **data)


@pytest.fixture
def tile_dir(tmp_path):
    d = tmp_path / "tiles"
    d.mkdir()
    names = []
    _write_tile(d / "harvest_with2016.npz", with_2016=True, harvest=True)
    _write_tile(d / "stable_with2016.npz", with_2016=True, harvest=False)
    _write_tile(d / "no2016.npz", with_2016=False)
    for n in ("harvest_with2016.npz", "stable_with2016.npz", "no2016.npz"):
        names.append(n)
    (d / "split_train.txt").write_text("\n".join(names) + "\n")
    (d / "split_val.txt").write_text("\n".join(names) + "\n")
    return d


def _ds(tile_dir):
    aux = ("height", "dem", "delta_vv", "delta_vh")
    return UnifiedDataset(
        lulc_dir=tile_dir, split="train", patch_size=H,
        augment_override=False, aux_channel_names=aux,
    ), aux


def test_dataset_emits_normalized_delta_for_harvest(tile_dir):
    """Harvest tile → negative, normalized delta_vv/delta_vh channels."""
    ds, aux = _ds(tile_dir)
    idx = ds.tile_names.index("harvest_with2016.npz")
    s = ds[idx]
    assert s["delta_vv"].shape == (1, H, W)
    assert s["delta_vh"].shape == (1, H, W)
    # season −14 dB, 2016 −8 dB → Δ = −6 dB → z = −6/4 = −1.5 (both bands).
    assert float(s["delta_vv"].mean()) == pytest.approx(-1.5, abs=1e-3)
    assert float(s["delta_vh"].mean()) == pytest.approx(-1.5, abs=1e-3)


def test_dataset_zero_delta_when_2016_absent(tile_dir):
    """Missing-2016 tile → delta channels are exactly 0 (neutral 'no change')."""
    ds, _ = _ds(tile_dir)
    idx = ds.tile_names.index("no2016.npz")
    s = ds[idx]
    assert float(s["delta_vv"].abs().max()) == 0.0
    assert float(s["delta_vh"].abs().max()) == 0.0


def test_dataset_zero_delta_for_stable(tile_dir):
    """Stable tile (season == 2016 backscatter... here −9 vs −8) → small Δ."""
    ds, _ = _ds(tile_dir)
    idx = ds.tile_names.index("stable_with2016.npz")
    s = ds[idx]
    # season −9, 2016 −8 → Δ = −1 dB → z = −0.25 (inside phenology band).
    assert float(s["delta_vv"].mean()) == pytest.approx(-0.25, abs=1e-3)
    assert abs(float(s["delta_vv"].mean())) < 0.5
