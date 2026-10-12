"""Inference aux stack must equal the training aux stack, channel for channel.

The eval path builds its aux tensor in ``inference_comparison`` while training
builds it in ``UnifiedDataset._load_aux_channels``. Two implementations of one
contract drift silently: a divergence keeps the channel COUNT correct, so the
aux conv accepts the tensor and the job exits 0 with a MANIFEST-stamped cache
that every downstream scorer treats as authoritative.

Three divergences existed and are pinned here:

  * ``delta_vv``/``delta_vh`` are COMPUTED from the S1 keys and are never
    stored arrays, so an ``if name in npz`` lookup missed on every tile and
    fed constant zero to the two channels that carry the clearcut signal.
  * an absent ``markfukt`` uses training's NaN sentinel (which normalizes to
    the channel mean), not a raw physical zero — and markfukt has genuine
    source-data gaps, so this fires on real tiles.
  * the absent-channel branch skipped ``normalize_aux_channel`` entirely, so
    even a legitimately-zero channel landed on a different scale.

Bit-equality against the training helpers is the assertion; an approximate
check would pass on exactly the drift this suite exists to catch.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


infcmp = _load("_infcmp_aux", ROOT / "scripts" / "inference_comparison.py")

from imint.training.unified_dataset import (  # noqa: E402
    AUX_CHANNEL_NAMES,
    AUX_NAN_NODATA_CHANNELS,
    compute_delta_sar,
    normalize_aux_channel,
)

TILE = 16
N_BANDS_TOTAL = 24  # 4 frames x 6 bands


def _tile(tmp_path: Path, *, with_sar: bool = True,
          drop: set[str] | None = None) -> str:
    """Synthetic tile carrying every canonical aux key (+ optional S1)."""
    rng = np.random.default_rng(1234)
    drop = drop or set()
    data = {
        "spectral": (rng.random((N_BANDS_TOTAL, TILE, TILE),
                                dtype=np.float32) * 0.3),
    }
    for name in list(AUX_CHANNEL_NAMES) + ["markfukt"]:
        if name not in drop:
            data[name] = rng.random((TILE, TILE), dtype=np.float32)
    if with_sar:
        # +0.1 floor keeps the linear γ⁰ composites strictly positive so the
        # dB difference is finite without relying on compute_delta_sar's floor.
        data["s1_vv_vh"] = rng.random((2, TILE, TILE), dtype=np.float32) + 0.1
        data["s1_vv_vh_2016"] = (rng.random((2, TILE, TILE),
                                            dtype=np.float32) + 0.1)
    path = tmp_path / "tile.npz"
    np.savez(path, **data)
    return str(path)


def _aux_stack(tile_path: str, aux_names: list[str]) -> np.ndarray:
    inp = infcmp._build_inference_inputs(
        tile_path, "cpu", TILE, aux_names, family="prithvi", num_frames=4)
    return inp["aux"].squeeze(0).numpy()


DELTA_NAMES = ["delta_vv", "delta_vh"]
FULL_NAMES = list(AUX_CHANNEL_NAMES) + ["markfukt"] + DELTA_NAMES


def test_delta_sar_is_computed_not_zero_filled(tmp_path: Path) -> None:
    """ΔSAR must equal training's compute_delta_sar, bit for bit."""
    path = _tile(tmp_path)
    stack = _aux_stack(path, FULL_NAMES)
    assert stack.shape[0] == len(FULL_NAMES)

    with np.load(path, allow_pickle=True) as raw:
        truth = compute_delta_sar({k: raw[k] for k in raw.files})
    assert truth is not None, "fixture must carry both S1 composites"

    for row, name in enumerate(DELTA_NAMES):
        got = stack[FULL_NAMES.index(name)]
        expected = normalize_aux_channel(name, truth[row].astype(np.float32))
        assert not np.allclose(got, 0.0), (
            f"{name} is constant zero — the computed-channel branch is gone "
            f"and the clearcut signal has been silently removed")
        np.testing.assert_array_equal(got, expected)


def test_delta_sar_zero_when_2016_composite_absent(tmp_path: Path) -> None:
    """No 2016 anchor → zeros, matching training's neutral 'no change'."""
    stack = _aux_stack(_tile(tmp_path, with_sar=False), FULL_NAMES)
    for name in DELTA_NAMES:
        got = stack[FULL_NAMES.index(name)]
        np.testing.assert_array_equal(
            got, normalize_aux_channel(name, np.zeros((TILE, TILE), np.float32)))


def test_absent_nan_nodata_channel_uses_training_sentinel(
    tmp_path: Path,
) -> None:
    """Absent markfukt → NaN sentinel → channel mean, not a raw zero."""
    assert "markfukt" in AUX_NAN_NODATA_CHANNELS
    stack = _aux_stack(_tile(tmp_path, drop={"markfukt"}), FULL_NAMES)
    got = stack[FULL_NAMES.index("markfukt")]
    expected = normalize_aux_channel(
        "markfukt", np.full((TILE, TILE), np.nan, dtype=np.float32))
    np.testing.assert_array_equal(got, expected)


def test_every_stored_channel_matches_training_normalization(
    tmp_path: Path,
) -> None:
    """Full-stack parity: each channel equals training's normalized array."""
    path = _tile(tmp_path)
    stack = _aux_stack(path, FULL_NAMES)
    with np.load(path, allow_pickle=True) as raw:
        stored = {k: raw[k] for k in raw.files}
    for i, name in enumerate(FULL_NAMES):
        if name in DELTA_NAMES:
            continue
        expected = normalize_aux_channel(
            name, stored[name].astype(np.float32))
        np.testing.assert_array_equal(stack[i], expected, err_msg=name)


def test_era5_channel_refuses_rather_than_substituting(
    tmp_path: Path,
) -> None:
    """ERA5 needs a sidecar + an explicit mode; inference has neither."""
    with pytest.raises(ValueError, match="ERA5 channel"):
        _aux_stack(_tile(tmp_path), list(AUX_CHANNEL_NAMES) + ["era5_gdd"])


# --------------------------------------------------------------------------
# run_inference derives the aux set from the checkpoint
#
# validate_against_nfi / validate_against_lucas / render_endgame_frames all
# call run_inference without an explicit aux list. Before this, that meant the
# canonical 10 regardless of what the checkpoint was trained with — the same
# defect the ladder eval died on, reached through a different door. Asserting
# on what run_inference PASSES DOWN (rather than on a source-code grep) is
# what makes these fail if the derivation is removed.
# --------------------------------------------------------------------------


class _Model:
    """Minimal stand-in carrying only what run_inference reads off a model."""

    def __init__(self, names, n_aux):
        self.ck_cfg = {"enabled_aux_names": list(names)} if names else {}
        self.n_aux_channels = n_aux
        self.num_frames = 4
        self.fm_spec = None


def _captured_aux_names(monkeypatch, model, explicit=None):
    seen = {}

    def _spy(tile_path, device, img_size, aux_channel_names, **kw):
        seen["names"] = aux_channel_names
        raise _Stop

    class _Stop(Exception):
        pass

    monkeypatch.setattr(infcmp, "_build_inference_inputs", _spy)
    with pytest.raises(_Stop):
        infcmp.run_inference(model, "unused.npz", "cpu", img_size=TILE,
                             aux_channel_names=explicit)
    return seen["names"]


def test_run_inference_takes_aux_set_from_checkpoint(monkeypatch) -> None:
    model = _Model(FULL_NAMES, len(FULL_NAMES))
    assert _captured_aux_names(monkeypatch, model) == FULL_NAMES


def test_run_inference_explicit_argument_still_wins(monkeypatch) -> None:
    model = _Model(FULL_NAMES, len(FULL_NAMES))
    override = list(AUX_CHANNEL_NAMES)
    assert _captured_aux_names(monkeypatch, model, override) == override


def test_run_inference_falls_back_when_checkpoint_records_nothing(
    monkeypatch,
) -> None:
    """Pre-config-era checkpoint → None → the builder's canonical default."""
    model = _Model(None, len(AUX_CHANNEL_NAMES))
    assert _captured_aux_names(monkeypatch, model) is None


def test_run_inference_refuses_name_count_conv_mismatch(monkeypatch) -> None:
    """Names and the aux conv must agree, or the stack is built wrong."""
    model = _Model(FULL_NAMES, len(FULL_NAMES) - 1)
    monkeypatch.setattr(
        infcmp, "_build_inference_inputs",
        lambda *a, **k: pytest.fail("must refuse before building inputs"))
    with pytest.raises(ValueError, match="aux conv takes"):
        infcmp.run_inference(model, "unused.npz", "cpu", img_size=TILE)
