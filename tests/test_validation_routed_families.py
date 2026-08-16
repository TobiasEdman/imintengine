"""Validation-inference wiring for the routed FM families (Clay NOW).

``scripts/inference_comparison._build_inference_inputs`` historically built
only Prithvi-style (spectral→img5d) or Tessera inputs, and ``run_inference`` /
``run_fraction_inference`` called the model with the Prithvi/Tessera
signature. Clay/CROMA/TerraMind have different forwards, so they could not be
field-validated against NFI/LUCAS.

This suite proves the extension routes Clay (and, structurally, CROMA/
TerraMind) through the SAME builder + forward router the trainer uses:

    synthetic clay tile (spectral + b08 + rededge)
      → _build_inference_inputs(family="clay")   # builds s2_clay batch
      → shapes MATCH UnifiedDataset(model_keys=("clay_v1_5",))[i]["s2_clay"]
      → run_inference / run_fraction_inference via family_forward
      → (H,W) pred + (4,cs,cs) fracs with a stub clay model

The stub model mirrors the real ``ClaySegmentationModel.forward`` contract
(``chips, timestamps, wavelengths, aux, return_fractions``) so the *plumbing*
— builder reuse, crop parity, router routing — is what is under test, with no
real Clay weights or GPU.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


infcmp = _load("_infcmp_routed", SCRIPTS / "inference_comparison.py")

from imint.training.unified_dataset import UnifiedDataset  # noqa: E402

H = W = 64
IMG_SIZE = 48  # < tile size so the centre-crop is exercised
N_CLASSES = 28
K_FRAC = 4
CLAY_BANDS = 10


# ── Synthetic clay-capable tile ───────────────────────────────────────────

def _write_clay_tile(path):
    """A 4-frame LULC tile carrying every key Clay's s2_clay stack reads.

    Clay is optical-only — no S1 needed. ``spectral`` (24,H,W)=4×6,
    ``b08`` (T,H,W), ``rededge`` (T*3,H,W). doy drives the peak-summer
    best-frame selection so the validation path picks the SAME frame the
    dataset does.
    """
    np.savez_compressed(
        str(path),
        spectral=(np.random.rand(24, H, W) * 0.4).astype(np.float32),
        b08=(np.random.rand(4, H, W) * 0.4).astype(np.float32),
        rededge=(np.random.rand(12, H, W) * 0.4).astype(np.float32),
        label=np.random.randint(0, N_CLASSES, (H, W)).astype(np.int64),
        doy=np.array([260, 130, 190, 220], dtype=np.float32),
        year=np.int32(2022),
        easting=np.float32(500000.0), northing=np.float32(6500000.0),
    )


@pytest.fixture
def clay_tile(tmp_path):
    d = tmp_path / "tiles"
    d.mkdir()
    name = "tile_clay_000.npz"
    _write_clay_tile(d / name)
    # split files so UnifiedDataset can list the tile for parity comparison
    (d / "split_train.txt").write_text(name + "\n")
    (d / "split_val.txt").write_text(name + "\n")
    return d / name, d


# ── Stub Clay model (mirrors ClaySegmentationModel.forward) ───────────────

class _StubClaySpec:
    family = "clay"


class _StubClayModel(nn.Module):
    """forward(chips, timestamps, wavelengths, aux, return_fractions).

    Upsamples to the chips' input resolution exactly like the real
    ClaySegmentationModel, so the (H,W) crop-size contract is asserted.
    """

    def __init__(self, n_classes=N_CLASSES, k_frac=K_FRAC, n_bands=CLAY_BANDS):
        super().__init__()
        self.classifier = nn.Conv2d(n_bands, n_classes, 1)
        self.frac_head = nn.Conv2d(n_bands, k_frac, 1)
        self.fm_spec = _StubClaySpec()
        self.num_frames = 1
        # Record what the router fed us so the test can assert the contract.
        self.seen = {}

    def forward(self, chips, timestamps, wavelengths, aux=None,
                return_fractions=False):
        self.seen = {
            "chips_shape": tuple(chips.shape),
            "ts_shape": tuple(timestamps.shape),
            "wls_shape": tuple(wavelengths.shape),
            "aux_shape": None if aux is None else tuple(aux.shape),
        }
        logits = self.classifier(chips)  # (B, C, H, W) — already input-res
        if return_fractions:
            return logits, self.frac_head(chips)
        return logits


# ── Tests ─────────────────────────────────────────────────────────────────

def test_build_inputs_clay_shapes_match_dataset(clay_tile):
    """_build_inference_inputs(family=clay) s2_clay MATCHES the dataset stack.

    The whole point of the reuse discipline: validation must feed the SAME
    s2_clay tensor the trainer built. We compare the (uncropped) dataset
    build against the validation build's crop-back, and check the crop is a
    strict centre-crop of the dataset tensor (same normalization, same bands,
    same best-frame).
    """
    tile_path, tile_dir = clay_tile

    # Dataset build at native resolution (no augment, patch=full tile).
    ds = UnifiedDataset(
        lulc_dir=tile_dir, split="val", patch_size=H,
        augment_override=False, enable_aux=False,
        model_keys=("clay_v1_5",), backbone_family="clay",
    )
    sample = ds[0]
    ds_clay = np.asarray(sample["s2_clay"])  # (10, H, W) raw
    assert ds_clay.shape == (CLAY_BANDS, H, W)

    inp = infcmp._build_inference_inputs(
        str(tile_path), device="cpu", img_size=IMG_SIZE,
        aux_channel_names=None, family="clay", num_frames=1,
    )
    val_clay = inp["batch"]["s2_clay"].squeeze(0).cpu().numpy()  # (10, cs, cs)
    cs = min(IMG_SIZE, H, W)
    assert val_clay.shape == (CLAY_BANDS, cs, cs)

    # The validation build is a centre-crop of the dataset build (bit-equal).
    y0 = (H - cs) // 2
    x0 = (W - cs) // 2
    np.testing.assert_allclose(
        val_clay, ds_clay[:, y0:y0 + cs, x0:x0 + cs], rtol=0, atol=0,
    )
    # No img5d for routed families; batch carries the family stack.
    assert inp["img5d"] is None
    assert inp["family"] == "clay"
    # Clay uses location coords; they must be built.
    assert inp["location_coords"] is not None
    assert tuple(inp["location_coords"].shape) == (1, 2)
    assert inp["temporal_coords"] is None


def test_run_inference_clay_returns_hw_pred(clay_tile):
    tile_path, _ = clay_tile
    model = _StubClayModel()
    pred = infcmp.run_inference(
        model, str(tile_path), device="cpu", img_size=IMG_SIZE,
    )
    cs = min(IMG_SIZE, H, W)
    assert pred.shape == (cs, cs)
    assert pred.dtype.kind in "iu"
    # Router fed the model the s2_clay stack + explicit wavelengths + ts.
    assert model.seen["chips_shape"] == (1, CLAY_BANDS, cs, cs)
    assert model.seen["ts_shape"] == (1, 4)
    assert model.seen["wls_shape"] == (1, CLAY_BANDS)


def test_run_inference_clay_probs_shape(clay_tile):
    """return_probs path (used by validate_against_nfi) → (C, cs, cs)."""
    tile_path, _ = clay_tile
    model = _StubClayModel()
    probs, raw_spec, _raw_aux = infcmp.run_inference(
        model, str(tile_path), device="cpu", img_size=IMG_SIZE,
        return_probs=True,
    )
    cs = min(IMG_SIZE, H, W)
    assert probs.shape == (N_CLASSES, cs, cs)
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-4)
    # raw_spectral is centre-cropped to the same size for downstream refine.
    assert raw_spec.shape[1:] == (cs, cs)


def test_run_fraction_inference_clay_shape(clay_tile):
    tile_path, _ = clay_tile
    model = _StubClayModel()
    fracs = infcmp.run_fraction_inference(
        model, str(tile_path), device="cpu", img_size=IMG_SIZE,
    )
    cs = min(IMG_SIZE, H, W)
    assert fracs.shape == (K_FRAC, cs, cs)
    assert (fracs >= 0).all() and (fracs <= 1).all()  # sigmoid range


def test_prithvi_path_unchanged_no_batch_key_routing(clay_tile):
    """Prithvi family still builds img5d and carries no routed batch.

    Guards the byte-unchanged promise: a non-routed family must not go
    through the family_forward path (batch stays None, img5d populated).
    """
    tile_path, _ = clay_tile
    inp = infcmp._build_inference_inputs(
        str(tile_path), device="cpu", img_size=IMG_SIZE,
        aux_channel_names=None, family="prithvi", num_frames=1,
    )
    assert inp["batch"] is None
    assert inp["img5d"] is not None
    assert inp["family"] == "prithvi"


def test_load_model_infers_n_aux_from_weights_and_loads_clean(tmp_path):
    """load_model recovers n_aux from the aux branch when config omits it.

    The trainer's minimal ``best_model.pt`` config can omit
    ``n_aux_channels``; the old default of 11 then built an 11-channel
    LiDARBranch that could not load a 10-aux checkpoint. This guards the
    weight-based inference: a Prithvi seg model saved with 10 aux channels
    and a config STRIPPED of ``n_aux_channels`` must still load with ZERO
    non-encoder missing keys (the exact clean-load contract the Clay/CROMA/
    TerraMind checkpoints need). Prithvi stands in for the family-agnostic
    aux-branch inference — no external FM package required.
    """
    from imint.fm.registry import MODEL_CONFIGS, build_backbone
    from imint.fm.upernet import build_segmentation_from_spec

    spec = MODEL_CONFIGS["prithvi_300m"]
    enc, _ = build_backbone(
        "prithvi_300m", num_frames=1, img_size=224, pretrained=False)
    model = build_segmentation_from_spec(
        spec, encoder=enc, num_classes=N_CLASSES, img_size=224,
        decoder_channels=256, n_aux_channels=10,
        enable_tradslag_head=True, num_tradslag=K_FRAC, device="cpu",
    )
    # Config deliberately omits n_aux_channels (mirrors the real minimal
    # trainer config) so the load MUST fall back to weight inference.
    ckpt = {
        "model_state_dict": {"model." + k: v for k, v in
                             model.state_dict().items()},
        "config": {
            "backbone_name": "prithvi_300m",
            "num_temporal_frames": 1,
            "num_classes": N_CLASSES,
            "enable_tradslag_head": True,
            "num_tradslag": K_FRAC,
            # NB: no "n_aux_channels" key.
        },
        "epoch": 7, "metrics": {"miou": 0.42},
    }
    ckpt_path = tmp_path / "prithvi10aux.pt"
    torch.save(ckpt, str(ckpt_path))

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        loaded, epoch, miou, img_size = infcmp.load_model(
            str(ckpt_path), device="cpu")
    log = buf.getvalue()

    # n_aux was recovered from the lidar_branch conv (10, not the 11 default).
    assert "n_aux_channels inferred from checkpoint: 10" in log
    # A clean load emits NO state_dict-mismatch warning (zero non-encoder
    # missing + zero unexpected keys).
    assert "WARN state_dict mismatch" not in log, log
    assert loaded.frac_head is not None  # frac head loaded
    assert epoch == 7


def test_load_model_reconciles_psp_pool_count(tmp_path):
    """load_model rebuilds the head with the checkpoint's PSP pool count.

    Reproduces the live Clay NFI load failure: a head trained at img=504 /
    patch=8 has 6 PSP pools (bottleneck in-ch = C_deep + 256×6 = 2560), but a
    checkpoint carrying NEITHER pos_embed NOR an img_size in its minimal
    config made load_model default to img=224, rebuilding a 5-pool head
    (bottleneck 256×2304) → an unrecoverable size mismatch on load.

    Prithvi stands in for the mechanism with NO external FM package: build a
    Prithvi head at img=448 (patch=16 → fm=28 → 5 pools), STRIP pos_embed and
    OMIT img_size from the config so load_model would otherwise default to 224
    (patch=16 → 4 pools). The reconciliation must recover the pool count (5)
    from the checkpoint's psp_modules indices, correct img_size to a 5-pool
    value, and load with ZERO non-encoder missing/unexpected keys.
    """
    from imint.fm.registry import MODEL_CONFIGS, build_backbone
    from imint.fm.upernet import build_segmentation_from_spec, get_default_pool_sizes

    # Sanity: the trap only exists if 448 and 224 give different pool counts.
    assert len(get_default_pool_sizes(device="cpu", img_size=448, patch_size=16)) == 5
    assert len(get_default_pool_sizes(device="cpu", img_size=224, patch_size=16)) == 4

    spec = MODEL_CONFIGS["prithvi_300m"]
    enc, _ = build_backbone(
        "prithvi_300m", num_frames=1, img_size=448, pretrained=False)
    model = build_segmentation_from_spec(
        spec, encoder=enc, num_classes=N_CLASSES, img_size=448,
        decoder_channels=256, n_aux_channels=10,
        enable_tradslag_head=True, num_tradslag=K_FRAC, device="cpu",
    )
    n_pools = len(model.decoder.psp_modules)
    assert n_pools == 5

    # Strip pos_embed (→ no grid inference) AND omit img_size from config
    # (→ would default to 224 / 4 pools) so the reconciliation MUST fire.
    sd = {"model." + k: v for k, v in model.state_dict().items()
          if k != "encoder.pos_embed"}
    ckpt = {
        "model_state_dict": sd,
        "config": {
            "backbone_name": "prithvi_300m",
            "num_temporal_frames": 1,
            "num_classes": N_CLASSES,
            "enable_tradslag_head": True,
            "num_tradslag": K_FRAC,
            # NB: no img_size, no n_aux_channels — the minimal-config trap.
        },
        "epoch": 3,
    }
    ckpt_path = tmp_path / "prithvi448_5pool.pt"
    torch.save(ckpt, str(ckpt_path))

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        loaded, epoch, miou, out_img_size = infcmp.load_model(
            str(ckpt_path), device="cpu", backbone_name="prithvi_300m")
    log = buf.getvalue()

    # The reconciliation corrected img_size to a 5-pool value (448 is the
    # smallest patch-16 multiple that yields 5 pools).
    assert "PSP pool count mismatch" in log, log
    assert out_img_size == 448
    assert len(loaded.decoder.psp_modules) == n_pools
    # Clean load: no non-encoder missing / unexpected keys.
    assert "WARN state_dict mismatch" not in log, log
