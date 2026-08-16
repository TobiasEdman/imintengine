"""End-to-end training-path wiring for CROMA / Clay / TerraMind members.

These are the three backbones that previously crashed on the first training
batch with a TypeError because the model↔trainer glue was never landed. This
suite exercises the FULL path with fake encoders (no real weights needed):

    synthetic tile (spectral + b08 + rededge + s1_vv_vh)
      → UnifiedDataset(model_keys=(<backbone>,))   # emits the per-model stacks
      → imint.fm.forward_router.family_forward(...) # per-family routing
      → (logits (B,28,H,W), frac_logits (B,4,H,W)) # frac head enabled
      → one loss.backward()                         # gradients flow

It also asserts:
  * the S1-required filter drops tiles missing s1_vv_vh (CROMA/TerraMind),
    while Clay keeps optical-only tiles,
  * the Prithvi default path is unaffected (no model_keys → no extra keys).

Real-weight end-to-end (CROMA_base.pt / TerraMind terratorch / Clay ckpt)
runs on the 2080ti k8s smoke jobs — this file is the fast local guard.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from imint.training.unified_dataset import UnifiedDataset
from imint.fm.forward_router import family_forward

H = W = 64
N_CLASSES = 28
K_FRAC = 4


# ── Synthetic tile ────────────────────────────────────────────────────────

def _write_tile(path, *, with_s1: bool, with_b01_b09: bool = True):
    """A 4-frame tile carrying every enrichment key the members read.

    ``with_b01_b09`` mirrors real unified_v2_512 tiles, which carry real
    per-frame B01 (coastal aerosol) and B09 (water vapour) + has-flags — so
    CROMA gets the FULL 12-band stack, not two zero-padded channels.
    """
    data = {
        "spectral": (np.random.rand(24, H, W) * 0.4).astype(np.float32),   # 4×6
        "b08":      (np.random.rand(4, H, W) * 0.4).astype(np.float32),     # (T,H,W)
        "rededge":  (np.random.rand(12, H, W) * 0.4).astype(np.float32),    # (T*3,H,W)
        "label":    np.random.randint(0, N_CLASSES, (H, W)).astype(np.int64),
        "doy":      np.array([260, 130, 190, 220], dtype=np.float32),
        "year":     np.int32(2022),
        "easting":  np.float32(500000.0),
        "northing": np.float32(6500000.0),
    }
    if with_b01_b09:
        # Distinct constants per band so the emission test can prove the REAL
        # band reached s2_croma (not a zero-pad).
        data["b01"] = np.full((4, H, W), 0.05, dtype=np.float32)   # (T,H,W)
        data["b09"] = np.full((4, H, W), 0.30, dtype=np.float32)
        data["has_b01"] = np.int32(1)
        data["has_b09"] = np.int32(1)
    if with_s1:
        # v2 layout: a single per-orbit season median composite (2, H, W) +
        # the version marker. The old (T*2, H, W) ±3-day stack is retired.
        data["s1_vv_vh"] = (np.random.rand(2, H, W) * 0.2).astype(np.float32)
        data["s1_orbit"] = np.bytes_("DESCENDING")
        data["has_s1"] = np.int32(1)
        data["s1_enrich_v"] = np.int32(2)
    np.savez_compressed(str(path), **data)


def _write_tile_nan_s1(path):
    """S1-complete v2 tile whose composite carries some NaN pixels (nodata).

    Mirrors real tiles where part of the composite is a genuine swath-edge
    gap; the dataset must scrub NaN→0 so the dB normalizer never sees NaN."""
    s1 = (np.random.rand(2, H, W) * 0.2).astype(np.float32)  # (2, H, W)
    s1[:, : H // 2, :] = np.nan  # top half is nodata
    np.savez_compressed(
        str(path),
        spectral=(np.random.rand(24, H, W) * 0.4).astype(np.float32),
        b08=(np.random.rand(4, H, W) * 0.4).astype(np.float32),
        rededge=(np.random.rand(12, H, W) * 0.4).astype(np.float32),
        label=np.random.randint(0, N_CLASSES, (H, W)).astype(np.int64),
        doy=np.array([260, 130, 190, 220], dtype=np.float32),
        year=np.int32(2022),
        easting=np.float32(500000.0), northing=np.float32(6500000.0),
        s1_vv_vh=s1, s1_orbit=np.bytes_("DESCENDING"),
        has_s1=np.int32(1), s1_enrich_v=np.int32(2),
    )


def _write_frac(path):
    np.savez_compressed(
        str(path),
        frac=(np.random.rand(K_FRAC, H, W) * 100).astype(np.float32),
        frac_unreliable=np.zeros((H, W), dtype=bool),
    )


@pytest.fixture
def tile_dir(tmp_path):
    """3 S1-complete tiles + 2 optical-only tiles, all listed for train/val."""
    d = tmp_path / "tiles"
    d.mkdir()
    names = []
    for i in range(3):
        n = f"tile_s1_{i:03d}.npz"
        _write_tile(d / n, with_s1=True)
        names.append(n)
    for i in range(2):
        n = f"tile_opt_{i:03d}.npz"
        _write_tile(d / n, with_s1=False)
        names.append(n)
    (d / "split_train.txt").write_text("\n".join(names) + "\n")
    (d / "split_val.txt").write_text("\n".join(names) + "\n")
    return d


@pytest.fixture
def frac_dir(tmp_path, tile_dir):
    fd = tmp_path / "fracs"
    fd.mkdir()
    for p in tile_dir.glob("*.npz"):
        _write_frac(fd / p.name)
    return fd


# ── Fake encoders (mirror the real forward contracts) ─────────────────────

class _FakeCroma(nn.Module):
    """PretrainedCROMA-like: forward(SAR_images=, optical_images=) → dict."""

    def __init__(self, embed_dim=768, patch=8, img=H):
        super().__init__()
        self.n = (img // patch) ** 2
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(12, embed_dim, patch, patch)

    def forward(self, SAR_images=None, optical_images=None):
        t = self.proj(optical_images).flatten(2).transpose(1, 2)  # (B,N,D)
        return {"joint_encodings": t, "optical_encodings": t,
                "SAR_encodings": t}


class _FakeClay(nn.Module):
    """Clay-like: .blocks ModuleList; forward(chips, ts, wls) pooled."""

    def __init__(self, embed_dim=1024, patch=8, n_bands=10):
        super().__init__()
        self.patch_embed = nn.Conv2d(n_bands, embed_dim, patch, patch)
        self.blocks = nn.ModuleList([nn.Identity()])
        self.embed_dim = embed_dim

    def forward(self, chips, timestamps, wavelengths):
        t = self.patch_embed(chips).flatten(2).transpose(1, 2)  # (B,N,D)
        # Route through the (hooked) last block so the wrapper captures tokens.
        t = self.blocks[-1](t)
        return t.mean(dim=1)  # pooled (B,D) — wrapper ignores, uses the hook


class _FakeTerraMind(nn.Module):
    """TerraMind-like: forward(dict) → (B, N, D) tokens."""

    def __init__(self, embed_dim=768, patch=16, img=W):
        super().__init__()
        self.n = (img // patch) ** 2
        self.proj = nn.Conv2d(6, embed_dim, patch, patch)

    def forward(self, inputs: dict):
        return self.proj(inputs["S2L2A"]).flatten(2).transpose(1, 2)


def _build(seg_cls, encoder, **kw):
    m = seg_cls(
        encoder=encoder, num_classes=N_CLASSES, img_size=H,
        enable_tradslag_head=True, num_tradslag=K_FRAC, **kw,
    )
    # Stash a minimal spec so callers that read fm_spec.family work.
    class _Spec:
        family = kw.get("_family", "prithvi")
    return m


# ── Dataset filter tests ──────────────────────────────────────────────────

class TestS1RequiredFilter:
    def test_croma_drops_optical_only_tiles(self, tile_dir):
        ds = UnifiedDataset(
            lulc_dir=tile_dir, split="train", patch_size=H,
            augment_override=False, model_keys=("croma_base",),
            backbone_family="croma",
        )
        assert len(ds) == 3  # only the 3 S1-complete tiles survive

    def test_terramind_drops_optical_only_tiles(self, tile_dir):
        ds = UnifiedDataset(
            lulc_dir=tile_dir, split="train", patch_size=H,
            augment_override=False, model_keys=("terramind_v1_base",),
            backbone_family="terramind",
        )
        assert len(ds) == 3

    def test_clay_keeps_all_tiles(self, tile_dir):
        ds = UnifiedDataset(
            lulc_dir=tile_dir, split="train", patch_size=H,
            augment_override=False, model_keys=("clay_v1_5",),
            backbone_family="clay",
        )
        assert len(ds) == 5  # optical-only fine — Clay needs no SAR

    def test_prithvi_default_unaffected(self, tile_dir):
        ds = UnifiedDataset(
            lulc_dir=tile_dir, split="train", patch_size=H,
            augment_override=False,
        )
        assert len(ds) == 5
        s = ds[0]
        assert "s2_croma" not in s and "s2_clay" not in s
        assert "s1_vv_vh" not in s and "s2_terramind" not in s


# ── Dataset emits the right per-model keys ────────────────────────────────

class TestDatasetEmitsModelKeys:
    def test_croma_keys(self, tile_dir):
        ds = UnifiedDataset(
            lulc_dir=tile_dir, split="train", patch_size=H,
            augment_override=False, model_keys=("croma_base",),
            backbone_family="croma",
        )
        s = ds[0]
        assert s["s2_croma"].shape == (12, H, W)
        assert s["s1_vv_vh"].shape == (2, H, W)

    def test_croma_uses_real_b01_b09(self, tile_dir):
        """CROMA_S2_BAND_ORDER puts B01 at index 0 and B09 at index 9. When
        the tile carries real B01/B09 (has-flags set), those channels must
        hold the real values — NOT zeros. This is the fairness fix: two real
        bands were previously discarded via zero-padding."""
        ds = UnifiedDataset(
            lulc_dir=tile_dir, split="train", patch_size=H,
            augment_override=False, model_keys=("croma_base",),
            backbone_family="croma",
        )
        s2 = ds[0]["s2_croma"]           # (12, H, W)
        b01_channel = s2[0]              # B01
        b09_channel = s2[9]              # B09
        # Real constants written by _write_tile: 0.05 and 0.30.
        assert torch.allclose(b01_channel, torch.full_like(b01_channel, 0.05))
        assert torch.allclose(b09_channel, torch.full_like(b09_channel, 0.30))
        assert (b01_channel != 0).all() and (b09_channel != 0).all()

    def test_croma_zero_pads_b01_b09_when_absent(self, tmp_path):
        """If the tile lacks B01/B09 (or has-flag=0), those channels fall
        back to zero-pad — the loader/dataset must not crash and must emit
        a finite 12-band stack."""
        d = tmp_path / "tiles"
        d.mkdir()
        _write_tile(d / "tile_0.npz", with_s1=True, with_b01_b09=False)
        (d / "split_train.txt").write_text("tile_0.npz\n")
        ds = UnifiedDataset(
            lulc_dir=d, split="train", patch_size=H,
            augment_override=False, model_keys=("croma_base",),
            backbone_family="croma",
        )
        s2 = ds[0]["s2_croma"]
        assert s2.shape == (12, H, W)
        assert torch.isfinite(s2).all()
        assert (s2[0] == 0).all() and (s2[9] == 0).all()  # B01, B09 padded

    def test_clay_keys(self, tile_dir):
        ds = UnifiedDataset(
            lulc_dir=tile_dir, split="train", patch_size=H,
            augment_override=False, model_keys=("clay_v1_5",),
            backbone_family="clay",
        )
        s = ds[0]
        assert s["s2_clay"].shape == (10, H, W)

    def test_s1_nan_frame_scrubbed(self, tmp_path):
        """A v2 composite carrying NaN nodata pixels must emit a finite SAR
        tensor (NaN → 0), else the S1 dB normalizer yields a NaN loss."""
        d = tmp_path / "tiles"
        d.mkdir()
        _write_tile_nan_s1(d / "tile_0000.npz")
        (d / "split_train.txt").write_text("tile_0000.npz\n")
        ds = UnifiedDataset(
            lulc_dir=d, split="train", patch_size=H,
            augment_override=False, model_keys=("croma_base",),
            backbone_family="croma",
        )
        s = ds[0]
        assert s["s1_vv_vh"].shape == (2, H, W)
        assert torch.isfinite(s["s1_vv_vh"]).all()

    def test_terramind_keys(self, tile_dir):
        ds = UnifiedDataset(
            lulc_dir=tile_dir, split="train", patch_size=H,
            augment_override=False, model_keys=("terramind_v1_base",),
            backbone_family="terramind",
        )
        s = ds[0]
        assert s["s2_terramind"].shape == (6, H, W)
        assert s["s1_vv_vh"].shape == (2, H, W)


# ── Full dataset → router → (logits, frac) → backward ─────────────────────

def _collate(samples):
    """Minimal stack collate for the keys the router reads."""
    out = {}
    for k in samples[0]:
        if isinstance(samples[0][k], torch.Tensor):
            out[k] = torch.stack([s[k] for s in samples])
    return out


class TestEndToEndForwardBackward:
    def _run(self, ds, model, family, aux_names):
        batch = _collate([ds[0], ds[1]])
        aux = torch.cat(
            [batch[n] for n in aux_names if n in batch], dim=1,
        )
        loc = batch.get("location_coords")
        logits, frac = family_forward(
            model, family, batch, torch.device("cpu"),
            aux=aux, location_coords=loc, return_fractions=True,
        )
        assert logits.shape == (2, N_CLASSES, H, W)
        assert frac.shape == (2, K_FRAC, H, W)
        loss = logits.float().mean() + frac.float().mean()
        loss.backward()
        # At least one trainable param got a gradient.
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters() if p.requires_grad
        )
        return logits.shape, frac.shape

    def test_croma_end_to_end(self, tile_dir):
        from imint.fm.croma_seg import CromaSegmentationModel
        ds = UnifiedDataset(
            lulc_dir=tile_dir, split="train", patch_size=H,
            augment_override=False, model_keys=("croma_base",),
            backbone_family="croma",
        )
        n_aux = len(ds.aux_channel_names)
        model = CromaSegmentationModel(
            encoder=_FakeCroma(), num_classes=N_CLASSES, img_size=H,
            patch_size=8, embed_dim=768, modality="joint",
            n_aux_channels=n_aux, enable_tradslag_head=True, num_tradslag=K_FRAC,
        )
        ls, fs = self._run(ds, model, "croma", ds.aux_channel_names)
        assert ls == (2, N_CLASSES, H, W) and fs == (2, K_FRAC, H, W)

    def test_clay_end_to_end(self, tile_dir):
        from imint.fm.clay_seg import ClaySegmentationModel
        ds = UnifiedDataset(
            lulc_dir=tile_dir, split="train", patch_size=H,
            augment_override=False, model_keys=("clay_v1_5",),
            backbone_family="clay",
        )
        n_aux = len(ds.aux_channel_names)
        model = ClaySegmentationModel(
            encoder=_FakeClay(), num_classes=N_CLASSES, img_size=H,
            patch_size=8, embed_dim=1024,
            n_aux_channels=n_aux, enable_tradslag_head=True, num_tradslag=K_FRAC,
        )
        ls, fs = self._run(ds, model, "clay", ds.aux_channel_names)
        assert ls == (2, N_CLASSES, H, W) and fs == (2, K_FRAC, H, W)

    def test_terramind_end_to_end(self, tile_dir):
        from imint.fm.terramind_seg import TerraMindSegmentationModel
        ds = UnifiedDataset(
            lulc_dir=tile_dir, split="train", patch_size=H,
            augment_override=False, model_keys=("terramind_v1_base",),
            backbone_family="terramind",
        )
        n_aux = len(ds.aux_channel_names)
        model = TerraMindSegmentationModel(
            encoder=_FakeTerraMind(patch=16, img=H), num_classes=N_CLASSES,
            img_size=H, embed_dim=768, patch_size=16,
            n_aux_channels=n_aux, enable_tradslag_head=True, num_tradslag=K_FRAC,
        )
        ls, fs = self._run(ds, model, "terramind", ds.aux_channel_names)
        assert ls == (2, N_CLASSES, H, W) and fs == (2, K_FRAC, H, W)
