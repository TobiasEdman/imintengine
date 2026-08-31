"""End-to-end wiring tests for the TESSERA member (D1).

Covers the three D1 gaps that connect the (already-existing)
``TesseraSegmentationModel`` to the training/inference pipeline:

1. ``UnifiedDataset`` reads the pre-baked ``tessera`` embedding — not
   ``spectral`` — when ``backbone_family="tessera"``, with NO Prithvi
   z-score and NO temporal reshape (the 4D embedding is emitted under the
   ``spectral`` batch key; the trainer routes on family).
2. The frac-head returns (B, 4, H, W) (covered in test_tessera_seg.py; a
   dataset→model round-trip is asserted here too).
3. Inference routing (``_build_inference_inputs``) reads the tessera key
   for tessera-family models and skips Prithvi normalization.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from imint.training.unified_dataset import UnifiedDataset

# A tessera tile is a normal Sentinel-2 training tile with the extra
# `tessera` (128, H, W) embedding baked in by enrich_tiles_tessera.py.
TILE_H = TILE_W = 64
TESSERA_DIM = 128
N_CLASSES = 28


def _write_tessera_tile(path: Path, *, with_frac: bool = False):
    """Write a synthetic tile carrying the exact keys the dataset reads.

    Mirrors scripts/enrich_tiles_tessera.py: `tessera` is (128, H, W) fp16,
    already normalized. `spectral` (24, H, W) is present too (the embedding
    is added to existing tiles) so the up-front spectral-key check passes.
    """
    data = {
        # 4-frame reflectance in [0,1] — present on every real tile.
        "spectral": (np.random.rand(24, TILE_H, TILE_W) * 0.4).astype(np.float32),
        # The pre-baked annual embedding — what the tessera path consumes.
        "tessera": np.random.randn(TESSERA_DIM, TILE_H, TILE_W).astype(np.float16),
        "has_tessera": np.int32(1),
        "tessera_year": np.int32(2022),
        # Unified 28-class hard label.
        "label": np.random.randint(0, N_CLASSES, (TILE_H, TILE_W)).astype(np.int64),
        "doy": np.array([260, 130, 190, 220], dtype=np.float32),
        "year": np.int32(2022),
    }
    np.savez_compressed(str(path), **data)


def _write_frac_sidecar(path: Path):
    """Trädslag fraction sidecar: (4, H, W) crown cover 0-100 + reliability."""
    np.savez_compressed(
        str(path),
        frac=(np.random.rand(4, TILE_H, TILE_W) * 100).astype(np.float32),
        frac_unreliable=np.zeros((TILE_H, TILE_W), dtype=bool),
    )


@pytest.fixture
def tessera_tile_dir(tmp_path):
    """A directory with a split_train.txt listing one tessera tile."""
    d = tmp_path / "tiles"
    d.mkdir()
    _write_tessera_tile(d / "tile_0001.npz")
    (d / "split_train.txt").write_text("tile_0001.npz\n")
    (d / "split_val.txt").write_text("tile_0001.npz\n")
    return d


class TestDatasetTesseraRouting:
    def test_reads_tessera_embedding_not_spectral(self, tessera_tile_dir):
        ds = UnifiedDataset(
            lulc_dir=tessera_tile_dir, split="train",
            patch_size=TILE_H, enable_aux=True,
            augment_override=False, backbone_family="tessera",
        )
        sample = ds[0]
        img = sample["spectral"]  # image tensor lives under `spectral` for all families
        # 128-channel embedding, NOT 24-channel reflectance.
        assert img.shape == (TESSERA_DIM, TILE_H, TILE_W)
        assert img.dtype == torch.float32

    def test_skips_prithvi_normalization(self, tessera_tile_dir):
        """Embedding passes through unchanged (no reflectance*10000 z-score).

        Compare the emitted tensor against the raw on-disk embedding after the
        same centre-crop the dataset applies (a no-op here since tile==patch).
        """
        raw = np.load(tessera_tile_dir / "tile_0001.npz")["tessera"].astype(np.float32)
        ds = UnifiedDataset(
            lulc_dir=tessera_tile_dir, split="train",
            patch_size=TILE_H, enable_aux=True,
            augment_override=False, backbone_family="tessera",
        )
        emitted = ds[0]["spectral"].numpy()
        np.testing.assert_allclose(emitted, raw, rtol=1e-3, atol=1e-3)

    def test_no_temporal_coords_emitted(self, tessera_tile_dir):
        ds = UnifiedDataset(
            lulc_dir=tessera_tile_dir, split="train",
            patch_size=TILE_H, augment_override=False,
            backbone_family="tessera",
        )
        sample = ds[0]
        # Coord keys omitted (not None) so default collate never sees a None.
        assert "temporal_coords" not in sample
        assert "location_coords" not in sample

    def test_prithvi_default_unchanged(self, tessera_tile_dir):
        """Default family still reads 24-ch reflectance + z-scores it."""
        ds = UnifiedDataset(
            lulc_dir=tessera_tile_dir, split="train",
            patch_size=TILE_H, augment_override=False,
        )
        sample = ds[0]
        # single-date extract → 6 bands, z-scored (values well outside [0,1]).
        assert sample["spectral"].shape[0] == 6
        assert sample["spectral"].max() > 1.0  # z-scored, not raw reflectance
        assert "temporal_coords" in sample

    def test_multitemporal_tessera_rejected(self, tessera_tile_dir):
        with pytest.raises(ValueError, match="single-frame"):
            UnifiedDataset(
                lulc_dir=tessera_tile_dir, split="train",
                multitemporal=True, backbone_family="tessera",
            )

    def test_frac_bundle_emitted(self, tmp_path, tessera_tile_dir):
        frac_dir = tmp_path / "fracs"
        frac_dir.mkdir()
        _write_frac_sidecar(frac_dir / "tile_0001.npz")
        ds = UnifiedDataset(
            lulc_dir=tessera_tile_dir, split="train",
            patch_size=TILE_H, augment_override=False,
            backbone_family="tessera", frac_dir=frac_dir,
        )
        sample = ds[0]
        assert sample["frac"].shape == (4, TILE_H, TILE_W)
        assert sample["frac_mask"].shape == (TILE_H, TILE_W)


class TestDatasetToModelRoundTrip:
    def test_embedding_feeds_tessera_seg_with_frac(self, tessera_tile_dir):
        from imint.fm.registry import build_backbone
        from imint.fm.upernet import build_segmentation_from_spec

        ds = UnifiedDataset(
            lulc_dir=tessera_tile_dir, split="train",
            patch_size=TILE_H, augment_override=False,
            backbone_family="tessera",
        )
        emb = ds[0]["spectral"].unsqueeze(0)  # (1, 128, H, W)

        enc, spec = build_backbone("tessera_v1", num_frames=1, pretrained=False)
        model = build_segmentation_from_spec(
            spec, encoder=enc, num_classes=N_CLASSES, img_size=TILE_H,
            enable_tradslag_head=True, num_tradslag=4,
        )
        logits, frac = model(emb, return_fractions=True)
        assert logits.shape == (1, N_CLASSES, TILE_H, TILE_W)
        assert frac.shape == (1, 4, TILE_H, TILE_W)


class TestInferenceRouting:
    """`_build_inference_inputs(..., family='tessera')` reads the embedding
    and skips Prithvi normalization + TL coords."""

    def _load_infcmp(self):
        import importlib.util
        p = Path(__file__).resolve().parents[1] / "scripts" / "inference_comparison.py"
        spec = importlib.util.spec_from_file_location("_infcmp", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_tessera_family_reads_embedding(self, tessera_tile_dir):
        infcmp = self._load_infcmp()
        tile = tessera_tile_dir / "tile_0001.npz"
        inp = infcmp._build_inference_inputs(
            str(tile), torch.device("cpu"), img_size=TILE_H,
            aux_channel_names=None, family="tessera",
        )
        # 4D embedding, NOT a 5D Prithvi Conv3d tensor.
        assert inp["img5d"].shape == (1, TESSERA_DIM, TILE_H, TILE_W)
        assert inp["temporal_coords"] is None
        assert inp["location_coords"] is None
        # Raw embedding preserved (no z-score).
        raw = np.load(tile)["tessera"].astype(np.float32)
        np.testing.assert_allclose(
            inp["img5d"].squeeze(0).numpy(), raw, rtol=1e-3, atol=1e-3,
        )

    def test_prithvi_family_still_5d(self, tessera_tile_dir):
        infcmp = self._load_infcmp()
        tile = tessera_tile_dir / "tile_0001.npz"
        inp = infcmp._build_inference_inputs(
            str(tile), torch.device("cpu"), img_size=TILE_H,
            aux_channel_names=None, family="prithvi",
        )
        # (1, 6, T, H, W) Conv3d layout — 5 dims, 6 spectral bands.
        assert inp["img5d"].dim() == 5
        assert inp["img5d"].shape[1] == 6

    def test_predict_fn_routes_on_stashed_family(self, tessera_tile_dir):
        """run_inference reads model.fm_spec.family and routes accordingly —
        a Prithvi-normalized path would blow up on a 128-D embedding."""
        infcmp = self._load_infcmp()
        from imint.fm.registry import build_backbone
        from imint.fm.upernet import build_segmentation_from_spec

        enc, spec = build_backbone("tessera_v1", num_frames=1, pretrained=False)
        model = build_segmentation_from_spec(
            spec, encoder=enc, num_classes=N_CLASSES, img_size=TILE_H,
        )
        model.fm_spec = spec  # stashed by load_model in production
        model.eval()

        probs, _raw_spec, _raw_aux = infcmp.run_inference(
            model, str(tessera_tile_dir / "tile_0001.npz"),
            torch.device("cpu"), img_size=TILE_H, return_probs=True,
        )
        assert probs.shape == (N_CLASSES, TILE_H, TILE_W)
