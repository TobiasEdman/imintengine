"""Tests for imint.fm.croma_seg.CromaSegmentationModel and
imint.fm.loaders.croma.build_s2_croma_tensor.

Uses a fake CROMA encoder that mimics PretrainedCROMA's dict return.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from imint.fm.croma_seg import CromaSegmentationModel
from imint.fm.loaders.croma import (
    CROMA_S2_BAND_ORDER,
    build_s2_croma_tensor,
)


class _FakeCromaEncoder(nn.Module):
    """Mimics PretrainedCROMA.forward which returns a dict."""

    def __init__(self, patch_size: int = 8, embed_dim: int = 768,
                 has_joint: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.has_joint = has_joint

    def _encode(self, img: torch.Tensor) -> torch.Tensor:
        B, _, H, W = img.shape
        gh, gw = H // self.patch_size, W // self.patch_size
        return torch.randn(B, gh * gw, self.embed_dim)

    def forward(self, SAR_images=None, optical_images=None):
        out = {}
        if SAR_images is not None:
            out["SAR_encodings"] = self._encode(SAR_images)
            out["SAR_GAP"] = out["SAR_encodings"].mean(dim=1)
        if optical_images is not None:
            out["optical_encodings"] = self._encode(optical_images)
            out["optical_GAP"] = out["optical_encodings"].mean(dim=1)
        if self.has_joint and SAR_images is not None and optical_images is not None:
            out["joint_encodings"] = (
                out["SAR_encodings"] + out["optical_encodings"]
            ) / 2
            out["joint_GAP"] = out["joint_encodings"].mean(dim=1)
        return out


class TestCromaSegmentationForward:
    def test_joint_modality_output_shape(self):
        enc = _FakeCromaEncoder(embed_dim=768)
        model = CromaSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=120, patch_size=8, embed_dim=768,
            modality="joint",
        )
        sar = torch.randn(2, 2, 120, 120)
        opt = torch.randn(2, 12, 120, 120)
        out = model(sar=sar, optical=opt)
        assert out.shape == (2, 23, 120, 120)

    def test_optical_only(self):
        enc = _FakeCromaEncoder(embed_dim=768, has_joint=False)
        model = CromaSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=120, patch_size=8, embed_dim=768,
            modality="optical",
        )
        opt = torch.randn(1, 12, 120, 120)
        out = model(sar=None, optical=opt)
        assert out.shape == (1, 23, 120, 120)

    def test_sar_only(self):
        enc = _FakeCromaEncoder(embed_dim=768, has_joint=False)
        model = CromaSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=120, patch_size=8, embed_dim=768,
            modality="sar",
        )
        sar = torch.randn(1, 2, 120, 120)
        out = model(sar=sar, optical=None)
        assert out.shape == (1, 23, 120, 120)

    def test_joint_fallback_to_optical(self):
        """If encoder doesn't return joint_encodings but we asked for
        joint, wrapper should fall back to optical."""
        enc = _FakeCromaEncoder(embed_dim=768, has_joint=False)
        model = CromaSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=120, patch_size=8, embed_dim=768,
            modality="joint",
        )
        sar = torch.randn(1, 2, 120, 120)
        opt = torch.randn(1, 12, 120, 120)
        out = model(sar=sar, optical=opt)
        assert out.shape == (1, 23, 120, 120)

    def test_joint_requires_both(self):
        enc = _FakeCromaEncoder(embed_dim=768)
        model = CromaSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=120, patch_size=8, embed_dim=768,
            modality="joint",
        )
        with pytest.raises(ValueError, match="sar"):
            model(sar=None, optical=torch.randn(1, 12, 120, 120))

    def test_output_size_arg_resizes(self):
        enc = _FakeCromaEncoder(embed_dim=768)
        model = CromaSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=120, patch_size=8, embed_dim=768,
            modality="joint",
        )
        sar = torch.randn(1, 2, 120, 120)
        opt = torch.randn(1, 12, 120, 120)
        out = model(sar=sar, optical=opt, output_size=(512, 512))
        assert out.shape == (1, 23, 512, 512)

    def test_aux_channels(self):
        enc = _FakeCromaEncoder(embed_dim=768)
        model = CromaSegmentationModel(
            encoder=enc, num_classes=23,
            img_size=120, patch_size=8, embed_dim=768,
            modality="joint", n_aux_channels=5,
        )
        sar = torch.randn(1, 2, 120, 120)
        opt = torch.randn(1, 12, 120, 120)
        aux = torch.randn(1, 5, 120, 120)
        out = model(sar=sar, optical=opt, aux=aux)
        assert out.shape == (1, 23, 120, 120)

    def test_indivisible_img_size_raises(self):
        enc = _FakeCromaEncoder(embed_dim=768)
        with pytest.raises(ValueError, match="divisible"):
            CromaSegmentationModel(
                encoder=enc, num_classes=23,
                img_size=125, patch_size=8, embed_dim=768,
            )


class TestBuildS2CromaTensor:
    def _make_inputs(self, h=8, w=8):
        spectral = np.stack([
            np.full((h, w), 102.0, dtype=np.float32),  # B02
            np.full((h, w), 103.0, dtype=np.float32),  # B03
            np.full((h, w), 104.0, dtype=np.float32),  # B04
            np.full((h, w), 108.5, dtype=np.float32),  # B8A
            np.full((h, w), 111.0, dtype=np.float32),  # B11
            np.full((h, w), 112.0, dtype=np.float32),  # B12
        ], axis=0)
        b08 = np.full((h, w), 108.0, dtype=np.float32)
        rededge = np.stack([
            np.full((h, w), 105.0, dtype=np.float32),  # B05
            np.full((h, w), 106.0, dtype=np.float32),  # B06
            np.full((h, w), 107.0, dtype=np.float32),  # B07
        ], axis=0)
        return spectral, b08, rededge

    def test_default_12band_order(self):
        spectral, b08, rededge = self._make_inputs()
        t = build_s2_croma_tensor(spectral, b08, rededge)  # B01, B09 padded
        # CROMA_S2_BAND_ORDER: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
        first_pixel = [float(t[i, 0, 0]) for i in range(12)]
        assert first_pixel == [
            0.0,    # B01 padded
            102.0,  # B02
            103.0,  # B03
            104.0,  # B04
            105.0,  # B05
            106.0,  # B06
            107.0,  # B07
            108.0,  # B08
            108.5,  # B8A
            0.0,    # B09 padded
            111.0,  # B11
            112.0,  # B12
        ]

    def test_with_b01_b09(self):
        spectral, b08, rededge = self._make_inputs()
        b01 = np.full((8, 8), 101.0, dtype=np.float32)
        b09 = np.full((8, 8), 109.0, dtype=np.float32)
        t = build_s2_croma_tensor(spectral, b08, rededge, b01=b01, b09=b09)
        assert float(t[0, 0, 0]) == 101.0
        assert float(t[9, 0, 0]) == 109.0

    def test_missing_rededge_raises(self):
        spectral, b08, _ = self._make_inputs()
        with pytest.raises(KeyError, match="rededge"):
            build_s2_croma_tensor(spectral, b08, rededge=None)

    def test_shape(self):
        spectral, b08, rededge = self._make_inputs(h=16, w=16)
        t = build_s2_croma_tensor(spectral, b08, rededge)
        assert t.shape == (12, 16, 16)

    def test_torch_path(self):
        spectral, b08, rededge = self._make_inputs()
        t = build_s2_croma_tensor(
            torch.from_numpy(spectral),
            torch.from_numpy(b08),
            torch.from_numpy(rededge),
        )
        assert isinstance(t, torch.Tensor)
        assert t.shape == (12, 8, 8)
