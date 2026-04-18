"""Tests for imint.fm.registry — ModelSpec + build_backbone + legacy aliases."""
from __future__ import annotations

import pytest

from imint.fm.registry import (
    LEGACY_BACKBONE_ALIAS,
    MODEL_CONFIGS,
    ModelSpec,
    build_backbone,
    default_feature_indices,
    resolve_backbone_name,
)


class TestDefaultFeatureIndices:
    def test_prithvi_300m_depth_24(self):
        assert default_feature_indices(24, 4) == (5, 11, 17, 23)

    def test_prithvi_600m_depth_32(self):
        assert default_feature_indices(32, 4) == (7, 15, 23, 31)

    def test_terramind_depth_12(self):
        assert default_feature_indices(12, 4) == (2, 5, 8, 11)


class TestRegistrySpecs:
    @pytest.mark.parametrize("name", sorted(MODEL_CONFIGS.keys()))
    def test_spec_is_frozen_dataclass(self, name):
        spec = MODEL_CONFIGS[name]
        assert isinstance(spec, ModelSpec)
        with pytest.raises(Exception):
            spec.name = "other"  # frozen=True → attribute error

    @pytest.mark.parametrize("name", sorted(MODEL_CONFIGS.keys()))
    def test_feature_indices_valid(self, name):
        spec = MODEL_CONFIGS[name]
        assert len(spec.feature_indices) == 4, \
            f"{name}: expected 4 feature indices for UPerNet"
        assert max(spec.feature_indices) < spec.depth, \
            f"{name}: feature index {max(spec.feature_indices)} >= depth {spec.depth}"
        assert min(spec.feature_indices) >= 0

    @pytest.mark.parametrize("name", sorted(MODEL_CONFIGS.keys()))
    def test_patch_size_supported(self, name):
        spec = MODEL_CONFIGS[name]
        # Known patch sizes across our registered FMs: 8 (Clay/CROMA),
        # 14 (Prithvi-600M), 16 (Prithvi-300M/TerraMind/THOR-default).
        assert spec.patch_size in (8, 14, 16), \
            f"{name}: unexpected patch_size {spec.patch_size}"

    @pytest.mark.parametrize("name", sorted(MODEL_CONFIGS.keys()))
    def test_input_bands_nonempty(self, name):
        spec = MODEL_CONFIGS[name]
        assert len(spec.input_bands) >= 1

    @pytest.mark.parametrize("name", sorted(MODEL_CONFIGS.keys()))
    def test_normalizer_family_wired(self, name):
        from imint.fm.normalize import NORMALIZERS
        spec = MODEL_CONFIGS[name]
        assert spec.normalizer_family in NORMALIZERS, \
            f"{name}: normalizer_family {spec.normalizer_family!r} not registered"

    @pytest.mark.parametrize("name", sorted(MODEL_CONFIGS.keys()))
    def test_native_num_frames_nonempty(self, name):
        spec = MODEL_CONFIGS[name]
        assert len(spec.native_num_frames) >= 1
        assert 1 in spec.native_num_frames, \
            f"{name}: must support num_frames=1 for single-date fallback"

    def test_temporal_consistency(self):
        # supports_temporal=True ⇒ native_num_frames includes >1
        for name, spec in MODEL_CONFIGS.items():
            if spec.supports_temporal:
                assert max(spec.native_num_frames) > 1, \
                    f"{name}: supports_temporal=True but only num_frames={spec.native_num_frames}"


class TestResolveBackboneName:
    def test_default_is_prithvi_300m(self):
        assert resolve_backbone_name(None, None) == "prithvi_300m"

    def test_new_name_takes_priority(self):
        assert resolve_backbone_name("prithvi_600m", "prithvi_eo_v2_300m_tl") == "prithvi_600m"

    def test_legacy_alias_map_300m(self):
        assert resolve_backbone_name(None, "prithvi_eo_v2_300m_tl") == "prithvi_300m"

    def test_legacy_alias_map_600m(self):
        assert resolve_backbone_name(None, "prithvi_eo_v2_600m_tl") == "prithvi_600m"

    def test_direct_registry_key_in_legacy_field(self):
        # If someone sets backbone="prithvi_300m" (not via alias), accept it.
        assert resolve_backbone_name(None, "prithvi_300m") == "prithvi_300m"

    def test_unknown_new_name_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone_name"):
            resolve_backbone_name("does_not_exist")


class TestLegacyAliasCoverage:
    def test_every_legacy_alias_points_to_valid_key(self):
        for legacy, registry_key in LEGACY_BACKBONE_ALIAS.items():
            assert registry_key in MODEL_CONFIGS, \
                f"Legacy alias {legacy!r} → {registry_key!r} not in MODEL_CONFIGS"


class TestBuildBackbone:
    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_backbone("not_a_model")

    def test_invalid_num_frames_raises(self):
        # TerraMind only supports num_frames=1
        with pytest.raises(ValueError, match="supports num_frames"):
            build_backbone("terramind_v1_base", num_frames=4, pretrained=False)

    def test_prithvi_300m_builds_without_pretrained(self):
        model, spec = build_backbone(
            "prithvi_300m", num_frames=1, img_size=224, pretrained=False,
        )
        assert spec.name == "prithvi_300m"
        assert spec.embed_dim == 1024
        # Param count: ~303M for Prithvi-300M (encoder+decoder MAE).
        n = sum(p.numel() for p in model.parameters())
        assert 300_000_000 < n < 350_000_000, f"got {n:,} params"

    def test_prithvi_600m_builds_without_pretrained(self):
        # 600M uses patch_size=14, so img_size must be a multiple of 14.
        # 448 = 14 × 32 (native training resolution for Prithvi at 448).
        model, spec = build_backbone(
            "prithvi_600m", num_frames=1, img_size=448, pretrained=False,
        )
        assert spec.name == "prithvi_600m"
        assert spec.embed_dim == 1280
        assert spec.depth == 32
        assert spec.patch_size == 14
        n = sum(p.numel() for p in model.parameters())
        assert 600_000_000 < n < 700_000_000, f"got {n:,} params (~646M expected)"

    def test_prithvi_600m_multitemporal(self):
        # Prithvi family must support 1-4 frames
        model, spec = build_backbone(
            "prithvi_600m", num_frames=4, img_size=448, pretrained=False,
        )
        assert 4 in spec.native_num_frames

    def test_stub_loaders_raise_notimplemented(self):
        # Clay and CROMA are stubs
        with pytest.raises(NotImplementedError):
            build_backbone("clay_v1_5", num_frames=1, pretrained=False)
        with pytest.raises(NotImplementedError):
            build_backbone("croma_base", num_frames=1, pretrained=False)
