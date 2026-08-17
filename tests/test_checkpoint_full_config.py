"""Checkpoint FULL-config persistence round-trip (the permanent load_model fix).

Historically the trainer wrote a MINIMAL `config` dict into best_model.pt, so
every loader had to weight-infer img_size / patch / aux_fusion / the PSP pool
count — the root cause of the repeated load_model fixes (ea9e79c … d1d0f0a).
`_save_checkpoint` now records the FULL model-build parameters. This test
saves via the REAL method (on a lightweight stub trainer that supplies only
the attributes the method reads) and asserts the round-trip carries every
build key, additively (the pre-existing keys are untouched).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from imint.training.config import TrainingConfig
from imint.training.trainer import LULCTrainer


class _Spec:
    patch_size = 8  # CROMA/Clay-style patch


def _stub_trainer(cfg: TrainingConfig) -> LULCTrainer:
    """A LULCTrainer whose __init__ is bypassed; only the attributes read by
    _save_checkpoint are populated. Avoids building a real FM backbone."""
    t = LULCTrainer.__new__(LULCTrainer)
    t.config = cfg
    t.device = torch.device("cpu")
    t._registry_name = "croma_base"
    model = nn.Conv2d(3, cfg.num_classes, 1)
    model.fm_spec = _Spec()
    t.model = model
    return t


def test_full_config_round_trip(tmp_path):
    cfg = TrainingConfig(
        num_classes=28,
        img_size=504,
        decoder_channels=256,
        aux_fusion="concat",
        enable_tradslag_head=True,
        num_tradslag=4,
        enable_height_channel=True,
        enable_vpp_channels=True,
        enable_delta_sar_channels=True,
        enable_temporal_pooling=True,
        enable_multilevel_aux=True,
    )
    t = _stub_trainer(cfg)
    path = tmp_path / "best_model.pt"
    t._save_checkpoint(path, epoch=7, metrics={"mIoU": 0.5})

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    c = ckpt["config"]

    # Every FULL build parameter the task requires must be present.
    for key in (
        "backbone_name", "img_size", "patch_size", "num_classes",
        "n_aux_channels", "enable_tradslag_head", "num_tradslag",
        "aux_fusion", "decoder_channels", "pool_sizes", "enabled_aux_names",
    ):
        assert key in c, f"missing full-config key: {key}"

    # Values reflect the real build.
    assert c["backbone_name"] == "croma_base"
    assert c["img_size"] == 504
    assert c["patch_size"] == 8
    assert c["num_classes"] == 28
    assert c["aux_fusion"] == "concat"
    assert c["enable_tradslag_head"] is True
    assert c["num_tradslag"] == 4
    # enabled_aux_names must include the ΔSAR channels LAST + match n_aux.
    names = c["enabled_aux_names"]
    assert names[-2:] == ["delta_vv", "delta_vh"]
    assert c["n_aux_channels"] == len(names)
    # pool_sizes derived for img 504 / patch 8 (fm=63 → the 56+ bucket).
    assert isinstance(c["pool_sizes"], list) and len(c["pool_sizes"]) >= 4


def test_additive_old_keys_unchanged(tmp_path):
    """The pre-existing minimal keys are still present + unchanged in shape."""
    cfg = TrainingConfig(num_classes=23)
    t = _stub_trainer(cfg)
    path = tmp_path / "best_model.pt"
    t._save_checkpoint(path, epoch=1, metrics={"mIoU": 0.1})
    c = torch.load(path, map_location="cpu", weights_only=False)["config"]
    # Legacy keys a loader may already read.
    for key in ("backbone_name", "num_classes", "decoder_type",
                "decoder_channels", "feature_indices", "dropout",
                "n_aux_channels", "num_temporal_frames",
                "enable_multitemporal", "enable_tradslag_head", "num_tradslag"):
        assert key in c
    assert c["num_classes"] == 23
