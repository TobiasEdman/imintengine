"""
imint/fm/loaders/tessera.py — TESSERA "loader" (no encoder)

TESSERA is unlike other foundation models in our registry: there is no
encoder to call at train/inference time. It publishes **pre-computed
annual 128-D per-pixel Sentinel-1/2 embeddings** that we bake into each
tile's .npz via ``scripts/enrich_tiles_tessera.py``. The "loader" here
just returns an identity passthrough — the real "encoder" ran once on
the TESSERA authors' HPC cluster.

At train time, the model consumes the ``tessera`` key from each tile
(shape ``(128, H, W)`` float16) and runs a light segmentation head
(2-3 conv layers + classifier). See ``imint.fm.tessera_seg`` for the
head.

Why this is in the registry at all: uniform interface for the ensemble
code path. ``build_backbone("tessera_v1")`` → (IdentityEncoder, spec);
``build_segmentation_from_spec(spec, ...)`` → TesseraSegmentationModel.
"""
from __future__ import annotations

import torch.nn as nn


class _TesseraPassthrough(nn.Module):
    """No-op 'encoder' — embeddings are already computed on disk.

    Has a trivial learnable scale so standard optimizers have something
    to optimize and parameter-counting code doesn't crash.
    """

    def __init__(self):
        super().__init__()
        import torch
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, embeddings):
        # embeddings shape: (B, 128, H, W) float32 (or fp16 upcast)
        return embeddings * self.scale


def load_tessera(
    pretrained: bool = True,
    num_frames: int = 1,
    img_size: int = 512,
    **kwargs,
):
    """Return a passthrough 'encoder' for TESSERA embeddings.

    Args:
        pretrained: Ignored (TESSERA has no weights to load at our
            train time — the real pretraining is already baked into
            the on-disk embeddings).
        num_frames: Must be 1. TESSERA embeddings are annual, not
            per-frame.
        img_size: Informational only; the passthrough operates on
            whatever shape the dataset emits.

    Returns:
        ``_TesseraPassthrough`` module.
    """
    if num_frames != 1:
        raise ValueError(
            f"TESSERA embeddings are annual (num_frames=1), got {num_frames}."
        )
    return _TesseraPassthrough()
