#!/usr/bin/env python3
"""Run inference on viz tiles with multiple model checkpoints.

Loads each best_model.pt, runs on the 5 viz tiles, and outputs
a JSON file per model with base64-encoded colored predictions.

Usage (on K8s pod or locally with checkpoints):
    python scripts/inference_comparison.py \
        --viz-dir /data/viz_tiles \
        --checkpoints /checkpoints/unified_v5a/best_model.pt \
                      /checkpoints/unified_v5b/best_model.pt \
                      /checkpoints/unified_v5c/best_model.pt \
        --output-dir /data/viz_tiles/predictions
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re as _re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.unified_schema import UNIFIED_CLASSES

# 23-class RGB color palette
CLASS_COLORS = np.array([
    (0, 0, 0),         # 0  bakgrund
    (26, 92, 53),       # 1  tallskog
    (45, 138, 91),      # 2  granskog
    (123, 198, 126),    # 3  lövskog
    (77, 179, 128),     # 4  blandskog
    (107, 142, 90),     # 5  sumpskog
    (201, 223, 110),    # 6  tillfälligt ej skog
    (155, 119, 34),     # 7  våtmark
    (212, 180, 74),     # 8  öppen mark
    (192, 57, 43),      # 9  bebyggelse
    (36, 113, 163),     # 10 vatten
    (232, 184, 0),      # 11 vete
    (212, 120, 10),     # 12 korn
    (240, 208, 96),     # 13 havre
    (212, 198, 0),      # 14 oljeväxter
    (145, 200, 76),     # 15 slåttervall
    (184, 222, 134),    # 16 bete
    (155, 89, 182),     # 17 potatis
    (214, 51, 129),     # 18 sockerbetor
    (224, 112, 32),     # 19 trindsäd
    (139, 32, 32),      # 20 råg
    (220, 200, 0),      # 21 majs
    (0, 168, 198),      # 22 hygge
], dtype=np.uint8)


def pred_to_rgb(pred: np.ndarray) -> np.ndarray:
    """Convert (H, W) class indices to (H, W, 3) RGB."""
    return CLASS_COLORS[np.clip(pred, 0, len(CLASS_COLORS) - 1)]


def rgb_to_b64png(rgb: np.ndarray) -> str:
    """(H, W, 3) uint8 → base64 PNG string."""
    from PIL import Image
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def model_has_frac_head(model) -> bool:
    """True if the model carries a Trädslag fraction head, wherever it lives.

    Prithvi/Tessera attach ``frac_head`` on the top-level module; the routed
    families (clay/croma) nest it inside the ViTUPerNetHead as
    ``decoder_head.frac_head``. A top-level-only getattr misses the nested one
    and falsely reports "checkpoint has no fraction head" for a checkpoint
    that demonstrably carries frac weights.
    """
    if getattr(model, "frac_head", None) is not None:
        return True
    return getattr(getattr(model, "decoder_head", None),
                   "frac_head", None) is not None


def load_model(ckpt_path: str, device, backbone_name: str | None = None,
               img_size: int | None = None):
    """Load a segmentation model from checkpoint.

    Routes through the model registry (``imint.fm.registry.MODEL_CONFIGS``)
    so the correct backbone variant — including its ``patch_size`` —
    is read from the checkpoint's saved config rather than hardcoded.
    This is what makes Prithvi-600M (patch_size=14) work alongside
    Prithvi-300M (patch_size=16) without a code edit per run.

    Fallback chain for ``backbone_name``:
      1. ``ck_cfg["backbone_name"]`` — preferred (set by trainer since
         the registry refactor)
      2. ``LEGACY_BACKBONE_ALIAS`` mapping on ``ck_cfg["backbone"]`` —
         old TrainingConfig field
      3. Default ``"prithvi_300m"`` — pre-registry runs were 300M-only

    ``img_size``: optional runtime input resolution the model will actually
    receive (e.g. the validator's ``--img-size 504``). Prithvi recovers its
    img_size from ``pos_embed``, but the FM families that carry no pos_embed
    AND omit ``img_size`` in their minimal config (clay/croma) otherwise
    default to 224 — which fixes the WRONG grid_size / PSP pool count. When
    given and self-consistent with the checkpoint's PSP pool count, this is
    used verbatim so the head is built at the exact resolution inference
    feeds (grid_size + expected token count match the encoder output). None →
    the historical behaviour (pos_embed / config / pool-count reconciliation).
    """
    import torch
    from imint.fm.registry import (
        MODEL_CONFIGS, build_backbone, resolve_backbone_name,
    )
    from imint.fm.upernet import build_segmentation_from_spec
    from imint.training.config import TrainingConfig

    cfg = TrainingConfig()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ck_cfg = ck.get("config", {})
    num_frames = ck_cfg.get("num_temporal_frames", cfg.num_temporal_frames)
    # num_classes from the checkpoint (23 for v8b, 28 for the NMD2023 finetune),
    # so the head is built to match the saved weights rather than the default.
    n_classes = ck_cfg.get("num_classes", cfg.num_classes)

    # State-dict first so we can inspect shapes for backbone detection
    sd = {k.replace("model.", "", 1): v for k, v in
          ck.get("model_state_dict", ck.get("state_dict", {})).items()}
    pos_embed = sd.get("encoder.pos_embed")

    # n_aux_channels: infer from the aux branch's first conv rather than
    # trusting ck_cfg. The trainer's minimal best_model.pt config omits
    # n_aux_channels entirely (the clay/croma/terramind runs, and any run
    # that logged only enable_*_channel flags), so the config default of 11
    # builds an 11-channel LiDARBranch that CANNOT load a 10-aux checkpoint
    # (silent gated_fusions / lidar_branch drop under strict=False, which the
    # mismatch warning below would then flag). The aux branch's first conv is
    # Conv2d(n_aux, 32, 3): its in-channel dim IS n_aux_channels. Search both
    # the Prithvi wrapper key (`aux_branch.*` / `lidar_branch.net.0.conv`) and
    # the ViTUPerNetHead key (`decoder_head.lidar_branch.net.0.conv`).
    _naux = None
    for _k in sd:
        if _k.endswith("lidar_branch.net.0.conv.weight") and sd[_k].dim() == 4:
            _naux = sd[_k].shape[1]; break
    n_aux = _naux if _naux is not None else ck_cfg.get("n_aux_channels", 11)
    if _naux is not None and _naux != ck_cfg.get("n_aux_channels", 11):
        print(f"  [load_model] n_aux_channels inferred from checkpoint: "
              f"{_naux} (config default was "
              f"{ck_cfg.get('n_aux_channels', 11)})")

    # Resolve backbone_name. Fallback chain:
    #  (1) ck_cfg["backbone_name"] — set by trainer since registry refactor
    #  (2) ck_cfg["backbone"]      — legacy field
    #  (3) infer from pos_embed embed_dim:
    #        1024 → prithvi_300m  (depth 24)
    #        1280 → prithvi_600m  (depth 32)
    #      Trainer's best_model.pt config is minimal and omits backbone_name
    #      (trainer.py:1000-1009), so (1)+(2) are None for any checkpoint
    #      saved before that gets fixed. Embed-dim inference is robust as
    #      long as the checkpoint actually loaded a registered backbone.
    # An explicit override wins over the ck_cfg chain. Required for
    # non-Prithvi families (e.g. tessera_v1): their checkpoints have no
    # pos_embed to infer from and the trainer's minimal config omits
    # backbone_name, so neither the ck_cfg lookup nor the embed-dim
    # inference below can recover it.
    if backbone_name is not None:
        backbone_name = resolve_backbone_name(backbone_name, None)
    else:
        backbone_name = resolve_backbone_name(
            ck_cfg.get("backbone_name"), ck_cfg.get("backbone"),
        )
    if backbone_name == "prithvi_300m" and pos_embed is not None:
        # only trust the default if we can't infer otherwise
        embed_dim = pos_embed.shape[-1]
        inferred = None
        for name, spec in MODEL_CONFIGS.items():
            if spec.family == "prithvi" and spec.embed_dim == embed_dim:
                inferred = name
                break
        if inferred and inferred != backbone_name:
            print(f"  [load_model] backbone_name absent in ck.config; "
                  f"inferred from pos_embed embed_dim={embed_dim} → {inferred}")
            backbone_name = inferred

    spec = MODEL_CONFIGS[backbone_name]
    patch_size = spec.patch_size  # 16 for 300M, 14 for 600M, etc.

    # Infer img_size from pos_embed grid using the CORRECT patch_size
    # for this backbone — the previous hardcoded `* 16` silently produced
    # img_size=576 for Prithvi-600M (grid 36) instead of the true 504.
    if pos_embed is not None:
        n_tokens = pos_embed.shape[1] - 1  # subtract CLS token
        n_spatial = n_tokens // max(num_frames, 1)
        grid_size = int(n_spatial ** 0.5)
        img_size = grid_size * patch_size
    elif img_size is not None:
        # Caller passed the runtime resolution (e.g. validator --img-size 504).
        # For the FM families with no pos_embed, this is the authoritative
        # img_size — it fixes grid_size / expected token count to the exact
        # resolution inference feeds, not a config default. The pool-count
        # reconciliation below still guards it against the checkpoint.
        pass
    else:
        img_size = ck_cfg.get("img_size", cfg.img_size)

    # PSP pool-count reconciliation — the decoder head's PSP module count is a
    # deterministic function of `img_size`+`patch_size` via get_default_pool_
    # sizes. The head is built from that img_size, so a wrong img_size builds
    # the WRONG number of PSP pools, and the PSP bottleneck's in-channels
    # (C_deep + decoder_channels × N_pools) then mismatches the checkpoint —
    # an unrecoverable size-mismatch on load, not a droppable key.
    #
    # This bit the FM families that carry NO pos_embed AND no `img_size` in
    # their minimal config (clay/croma): img_size defaulted to 224, which at
    # patch=8 yields 5 pools, while the trainer built at img=504 → 6 pools
    # (bottleneck 256×2560 vs the rebuilt 256×2304 — exactly one pool branch).
    #
    # Self-describing fix (same discipline as decoder_channels / aux_fusion /
    # n_aux above): recover the pool COUNT from the checkpoint's
    # `decoder.psp_modules.N.*` indices — robust and exact — then, if the
    # provisional img_size does not reproduce that count, correct img_size to
    # the smallest patch-multiple that DOES. Because get_default_pool_sizes
    # returns a single tuple per (count, patch) band for the FM patch sizes,
    # matching the count also fixes the pool GRID SIZES — which are load-
    # bearing for inference correctness (each PSP AdaptiveAvgPool2d pools the
    # deepest feature map to pool_size², and the learned 256→256 conv was
    # trained on that specific grid; the conv weights are pool_size-agnostic
    # so a wrong grid loads cleanly but feeds the conv an off-distribution
    # input). Prithvi/Tessera are untouched: their provisional img_size (from
    # pos_embed / a present config) already reproduces the checkpoint count,
    # so the correction is a no-op for them.
    from imint.fm.upernet import get_default_pool_sizes as _pool_sizes
    _psp_idxs = set()
    for _k in sd:
        _m = _re.search(r"decoder\.psp_modules\.(\d+)\.", _k)
        if _m:
            _psp_idxs.add(int(_m.group(1)))
    _n_pools_ckpt = (max(_psp_idxs) + 1) if _psp_idxs else 0
    if _n_pools_ckpt:
        _built = len(_pool_sizes(device=device, img_size=img_size,
                                 patch_size=patch_size))
        if _built != _n_pools_ckpt:
            _fixed = None
            for _cand in range(patch_size, 1024 + 1, patch_size):
                if len(_pool_sizes(device=device, img_size=_cand,
                                   patch_size=patch_size)) == _n_pools_ckpt:
                    _fixed = _cand
                    break
            if _fixed is None:
                raise ValueError(
                    f"[load_model] checkpoint has {_n_pools_ckpt} PSP pools but "
                    f"no img_size (patch={patch_size}) reproduces that count — "
                    f"cannot rebuild the head to match the checkpoint."
                )
            print(f"  [load_model] PSP pool count mismatch: provisional "
                  f"img_size={img_size} → {_built} pools, checkpoint has "
                  f"{_n_pools_ckpt}; corrected img_size to {_fixed} "
                  f"(pool_sizes={_pool_sizes(device=device, img_size=_fixed, patch_size=patch_size)})")
            img_size = _fixed

    # build_backbone returns (encoder, spec) — use the spec we already
    # resolved so feature_indices / embed_dim / etc. are consistent.
    backbone, _ = build_backbone(
        backbone_name, num_frames=num_frames, img_size=img_size,
        pretrained=False,
    )

    # decoder_channels: infer from the checkpoint's head weights rather than
    # trusting ck_cfg. The logged config can disagree with the model that was
    # actually built (e.g. the tessera_gated run logs 256 but its head/frac_head
    # are 128-wide), and every decoder conv keys off this value — a wrong width
    # is an unrecoverable shape-mismatch on load, not a droppable key. The head
    # classifier and frac_head are Conv2d(decoder_channels, ...), so their
    # in-channel dim IS decoder_channels.
    _dc = None
    for _k in sd:
        if _k.endswith("frac_head.weight") and sd[_k].dim() == 4:
            _dc = sd[_k].shape[1]; break
    if _dc is None:
        for _k in sd:
            if _k.endswith("classifier.weight") and sd[_k].dim() == 4:
                _dc = sd[_k].shape[1]; break
    decoder_channels = _dc if _dc is not None else cfg.decoder_channels
    if _dc is not None and _dc != cfg.decoder_channels:
        print(f"  [load_model] decoder_channels inferred from checkpoint: "
              f"{_dc} (config default was {cfg.decoder_channels})")

    model = build_segmentation_from_spec(
        spec,
        encoder=backbone,
        num_classes=n_classes,
        img_size=img_size,
        decoder_channels=decoder_channels,
        dropout=getattr(cfg, "dropout", 0.1),
        n_aux_channels=n_aux,
        enable_temporal_pooling=cfg.enable_temporal_pooling,
        enable_multilevel_aux=cfg.enable_multilevel_aux,
        # Fraction head: the trainer persists these in the checkpoint config —
        # without threading them the frac_head weights are silently dropped by
        # strict=False and --use-fraction-head cannot run.
        enable_tradslag_head=ck_cfg.get("enable_tradslag_head", False),
        num_tradslag=ck_cfg.get("num_tradslag", 4),
        # Aux fusion ("concat" | "gated"): detect from the weights, not the
        # config. best_model.pt's embedded config is minimal and can omit
        # aux_fusion (the tessera_gated run does), so a config-only read
        # defaults to concat and builds a 256-wide head that can't accept the
        # gated checkpoint's 128-wide one. The gated path is self-identifying:
        # it carries gated_fusion(s).* modules. Fall back to config/concat when
        # no such keys exist.
        aux_fusion=("gated" if any("gated_fusion" in k for k in sd)
                    else ck_cfg.get("aux_fusion", "concat")),
        device=device,
    )
    incompat = model.load_state_dict(sd, strict=False)
    # Surface load mismatches: with strict=False a mismatched architecture
    # (e.g. gated checkpoint into a concat model) loads "successfully" but
    # drops weights silently. Warn loudly on anything beyond the expected
    # backbone-buffer misses so a bad load can't masquerade as a valid run.
    _miss = [k for k in incompat.missing_keys if "encoder." not in k]
    if _miss or incompat.unexpected_keys:
        print(f"  [load_model] WARN state_dict mismatch — "
              f"missing(non-encoder)={len(_miss)}, "
              f"unexpected={len(incompat.unexpected_keys)}")
        for k in (_miss[:8] + list(incompat.unexpected_keys)[:8]):
            print(f"    · {k}")
    model = model.to(device).eval()
    # Stash the spec so the inference input builder can route on family
    # (tessera reads the pre-baked embedding; Prithvi reads reflectance).
    model.fm_spec = spec
    # Stash the temporal-frame count so the inference input builder can
    # mirror training's single-frame selection for num_temporal_frames=1
    # checkpoints (e.g. Prithvi-300M) instead of feeding the full 4-frame
    # stack (which 4x:es the token grid and breaks the pos_embed add).
    model.num_frames = num_frames

    epoch = ck.get("epoch", "?")
    miou = ck.get("metrics", {}).get("miou", "?")
    return model, epoch, miou, img_size


# Families whose forward is routed through imint.fm.forward_router.family_forward
# (multi-modal / dict / dynamic-embedding signatures that the direct
# model(img5d, aux=, temporal_coords=, location_coords=) call cannot express).
# Prithvi/Tessera keep the direct call — byte-unchanged.
_ROUTED_FAMILIES = ("clay", "croma", "terramind")

# Registry name each routed family emits its per-model tensors under. Mirrors
# scripts/train_unified.py: model_keys = (registry_name,) when the active
# backbone is one of the multi-modal members. Used so validation builds the
# SAME dataset stacks (s2_clay / s2_croma / s2_terramind / s1_vv_vh) the
# trainer fed, via the SAME UnifiedDataset._build_model_specific_tensors.
_FAMILY_MODEL_KEY = {
    "clay": "clay_v1_5",
    "croma": "croma_base",
    "terramind": "terramind_v1_base",
}


def _tile_source(tile_path) -> str:
    """LULC vs crop source string for the dataset's best-frame selector.

    Mirrors the single-frame path's ``stem.startswith("crop_")`` convention:
    crop_* tiles → "crop" (frame-1 / seasons_valid selection), everything
    else → "lulc" (peak-summer DOY selection). The routed families read a
    single best frame, so the source string must match training's choice.
    """
    return "crop" if Path(str(tile_path)).stem.startswith("crop_") else "lulc"


def _build_routed_batch(data, tile_path, device, img_size, family):
    """Build the family's model_keys tensors + centre-crop them to img_size.

    Reuses ``UnifiedDataset._build_model_specific_tensors`` (the exact builder
    the trainer uses) so the emitted stacks — s2_clay / s2_croma /
    s2_terramind / s1_vv_vh — are bit-identical to training. Then applies the
    SAME centre-crop the Prithvi/Tessera paths use, so the routed families see
    inputs cropped identically to the hard-head validation.

    Returns ``(batch, y0, x0, crop_sz)`` where ``batch`` carries (1, C, cs, cs)
    tensors keyed as ``family_forward`` expects.

    Fail-loud contract (mirrors unified_dataset): CROMA/TerraMind require the
    v2 season-composite S1 (``s1_enrich_v==2``); a v1 leftover raises inside
    ``_build_model_specific_tensors`` rather than silently feeding a
    mis-composited SAR stack. Clay is optical-only and needs no S1.
    """
    import torch
    from imint.training.unified_dataset import UnifiedDataset

    # Bare instance carrying only the state _build_model_specific_tensors
    # reads: model_keys (+ the lazily-created _band_miss_logged cache). Avoids
    # UnifiedDataset.__init__'s tile-directory discovery / I/O — we already
    # hold the loaded npz.
    builder = object.__new__(UnifiedDataset)
    builder.model_keys = (_FAMILY_MODEL_KEY[family],)
    source = _tile_source(tile_path)

    stacks = builder._build_model_specific_tensors(data, source)  # {key: (C,H,W)}

    # Centre-crop every stack to img_size with the SAME offset math the
    # reflectance path uses (crop_sz = min(img_size, H, W); TL-centred).
    first = next(iter(stacks.values()))
    _, h, w = first.shape
    crop_sz = min(img_size, h, w)
    y0 = (h - crop_sz) // 2
    x0 = (w - crop_sz) // 2

    batch = {}
    for key, arr in stacks.items():
        cropped = arr[:, y0:y0 + crop_sz, x0:x0 + crop_sz]
        batch[key] = torch.from_numpy(
            np.ascontiguousarray(cropped)
        ).unsqueeze(0).to(device)
    return batch, y0, x0, crop_sz


def _build_inference_inputs(tile_path, device, img_size, aux_channel_names,
                            family="prithvi", num_frames=None):
    """Build (img5d, aux, temporal_coords, location_coords, crop meta) for a tile.

    Shared preprocessing for both the class-head and fraction-head inference
    paths — identical spectral normalization, aux stacking, centre-crop and TL
    coordinate construction, so the fraction head sees exactly the inputs the
    hard head was validated on. Returns a dict of the built tensors plus the
    crop offsets ``(y0, x0, crop_sz)`` and the raw ``data`` handle.

    ``family`` routes the image tensor:

      * ``"tessera"`` reads the pre-baked ``tessera`` embedding (128, H, W)
        with NO Prithvi normalization (it ships normalized) and NO
        temporal/location coords (annual, single-frame).
      * ``"clay"`` / ``"croma"`` / ``"terramind"`` build the per-model stacks
        (s2_clay / s2_croma / s2_terramind / s1_vv_vh) via the dataset's own
        ``_build_model_specific_tensors`` — the SAME builder the trainer used —
        and return them under the ``batch`` key for ``family_forward`` routing.
        These families still receive location coords (Clay uses them; the
        others ignore them). No ``img5d`` is built for them.
      * any other family (``"prithvi"``) reads Sentinel-2 reflectance and
        z-scores it into the 5D Conv3d layout.

    The ``img5d`` key holds a 5D tensor for Prithvi and the 4D (1, 128, H, W)
    embedding for tessera — the model's forward consumes each family's native
    rank. Routed families carry ``img5d=None`` and a populated ``batch``.
    """
    import torch
    from imint.training.unified_dataset import (
        PRITHVI_MEAN, PRITHVI_STD, N_BANDS,
        AUX_CHANNEL_NAMES, normalize_aux_channel,
    )
    aux_names = aux_channel_names if aux_channel_names is not None else AUX_CHANNEL_NAMES

    data = np.load(tile_path, allow_pickle=True)

    routed = family in _ROUTED_FAMILIES
    img5d = None
    batch = None

    if routed:
        # Multi-modal / dynamic-embedding families: build the dataset's
        # per-model stacks and crop them; forward runs via family_forward.
        batch, y0, x0, crop_sz = _build_routed_batch(
            data, tile_path, device, img_size, family)
        spectral = None
        single_frame = False
        n_frames = 1
    elif family == "tessera":
        # Pre-baked (128, H, W) embedding — already normalized on the TESSERA
        # cluster; skip the reflectance z-score entirely.
        emb = np.asarray(data["tessera"], dtype=np.float32)
        _, h, w = emb.shape
        crop_sz = min(img_size, h, w)
        y0 = (h - crop_sz) // 2
        x0 = (w - crop_sz) // 2
        emb = emb[:, y0:y0+crop_sz, x0:x0+crop_sz]
        img5d = torch.from_numpy(emb).unsqueeze(0).to(device)  # (1, 128, H, W)
    else:
        spectral = data.get("spectral", data.get("image")).astype(np.float32)

        # Single-frame checkpoints (num_temporal_frames=1, e.g. Prithvi-300M)
        # must see the SAME one frame training selected — feeding the tile's
        # full 4-frame stack builds a 4x token grid the model can't add its
        # pos_embed to. Reuse the dataset's own selectors verbatim (peak-summer
        # DOY for lulc tiles, frame 1 for crop_* tiles) so the choice is
        # bit-identical to training rather than a re-implementation.
        single_frame = (num_frames == 1
                        and spectral.shape[0] // N_BANDS > 1)
        if single_frame:
            from imint.training.unified_dataset import UnifiedDataset
            stem = Path(str(tile_path)).stem
            if stem.startswith("crop_"):
                spectral = UnifiedDataset._extract_crop_frame(data)
            else:
                spectral = UnifiedDataset._extract_lulc_frame(data)

        # Normalize: reflectance → DN → Prithvi z-score
        n_frames = spectral.shape[0] // N_BANDS
        mean = np.tile(PRITHVI_MEAN.reshape(N_BANDS, 1, 1), (n_frames, 1, 1))
        std = np.tile(PRITHVI_STD.reshape(N_BANDS, 1, 1), (n_frames, 1, 1))
        spectral = (spectral * 10000.0 - mean) / std

        # Center crop to img_size (no crop if tile == img_size)
        _, h, w = spectral.shape
        crop_sz = min(img_size, h, w)
        y0 = (h - crop_sz) // 2
        x0 = (w - crop_sz) // 2
        spectral = spectral[:, y0:y0+crop_sz, x0:x0+crop_sz]

    # Aux channels
    aux_list = []
    for ch_name in aux_names:
        if ch_name in data:
            arr = data[ch_name].astype(np.float32)
            arr = arr[y0:y0+crop_sz, x0:x0+crop_sz]
            arr = normalize_aux_channel(ch_name, arr)
            aux_list.append(arr[np.newaxis])
        else:
            aux_list.append(np.zeros((1, crop_sz, crop_sz), dtype=np.float32))

    aux = torch.from_numpy(np.concatenate(aux_list, axis=0)).unsqueeze(0).to(device)

    temporal_coords = None
    location_coords = None

    if routed:
        # Routed families carry no temporal_coords (single-date / annual).
        # Location coords ARE built — Clay's dynamic embedding uses them
        # (CROMA/TerraMind ignore them in family_forward). Same SWEREF→WGS84
        # transform as the Prithvi path so lat/lon are identical.
        from imint.training.sampler import _sweref99_to_wgs84
        easting = float(data.get("easting", 500_000))
        northing = float(data.get("northing", 6_500_000))
        lat, lon = _sweref99_to_wgs84(easting, northing)
        location_coords = torch.from_numpy(
            np.array([[lat, lon]], dtype=np.float32)
        ).to(device)
    elif family != "tessera":
        # Prithvi image tensor + TL coords. Tessera already built img5d above
        # and has no per-frame/location coords (annual embedding).
        img = torch.from_numpy(spectral).unsqueeze(0).to(device)
        T = img.shape[1] // 6
        img5d = img.view(1, T, 6, crop_sz, crop_sz).permute(0, 2, 1, 3, 4)

        doy = data.get("doy")
        if doy is not None:
            from imint.training.sampler import _sweref99_to_wgs84
            year = int(data.get("year", data.get("lpis_year", 2022)))
            tc = np.zeros((n_frames, 2), dtype=np.float32)
            tc[:, 0] = float(year)
            if not single_frame:
                # Multitemporal: per-frame DOY, as trained.
                tc[:len(doy), 1] = doy[:n_frames].astype(np.float32)
            # Single-frame mirrors training's non-multitemporal path, which
            # builds coords with doy=None → [[year, 0]] (unified_dataset
            # ~L582/L834). Leaving tc[:,1]=0 here matches it exactly.
            temporal_coords = torch.from_numpy(tc).unsqueeze(0).to(device)

            easting = float(data.get("easting", 500_000))
            northing = float(data.get("northing", 6_500_000))
            lat, lon = _sweref99_to_wgs84(easting, northing)
            location_coords = torch.from_numpy(
                np.array([[lat, lon]], dtype=np.float32)
            ).to(device)

    return {
        "img5d": img5d, "aux": aux, "batch": batch, "family": family,
        "temporal_coords": temporal_coords, "location_coords": location_coords,
        "y0": y0, "x0": x0, "crop_sz": crop_sz, "data": data,
        "aux_names": aux_names,
    }


def _forward_from_inputs(model, inp, device, *, return_fractions=False):
    """Run the model on a built ``_build_inference_inputs`` dict.

    Routes on family: prithvi/tessera call the model directly with the
    (img5d, aux, temporal_coords, location_coords) signature they were
    validated on — byte-unchanged. clay/croma/terramind route through
    ``imint.fm.forward_router.family_forward`` (the SAME router the trainer
    and evaluate loop use), building the batch dict the router expects from
    the pre-cropped per-model stacks.

    Returns the model's raw return: ``logits`` or ``(logits, frac_logits)``
    when ``return_fractions`` is True.
    """
    family = inp["family"]
    if family in _ROUTED_FAMILIES:
        from imint.fm.forward_router import family_forward
        return family_forward(
            model, family, inp["batch"], device,
            aux=inp["aux"],
            temporal_coords=inp["temporal_coords"],
            location_coords=inp["location_coords"],
            return_fractions=return_fractions,
        )
    return model(
        inp["img5d"], aux=inp["aux"],
        temporal_coords=inp["temporal_coords"],
        location_coords=inp["location_coords"],
        return_fractions=return_fractions,
    )


def run_fraction_inference(model, tile_path: str, device, img_size: int = 224,
                           aux_channel_names=None):
    """Run the FRACTION head on a single tile → (4, cs, cs) sigmoid fractions.

    Uses the same preprocessing/crop as ``run_inference`` and calls the model
    with ``return_fractions=True``. Returns the per-species crown-cover in
    [0, 1] (order tall/gran/trivial/adel). Raises if the model has no fraction
    head (build it with ``enable_tradslag_head=True``).
    """
    import torch
    if getattr(model, "frac_head", None) is None:
        raise ValueError("model has no fraction head (enable_tradslag_head=False)")
    family = getattr(getattr(model, "fm_spec", None), "family", "prithvi")
    inp = _build_inference_inputs(
        tile_path, device, img_size, aux_channel_names, family=family,
        num_frames=getattr(model, "num_frames", None))
    with torch.no_grad():
        _logits, frac_logits = _forward_from_inputs(
            model, inp, device, return_fractions=True)
        fracs = torch.sigmoid(frac_logits).squeeze(0).cpu().numpy()  # (4, cs, cs)
    return fracs


def run_inference(model, tile_path: str, device, img_size: int = 224,
                  return_probs: bool = False, aux_channel_names=None):
    """Run inference on a single tile.

    Returns (H, W) prediction, or ((C, H, W) softmax, (B, H, W) spectral_raw, (N, H, W) aux)
    when return_probs=True (for superpixel refinement).

    ``aux_channel_names`` overrides the aux stack (default: the canonical 10).
    Pass the 11-channel list (…+"markfukt") for a wetness-aux checkpoint so the
    fed aux count matches the model's input embedding.
    """
    import torch
    family = getattr(getattr(model, "fm_spec", None), "family", "prithvi")
    inp = _build_inference_inputs(
        tile_path, device, img_size, aux_channel_names, family=family,
        num_frames=getattr(model, "num_frames", None))
    y0 = inp["y0"]; x0 = inp["x0"]; crop_sz = inp["crop_sz"]
    data = inp["data"]; aux_names = inp["aux_names"]

    with torch.no_grad():
        logits = _forward_from_inputs(model, inp, device)
        if return_probs:
            import torch.nn.functional as F
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (C, H, W)
            # Return raw spectral (before normalization) + aux for superpixel generation
            raw_spectral = data.get("spectral", data.get("image")).astype(np.float32)
            _, h_full, w_full = raw_spectral.shape
            raw_spectral = raw_spectral[:, y0:y0+crop_sz, x0:x0+crop_sz]
            # Collect raw aux
            raw_aux_list = []
            for ch_name in aux_names:
                if ch_name in data:
                    a = data[ch_name].astype(np.float32)[y0:y0+crop_sz, x0:x0+crop_sz]
                    raw_aux_list.append(a[np.newaxis])
            raw_aux = np.concatenate(raw_aux_list, axis=0) if raw_aux_list else None
            return probs, raw_spectral, raw_aux
        pred = logits.argmax(1).squeeze(0).cpu().numpy()
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz-dir", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", help="Labels for each checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--superpixel", action="store_true",
                        help="Apply SLIC superpixel refinement")
    parser.add_argument("--superpixel-segments", type=int, default=500)
    parser.add_argument("--guided-filter", action="store_true",
                        help="Apply guided filter refinement (pixel-level, spectral edge transfer)")
    parser.add_argument("--gf-radius", type=int, default=2,
                        help="Guided filter radius (2=fine, 4=moderate)")
    parser.add_argument("--gf-eps", type=float, default=0.01,
                        help="Guided filter eps (0.001=sharp, 0.01=balanced)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Apply morphological cleanup (remove < MMU)")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    viz_dir = Path(args.viz_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = sorted(viz_dir.glob("*.npz"))
    if not tiles:
        print(f"ERROR: no .npz files in {viz_dir}")
        sys.exit(1)
    print(f"Tiles: {len(tiles)}")

    labels = args.labels or [Path(c).parent.name for c in args.checkpoints]

    for ckpt_path, label in zip(args.checkpoints, labels):
        print(f"\n=== {label}: {ckpt_path} ===")
        model, epoch, miou, model_img_size = load_model(ckpt_path, device)
        print(f"  Loaded: epoch={epoch}, miou={miou}, img_size={model_img_size}")

        suffix = ""
        if args.superpixel:
            suffix += "_sp"
        if args.guided_filter:
            suffix += "_gf"
        if args.cleanup:
            suffix += "_clean"

        result = {"_label": label + suffix, "_epoch": str(epoch), "_miou": str(miou)}
        for tile_path in tiles:
            name = tile_path.stem
            print(f"  {name}...", end=" ", flush=True)

            if args.superpixel or args.guided_filter:
                from imint.inference.superpixel_refine import (
                    superpixel_refine, guided_filter_refine, morphological_cleanup,
                )
                probs, raw_spec, raw_aux = run_inference(
                    model, str(tile_path), device,
                    img_size=model_img_size, return_probs=True,
                )
                if args.guided_filter:
                    pred = guided_filter_refine(
                        probs, raw_spec,
                        radius=args.gf_radius, eps=args.gf_eps,
                    )
                elif args.superpixel:
                    pred = superpixel_refine(
                        probs, raw_spec, aux=raw_aux,
                        n_segments=args.superpixel_segments,
                    )
                if args.cleanup:
                    pred = morphological_cleanup(pred, min_pixels=25)
            else:
                pred = run_inference(model, str(tile_path), device, img_size=model_img_size)
                if args.cleanup:
                    from imint.inference.superpixel_refine import morphological_cleanup
                    pred = morphological_cleanup(pred, min_pixels=25)
            rgb = pred_to_rgb(pred)
            result[f"{name}_pred"] = rgb_to_b64png(rgb)
            result[f"{name}_shape"] = list(rgb.shape[:2])
            print(f"done ({pred.shape})")

        out_path = out_dir / f"{label}_predictions.json"
        with open(out_path, "w") as f:
            json.dump(result, f)
        print(f"  Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
