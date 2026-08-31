#!/usr/bin/env python3
"""scripts/extract_plot_features.py — dump pre-classifier features at NFI plots.

Hybrid-experiment step 1. Forward a NMD2023-pretrained UPerNet-on-Prithvi seg
model over each tile that carries NFI plots, hook the 256-dim feature map that
FEEDS the final 1×1 classifier (``head.head.2`` in ``imint/fm/upernet.py``), and
sample that 256-vector at every in-crop plot pixel. Also derive the NFI
field-truth forest class per plot (``derive_nfi_forest_class``). The result is a
per-plot feature parquet that ``nfi_head_cv.py`` k-fold-cross-validates a small
head on — testing whether a head trained DIRECTLY on field-truth can beat the
NMD label ceiling for forest-type classification.

Why hook the INPUT of ``head.head.2``? The seg head is
``Sequential(Identity, Dropout2d, Conv2d(256, num_classes, 1))``. The 256-ch
tensor entering that Conv2d is the decoder's fused feature — the representation
the network actually classifies from, before the (NMD-supervised) linear layer.
Training a fresh head on it against NFI truth isolates "is the information there"
from "did NMD supervision surface it".

Crop remap: ``run_inference`` centre-crops each tile to ``img_size`` (504 for the
600M patch-14 backbone on 512 tiles). Plot ``(row, col)`` are in full-tile
coords, so we subtract the crop offset ``off = (tile_h - crop_sz) // 2`` and drop
plots that fall in the discarded border — mirroring ``validate_against_nfi.main``
exactly, so the sampled pixel matches what the model scored.

    python scripts/extract_plot_features.py \
        --checkpoint /data/checkpoints/v8b_nmd2023_long/best_model.pt \
        --plot-index /data/nfi/nfi_index_unified_v2_512.parquet \
        --img-size 504 \
        --out /data/nfi_eval/nfi_plot_features_nmd2023.parquet
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_against_nfi import crop_offset, derive_nfi_forest_class

# Feature width is the classifier's in_channels — a BACKBONE property
# (256 for the UPerNet families, 128 for tessera), read from the model at
# runtime and carried by the parquet's column count. Never a constant.


def _load_inference_comparison():
    """Import ``inference_comparison`` as a module (it lives beside this file).

    Uses spec-from-file-location — the same pattern ``validate_against_nfi``
    uses — so this works whether or not ``scripts/`` is an importable package.
    """
    spec = importlib.util.spec_from_file_location(
        "_infcmp", str(Path(__file__).resolve().parent / "inference_comparison.py"),
    )
    infcmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infcmp)
    return infcmp


def register_preclassifier_hook(model) -> dict:
    """Hook ``head.head.2`` to capture its INPUT (the 256-dim feature map).

    ``head.head`` is ``Sequential(Identity, Dropout2d, Conv2d(256, C, 1))``, so
    ``head.head[2]`` is the final classifier. A forward pre-hook style capture
    via the module's ``forward`` input grabs the (1, 256, H, W) tensor before
    the class projection. Returns a mutable dict whose ``["feat"]`` holds the
    most recent capture (CPU numpy, batch-squeezed → (256, H, W)).

    The hook stores the feature at classifier resolution (pre-upsample). The
    logits are bilinearly upsampled to input resolution AFTER the head, so to
    keep plot (row,col) — which are input-resolution pixels — aligned with the
    feature grid we upsample the captured map to (crop_sz, crop_sz) at sample
    time (see ``_sample_feature``).
    """
    store: dict = {"feat": None}
    classifier = find_classifier(model)

    def hook(_module, inputs, _output):
        # inputs is a tuple; inputs[0] is the (B, 256, H, W) feature map.
        feat = inputs[0].detach()
        store["feat"] = feat  # keep on device; squeeze/np at sample time

    handle = classifier.register_forward_hook(hook)
    store["handle"] = handle
    store["n_features"] = int(classifier.in_channels)
    return store


def find_classifier(model):
    """The final per-pixel class-projection Conv2d, across all six families.

    The hook predates the multi-backbone ladder and hardcoded Prithvi's
    ``head.head[2]``. Known per-family paths are tried first so a family
    refactor fails loudly HERE; then a structural fallback.

    NOTE the in_channels is NOT always 256: tessera's is 128
    (final_in = hidden // 2). The feature dim is a property of the
    backbone's head and travels with the export — consumers read the
    width from the data, never from a constant.
    """
    import torch.nn as nn

    # Tessera: its own small head ends in self.classifier.
    mod = getattr(model, "classifier", None)
    if isinstance(mod, nn.Conv2d):
        return mod
    # Prithvi: model.head.head[2]; croma/terramind/clay wrap the shared
    # UPerNet as model.decoder_head -> .head.head[2].
    for holder_name in ("head", "decoder_head"):
        holder = getattr(model, holder_name, None)
        for seq in (getattr(holder, "head", None),
                    getattr(getattr(holder, "head", None), "head", None)):
            try:
                m = seq[2]
            except (TypeError, IndexError, KeyError):
                continue
            if isinstance(m, nn.Conv2d):
                return m
    # Structural fallback: among 1x1 convs the class projection has the
    # UNIQUE maximal out_channels (frac/binary heads carry 4/1). Guessing
    # between ambiguous candidates would silently hook the wrong feature.
    convs = [m for m in model.modules()
             if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1)]
    if convs:
        top = max(c.out_channels for c in convs)
        winners = [c for c in convs if c.out_channels == top]
        if len(winners) == 1:
            return winners[0]
    raise TypeError(
        f"cannot locate the classifier Conv2d on {type(model).__name__}; "
        f"1x1 convs seen: {[(c.in_channels, c.out_channels) for c in convs]}")

def _sample_feature(feat_map, rows, cols, crop_sz: int) -> np.ndarray:
    """Sample (N, 256) vectors at input-resolution (row,col) from (1,256,h,w).

    The captured feature grid is at classifier resolution (h,w) which may be
    smaller than ``crop_sz`` (the head's logits are upsampled AFTER the
    classifier). Bilinearly resize the feature map to (crop_sz, crop_sz) so the
    plot pixels — expressed in crop/input coordinates — index the right cell.
    """
    import torch
    import torch.nn.functional as F

    if feat_map.shape[-1] != crop_sz or feat_map.shape[-2] != crop_sz:
        feat_map = F.interpolate(
            feat_map, size=(crop_sz, crop_sz),
            mode="bilinear", align_corners=True,
        )
    fm = feat_map.squeeze(0).cpu().numpy()  # (256, crop_sz, crop_sz)
    r = np.asarray(rows, dtype=np.int64)
    c = np.asarray(cols, dtype=np.int64)
    return fm[:, r, c].T.astype(np.float32)  # (N, 256)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--plot-index", required=True,
                    help="parquet from nfi_tile_coverage.py (tile_name, tile_path, "
                         "row, col, Easting, Northing, TractID, PlotID + NFI cols)")
    ap.add_argument("--img-size", type=int, default=504,
                    help="inference crop (504 = 600M patch-14 on 512 tiles)")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--dominant-frac", type=float, default=0.7,
                    help="conifer/deciduous dominance threshold for NFI truth")
    ap.add_argument("--truth-col", default=None,
                    help="take truth directly from this index column "
                         "(e.g. unified_class for the LUCAS crop index) "
                         "instead of deriving NFI forest class from volume "
                         "columns")
    ap.add_argument("--backbone-name", default=None,
                    help="registry backbone for load_model; without it a "
                         "checkpoint lacking backbone_name silently "
                         "resolves to prithvi_300m")
    ap.add_argument("--enable-markfukt", action="store_true",
                    help="feed markfukt as the 11th aux (wetness-aux checkpoint)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    infcmp = _load_inference_comparison()

    index_df = pd.read_parquet(args.plot_index)
    print(f"plot index: {len(index_df):,} co-located plots on "
          f"{index_df['tile_name'].nunique()} tiles")

    # Drop plots whose tile no longer exists on disk (stale index rows) — same
    # guard as validate_against_nfi.main so one missing tile can't abort.
    exists = index_df["tile_path"].map(os.path.exists)
    if not exists.any():
        # Every path missing is not "the dataset shrank" — it is a mount
        # mismatch: the index stores absolute paths (e.g. /data/…) from
        # the environment that built it. Name the first path so the next
        # mis-mounted pod diagnoses itself in one line, not a traceback.
        raise SystemExit(
            f"0 of {len(index_df)} indexed tile paths exist — mount "
            f"mismatch? First expected: {index_df['tile_path'].iloc[0]}")
    if not exists.all():
        gone = int((~exists).sum())
        print(f"dropping {gone} plots on tiles no longer in the dataset "
              f"({index_df.loc[~exists, 'tile_name'].nunique()} tiles)")
        index_df = index_df[exists].copy()

    # Crop-remap: run_inference centre-crops to img_size; remap (row,col) into
    # crop coords and drop border plots. MIRRORS validate_against_nfi.main.
    sample_path = index_df["tile_path"].iloc[0]
    tile_h = int(np.load(sample_path, allow_pickle=True)["spectral"].shape[-1])
    off = crop_offset(tile_h, args.img_size)
    cs = min(args.img_size, tile_h)
    before = len(index_df)
    index_df = index_df[
        (index_df["row"] >= off) & (index_df["row"] < off + cs)
        & (index_df["col"] >= off) & (index_df["col"] < off + cs)
    ].copy()
    index_df["row"] -= off
    index_df["col"] -= off
    kept = len(index_df)
    print(f"crop offset={off} (tile {tile_h}→{cs}); kept {kept}/{before} plots "
          f"in-crop ({before - kept} border-dropped)")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # The checkpoint KNOWS its aux set — reconstructing it from flags is
    # how terramind died: its r2 trained with 13 aux (the usual 11 plus
    # delta_vv/delta_vh ΔSAR), while --enable-markfukt rebuilt 11 and the
    # lidar_branch conv rejected the tensor. The saved config is the
    # single source of truth; the flag survives only as a fallback for
    # pre-config-era checkpoints.
    import torch as _torch
    _cfg = _torch.load(args.checkpoint, map_location="cpu",
                       weights_only=False).get("config", {})
    aux_names = _cfg.get("enabled_aux_names")
    if aux_names:
        aux_names = list(aux_names)
        print(f"  aux from checkpoint config: {len(aux_names)} channels "
              f"({aux_names[-3:]}...)")
    elif args.enable_markfukt:
        from imint.training.unified_dataset import AUX_CHANNEL_NAMES
        aux_names = list(AUX_CHANNEL_NAMES) + ["markfukt"]
        print(f"  aux fallback (no config in ckpt): {len(aux_names)}")
    else:
        aux_names = None

    # BOTH kwargs must reach load_model — the full call shape of
    # infer_tiles.py:243. img_size: clay/croma carry no pos_embed, so the
    # backbone is otherwise BUILT at a 224 grid while run_inference feeds
    # it 504px tiles — the 256-dim features this script exists to capture
    # would come off a wrongly-shaped head. backbone_name: checkpoint-only
    # resolution defaults to prithvi_300m when the saved config lacks the
    # field (pre-2026-08-24 trainer) — the wrong backbone entirely.
    model, epoch, miou, model_img_size = infcmp.load_model(
        args.checkpoint, device, backbone_name=args.backbone_name,
        img_size=args.img_size)
    print(f"  [load_model] epoch={epoch} ckpt_mIoU={miou} native_img={model_img_size}")

    store = register_preclassifier_hook(model)

    n_features = store["n_features"]
    print(f"  classifier in_channels = {n_features} (native feature width)")
    feat_cols = [f"f{i:03d}" for i in range(n_features)]
    records: list[dict] = []

    for tile_name, grp in index_df.groupby("tile_name", sort=False):
        tile_path = grp["tile_path"].iloc[0]
        # Forward the tile once; the hook captures the 256-ch pre-classifier map.
        # return_probs=True runs the same preprocessing (spectral z-score + aux
        # + temporal/location coords) the model was trained with.
        infcmp.run_inference(
            model, tile_path, device, img_size=args.img_size,
            return_probs=True, aux_channel_names=aux_names,
        )
        feat_map = store["feat"]
        if feat_map is None:
            raise RuntimeError(f"hook captured no feature for tile {tile_name}")

        rows = grp["row"].to_numpy()
        cols = grp["col"].to_numpy()
        vecs = _sample_feature(feat_map, rows, cols, cs)  # (N, 256)

        for (_, r), vec in zip(grp.iterrows(), vecs):
            rec = {
                "TractID": r.get("TractID"),
                "PlotID": r.get("PlotID"),
                "point_id": r.get("point_id"),
                "Easting": r.get("Easting"),
                "Northing": r.get("Northing"),
                "tile_name": str(tile_name),
            }
            if args.truth_col:
                # Generic truth passthrough (LUCAS crop mode): the index
                # already carries the class; NFI volume-derivation would
                # be meaningless on these rows.
                rec[args.truth_col] = int(r[args.truth_col])
            else:
                nc = derive_nfi_forest_class(r, dominant_frac=args.dominant_frac)
                rec["nfi_forest"] = int(nc) if nc is not None else -1
            rec.update(dict(zip(feat_cols, vec.tolist())))
            records.append(rec)

    store["handle"].remove()

    out_df = pd.DataFrame.from_records(records, columns=(
        ["TractID", "PlotID", "Easting", "Northing", "tile_name", "nfi_forest"]
        + feat_cols
    ))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)

    n_by_class = out_df["nfi_forest"].value_counts().sort_index().to_dict()
    print(f"\nwrote {out} — {len(out_df)} plots × {n_features} features")
    print(f"  nfi_forest distribution (−1=treeless): {n_by_class}")


if __name__ == "__main__":
    main()
