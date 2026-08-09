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

N_FEATURES = 256  # channels feeding head.head.2 (decoder fpn_bottleneck output)


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

    classifier = model.head.head[2]
    if not isinstance(classifier, __import__("torch").nn.Conv2d):
        raise TypeError(
            f"expected head.head[2] to be Conv2d, got {type(classifier)!r}"
        )

    def hook(_module, inputs, _output):
        # inputs is a tuple; inputs[0] is the (B, 256, H, W) feature map.
        feat = inputs[0].detach()
        store["feat"] = feat  # keep on device; squeeze/np at sample time

    handle = classifier.register_forward_hook(hook)
    store["handle"] = handle
    return store


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

    aux_names = None
    if args.enable_markfukt:
        from imint.training.unified_dataset import AUX_CHANNEL_NAMES
        aux_names = list(AUX_CHANNEL_NAMES) + ["markfukt"]
        print(f"  markfukt enabled → {len(aux_names)} aux channels")

    model, epoch, miou, model_img_size = infcmp.load_model(args.checkpoint, device)
    print(f"  [load_model] epoch={epoch} ckpt_mIoU={miou} native_img={model_img_size}")

    store = register_preclassifier_hook(model)

    feat_cols = [f"f{i:03d}" for i in range(N_FEATURES)]
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
            nc = derive_nfi_forest_class(r, dominant_frac=args.dominant_frac)
            rec = {
                "TractID": r.get("TractID"),
                "PlotID": r.get("PlotID"),
                "Easting": r.get("Easting"),
                "Northing": r.get("Northing"),
                "tile_name": str(tile_name),
                "nfi_forest": int(nc) if nc is not None else -1,
            }
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
    print(f"\nwrote {out} — {len(out_df)} plots × {N_FEATURES} features")
    print(f"  nfi_forest distribution (−1=treeless): {n_by_class}")


if __name__ == "__main__":
    main()
