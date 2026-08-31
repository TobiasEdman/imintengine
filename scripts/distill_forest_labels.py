#!/usr/bin/env python3
"""scripts/distill_forest_labels.py — dense forest-type pseudo-labels (GPU job).

Hybrid-NFI distillation step 2 (cluster, GPU). Take the head trained by
``train_distill_head.py`` and run it DENSELY over every tile: forward the seg
checkpoint, hook the 256-dim pre-classifier feature map (same hook as
``extract_plot_features.py``), forward the small head per pixel, and OVERRIDE the
forest pixels of each tile's NMD2023 label sidecar with the head's forest-type
call. The result is a distilled label sidecar the seg model finetunes on — the
head's field-truth forest-type signal baked into dense labels.

Leakage: ALL tiles are processed, including the test-split tiles. That is safe —
the head never saw the test tiles' plots (grouped-by-tile split), so its dense
prediction there is an out-of-sample inference, not a memorised label. The
honest final eval is the seg model's accuracy at the held-out plots (parent's
job); this script only produces the training target.

Override rule (conservative): a pixel is overridden ONLY when
    original label ∈ {1,2,3,4 forest}  AND  head-pred ∈ {1,2,3,4 forest}.
Everything else keeps the original NMD2023 label — background, sumpskog(5),
tillfälligt-ej-skog(6), crops, hygge, the 23-27 NMD2023 classes, and any pixel
the head calls non-forest. We never *create* or *destroy* forest, only refine
forest TYPE within pixels NMD2023 already calls forest. This bounds the blast
radius: the distilled labels differ from NMD2023 only in tall/gran/löv/bland
reassignment inside the existing forest mask.

Grid: run_inference centre-crops 512→504 (offset 4). The head predicts the
504×504 window; the border ring (4px) keeps the original label. The sidecar
``label`` stays a full 512×512 raster (what the trainer's ``_LabelOverlay``
expects) with only the centre window's forest pixels rewritten.

    python scripts/distill_forest_labels.py \
        --checkpoint /data/checkpoints/v8b_nmd2023_long/best_model.pt \
        --head data/distill/distill_head.npz \
        --data-dir /data/unified_v2_512 \
        --label-dir /data/nmd2023_labels \
        --out-dir /data/nmd2023_distill_labels \
        --img-size 504
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

# Repo root for imint.*, own dir for sibling scripts. The second insert is
# what direct execution gets for free (sys.path[0] = script dir) but an
# importlib load — the test suite — does not.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_pinned_plot_set import npz_key_names
from extract_plot_features import register_preclassifier_hook

FOREST_CLASSES = frozenset({1, 2, 3, 4})  # tall, gran, löv, bland
# Sidecar keys the distilled npz must carry (superset of what any NMD2023
# sidecar holds; we copy whatever is present + always the replaced label).
_SIDECAR_KEYS = (
    "label", "nmd_label_raw", "nmd_area_ha", "label_mask",
    "parcel_area_ha", "n_parcels", "harvest_mask",
    "n_harvest_polygons", "n_mature_polygons", "tile_size_px",
)


def _load_inference_comparison():
    """Import ``inference_comparison`` from its file (same trick as elsewhere)."""
    spec = importlib.util.spec_from_file_location(
        "_infcmp", str(Path(__file__).resolve().parent / "inference_comparison.py"),
    )
    infcmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infcmp)
    return infcmp


def load_head(head_path: str) -> dict:
    """Load the exported distillation head npz into a plain dict of arrays.

    ``head_sha`` (hash of the npz bytes) travels with the dict and is
    stamped into every sidecar this head produces — the resume path skips
    an existing sidecar only when its stamp matches. Without it,
    resume-after-abort with a retrained head silently mixes two heads'
    labels into the rung-3/4 training target.
    """
    z = np.load(head_path)
    head = {
        "mean_": z["mean_"].astype(np.float32),
        "scale_": z["scale_"].astype(np.float32),
        "W1": z["W1"].astype(np.float32),
        "b1": z["b1"].astype(np.float32),
        "W2": z["W2"].astype(np.float32),
        "b2": z["b2"].astype(np.float32),
        "classes_": z["classes_"].astype(np.int64),
    }
    assert head["W1"].shape[0] == head["mean_"].shape[0], "head/scaler dim mismatch"
    head["head_sha"] = hashlib.sha256(
        Path(head_path).read_bytes()).hexdigest()[:16]
    return head


def sidecar_is_current(out_path: str, head_sha: str) -> bool:
    """True iff an existing sidecar was written by THIS head.

    A sidecar without a stamp (pre-provenance era) or with a different
    stamp is stale and must be rewritten, not counted as done. Unreadable
    counts as stale — the rewrite is atomic either way.
    """
    try:
        with np.load(out_path) as z:
            return "head_sha" in z.files and str(z["head_sha"]) == head_sha
    except (OSError, ValueError):
        return False


def head_predict_dense(feat_hwc: np.ndarray, head: dict) -> np.ndarray:
    """Forward the head over a dense (H, W, 256) feature map → (H, W) class ids.

    Vectorised numpy port of MLPClassifier.predict: standardize → relu hidden →
    output logits → argmax → classes_ map. Identical math to
    ``train_distill_head.head_forward_npz`` (verified there to match sklearn
    exactly); flattened over pixels here. Returns labels in {0,1,2,3,4}.
    """
    h, w, c = feat_hwc.shape
    X = feat_hwc.reshape(-1, c)
    Xs = (X - head["mean_"]) / head["scale_"]
    hid = np.maximum(Xs @ head["W1"] + head["b1"], 0.0)
    logits = hid @ head["W2"] + head["b2"]
    pred = head["classes_"][logits.argmax(axis=1)]
    return pred.reshape(h, w).astype(np.uint8)


def _upsample_feat_to(feat_map, crop_sz: int) -> np.ndarray:
    """Bilinear-upsample captured (1,256,h,w) → (crop_sz, crop_sz, 256) numpy.

    Mirrors ``extract_plot_features._sample_feature``'s resize (align_corners=
    True) so the dense feature grid matches the plot-sampled grid the head was
    trained on, then moves channels last for the per-pixel matmul.
    """
    import torch
    import torch.nn.functional as F

    if feat_map.shape[-1] != crop_sz or feat_map.shape[-2] != crop_sz:
        feat_map = F.interpolate(
            feat_map, size=(crop_sz, crop_sz),
            mode="bilinear", align_corners=True,
        )
    fm = feat_map.squeeze(0).cpu().numpy()          # (256, crop_sz, crop_sz)
    return np.transpose(fm, (1, 2, 0))               # (crop_sz, crop_sz, 256)


def apply_override(orig_label: np.ndarray, head_pred: np.ndarray,
                   off: int, crop_sz: int) -> tuple[np.ndarray, int, int]:
    """Build the distilled full-grid label from an original + a crop-window pred.

    ``orig_label`` is the full (H, W) NMD2023 raster; ``head_pred`` is the
    (crop_sz, crop_sz) head call for the centre window at top-left ``off``.
    Override only where original ∈ forest AND head ∈ forest. Returns
    (new_label, n_overridden, n_forest_in_window).
    """
    new_label = orig_label.copy()
    win = new_label[off:off + crop_sz, off:off + crop_sz]
    orig_win = orig_label[off:off + crop_sz, off:off + crop_sz]

    orig_forest = np.isin(orig_win, list(FOREST_CLASSES))
    head_forest = np.isin(head_pred, list(FOREST_CLASSES))
    mask = orig_forest & head_forest

    win[mask] = head_pred[mask]
    new_label[off:off + crop_sz, off:off + crop_sz] = win
    return new_label, int(mask.sum()), int(orig_forest.sum())


def _atomic_savez(out_path: str, payload: dict) -> None:
    """tmp + os.replace with the savez ``.npz``-suffix quirk (build_labels.py)."""
    tmp_base = out_path + ".tmp"
    np.savez_compressed(tmp_base, **payload)
    os.replace(tmp_base + ".npz", out_path)


def process_tile(tile_path: str, model, head: dict, device, img_size: int,
                 label_dir: str, out_dir: str, infcmp, store,
                 aux_channel_names=None) -> dict:
    """Distill one tile. Returns a status dict (skip reasons logged by caller)."""
    name = os.path.basename(tile_path).replace(".npz", "")
    out_path = os.path.join(out_dir, name + ".npz")
    if os.path.exists(out_path):
        if sidecar_is_current(out_path, head["head_sha"]):
            return {"name": name, "status": "exists"}
        print(f"    stale sidecar {name} (head changed) — rewriting")

    sidecar_path = os.path.join(label_dir, name + ".npz")
    if not os.path.exists(sidecar_path):
        return {"name": name, "status": "no_sidecar"}

    # Forward the tile; the hook captures the 256-ch pre-classifier feature
    # map. aux_channel_names must match the checkpoint's aux count — an
    # 11-aux (markfukt) model fed the default 10-aux input dies on the
    # input conv.
    infcmp.run_inference(model, tile_path, device, img_size=img_size,
                         return_probs=True, aux_channel_names=aux_channel_names)
    feat_map = store["feat"]
    if feat_map is None:
        return {"name": name, "status": "no_feature"}

    # Determine crop geometry from the tile itself (must equal run_inference's).
    data0 = np.load(tile_path, allow_pickle=True)
    tile_h = int(data0.get("spectral", data0.get("image")).shape[-1])
    crop_sz = min(img_size, tile_h)
    off = (tile_h - crop_sz) // 2

    feat_hwc = _upsample_feat_to(feat_map, crop_sz)          # (cs, cs, 256)
    head_pred = head_predict_dense(feat_hwc, head)           # (cs, cs) in {0..4}

    sidecar = np.load(sidecar_path, allow_pickle=True)
    orig_label = np.asarray(sidecar["label"]).astype(np.uint8)
    if orig_label.shape[-1] != tile_h:
        return {"name": name, "status": "shape_mismatch"}

    new_label, n_over, n_forest = apply_override(orig_label, head_pred, off, crop_sz)

    # Copy every original sidecar key, replace only `label`; stamp which
    # head produced this file so a resume can tell current from stale.
    payload = {k: sidecar[k] for k in _SIDECAR_KEYS if k in sidecar}
    payload["label"] = new_label
    payload["head_sha"] = np.str_(head["head_sha"])
    _atomic_savez(out_path, payload)

    return {
        "name": name, "status": "ok",
        "n_over": n_over, "n_forest": n_forest,
        "frac_over": (n_over / n_forest) if n_forest else 0.0,
        "pred_hist": {int(c): int((head_pred == c).sum())
                      for c in (0, 1, 2, 3, 4)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--head", required=True, help="npz from train_distill_head.py")
    ap.add_argument("--data-dir", required=True, help="tiles (.npz) directory")
    ap.add_argument("--label-dir", required=True, help="NMD2023 label sidecars")
    ap.add_argument("--out-dir", required=True, help="distilled sidecar out dir")
    ap.add_argument("--img-size", type=int, default=504)
    ap.add_argument("--backbone-name", default=None,
                    help="registry backbone for load_model. Without it, "
                         "resolution falls back to the checkpoint config; a "
                         "checkpoint lacking backbone_name (pre-2026-08-24 "
                         "trainer) silently defaults to prithvi_300m — the "
                         "wrong backbone entirely for clay/croma")
    ap.add_argument("--enable-markfukt", action="store_true",
                    help="forward with the 11th (SLU soil moisture) aux — "
                         "REQUIRED for checkpoints trained with it; the "
                         "input conv otherwise rejects the 10-aux tensor")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N tiles")
    ap.add_argument("--require-npz-key", action="append", default=[],
                    help="skip tiles whose npz lacks this key (repeatable). "
                         "SAR backbones (croma/terramind) pass s1_vv_vh so "
                         "the pass covers exactly their trainable cohort "
                         "instead of crashing on the first optical-only tile")
    args = ap.parse_args()

    import torch

    infcmp = _load_inference_comparison()
    head = load_head(args.head)
    print(f"head: hidden {head['W1'].shape} → {head['W2'].shape}, "
          f"classes {head['classes_'].tolist()}")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # BOTH kwargs must reach load_model — the full call shape of
    # infer_tiles.py:243. img_size: clay/croma carry no pos_embed, so
    # without it the backbone is BUILT at a 224 grid and then fed 504px
    # tiles. backbone_name: without it, resolution falls back to the
    # checkpoint config, and a checkpoint that lacks the field
    # (pre-2026-08-24 trainer) silently defaults to prithvi_300m — the
    # wrong backbone entirely. Relying on checkpoint recency is an
    # accident, not a contract.
    model, epoch, miou, model_img_size = infcmp.load_model(
        args.checkpoint, device, backbone_name=args.backbone_name,
        img_size=args.img_size)
    print(f"[load_model] epoch={epoch} ckpt_mIoU={miou} native_img={model_img_size}")
    if model_img_size != args.img_size:
        print(f"  WARN: --img-size {args.img_size} != model native "
              f"{model_img_size}; using --img-size for the crop geometry")

    store = register_preclassifier_hook(model)

    aux_names = None
    if args.enable_markfukt:
        from imint.training.unified_dataset import AUX_CHANNEL_NAMES
        aux_names = list(AUX_CHANNEL_NAMES) + ["markfukt"]
        print(f"markfukt enabled → {len(aux_names)} aux channels")

    os.makedirs(args.out_dir, exist_ok=True)
    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    unreadable: list[str] = []
    if args.require_npz_key:
        # Cheap zip-directory probe (shared with build_pinned_plot_set —
        # one implementation, not two drifting copies). This defines the
        # column's cohort (e.g. the ~6011-tile SAR subset) rather than
        # discovering it via a KeyError mid-run. Unreadable is NOT
        # "lacks the key": it drops from the cohort here but is carried
        # to the final gate, so a degraded PVC cannot pass silently.
        before = len(tiles)
        kept: list[str] = []
        for t in tiles:
            names = npz_key_names(Path(t))
            if names is None:
                unreadable.append(os.path.basename(t))
            elif all(k in names for k in args.require_npz_key):
                kept.append(t)
        tiles = kept
        print(f"require {args.require_npz_key}: {len(tiles)}/{before} tiles"
              + (f" ({len(unreadable)} UNREADABLE)" if unreadable else ""))
    if args.limit:
        tiles = tiles[:args.limit]
    print(f"tiles: {len(tiles)}  →  out {args.out_dir}\n")

    counts = {"ok": 0, "exists": 0, "no_sidecar": 0, "no_feature": 0,
              "shape_mismatch": 0, "error": 0}
    fracs: list[float] = []
    total_hist = {c: 0 for c in (0, 1, 2, 3, 4)}

    for i, tp in enumerate(tiles):
        # One corrupt tile must not kill an hours-long pass: completed
        # sidecars persist and the "exists" skip makes reruns idempotent,
        # so contain the failure, keep going, and fail the JOB at the end.
        try:
            res = process_tile(tp, model, head, device, args.img_size,
                               args.label_dir, args.out_dir, infcmp, store,
                               aux_channel_names=aux_names)
        except Exception as exc:  # noqa: BLE001 — summarized + fatal below
            res = {"status": "error", "name": os.path.basename(tp)}
            print(f"  [{i+1}/{len(tiles)}] ERROR {res['name']}: "
                  f"{type(exc).__name__}: {exc}")
        counts[res["status"]] = counts.get(res["status"], 0) + 1
        if res["status"] == "ok":
            fracs.append(res["frac_over"])
            for c, v in res["pred_hist"].items():
                total_hist[c] += v
            if i % 50 == 0 or res["frac_over"] > 0:
                print(f"  [{i+1}/{len(tiles)}] {res['name']}: "
                      f"{res['n_over']}/{res['n_forest']} forest px overridden "
                      f"({100*res['frac_over']:.1f}%)")
        elif res["status"] != "exists":
            print(f"  [{i+1}/{len(tiles)}] SKIP {res['name']}: {res['status']}")

    store["handle"].remove()

    print("\n─── summary ─────────────────────────────────────────────")
    print(f"  processed ok : {counts['ok']}")
    print(f"  already exist: {counts['exists']}")
    print(f"  no sidecar   : {counts['no_sidecar']}")
    print(f"  no feature   : {counts['no_feature']}")
    print(f"  shape mismatch: {counts['shape_mismatch']}")
    if fracs:
        print(f"  mean % forest pixels overridden: {100*np.mean(fracs):.1f}% "
              f"(median {100*np.median(fracs):.1f}%)")
    tot = sum(total_hist.values()) or 1
    print(f"  head-pred distribution over all crop windows:")
    names = {0: "non-forest", 1: "tall", 2: "gran", 3: "löv", 4: "bland"}
    for c in (0, 1, 2, 3, 4):
        print(f"    {c} {names[c]:<11}: {total_hist[c]:>12,} "
              f"({100*total_hist[c]/tot:.1f}%)")

    # The ladder gate (label_source_ladder.md, execution order step 2):
    # a column may proceed to rungs 3/4 only when its sidecar count matches
    # its cohort count. ok+exists is exactly that; anything else means a
    # tile this column trains on has no distilled sidecar. Exit non-zero so
    # the k8s Job goes red instead of a short set silently shrinking the
    # rung-3 training set.
    covered = counts["ok"] + counts["exists"]
    if covered != len(tiles) or unreadable:
        raise SystemExit(
            f"sidecar gate FAILED: {covered}/{len(tiles)} cohort tiles "
            f"covered (no_sidecar={counts['no_sidecar']} "
            f"no_feature={counts['no_feature']} "
            f"shape_mismatch={counts['shape_mismatch']} "
            f"error={counts['error']} unreadable={len(unreadable)}). "
            f"Rungs 3/4 must not start.")
    print(f"\n  sidecar gate OK: {covered}/{len(tiles)} cohort tiles covered")


if __name__ == "__main__":
    main()
