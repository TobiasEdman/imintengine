#!/usr/bin/env python3
"""Score every 28-class ladder cell on the held-out tiles, per pixel.

The ladder's training logs carry mIoU and per-class IoU only — no overall
accuracy — and they are computed on the *validation* split. This scores the
frozen holdout set (``/cephfs/holdout_val_512``, never trained on) against the
28-class NMD2023 sidecars, which is the only place a real 28-class overall
accuracy can come from.

Rung 1 is deliberately out of scope: it is 23-class, and its predictions do
not live in the same label space.

Nothing is rendered. The matrix job draws the pictures; this one only counts,
accumulating a 28x28 confusion matrix per cell so memory stays flat however
many tiles are scored.

    python scripts/ladder_holdout_eval.py \
        --holdout-dir /cephfs/holdout_val_512 \
        --label-dir /cephfs/nmd2023_labels \
        --ckpt-root /cephfs/checkpoints/ladder \
        --out /cephfs/ladder_eval/holdout28.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imint.training.evaluate import compute_miou  # noqa: E402
from imint.training.unified_schema import NUM_UNIFIED_CLASSES  # noqa: E402
from gen_ladder_manifests import DISTILL  # noqa: E402
from infer_tiles import ckpt_sha256  # noqa: E402

NUM_CLASSES_28 = 28
EVAL_RUNGS = (2, 3, 4)


def _load_infcmp():
    spec = importlib.util.spec_from_file_location(
        "_infcmp",
        str(Path(__file__).resolve().parent / "inference_comparison.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def select_tiles(holdout_dir: Path, label_dir: Path, limit: int | None) -> list[str]:
    """Deterministic tile list: every holdout tile that has 28-class truth.

    Sorted by name and then strided, so a --limit run is spread across the
    whole set rather than taking one alphabetical corner of the country. The
    same limit always yields the same tiles.
    """
    names = sorted(p.stem for p in holdout_dir.glob("*.npz")
                   if (label_dir / f"{p.stem}.npz").exists())
    if not names:
        raise SystemExit(
            f"no holdout tile in {holdout_dir} has a 28-class sidecar in "
            f"{label_dir} — run k8s/build-labels-holdout-nmd2023-job.yaml first")
    if limit is None or limit >= len(names):
        return names
    step = len(names) / limit
    return [names[int(i * step)] for i in range(limit)]


def derive(confusion: np.ndarray, ignore_index: int = 0) -> dict:
    """Overall accuracy, per-class IoU/F1 and the macro summaries.

    ``ignore_index`` drops class 0 (background/nodata) from every aggregate but
    keeps it in the matrix, so confusions *into* background stay visible.
    """
    keep = [c for c in range(confusion.shape[0]) if c != ignore_index]
    # Denominator is every pixel whose TRUTH is a real class — including the
    # ones the model called background. Excluding class 0 from both axes
    # instead would silently drop those errors and inflate the headline.
    total = confusion[keep, :].sum()
    correct = sum(int(confusion[c, c]) for c in keep)
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
    per_class = {
        int(c): {
            "iou": None if np.isnan(iou[c]) else round(float(iou[c]), 4),
            "f1": None if np.isnan(f1[c]) else round(float(f1[c]), 4),
            "precision": None if np.isnan(prec[c]) else round(float(prec[c]), 4),
            "recall": None if np.isnan(rec[c]) else round(float(rec[c]), 4),
            "support": int(confusion[c].sum()),
        }
        for c in keep
    }
    present = [c for c in keep if confusion[c].sum() > 0]
    return {
        "overall_accuracy": round(float(correct / total), 4) if total else None,
        "mean_iou": round(float(np.nanmean([iou[c] for c in present])), 4) if present else None,
        "mean_f1": round(float(np.nanmean([f1[c] for c in present])), 4) if present else None,
        "classes_present": len(present),
        "pixels_scored": int(total),
        "per_class": per_class,
    }


def centre_crop_truth(truth: np.ndarray, img_size: int) -> np.ndarray:
    """Crop the label to the exact window the model actually saw.

    ``run_inference`` centre-crops its input with
    ``crop_sz = min(img_size, H, W)`` and ``y0 = (H - crop_sz) // 2``, so a
    512x512 tile scored at img_size 504 yields a prediction covering
    ``truth[4:508, 4:508]`` — not the whole tile. This repeats that arithmetic
    exactly rather than resizing: labels are categorical, so any interpolation
    would invent classes that were never there, and an off-by-four alignment
    would still produce a plausible-looking accuracy computed against shifted
    ground.
    """
    h, w = truth.shape
    crop_sz = min(img_size, h, w)
    y0 = (h - crop_sz) // 2
    x0 = (w - crop_sz) // 2
    return truth[y0:y0 + crop_sz, x0:x0 + crop_sz]


def score_cell(model: str, rung: int, ckpt: Path, tiles: list[str],
               holdout_dir: Path, label_dir: Path, device: str) -> dict:
    import torch

    infcmp = _load_infcmp()
    cfg = DISTILL[model]
    dev = torch.device(device)
    # Aux set comes from the checkpoint's own config, never from flags — the
    # same rule the matrix job follows, and the reason terramind (13 aux) and
    # the 11-aux ladder columns do not silently get the canonical 10.
    model_obj, epoch, ckpt_miou, _, ck_cfg = infcmp.load_model(
        str(ckpt), dev, backbone_name=cfg["backbone"],
        img_size=cfg["img_size"], return_checkpoint_config=True)
    aux_names = (ck_cfg or {}).get("enabled_aux_names")
    aux_names = list(aux_names) if aux_names else None

    confusion = np.zeros((NUM_CLASSES_28, NUM_CLASSES_28), dtype=np.int64)
    t0 = time.time()
    skipped: list[str] = []
    for i, name in enumerate(tiles, 1):
        pred = np.asarray(infcmp.run_inference(
            model_obj, str(holdout_dir / f"{name}.npz"), dev,
            img_size=cfg["img_size"], aux_channel_names=aux_names))
        with np.load(label_dir / f"{name}.npz", allow_pickle=False) as z:
            truth = z["label"]
        truth = centre_crop_truth(truth, cfg["img_size"])
        if pred.shape != truth.shape:
            # The prediction should now be exactly the cropped window. If it
            # is not, the tile's geometry disagrees with the crop contract and
            # scoring it would compare different ground.
            skipped.append(name)
            continue
        if truth.max() >= NUM_CLASSES_28 or pred.max() >= NUM_CLASSES_28:
            skipped.append(name)
            continue
        confusion += compute_miou(
            pred, truth, NUM_CLASSES_28, ignore_index=0)["confusion_matrix"]
        if i % 25 == 0 or i == len(tiles):
            rate = (time.time() - t0) / i
            print(f"  [{model}_r{rung}] {i}/{len(tiles)} "
                  f"{rate:.2f}s/tile eta={(len(tiles)-i)*rate/60:.1f}min",
                  flush=True)
    del model_obj
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    out = derive(confusion)
    out.update({
        "cell": f"{model}_r{rung}", "backbone": cfg["backbone"],
        "img_size": cfg["img_size"], "ckpt_sha256": ckpt_sha256(str(ckpt)),
        "ckpt_epoch": epoch, "ckpt_val_miou": ckpt_miou,
        "tiles_scored": len(tiles) - len(skipped), "tiles_skipped": skipped,
        "seconds": round(time.time() - t0, 1),
        "confusion_matrix": confusion.tolist(),
    })
    print(f"[{model}_r{rung}] OA={out['overall_accuracy']} "
          f"mIoU={out['mean_iou']} over {out['pixels_scored']:,} px", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--holdout-dir", type=Path, default=Path("/cephfs/holdout_val_512"))
    ap.add_argument("--label-dir", type=Path, default=Path("/cephfs/nmd2023_labels"))
    ap.add_argument("--ckpt-root", type=Path, default=Path("/cephfs/checkpoints/ladder"))
    ap.add_argument("--out", type=Path, default=Path("/cephfs/ladder_eval/holdout28.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-tiles", type=int, default=None,
                    help="score a deterministic spread of this many tiles")
    ap.add_argument("--models", default=None,
                    help="comma-separated subset of ladder columns")
    ap.add_argument("--rungs", default=",".join(str(r) for r in EVAL_RUNGS),
                    help="comma-separated 28-class rungs (rung 1 is 23-class)")
    ap.add_argument("--git-sha", default="unknown")
    args = ap.parse_args()

    rungs = [int(r) for r in args.rungs.split(",") if r.strip()]
    if 1 in rungs:
        ap.error("rung 1 is 23-class and cannot be scored in the 28-class space")
    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else list(DISTILL))
    unknown = [m for m in models if m not in DISTILL]
    if unknown:
        ap.error(f"unknown ladder column(s): {unknown}")

    tiles = select_tiles(args.holdout_dir, args.label_dir, args.limit_tiles)
    print(f"=== 28-class holdout eval — {len(models)} columns x {len(rungs)} "
          f"rungs x {len(tiles)} tiles ===", flush=True)

    results, missing = [], []
    for model in models:
        for rung in rungs:
            ckpt = args.ckpt_root / f"{model}_r{rung}" / "best_model.pt"
            if not ckpt.exists():
                missing.append(f"{model}_r{rung}")
                print(f"[{model}_r{rung}] no checkpoint — skipping", flush=True)
                continue
            results.append(score_cell(model, rung, ckpt, tiles,
                                      args.holdout_dir, args.label_dir,
                                      args.device))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({
                "schema": "ladder-holdout28-v1",
                "git_sha": args.git_sha,
                "num_classes": NUM_CLASSES_28,
                "unified_classes": NUM_UNIFIED_CLASSES,
                "holdout_dir": str(args.holdout_dir),
                "label_dir": str(args.label_dir),
                "tiles": tiles,
                "missing_cells": missing,
                "cells": results,
            }, indent=1))

    if not results:
        raise SystemExit("no cell scored — nothing was written")
    best = max(results, key=lambda r: r["overall_accuracy"] or 0)
    print(f"\nBEST 28-class OA: {best['cell']} = {best['overall_accuracy']} "
          f"(mIoU {best['mean_iou']})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
