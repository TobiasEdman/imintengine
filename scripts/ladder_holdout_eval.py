#!/usr/bin/env python3
"""Score 28- or 29-class checkpoints on the held-out tiles, per pixel.

The ladder's training logs carry mIoU and per-class IoU only — no overall
accuracy — and they are computed on the *validation* split. This scores the
frozen holdout set (``/cephfs/holdout_val_512``, never trained on) against the
NMD2023 label sidecars, which is the only place a real holdout overall
accuracy can come from.

Two ways to say what is scored:

* the ladder grid (default): every ``{column}_r{rung}/best_model.pt`` under
  ``--ckpt-root``, rungs 2–4, all 28-class. Rung 1 is out of scope: it is
  23-class, and its predictions do not live in the same label space.
* one checkpoint: ``--checkpoint PATH --cell-name NAME --models COLUMN
  --label-dir DIR --num-classes N --out FILE``, for models outside the ladder
  layout (IM-023 main and control). ``COLUMN`` supplies backbone and
  img_size, which must match the checkpoint's own config.

``--num-classes`` is the checkpoint's own class count and must equal its
config's ``num_classes``; nothing is inferred from the data. ``--fold 28:3``
maps a 29-class model's class 28 onto 3 before scoring, so it is scored in
the 28-class space of the original labels and its matrices line up with R4's.
The label directory must be in the scored space: 29-class scoring needs the
barrblandskog labels (identified by their relabel manifest), 28-class scoring
refuses them. Every truth and prediction value must lie in the space it is
scored in; anything else fails the cell, it is never dropped. An existing
report is only continued with ``--resume``, never overwritten.

Nothing is rendered. Each cell keeps one confusion matrix per tile, written
next to the report as ``<out stem>.<cell>.tiles.npz`` for the paired tile
bootstrap (``imint.eval.tile_bootstrap``), and their sum in the report.

    # the ladder grid (R4 is tessera_r4) on the original labels
    python scripts/ladder_holdout_eval.py \\
        --holdout-dir /cephfs/holdout_val_512 \\
        --label-dir /cephfs/nmd2023_labels \\
        --ckpt-root /cephfs/checkpoints/ladder \\
        --out /cephfs/ladder_eval/holdout28.json

    # IM-023 main, class 28 folded into 3, on the original labels
    python scripts/ladder_holdout_eval.py \\
        --checkpoint /cephfs/checkpoints/im023/tessera_b28/best_model.pt \\
        --cell-name main_fold28to3 --models tessera --num-classes 29 \\
        --fold 28:3 --label-dir /cephfs/nmd2023_labels \\
        --out /cephfs/ladder_eval/im023/main_fold28to3.json

    # IM-023 main with 28 as its own class, on the barrblandskog labels
    python scripts/ladder_holdout_eval.py \\
        --checkpoint /cephfs/checkpoints/im023/tessera_b28/best_model.pt \\
        --cell-name main_b28 --models tessera --num-classes 29 \\
        --label-dir /cephfs/nmd2023_labels_b28 \\
        --out /cephfs/ladder_eval/im023/main_b28.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imint.metrics import compute_miou  # noqa: E402
from imint.schema.unified_schema import (  # noqa: E402
    NUM_CLASSES_BEFORE_BARRBLANDSKOG, NUM_UNIFIED_CLASSES,
)
from imint.eval.tile_bootstrap import save_tile_confusions  # noqa: E402
from gen_ladder_manifests import DISTILL  # noqa: E402
from relabel_barrblandskog import MANIFEST as RELABEL_MANIFEST  # noqa: E402
from relabel_barrblandskog import RULE as RELABEL_RULE  # noqa: E402
from scripts.cache_predict import file_identity  # noqa: E402
from imint.eval.fieldtruth import sha256_file  # noqa: E402

NUM_CLASSES_CHOICES = (NUM_CLASSES_BEFORE_BARRBLANDSKOG, NUM_UNIFIED_CLASSES)
LADDER_NUM_CLASSES = NUM_CLASSES_BEFORE_BARRBLANDSKOG
# The one fold: a 29-class model's class 28 (barrblandskog), back into 28 classes.
FOLD_SOURCE = NUM_UNIFIED_CLASSES - 1
EVAL_RUNGS = (2, 3, 4)
DEFAULT_LABEL_DIR = Path("/cephfs/nmd2023_labels")
DEFAULT_CKPT_ROOT = Path("/cephfs/checkpoints/ladder")
DEFAULT_OUT = Path("/cephfs/ladder_eval/holdout28.json")
# A cell name becomes part of a file name next to the report.
CELL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SCHEMA = "ladder-holdout-v3"


def _load_infcmp():
    spec = importlib.util.spec_from_file_location(
        "_infcmp",
        str(Path(__file__).resolve().parent / "inference_comparison.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def select_tiles(holdout_dir: Path, label_dir: Path, limit: int | None) -> list[str]:
    """Deterministic tile list: every holdout tile that has a label sidecar.

    Sorted by name and then strided, so a --limit run is spread across the
    whole set rather than taking one alphabetical corner of the country. The
    same limit always yields the same tiles.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    names = sorted(p.stem for p in holdout_dir.glob("*.npz")
                   if (label_dir / f"{p.stem}.npz").exists())
    if not names:
        raise SystemExit(
            f"no holdout tile in {holdout_dir} has a label sidecar in "
            f"{label_dir} — build the holdout labels first "
            f"(k8s/build-labels-holdout-nmd2023-job.yaml)")
    if limit is None or limit >= len(names):
        return names
    step = len(names) / limit
    return [names[int(i * step)] for i in range(limit)]


def parse_fold(text: str) -> tuple[int, int]:
    """``PRED:TARGET`` → ``(pred, target)``; an argparse ``type``."""
    try:
        source, target = (int(v) for v in text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--fold takes PRED:TARGET, e.g. 28:3, not {text!r}") from None
    return source, target


def fold_record(fold: tuple[int, int] | None) -> dict | None:
    return None if fold is None else {"prediction": fold[0], "scored_as": fold[1]}


def scored_classes(num_classes: int, fold: tuple[int, int] | None) -> int:
    """Size of the label space the confusion matrices are kept in.

    The only fold is a 29-class model's class 28 onto a class below it, so
    the folded model is scored in the 28-class space of the original labels.
    """
    if num_classes not in NUM_CLASSES_CHOICES:
        raise ValueError(f"num_classes must be one of {NUM_CLASSES_CHOICES}, got {num_classes}")
    if fold is None:
        return num_classes
    source, target = fold
    if (num_classes, source) != (NUM_UNIFIED_CLASSES, FOLD_SOURCE) or not 0 <= target < source:
        raise ValueError(
            f"--fold {source}:{target}: only {FOLD_SOURCE}:<class below it> with "
            f"--num-classes {NUM_UNIFIED_CLASSES}")
    return FOLD_SOURCE


def label_set(label_dir: Path, n_scored: int) -> dict | None:
    """Bind the label directory to the scored class space.

    A barrblandskog (b28) label directory is identified by the manifest
    ``scripts/relabel_barrblandskog.py run`` writes into it, naming it as its
    NMD output; original labels carry none. 29-class scoring requires b28
    labels and 28-class scoring (the fold included) refuses them, so a model
    is never scored against labels from the other class space. Returns the
    manifest's identity, or None for original labels.
    """
    path = label_dir / RELABEL_MANIFEST
    if not path.is_file():
        if n_scored == NUM_UNIFIED_CLASSES:
            raise ValueError(
                f"{n_scored}-class scoring needs barrblandskog labels, but {label_dir} "
                f"has no {RELABEL_MANIFEST}")
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("rule") != RELABEL_RULE:
        raise ValueError(f"{path}: rule {manifest.get('rule')!r}, expected {RELABEL_RULE!r}")
    if Path(manifest["outputs"]["nmd"]) != label_dir:
        raise ValueError(f"{path} names {manifest['outputs']['nmd']} as its NMD output, "
                         f"not {label_dir}")
    if n_scored != NUM_UNIFIED_CLASSES:
        raise ValueError(f"{label_dir} holds barrblandskog labels (class 28); "
                         f"{n_scored}-class scoring needs the original labels")
    return {"manifest": str(path), "sha256": sha256_file(path), "rule": manifest["rule"]}


def _require_in_range(what: str, values: np.ndarray, n: int) -> None:
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{what} must be integer class ids, got {values.dtype}")
    outside = (values < 0) | (values >= n)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} {what} pixel(s) outside [0, {n}): "
            f"values {np.unique(values[outside])[:10].tolist()}")


def tile_confusion(pred: np.ndarray, truth: np.ndarray, *, num_classes: int,
                   fold: tuple[int, int] | None) -> np.ndarray:
    """One tile's (truth, prediction) counts in the scored label space.

    Predictions must lie in [0, num_classes) before the fold and truth in the
    scored space after it; anything else raises. ``compute_miou`` alone would
    drop truth >= n without a trace, which is how a whole class vanishes from
    a score. Background truth (0) is excluded under the metric's v2 contract.
    """
    n = scored_classes(num_classes, fold)
    pred, truth = np.asarray(pred), np.asarray(truth)
    _require_in_range("prediction", pred, num_classes)
    if fold is not None:
        pred = np.where(pred == fold[0], fold[1], pred)
    _require_in_range("truth", truth, n)
    return compute_miou(pred, truth, n, ignore_index=0)["confusion_matrix"]


def tile_confusions_path(out: Path, cell: str) -> Path:
    return out.with_name(f"{out.stem}.{cell}.tiles.npz")


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


def score_cell(cell: str, column: str, ckpt: Path, tiles: list[str],
               holdout_dir: Path, label_dir: Path, device: str, *,
               num_classes: int, fold: tuple[int, int] | None,
               tiles_out: Path, check_column: bool) -> dict:
    """Score one checkpoint. ``check_column`` also requires the checkpoint's
    own backbone and img_size to be the column's (single-checkpoint mode)."""
    import torch

    infcmp = _load_infcmp()
    cfg = DISTILL[column]
    n = scored_classes(num_classes, fold)
    dev = torch.device(device)
    # Hashed before loading, and the loader refuses other bytes: the recorded
    # hash is of the weights that were scored, even on a mutable volume.
    identity = file_identity(ckpt)
    # Aux set comes from the checkpoint's own config, never from flags — the
    # same rule the matrix job follows, and the reason terramind (13 aux) and
    # the 11-aux ladder columns do not silently get the canonical 10.
    model_obj, epoch, ckpt_miou, _, ck_cfg = infcmp.load_model(
        str(ckpt), dev, backbone_name=cfg["backbone"],
        img_size=cfg["img_size"], return_checkpoint_config=True,
        expected_checkpoint_sha256=identity["sha256"],
        expected_checkpoint_size=identity["size"])
    per_tile: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    t0 = time.time()
    try:
        ck_cfg = ck_cfg or {}
        if ck_cfg.get("num_classes") != num_classes:
            raise ValueError(
                f"{cell}: checkpoint config num_classes={ck_cfg.get('num_classes')!r}, "
                f"but --num-classes is {num_classes}")
        if check_column:
            found = (ck_cfg.get("backbone_name"), ck_cfg.get("img_size"))
            if found != (cfg["backbone"], cfg["img_size"]):
                raise ValueError(
                    f"{cell}: checkpoint config backbone/img_size {found} is not "
                    f"column {column}'s {(cfg['backbone'], cfg['img_size'])}")
        aux_names = ck_cfg.get("enabled_aux_names")
        aux_names = list(aux_names) if aux_names else None
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
                # scoring it would compare different ground. The only skip.
                skipped.append(name)
                continue
            try:
                per_tile[name] = tile_confusion(
                    pred, truth, num_classes=num_classes, fold=fold)
            except ValueError as exc:
                raise ValueError(f"{cell}: tile {name}: {exc}") from exc
            if i % 25 == 0 or i == len(tiles):
                rate = (time.time() - t0) / i
                print(f"  [{cell}] {i}/{len(tiles)} "
                      f"{rate:.2f}s/tile eta={(len(tiles)-i)*rate/60:.1f}min",
                      flush=True)
    finally:
        del model_obj
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    confusion = np.zeros((n, n), dtype=np.int64)
    for matrix in per_tile.values():
        confusion += matrix
    if not confusion[1:, :].sum():
        # Every tile was skipped or unusable. An all-zero matrix yields None
        # headline metrics that would still enter the results list as a
        # scored cell; the caller records it as failed instead.
        raise RuntimeError(
            f"{cell}: no pixel scored — "
            f"{len(skipped)} of {len(tiles)} tile(s) skipped")
    out = derive(confusion)
    out.update({
        "cell": cell, "column": column, "backbone": cfg["backbone"],
        "img_size": cfg["img_size"], "checkpoint": str(ckpt),
        "checkpoint_sha256": identity["sha256"], "checkpoint_size": identity["size"],
        "ckpt_epoch": epoch, "ckpt_val_miou": ckpt_miou,
        "num_classes": num_classes, "scored_classes": n,
        "fold": fold_record(fold), "label_dir": str(label_dir),
        "tiles_scored": len(per_tile), "tiles_skipped": skipped,
        "tile_confusions": {"file": tiles_out.name,
                            "sha256": save_tile_confusions(tiles_out, per_tile)},
        "seconds": round(time.time() - t0, 1),
        "confusion_matrix": confusion.tolist(),
    })
    print(f"[{cell}] OA={out['overall_accuracy']} "
          f"mIoU={out['mean_iou']} over {out['pixels_scored']:,} px", flush=True)
    return out


def run_fingerprint(tiles: list[str], args) -> dict:
    """What a prior report must agree with before its cells may be reused.

    Cell name alone is not identity. A first run with --limit-tiles 1, then a
    changed checkpoint and an unlimited resume, would skip scoring entirely
    and write the new tile list over a result whose tiles_scored is 1 and
    whose checkpoint hash is the old one.
    """
    root = Path(__file__).resolve().parents[1]
    return {
        "protocol": SCHEMA,
        "code_sha256": {
            str(p.relative_to(root)): sha256_file(p)
            for p in [
                Path(__file__).resolve(),
                Path(__file__).with_name("inference_comparison.py"),
                root / "imint/training/unified_dataset.py",
                root / "imint/metrics.py",
                root / "imint/eval/tile_bootstrap.py",
            ]
        },
        "model_configs": json.loads(json.dumps(DISTILL)),
        "inputs": {
            name: {"tile": sha256_file(args.holdout_dir / f"{name}.npz"),
                   "label": sha256_file(args.label_dir / f"{name}.npz")}
            for name in tiles
        },
        "num_classes": args.num_classes,
        "fold": fold_record(args.fold),
        "holdout_dir": str(args.holdout_dir),
        "label_dir": str(args.label_dir),
        "label_manifest_sha256": (sha256_file(args.label_dir / RELABEL_MANIFEST)
                                  if (args.label_dir / RELABEL_MANIFEST).is_file() else None),
        "ckpt_root": None if args.ckpt_root is None else str(args.ckpt_root),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "cell_name": args.cell_name,
        "n_tiles": len(tiles),
        "tiles_sha256": hashlib.sha256("\n".join(tiles).encode()).hexdigest()[:16],
    }


def _reusable(cell: dict, ckpt: Path, out_dir: Path) -> bool:
    """A prior cell stands only if its checkpoint and per-tile file still do."""
    files = cell.get("tile_confusions")
    if not files or not ckpt.is_file():
        return False
    matrices = out_dir / files["file"]
    return (cell.get("checkpoint_sha256") == sha256_file(ckpt)
            and matrices.is_file() and files["sha256"] == sha256_file(matrices))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--holdout-dir", type=Path, default=Path("/cephfs/holdout_val_512"))
    ap.add_argument("--label-dir", type=Path, default=None,
                    help=f"label sidecars (ladder grid default {DEFAULT_LABEL_DIR}; "
                         "required with --checkpoint)")
    ap.add_argument("--ckpt-root", type=Path, default=None,
                    help=f"ladder grid root (default {DEFAULT_CKPT_ROOT})")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="score this one checkpoint instead of the ladder grid")
    ap.add_argument("--cell-name", default=None,
                    help="report name of the --checkpoint cell")
    ap.add_argument("--num-classes", type=int, choices=NUM_CLASSES_CHOICES, default=None,
                    help=f"the checkpoint's class count (ladder grid: "
                         f"{LADDER_NUM_CLASSES}; required with --checkpoint)")
    ap.add_argument("--fold", type=parse_fold, default=None, metavar="PRED:TARGET",
                    help="score predicted class PRED as TARGET, e.g. 28:3 "
                         "(--checkpoint only)")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"report (ladder grid default {DEFAULT_OUT}; required with "
                         "--checkpoint); an existing one is only continued with --resume")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-tiles", type=int, default=None,
                    help="score a deterministic spread of this many tiles")
    ap.add_argument("--models", default=None,
                    help="comma-separated subset of ladder columns; with "
                         "--checkpoint exactly one (supplies backbone and img_size)")
    ap.add_argument("--rungs", default=None,
                    help="comma-separated 28-class rungs (default 2,3,4; rung 1 is 23-class)")
    ap.add_argument("--git-sha", default="unknown")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells already present in --out")
    args = ap.parse_args()

    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else list(DISTILL))
    unknown = [m for m in models if m not in DISTILL]
    if unknown:
        ap.error(f"unknown ladder column(s): {unknown}")
    if args.checkpoint is not None:
        needed = [flag for flag, value in (
            ("--cell-name", args.cell_name), ("--label-dir", args.label_dir),
            ("--num-classes", args.num_classes), ("--models", args.models),
            ("--out", args.out))
            if value is None]
        if needed:
            ap.error(f"--checkpoint also needs {', '.join(needed)}")
        if args.ckpt_root is not None or args.rungs is not None:
            ap.error("--checkpoint scores one cell; --ckpt-root and --rungs "
                     "belong to the ladder grid")
        if not CELL_NAME.fullmatch(args.cell_name):
            ap.error(f"--cell-name {args.cell_name!r} must match {CELL_NAME.pattern}")
        if len(models) != 1:
            ap.error("--checkpoint takes exactly one --models column")
        plan = {args.cell_name: (models[0], args.checkpoint)}
    else:
        if args.cell_name is not None:
            ap.error("--cell-name names a --checkpoint cell")
        if args.fold is not None:
            ap.error("--fold applies to a --checkpoint cell; the ladder grid is 28-class")
        if args.num_classes not in (None, LADDER_NUM_CLASSES):
            ap.error(f"the ladder grid is {LADDER_NUM_CLASSES}-class")
        args.num_classes = LADDER_NUM_CLASSES
        if args.label_dir is None:
            args.label_dir = DEFAULT_LABEL_DIR
        if args.ckpt_root is None:
            args.ckpt_root = DEFAULT_CKPT_ROOT
        if args.out is None:
            args.out = DEFAULT_OUT
        rungs = ([int(r) for r in args.rungs.split(",") if r.strip()]
                 if args.rungs is not None else list(EVAL_RUNGS))
        if not rungs or any(r not in EVAL_RUNGS for r in rungs):
            ap.error("rungs must be a nonempty subset of 2,3,4; rung 1 is 23-class")
        plan = {f"{m}_r{r}": (m, args.ckpt_root / f"{m}_r{r}" / "best_model.pt")
                for m in models for r in rungs}
    try:
        n_scored = scored_classes(args.num_classes, args.fold)
        labels = label_set(args.label_dir, n_scored)
    except ValueError as exc:
        ap.error(str(exc))
    if args.out.exists() and not args.resume:
        ap.error(f"{args.out} exists: --resume continues it, or choose another --out")

    tiles = select_tiles(args.holdout_dir, args.label_dir, args.limit_tiles)
    print(f"=== {n_scored}-class holdout eval — {len(plan)} cell(s) x "
          f"{len(tiles)} tiles ===", flush=True)

    # Resume: a sweep is 20+ cells over many hours, and cells already scored
    # into --out are not re-run. Without this a single late failure costs the
    # whole sweep on restart.
    results, missing, failed = [], [], []
    done: set[str] = set()
    fingerprint = run_fingerprint(tiles, args)
    if args.resume and args.out.exists():
        try:
            prior = json.loads(args.out.read_text())
            if prior.get("fingerprint") != fingerprint:
                # Different inputs produced that file. Reusing its cells would
                # mix two scoring runs under one report.
                raise SystemExit(
                    f"--resume refused: {args.out} was written for different "
                    f"inputs.\n  prior: {prior.get('fingerprint')}\n  now:   "
                    f"{fingerprint}\nDelete it or point --out elsewhere.")
            # A cell is reusable only if its checkpoint is still the one that
            # produced it and its per-tile matrices are intact; a retrained
            # cell must be re-scored.
            reusable = []
            for c in prior.get("cells", []):
                if c["cell"] not in plan:
                    continue
                if _reusable(c, plan[c["cell"]][1], args.out.parent):
                    reusable.append(c)
                else:
                    print(f"[{c['cell']}] checkpoint or per-tile matrices "
                          f"changed — re-scoring", flush=True)
            results = reusable
            done = {c["cell"] for c in results}
            print(f"resuming — {len(done)} cell(s) reusable", flush=True)
        except (OSError, ValueError, KeyError) as exc:
            print(f"could not resume from {args.out}: {exc!r}", flush=True)

    # Per-tile matrices land here as each cell finishes, before the report.
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def _write() -> None:
        # Called after every cell, success or failure: a failure
        # after the last success would otherwise never reach the
        # file, which is exactly when the record matters most.
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(json.dumps({
            "schema": SCHEMA,
            "fingerprint": fingerprint,
            "git_sha": args.git_sha,
            "num_classes": args.num_classes,
            "scored_classes": n_scored,
            "fold": fold_record(args.fold),
            "unified_classes": NUM_UNIFIED_CLASSES,
            "holdout_dir": str(args.holdout_dir),
            "label_dir": str(args.label_dir),
            "label_set": labels,
            "tiles": tiles,
            "missing_cells": missing,
            "failed_cells": failed,
            "cells": results,
        }, indent=1))
        temporary.replace(args.out)

    for cell, (column, ckpt) in plan.items():
        if cell in done:
            print(f"[{cell}] already scored — skipping", flush=True)
            continue
        if not ckpt.exists():
            missing.append(cell)
            print(f"[{cell}] no checkpoint — skipping", flush=True)
            continue
        try:
            results.append(score_cell(
                cell, column, ckpt, tiles, args.holdout_dir, args.label_dir,
                args.device, num_classes=args.num_classes, fold=args.fold,
                tiles_out=tile_confusions_path(args.out, cell),
                check_column=args.checkpoint is not None))
        except Exception as exc:
            # One unscoreable cell must not cost the other twenty. The
            # dataset's fail-loud contracts are the expected case here:
            # croma/terramind refuse a tile whose s1_enrich_v predates
            # what they were trained on, which is a property of the
            # holdout set, not of this sweep.
            failed.append({"cell": cell, "error": repr(exc)[:400]})
            print(f"[{cell}] FAILED — {exc!r}"[:500], flush=True)
            _write()
            continue
        _write()

    _write()
    if not results:
        raise SystemExit("no cell scored — see failed_cells in the report")
    if failed:
        print(f"\n{len(failed)} cell(s) could not be scored:")
        for f in failed:
            print(f"  {f['cell']}: {f['error'][:140]}")
    best = max(results, key=lambda r: r["overall_accuracy"] or 0)
    print(f"\nBEST {n_scored}-class OA: {best['cell']} = {best['overall_accuracy']} "
          f"(mIoU {best['mean_iou']})")
    print(f"wrote {args.out}")
    return 1 if failed or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
