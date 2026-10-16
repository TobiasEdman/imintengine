#!/usr/bin/env python3
"""Rescore the distillation-ladder checkpoints under ``miou-ignore-target-v2``.

Each ladder cell (``<backbone>_r<N>``) stores the full validation confusion
matrix of every new-best epoch in ``training_log.json``. Both the historical
(unversioned) and the current mIoU are pure functions of that matrix, so the
best epoch can be rescored exactly without inference or a GPU.

Per cell the script

* recomputes the historical mIoU from the stored matrix and refuses the cell
  unless it reproduces the stored ``best_metric`` to 4 dp;
* loads ``best_model.pt`` (CPU, ``weights_only=True``) and refuses the cell
  unless its ``epoch`` and ``metrics.miou`` are the scored epoch's — the
  reverse check that the checkpoint on disk is the one being rescored;
* scores the same matrix with :func:`imint.metrics.miou_from_confusion`.

Default is a dry run that writes nothing. ``--write`` creates one
``miou_v2_rescore.json`` per cell beside ``training_log.json`` and the
``--summary-out`` file, all with exclusive create (mode ``'x'``), and only if
every cell passed and no target already exists. No existing file is opened
for writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.metrics import SEGMENTATION_METRIC_VERSION, miou_from_confusion

# docs/training/metric_migration.md calls pre-v2 scores "unversioned".
METRIC_VERSION_OLD = "miou-unversioned-v1"
OUTPUT_NAME = "miou_v2_rescore.json"
CELL_PATTERN = re.compile(r"^(?P<backbone>[a-z0-9]+)_r(?P<rung>\d+)$")
# The trainer's split when no split_<name>.txt exists beside the tiles
# (imint/training/unified_dataset.py: md5 of Path.name, which includes .npz).
SPLIT_RULE = "int(md5(tile_file_name_with_.npz).hexdigest(), 16) % 100 < 10"
IGNORE_INDEX = 0
_READ_BLOCK = 1 << 20


class CellError(RuntimeError):
    """A cell's stored evidence is inconsistent; it must not be rescored."""


def legacy_miou_from_confusion(cm: np.ndarray) -> tuple[float, dict[int, float]]:
    """Frozen pre-v2 mIoU: ``compute_miou`` at 905c656^1 (before PR #76).

    That version kept target==0 rows in the matrix, so a foreground
    prediction on a background target was a false positive for that class.
    Class 0 was skipped in the mean; zero-denominator classes were skipped.
    Do not "fix" this function — it exists to reproduce stored scores.
    """
    cm = np.asarray(cm, dtype=np.int64)
    per_class: dict[int, float] = {}
    ious = []
    for c in range(cm.shape[0]):
        if c == 0:
            continue
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else float("nan")
        per_class[c] = iou
        if not np.isnan(iou):
            ious.append(iou)
    return (float(np.mean(ious)) if ious else 0.0), per_class


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _iou_json(per_class: dict[int, float]) -> dict[str, float | None]:
    """Class index → IoU (6 dp); NaN (empty class) becomes null."""
    return {
        str(c): (None if math.isnan(iou) else round(float(iou), 6))
        for c, iou in per_class.items()
    }


def val_fingerprint(cm: np.ndarray) -> dict:
    """Aggregate target counts; not sample, pixel, or label identity.

    ``target_row_sums_sha256`` hashes the UTF-8 JSON list of per-class target
    pixel counts (``json.dumps([int, ...])``, default separators).
    """
    row_sums = [int(v) for v in cm.sum(axis=1)]
    total = int(sum(row_sums))
    return {
        "cm_total_pixels": total,
        "target_row_sums": row_sums,
        "target_row_sums_sha256": sha256_bytes(json.dumps(row_sums).encode()),
        "target0_share": round(row_sums[0] / total, 6) if total else None,
        "confusion_matrix_sha256": sha256_bytes(json.dumps(cm.tolist()).encode()),
    }


def split_identity(tiles_root: Path) -> dict:
    split_file = tiles_root / "split_val.txt"
    if split_file.is_file():
        return {"split_hash": sha256_file(split_file), "split_file": str(split_file),
                "split_note": "Current split-file bytes only; historical use is not established."}
    return {
        "split_hash": None,
        "split_rule": SPLIT_RULE,
        "split_note": (
            f"{split_file} does not exist at rescore time. The hash rule is the "
            "trainer fallback, but historical use and sample membership are not "
            "established. Target row sums describe class counts, not set identity."
        ),
    }


def load_checkpoint_cpu(path: Path, expected_sha256: str, expected_size: int) -> dict:
    """``weights_only=True`` CPU load of the exact bytes hashed for the record.

    Reuses the inference loader: it copies the bytes to a private TMPDIR file,
    re-verifies sha256/size, and allowlists both numpy pickle spellings.
    """
    from scripts.inference_comparison import _load_checkpoint_for_inference

    return _load_checkpoint_for_inference(
        path,
        map_location="cpu",
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )


def _best_epoch_record(log: dict, best_epoch: int) -> dict:
    records = [e for e in log.get("epochs", []) if e.get("epoch") == best_epoch]
    with_cm = [e for e in records if e.get("confusion_matrix") is not None]
    if not with_cm:
        raise CellError(f"no epoch record with a confusion_matrix for best_epoch={best_epoch}")
    if any(e["confusion_matrix"] != with_cm[0]["confusion_matrix"] for e in with_cm[1:]):
        raise CellError(f"{len(with_cm)} differing records for best_epoch={best_epoch}")
    return with_cm[0]


def _runner_up(log: dict, best_epoch: int) -> dict | None:
    """Best other epoch by stored (old) val_miou — selection-margin context."""
    others = [e for e in log.get("epochs", [])
              if e.get("epoch") != best_epoch and e.get("val_miou") is not None]
    if not others:
        return None
    top = max(others, key=lambda e: e["val_miou"])
    return {"epoch": top["epoch"], "old_val_miou": top["val_miou"],
            "has_confusion_matrix": top.get("confusion_matrix") is not None}


def rescore_cell(
    cell_dir: Path,
    *,
    load_checkpoint: Callable[[Path, str, int], dict] = load_checkpoint_cpu,
) -> dict:
    """Return the rescore record for one cell or raise :class:`CellError`."""
    match = CELL_PATTERN.match(cell_dir.name)
    if not match:
        raise CellError(f"not a ladder cell name: {cell_dir.name}")
    log_path = cell_dir / "training_log.json"
    ckpt_path = cell_dir / "best_model.pt"
    log_bytes = log_path.read_bytes()      # hash and parse the same bytes
    log = json.loads(log_bytes)

    best_epoch = log.get("best_epoch")
    stored = log.get("best_metric")
    if not isinstance(best_epoch, int) or best_epoch < 1 or stored is None:
        raise CellError(f"best_epoch/best_metric missing: {best_epoch!r}/{stored!r}")
    record = _best_epoch_record(log, best_epoch)
    cm = np.asarray(record["confusion_matrix"], dtype=np.int64)
    n = cm.shape[0]
    if cm.ndim != 2 or cm.shape != (n, n):
        raise CellError(f"confusion matrix is not square: {cm.shape}")

    old, old_per_class = legacy_miou_from_confusion(cm)
    new, new_per_class = miou_from_confusion(cm, ignore_index=IGNORE_INDEX)
    errors = []
    if round(old, 4) != round(float(stored), 4):
        errors.append(f"old mIoU {old:.6f} does not reproduce stored best_metric {stored}")
    if record.get("val_miou") is not None and round(old, 4) != round(record["val_miou"], 4):
        errors.append(f"old mIoU {old:.6f} != epoch val_miou {record['val_miou']}"
                      " (selection metric was not mIoU?)")

    ckpt_size = ckpt_path.stat().st_size
    ckpt_sha = sha256_file(ckpt_path)
    ckpt = load_checkpoint(ckpt_path, ckpt_sha, ckpt_size)
    ckpt_epoch = ckpt.get("epoch")
    ckpt_miou = (ckpt.get("metrics") or {}).get("miou")
    ckpt_classes = (ckpt.get("config") or {}).get("num_classes")
    if ckpt_epoch is None or int(ckpt_epoch) != best_epoch:
        errors.append(f"best_model.pt epoch {ckpt_epoch!r} != best_epoch {best_epoch}")
    if ckpt_miou is None or round(float(ckpt_miou), 4) != round(float(stored), 4):
        errors.append(f"best_model.pt metrics.miou {ckpt_miou!r} != best_metric {stored}")
    if ckpt_classes is not None and int(ckpt_classes) != n:
        errors.append(f"best_model.pt num_classes {ckpt_classes} != matrix size {n}")
    if errors:
        raise CellError("; ".join(errors))

    epochs = log.get("epochs", [])
    # Earlier new-best epochs also stored their matrix: score them too, so
    # "would v2 have selected another epoch" is answered where evidence exists.
    recorded = []
    v2_best, best_selection_score = None, float("-inf")
    for e in sorted(epochs, key=lambda e: e["epoch"]):
        if e.get("confusion_matrix") is None:
            continue
        score = miou_from_confusion(
            np.asarray(e["confusion_matrix"], dtype=np.int64), IGNORE_INDEX)[0]
        entry = {"epoch": e["epoch"],
                 "old_miou": round(legacy_miou_from_confusion(e["confusion_matrix"])[0], 6),
                 "new_miou": round(score, 6)}
        recorded.append(entry)
        # Trainer compares its 4-dp metric with strict >: the first tie wins.
        selection_score = round(score, 4)
        if selection_score > best_selection_score:
            v2_best, best_selection_score = entry, selection_score
    return {
        "cell": cell_dir.name,
        "cell_path": str(cell_dir),
        "backbone": match["backbone"],
        "rung": int(match["rung"]),
        "num_classes": n,
        "best_epoch": best_epoch,
        "stored_best_miou": float(stored),
        "old_miou_recomputed": round(old, 6),
        "new_miou": round(new, 6),
        "delta": round(new - old, 6),
        "old_per_class_iou": _iou_json(old_per_class),
        "new_per_class_iou": _iou_json(new_per_class),
        "metric_version_old": METRIC_VERSION_OLD,
        "metric_version_new": SEGMENTATION_METRIC_VERSION,
        "reverse_check": {
            "old_reproduces_stored_4dp": True,
            "checkpoint_epoch": int(ckpt_epoch),
            "checkpoint_metrics_miou": float(ckpt_miou),
            "checkpoint_metric_version": ckpt.get("metric_version"),
        },
        "best_model_pt": {"sha256": ckpt_sha, "size": ckpt_size},
        "training_log_json": {"sha256": sha256_bytes(log_bytes), "size": len(log_bytes)},
        "val_fingerprint": val_fingerprint(cm),
        "log_context": {
            "n_epochs": len(epochs),
            "n_epochs_with_confusion_matrix": len(recorded),
            "recorded_epochs": recorded,
            "v2_best_recorded_epoch": v2_best["epoch"],
            "runner_up_by_old_val_miou": _runner_up(log, best_epoch),
            "status": log.get("status"),
            "started_at": log.get("started_at"),
            "updated_at": log.get("updated_at"),
            "last_epoch_elapsed_s": epochs[-1].get("elapsed_s") if epochs else None,
        },
    }


def write_exclusive(path: Path, payload: dict) -> None:
    """Create ``path`` with ``payload``; raise FileExistsError if it exists."""
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


def discover_cells(ladder_root: Path) -> list[Path]:
    return sorted(p for p in ladder_root.iterdir()
                  if p.is_dir() and CELL_PATTERN.match(p.name)
                  and (p / "training_log.json").is_file())


def run(args: argparse.Namespace) -> int:
    ladder_root = Path(args.ladder_root)
    cells = discover_cells(ladder_root)
    print(f"ladder root {ladder_root}: {len(cells)} cells "
          f"(expected {args.expected_cells}); mode={'WRITE' if args.write else 'DRY-RUN'}")
    if len(cells) != args.expected_cells:
        print(f"REFUSING: found {len(cells)} cells, expected {args.expected_cells}")
        return 2
    if args.write:
        targets = [c / OUTPUT_NAME for c in cells]
        if args.summary_out:
            targets.append(Path(args.summary_out))
        existing = [str(t) for t in targets if t.exists()]
        if existing:
            print(f"REFUSING: outputs already exist: {existing}")
            return 2

    script_sha = sha256_file(Path(__file__).resolve())
    run_meta = {
        "method": "stored-confusion-matrix",
        "gpu_seconds": 0,
        "git_commit": os.environ.get("RESCORE_GIT_COMMIT"),
        "script": "scripts/ladder_miou_v2_rescore.py",
        "script_sha256": script_sha,
        "metric_version_old": METRIC_VERSION_OLD,
        "metric_version_new": SEGMENTATION_METRIC_VERSION,
        "ignore_index": IGNORE_INDEX,
        "produced_at": datetime.now(timezone.utc).isoformat(),
        **split_identity(Path(args.tiles_root)),
    }

    results, failures = [], {}
    for cell in cells:
        try:
            rec = rescore_cell(cell, load_checkpoint=load_checkpoint_cpu)
        except CellError as exc:
            failures[cell.name] = str(exc)
            print(f"FAIL {cell.name}: {exc}")
            continue
        results.append({**rec, **run_meta})
        print(f"OK   {cell.name:<18} n={rec['num_classes']} ep={rec['best_epoch']:>2} "
              f"stored={rec['stored_best_miou']:.4f} old={rec['old_miou_recomputed']:.4f} "
              f"new={rec['new_miou']:.4f} delta={rec['delta']:+.4f}")

    summary = {**run_meta, "ladder_root": str(ladder_root),
               "n_cells": len(cells), "n_ok": len(results),
               "failures": failures, "cells": results}
    if failures:
        print(f"{len(failures)} cell(s) failed; nothing written")
    elif args.write:
        for rec in results:
            write_exclusive(Path(rec["cell_path"]) / OUTPUT_NAME, rec)
        if args.summary_out:
            write_exclusive(Path(args.summary_out), summary)
        print(f"wrote {len(results)} x {OUTPUT_NAME}"
              + (f" and {args.summary_out}" if args.summary_out else ""))
    print("=== RESCORE_SUMMARY_BEGIN ===")
    print(json.dumps(summary, sort_keys=True))
    print("=== RESCORE_SUMMARY_END ===")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ladder-root", default="/cephfs/checkpoints/ladder")
    parser.add_argument("--expected-cells", type=int, required=True)
    parser.add_argument("--tiles-root", default="/cephfs/unified_v2_512",
                        help="Directory the trainer would read split_val.txt from.")
    parser.add_argument("--write", action="store_true",
                        help="Create the output files (default: dry run).")
    parser.add_argument("--summary-out", default=None)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
