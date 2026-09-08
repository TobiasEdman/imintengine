#!/usr/bin/env python3
"""Render the ladder's inference matrix: K spread holdout tiles × every cell.

The dashboard's numbers say WHICH cell is best; this shows WHAT each cell
actually paints. K (~10) geographically spread tiles are picked ONCE from
the held-out validation set (``/cephfs/holdout_val_512`` — never trained on
by any rung) and frozen to ``tiles.json``; every completed ladder checkpoint
then renders its prediction on exactly those tiles, so cells are visually
comparable along both axes (same tile across models, same model across
rungs).

Selection is farthest-point (max-min) on the tiles' SWEREF99 TM centers,
seeded at the southernmost candidate — deterministic, no RNG. Frozen-once:
a re-run reuses the existing ``tiles.json`` (state on disk), so later cells
land on the same tiles even if the holdout pool changes.

Per cell: predictions keyed to the exact checkpoint bytes (sha256) — a
re-trained ``best_model.pt`` invalidates and re-renders the cell; an
unchanged one is skipped. Shared per-tile RGB (summer frame) and in-tile
truth panels render once. Everything the dashboard needs lands in
``matrix.json``.

    python3 scripts/ladder_inference_matrix.py \
        --holdout-dir /cephfs/holdout_val_512 \
        --checkpoint-root /cephfs/checkpoints/ladder \
        --out-dir /cephfs/ladder_inference \
        --device cuda
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imint.training.unified_schema import NUM_UNIFIED_CLASSES, UNIFIED_COLORS

from gen_ladder_manifests import DISTILL, RUNGS
from infer_tiles import ckpt_sha256

TILES_NAME = "tiles.json"
MATRIX_NAME = "matrix.json"
K_DEFAULT = 10


def _load_infcmp():
    spec = importlib.util.spec_from_file_location(
        "_infcmp", str(Path(__file__).resolve().parent / "inference_comparison.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tile_center(path: Path) -> tuple[float, float] | None:
    """(easting, northing) from the npz, or None if either key is absent.

    ``np.load`` inflates only the members it is asked for — two scalars —
    so scanning ~1.5k tiles stays cheap.
    """
    try:
        with np.load(path, allow_pickle=False) as z:
            if "easting" not in z or "northing" not in z:
                return None
            return float(z["easting"]), float(z["northing"])
    except (OSError, ValueError, zipfile.BadZipFile):
        return None


def select_spread(centers: dict[str, tuple[float, float]], k: int) -> list[str]:
    """Farthest-point subset of ``centers`` — deterministic, no RNG.

    Seeded at the southernmost tile (ties broken by name), then greedily
    adds the candidate maximizing its minimum distance to the picked set.
    Max-min beats random sampling here: K is tiny and the pool clusters
    around agricultural regions, so a random draw routinely lands two
    tiles in the same cluster and leaves whole regions unseen.
    """
    if k >= len(centers):
        return sorted(centers)
    start = min(centers, key=lambda n: (centers[n][1], n))
    picked = [start]
    rest = {n for n in centers if n != start}
    while len(picked) < k:
        best_name, best_d = None, -1.0
        for n in sorted(rest):
            e, no = centers[n]
            d = min((e - centers[p][0]) ** 2 + (no - centers[p][1]) ** 2
                    for p in picked)
            if d > best_d:
                best_name, best_d = n, d
        picked.append(best_name)
        rest.remove(best_name)
    return picked


def freeze_tiles(holdout_dir: Path, out_dir: Path, k: int,
                 git_sha: str | None) -> list[dict]:
    """Pick + freeze the tile panel, or reuse the existing freeze."""
    tiles_p = out_dir / TILES_NAME
    if tiles_p.exists():
        doc = json.loads(tiles_p.read_text())
        print(f"tile panel already frozen ({len(doc['tiles'])} tiles) — reusing")
        return doc["tiles"]

    centers: dict[str, tuple[float, float]] = {}
    skipped = 0
    for p in sorted(holdout_dir.glob("*.npz")):
        c = _tile_center(p)
        if c is None:
            skipped += 1
            continue
        centers[p.stem] = c
    if len(centers) < k:
        raise SystemExit(f"only {len(centers)} locatable holdout tiles "
                         f"(need {k}); {skipped} skipped")
    print(f"{len(centers)} candidates ({skipped} without coordinates)")

    names = select_spread(centers, k)
    tiles = [{
        "name": n,
        "easting": centers[n][0],
        "northing": centers[n][1],
        # holdoutval_<id>_<id>_<year>.npz — the year is display metadata.
        "year": int(n.rsplit("_", 1)[1]) if n.rsplit("_", 1)[1].isdigit() else None,
    } for n in names]
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_p.write_text(json.dumps({
        "k": k, "n_candidates": len(centers),
        "algorithm": "farthest-point, southernmost seed",
        "holdout_dir": str(holdout_dir), "git_sha": git_sha,
        "tiles": tiles,
    }, indent=1))
    print(f"froze {k} tiles → {tiles_p}")
    return tiles


def colorize(label: np.ndarray) -> np.ndarray:
    """Class map → RGB via the unified palette (LUT, not a per-class loop)."""
    lut = np.zeros((max(NUM_UNIFIED_CLASSES, int(label.max()) + 1, 256), 3),
                   dtype=np.uint8)
    for cls, c in UNIFIED_COLORS.items():
        lut[cls] = c
    return lut[label.astype(np.int64)]


def summer_rgb(spectral: np.ndarray) -> np.ndarray:
    """2–98% stretched RGB from the summer frame (display only — the raw
    reflectance rule applies to MODEL inputs, not thumbnails)."""
    bands_per_frame = 6
    n_frames = spectral.shape[0] // bands_per_frame
    summer = min(2, n_frames - 1) if n_frames >= 4 else min(1, n_frames - 1)
    base = summer * bands_per_frame
    rgb = np.stack([spectral[base + 2], spectral[base + 1], spectral[base + 0]],
                   axis=-1).astype(np.float32)
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def render_shared(tiles: list[dict], holdout_dir: Path, out_dir: Path,
                  nmd2018_label_dir: Path, nmd2023_label_dir: Path) -> None:
    """RGB + both training-truth panels, once per tile (idempotent).

    Two truths because the ladder trains against two vocabularies —
    BOTH read from reference-built sidecars, never the in-tile ``label``:
    the holdout tiles' baked label carries RAW NMD2018 codes (max 128,
    never unified-mapped at fetch) and no LPIS/SKS masks, so it can
    render neither vocabulary. ``_truth`` is the 23-class unified v5
    (rung 1's target, NMD2018 + LPIS crops + SKS clearcuts) from the
    nmd2018 sidecar; ``_truth28`` is the 28-class (rungs 2-4's target)
    from the nmd2023 sidecar. Both use UNIFIED_COLORS — 23-class values
    are a subset of 0-27, so shared classes keep identical colours.

    A missing sidecar is a broken precondition, not a skippable gap: the
    panels claim to show what the rungs trained on, so rendering without
    them would misrepresent the matrix. Build both holdout sidecar sets
    first (k8s/build-labels-holdout-nmd2018-job.yaml and
    k8s/build-labels-holdout-nmd2023-job.yaml).
    """
    for t in tiles:
        rgb_p = out_dir / "_rgb" / f"{t['name']}.png"
        truth_p = out_dir / "_truth23" / f"{t['name']}.png"
        truth28_p = out_dir / "_truth28" / f"{t['name']}.png"
        if rgb_p.exists() and truth_p.exists() and truth28_p.exists():
            continue
        side23 = nmd2018_label_dir / f"{t['name']}.npz"
        side28 = nmd2023_label_dir / f"{t['name']}.npz"
        for side, job in ((side23, "build-labels-holdout-nmd2018-job.yaml"),
                          (side28, "build-labels-holdout-nmd2023-job.yaml")):
            if not side.exists():
                raise FileNotFoundError(
                    f"truth sidecar missing for frozen tile {t['name']}: "
                    f"{side} — run k8s/{job} before the matrix job")
        with np.load(holdout_dir / f"{t['name']}.npz", allow_pickle=False) as z:
            _save_png(summer_rgb(z["spectral"]), rgb_p)
        with np.load(side23, allow_pickle=False) as z:
            _save_png(colorize(z["label"]), truth_p)
        with np.load(side28, allow_pickle=False) as z:
            _save_png(colorize(z["label"]), truth28_p)
        print(f"  shared panels: {t['name']}")


def render_cell(model: str, rung: int, ckpt: Path, tiles: list[dict],
                holdout_dir: Path, out_dir: Path, device: str) -> dict | None:
    """One (model, rung) cell: K prediction PNGs keyed to the ckpt bytes."""
    import torch

    cell_dir = out_dir / f"{model}_r{rung}"
    cell_p = cell_dir / "_cell.json"
    sha = ckpt_sha256(str(ckpt))
    if cell_p.exists():
        cell = json.loads(cell_p.read_text())
        if (cell.get("ckpt_sha") == sha
                and all((cell_dir / f"{t['name']}.png").exists() for t in tiles)):
            print(f"[{model}_r{rung}] unchanged (sha {sha[:8]}) — skipping")
            return cell

    infcmp = _load_infcmp()
    cfg = DISTILL[model]
    dev = torch.device(device)
    # Aux set from the checkpoint's own config — reconstructing it from
    # flags is how terramind died in the distill stage (13 vs 11 aux).
    # ONE load through the reviewed safe loader (weights_only=True +
    # descriptor sealing); a raw unsafe torch.load on the shared PVC
    # would hand code execution to any checkpoint writer.
    model_obj, epoch, miou, _, ck_cfg = infcmp.load_model(
        str(ckpt), dev, backbone_name=cfg["backbone"],
        img_size=cfg["img_size"], return_checkpoint_config=True)
    aux_names = (ck_cfg or {}).get("enabled_aux_names")
    aux_names = list(aux_names) if aux_names else None
    print(f"[{model}_r{rung}] epoch={epoch} mIoU={miou} sha={sha[:8]}")

    for t in tiles:
        pred = infcmp.run_inference(
            model_obj, str(holdout_dir / f"{t['name']}.npz"), dev,
            img_size=cfg["img_size"], aux_channel_names=aux_names)
        _save_png(colorize(np.asarray(pred)), cell_dir / f"{t['name']}.png")
        print(f"  {t['name']}")
    del model_obj
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    cell = {"ckpt_sha": sha, "epoch": epoch, "ckpt_miou": miou,
            "img_size": cfg["img_size"], "backbone": cfg["backbone"]}
    cell_p.write_text(json.dumps(cell, indent=1))
    return cell


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout-dir", default="/cephfs/holdout_val_512")
    ap.add_argument("--checkpoint-root", default="/cephfs/checkpoints/ladder")
    ap.add_argument("--out-dir", default="/cephfs/ladder_inference")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--git-sha", default=None)
    ap.add_argument("--nmd2018-label-dir", default="/cephfs/nmd2018_labels",
                    help="sidecar dir with 23-class unified v5 labels "
                         "(rung 1's training truth incl. LPIS+SKS) for the "
                         "truth23 panel")
    ap.add_argument("--nmd2023-label-dir", default="/cephfs/nmd2023_labels",
                    help="sidecar dir with 28-class unified labels "
                         "(rungs 2-4's training truth) for the truth28 panel")
    args = ap.parse_args()

    holdout = Path(args.holdout_dir)
    root = Path(args.checkpoint_root)
    out_dir = Path(args.out_dir)

    tiles = freeze_tiles(holdout, out_dir, args.k, args.git_sha)
    render_shared(tiles, holdout, out_dir, Path(args.nmd2018_label_dir),
                  Path(args.nmd2023_label_dir))

    cells: dict[str, dict] = {}
    for model in DISTILL:
        for rung in RUNGS:
            ckpt = root / f"{model}_r{rung}" / "best_model.pt"
            if not ckpt.exists():
                continue
            cell = render_cell(model, rung, ckpt, tiles, holdout,
                               out_dir, args.device)
            if cell:
                cells[f"{model}_r{rung}"] = cell

    (out_dir / MATRIX_NAME).write_text(json.dumps({
        "git_sha": args.git_sha,
        "tiles": tiles,
        "cells": cells,
    }, indent=1))
    print(f"matrix: {len(cells)} cells × {len(tiles)} tiles → "
          f"{out_dir / MATRIX_NAME}")


if __name__ == "__main__":
    main()
