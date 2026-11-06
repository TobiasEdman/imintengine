"""Per-tile confusion matrices and a paired tile bootstrap over them.

``scripts/ladder_holdout_eval.py`` keeps one (truth, prediction) confusion
matrix per holdout tile so an interval can be put on the difference between
two models scored on the same tiles. A draw resamples whole tiles, because the
pixels of one tile are not independent, and both runs get the same draw, so
the comparison stays paired.

IoU follows the eval's ``derive``: per class ``tp / (tp + fp + fn)``, row
``ignore_index`` (background truth) dropped, and mean IoU over the classes
that have truth support in that matrix.

    python -m imint.eval.tile_bootstrap A.tiles.npz B.tiles.npz \\
        --draws 1000 --seed 42 --out a_minus_b.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _check_stack(stack: np.ndarray) -> None:
    if stack.ndim != 3 or stack.shape[1] != stack.shape[2] or not len(stack):
        raise ValueError(f"tile confusions must be (tiles, n, n), got {stack.shape}")
    if not np.issubdtype(stack.dtype, np.integer) or (stack < 0).any():
        raise ValueError("tile confusions must hold non-negative integer counts")


def save_tile_confusions(path: Path, matrices: Mapping[str, np.ndarray]) -> str:
    """Write ``{tile: matrix}`` atomically as ``tiles`` + ``confusion`` arrays.

    Two arrays rather than one key per tile: a tile name never has to be a
    valid ``savez`` keyword. Returns the written file's sha256.
    """
    names = sorted(matrices)
    if not names:
        raise ValueError("no tile matrices to save")
    stack = np.stack([np.asarray(matrices[name]) for name in names])
    _check_stack(stack)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, tiles=np.array(names), confusion=stack.astype(np.int64))
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tile_confusions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        names = [str(name) for name in archive["tiles"]]
        stack = archive["confusion"]
    _check_stack(stack)
    if len(names) != len(stack) or len(set(names)) != len(names):
        raise ValueError(f"{path}: tile names do not match the matrices one to one")
    return dict(zip(names, stack))


def _iou(matrices: np.ndarray, ignore_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-class IoU and mean IoU of a (..., n, n) stack; NaN where undefined."""
    m = matrices.astype(np.float64, copy=True)
    m[..., ignore_index, :] = 0
    tp = np.diagonal(m, axis1=-2, axis2=-1)
    support = m.sum(axis=-1)
    denominator = support + m.sum(axis=-2) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denominator > 0, tp / denominator, np.nan)
    present = support > 0
    present[..., ignore_index] = False
    with np.errstate(divide="ignore", invalid="ignore"):
        miou = np.where(present, iou, 0).sum(axis=-1) / present.sum(axis=-1)
    return iou, miou


def _interval(differences: np.ndarray, ci: float) -> list[float] | None:
    defined = differences[np.isfinite(differences)]
    if not len(defined):
        return None
    tail = 50 * (1 - ci)
    return [float(v) for v in np.percentile(defined, [tail, 100 - tail])]


def _value(x: float) -> float | None:
    return None if np.isnan(x) else float(x)


def paired_iou_difference(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], *,
                          draws: int = 1000, seed: int = 42, ci: float = 0.95,
                          ignore_index: int = 0) -> dict:
    """IoU of run ``a`` minus run ``b``, per class and mean, with percentile intervals.

    Both runs must cover the same tiles in the same label space and must have
    scored the same truth pixels on every tile (equal row sums): a bootstrap
    over two different grounds would not be paired.

    A class's interval uses only the draws where its IoU is defined in both
    runs (a nonzero denominator); ``defined_draws`` says how many that was,
    and the interval is None when there were none. A rare class can thus
    rest on fewer draws than ``draws``.
    """
    if draws < 1 or not 0 < ci < 1:
        raise ValueError("draws must be positive and ci in (0, 1)")
    if set(a) != set(b):
        raise ValueError(f"runs cover different tiles: only in a {sorted(set(a) - set(b))[:5]}, "
                         f"only in b {sorted(set(b) - set(a))[:5]}")
    tiles = sorted(a)
    if not tiles:
        raise ValueError("no tiles to bootstrap")
    stack_a = np.stack([np.asarray(a[t]) for t in tiles])
    stack_b = np.stack([np.asarray(b[t]) for t in tiles])
    _check_stack(stack_a)
    _check_stack(stack_b)
    if stack_a.shape != stack_b.shape:
        raise ValueError(f"runs are in different label spaces: {stack_a.shape[1:]} vs {stack_b.shape[1:]}")
    differs = [t for t, ra, rb in zip(tiles, stack_a.sum(axis=2), stack_b.sum(axis=2))
               if not np.array_equal(ra, rb)]
    if differs:
        raise ValueError(f"runs scored different truth on {len(differs)} tile(s), e.g. {differs[:3]}")
    n_tiles, n = stack_a.shape[:2]
    rng = np.random.default_rng(seed)
    # How many times each tile is drawn, per resample. Float matmul is exact
    # here: every resampled count stays far below 2**53.
    weights = np.stack([np.bincount(rng.integers(n_tiles, size=n_tiles), minlength=n_tiles)
                        for _ in range(draws)]).astype(np.float64)
    resampled_a = (weights @ stack_a.reshape(n_tiles, -1)).reshape(draws, n, n)
    resampled_b = (weights @ stack_b.reshape(n_tiles, -1)).reshape(draws, n, n)
    iou_a, miou_a = _iou(resampled_a, ignore_index)
    iou_b, miou_b = _iou(resampled_b, ignore_index)
    point_a, point_miou_a = _iou(stack_a.sum(axis=0), ignore_index)
    point_b, point_miou_b = _iou(stack_b.sum(axis=0), ignore_index)
    per_class = {}
    for c in range(n):
        if c == ignore_index:
            continue
        difference = iou_a[:, c] - iou_b[:, c]
        per_class[c] = {"a": _value(point_a[c]), "b": _value(point_b[c]),
                        "difference": _value(point_a[c] - point_b[c]),
                        "interval": _interval(difference, ci),
                        "defined_draws": int(np.isfinite(difference).sum())}
    return {"tiles": n_tiles, "classes": n, "draws": draws, "seed": seed, "ci": ci,
            "ignore_index": ignore_index,
            "mean_iou": {"a": _value(point_miou_a), "b": _value(point_miou_b),
                         "difference": _value(point_miou_a - point_miou_b),
                         "interval": _interval(miou_a - miou_b, ci)},
            "per_class": per_class}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a", type=Path, help="per-tile confusions of run a")
    ap.add_argument("b", type=Path, help="per-tile confusions of run b (subtracted)")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = paired_iou_difference(load_tile_confusions(args.a), load_tile_confusions(args.b),
                                   draws=args.draws, seed=args.seed, ci=args.ci)
    report["inputs"] = {side: {"file": str(path),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for side, path in (("a", args.a), ("b", args.b))}
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
