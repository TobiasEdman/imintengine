"""scripts/score_against_truth.py — Stage B: CPU scoring from the cache.

Reads the per-tile prediction cache written by ``infer_tiles.py`` and scores it
against a truth source using the EXISTING scoring cores unchanged:

  * ``--truth nfi``   → ``validate_against_nfi.score_against_nfi``
  * ``--truth lucas`` → ``validate_against_lucas.score_against_lucas``

No GPU, no model load, no re-implementation of any metric. The only new code is
the cache-backed ``predict_fn`` (``cache_predict.make_cached_predict_fn``); it
satisfies the exact contract those scorers call, so the JSON/parquet outputs are
identical to the fused validators (the bit-parity gate). Because scoring is pure
CPU on cached arrays, every truth source scores from the SAME cache — inference
is paid once, not once per truth.

    # NFI
    python scripts/score_against_truth.py --truth nfi \
        --cache-dir /cephfs/pred_cache --ckpt-sha <sha> \
        --plot-index /data/nfi/nfi_index_unified_v2_512.parquet \
        --num-classes 28 --out /data/nfi_eval/nfi_validation_tessera.json

    # LUCAS (fraction member)
    python scripts/score_against_truth.py --truth lucas \
        --cache-dir /cephfs/pred_cache --ckpt-sha <sha> \
        --lucas-index /data/lucas/lucas_tile_index.parquet \
        --data-dir /data/unified_v2_512 --num-classes 28 --auto-fraction \
        --out /data/nfi_eval/lucas-validation-tessera.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache_predict import make_cached_predict_fn  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).resolve().parent / f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cache_has_fracs(cache_dir, ckpt_sha) -> bool:
    """True if the cache stores a real (non-degenerate) fraction head.

    A checkpoint with no fraction head still gets a ``fracs`` array written by
    ``infer_tiles.py`` (the model forward always returns a frac_logits tensor);
    for auto-detection we treat presence of the key as the frac signal, matching
    ``--auto-fraction`` on the validators. The MANIFEST is authoritative for the
    checkpoint identity; the per-tile key presence is the runtime signal.
    """
    sha_dir = Path(cache_dir) / ckpt_sha
    sample = next(sha_dir.glob("*.npz"), None)
    if sample is None:
        return False
    with np.load(sample) as z:
        return "fracs" in z.files


def score_nfi(args) -> dict:
    van = _load_script("validate_against_nfi")
    index_df = pd.read_parquet(args.plot_index)
    print(f"plot index: {len(index_df):,} plots on "
          f"{index_df['tile_name'].nunique()} tiles")

    exists = index_df["tile_path"].map(os.path.exists)
    if not exists.all():
        gone = int((~exists).sum())
        print(f"dropping {gone} plots on tiles no longer in the dataset")
        index_df = index_df[exists].copy()

    # Same centre-crop remap the fused validator applies (run_inference crops to
    # img_size; the cache is stored in crop coords, so plot (row,col) must be
    # shifted into crop space and border plots dropped).
    sample_path = index_df["tile_path"].iloc[0]
    tile_h = int(np.load(sample_path, allow_pickle=True)["spectral"].shape[-1])
    off = van.crop_offset(tile_h, args.img_size)
    cs = min(args.img_size, tile_h)
    before = len(index_df)
    index_df = index_df[
        (index_df["row"] >= off) & (index_df["row"] < off + cs)
        & (index_df["col"] >= off) & (index_df["col"] < off + cs)
    ].copy()
    index_df["row"] -= off
    index_df["col"] -= off
    print(f"crop offset={off} (tile {tile_h}→{cs}); kept {len(index_df)}/{before}")

    predict_fn = make_cached_predict_fn(
        args.cache_dir, args.ckpt_sha, want_fracs=False)
    per_plot = [] if args.dump_per_plot else None
    results = van.score_against_nfi(
        index_df, predict_fn, num_classes=args.num_classes,
        dominant_frac=args.dominant_frac, per_plot_sink=per_plot)
    results["_meta"] = {
        "cache_dir": str(args.cache_dir), "ckpt_sha": args.ckpt_sha,
        "img_size": args.img_size, "plots_in_crop": len(index_df),
        "plots_total": before,
    }
    if args.dump_per_plot:
        _dump_parquet(per_plot, args.dump_per_plot, "plots")
    return results


def score_lucas(args) -> dict:
    val = _load_script("validate_against_lucas")
    index_df = pd.read_parquet(args.lucas_index)
    index_df["tile_path"] = index_df["tile_name"].map(
        lambda n: os.path.join(args.data_dir, f"{n}.npz"))
    print(f"LUCAS index: {len(index_df):,} points on "
          f"{index_df['tile_name'].nunique()} tiles")

    exists = index_df["tile_path"].map(os.path.exists)
    if not exists.all():
        gone = int((~exists).sum())
        print(f"dropping {gone} points on tiles not in {args.data_dir}")
        index_df = index_df[exists].copy()
    if index_df.empty:
        raise SystemExit(f"no LUCAS tiles found under {args.data_dir}")

    index_df = val._apply_crop_offset(index_df, args.img_size)

    is_fraction = args.use_fraction_head
    if args.auto_fraction and not is_fraction:
        is_fraction = _cache_has_fracs(args.cache_dir, args.ckpt_sha)
        print(f"  auto-fraction: cached fracs "
              f"{'present → L2a on' if is_fraction else 'absent → L2b only'}")

    predict_fn = make_cached_predict_fn(
        args.cache_dir, args.ckpt_sha, want_fracs=is_fraction)
    tile_years = val.build_tile_years(index_df)

    per_point = [] if args.dump_per_point else None
    results = val.score_against_lucas(
        index_df, predict_fn, num_classes=args.num_classes,
        min_support=args.min_support, is_fraction=is_fraction,
        tile_years=tile_years, per_point_sink=per_point)
    results["_meta"] = {
        "cache_dir": str(args.cache_dir), "ckpt_sha": args.ckpt_sha,
        "img_size": args.img_size, "num_classes": args.num_classes,
        "min_support": args.min_support, "is_fraction_member": is_fraction,
        "points_scored": len(index_df),
        "n_tiles": int(index_df["tile_name"].nunique()),
    }
    if args.dump_per_point:
        _dump_parquet(per_point, args.dump_per_point, "points")
    return results


def _dump_parquet(records, path, unit):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(p, index=False)
    print(f"wrote {p} ({len(records)} {unit})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", required=True, choices=["nfi", "lucas"])
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ckpt-sha", required=True,
                    help="checkpoint sha subdir name (from infer_tiles.py)")
    ap.add_argument("--img-size", type=int, default=504,
                    help="must match the crop infer_tiles.py used")
    ap.add_argument("--num-classes", type=int, default=28)
    ap.add_argument("--out", required=True)
    # NFI
    ap.add_argument("--plot-index", default=None,
                    help="(nfi) parquet from nfi_tile_coverage.py")
    ap.add_argument("--dominant-frac", type=float, default=0.7)
    ap.add_argument("--dump-per-plot", default=None)
    # LUCAS
    ap.add_argument("--lucas-index", default=None,
                    help="(lucas) L1 parquet lucas_tile_index.parquet")
    ap.add_argument("--data-dir", default=None,
                    help="(lucas) tile root for tile_path + tile_year")
    ap.add_argument("--min-support", type=int, default=20)
    ap.add_argument("--use-fraction-head", action="store_true",
                    help="(lucas) force L2a fraction metrics on")
    ap.add_argument("--auto-fraction", action="store_true",
                    help="(lucas) enable L2a if the cache carries fractions")
    ap.add_argument("--dump-per-point", default=None)
    args = ap.parse_args()

    if args.truth == "nfi":
        if not args.plot_index:
            ap.error("--truth nfi requires --plot-index")
        results = score_nfi(args)
    else:
        if not (args.lucas_index and args.data_dir):
            ap.error("--truth lucas requires --lucas-index and --data-dir")
        results = score_lucas(args)

    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
