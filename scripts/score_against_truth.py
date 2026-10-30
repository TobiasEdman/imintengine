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

    # LUCAS (fractions allowed only by a reviewed, pinned policy)
    python scripts/score_against_truth.py --truth lucas \
        --cache-dir /cephfs/pred_cache --ckpt-sha <sha> \
        --lucas-index /data/lucas/lucas_tile_index.parquet \
        --data-dir /data/unified_v2_512 --num-classes 28 --auto-fraction \
        --fraction-policy /data/audits/fraction-policy.json \
        --fraction-policy-sha256 <reviewed-policy-sha256> \
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

from scripts.cache_predict import make_cached_predict_fn  # noqa: E402
from scripts import prediction_cache as cache  # noqa: E402
from scripts.fraction_scoring_policy import select_fraction_mode  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).resolve().parent / f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fraction_decision(args, manifest):
    decision = select_fraction_mode(
        manifest["contract"],
        requested=getattr(args, "use_fraction_head", False),
        automatic=getattr(args, "auto_fraction", False),
        policy_path=getattr(args, "fraction_policy", None),
        policy_sha256=getattr(args, "fraction_policy_sha256", None))
    print(f"fraction scoring: {decision['decision']} "
          f"(enabled={decision['enabled']}) — {decision['reason']}")
    return decision


def _validate_cache_options(args):
    manifest = cache.load_manifest(args.cache_dir, args.ckpt_sha)
    for field in ("img_size", "num_classes"):
        if getattr(args, field) != manifest["contract"]["options"][field]:
            raise cache.CacheError(f"scoring {field} disagrees with prediction cache")
    return manifest



def _prepare_cache_index(index_df, manifest, *, include_year=False):
    """Authenticate metadata and map each tile's own rectangular crop."""
    if index_df.empty:
        raise cache.CacheError("empty scoring population")
    contract = manifest["contract"]
    rows = index_df["row"].to_numpy(dtype=float).copy()
    cols = index_df["col"].to_numpy(dtype=float).copy()
    keep = np.zeros(len(index_df), dtype=bool)
    years = {}
    if index_df[["tile_name", "tile_path"]].isna().any().any():
        raise cache.CacheError("missing tile identity in scoring index")
    for (name, path), positions in index_df.groupby(
            ["tile_name", "tile_path"], sort=False).indices.items():
        stem = Path(path).stem
        if Path(str(name)).stem != stem or stem not in contract["tiles"]:
            raise cache.CacheError(f"scoring tile is outside the cache inventory: {name}")
        entry = contract["tiles"][stem]
        metadata = cache.tile_metadata(
            path, entry["input"], family=contract["model"]["family"],
            include_year=include_year)
        height, width = metadata["height"], metadata["width"]
        crop = min(contract["options"]["img_size"], height, width)
        if crop != entry["crop_size"]:
            raise cache.CacheError(f"cached crop disagrees with authenticated tile: {name}")
        rows[positions] -= (height - crop) // 2
        cols[positions] -= (width - crop) // 2
        keep[positions] = (
            (rows[positions] >= 0) & (rows[positions] < crop)
            & (cols[positions] >= 0) & (cols[positions] < crop))
        if include_year:
            years[name] = metadata["year"]
    result = index_df.copy()
    result["row"], result["col"] = rows, cols
    result = result.iloc[np.flatnonzero(keep)].copy()
    if result.empty:
        raise cache.CacheError("no scoring observations within cached support")
    print(f"authenticated crop: kept {len(result)}/{len(index_df)} observations")
    return result, years


def score_nfi(args) -> dict:
    manifest = _validate_cache_options(args)
    fraction = _fraction_decision(args, manifest)
    van = _load_script("validate_against_nfi")
    index_df = pd.read_parquet(args.plot_index)
    print(f"plot index: {len(index_df):,} plots on "
          f"{index_df['tile_name'].nunique()} tiles")

    before = len(index_df)
    index_df, _years = _prepare_cache_index(index_df, manifest)

    if fraction["enabled"]:
        # Fraction-collapse mode: identical rule to the fused validator via
        # the SHARED van.fracs_to_class_and_probs (parity by construction).
        base = make_cached_predict_fn(
            args.cache_dir, args.ckpt_sha, want_fracs=True)
        print(f"  fraction-head collapse: dominant_frac={args.dominant_frac}, "
              f"forest_floor={args.forest_floor}")

        def predict_fn(tile_path):
            _cls, _probs, fracs = base(tile_path)
            return van.fracs_to_class_and_probs(
                fracs, dominant_frac=args.dominant_frac,
                forest_floor=args.forest_floor, num_classes=args.num_classes)
    else:
        predict_fn = make_cached_predict_fn(
            args.cache_dir, args.ckpt_sha, want_fracs=False)
    per_plot = [] if args.dump_per_plot else None
    results = van.score_against_nfi(
        index_df, predict_fn, num_classes=args.num_classes,
        dominant_frac=args.dominant_frac, per_plot_sink=per_plot)
    results["_meta"] = {
        "cache_dir": str(args.cache_dir), "ckpt_sha": args.ckpt_sha,
        "img_size": args.img_size, "plots_in_crop": len(index_df),
        "plots_total": before, "fraction_scoring": fraction,
    }
    if args.dump_per_plot:
        _dump_parquet(per_plot, args.dump_per_plot, "plots")
    return results


def score_lucas(args) -> dict:
    manifest = _validate_cache_options(args)
    fraction = _fraction_decision(args, manifest)
    val = _load_script("validate_against_lucas")
    index_df = pd.read_parquet(args.lucas_index)
    index_df["tile_path"] = index_df["tile_name"].map(
        lambda n: os.path.join(args.data_dir, f"{n}.npz"))
    print(f"LUCAS index: {len(index_df):,} points on "
          f"{index_df['tile_name'].nunique()} tiles")

    index_df, tile_years = _prepare_cache_index(index_df, manifest, include_year=True)

    is_fraction = fraction["enabled"]

    predict_fn = make_cached_predict_fn(
        args.cache_dir, args.ckpt_sha, want_fracs=is_fraction)

    per_point = [] if args.dump_per_point else None
    results = val.score_against_lucas(
        index_df, predict_fn, num_classes=args.num_classes,
        min_support=args.min_support, is_fraction=is_fraction,
        tile_years=tile_years, per_point_sink=per_point)
    results["_meta"] = {
        "cache_dir": str(args.cache_dir), "ckpt_sha": args.ckpt_sha,
        "img_size": args.img_size, "num_classes": args.num_classes,
        "min_support": args.min_support, "is_fraction_member": is_fraction,
        "fraction_scoring": fraction,
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
                    help="full prediction-contract key returned by infer_tiles.py")
    ap.add_argument("--img-size", type=int, default=504,
                    help="must match the crop infer_tiles.py used")
    ap.add_argument("--num-classes", type=int, default=28)
    ap.add_argument("--out", required=True)
    # NFI
    ap.add_argument("--plot-index", default=None,
                    help="(nfi) parquet from nfi_tile_coverage.py")
    ap.add_argument("--dominant-frac", type=float, default=0.7)
    ap.add_argument("--forest-floor", type=float, default=0.1)
    ap.add_argument("--dump-per-plot", default=None)
    # LUCAS
    ap.add_argument("--lucas-index", default=None,
                    help="(lucas) L1 parquet lucas_tile_index.parquet")
    ap.add_argument("--data-dir", default=None,
                    help="(lucas) tile root for tile_path + tile_year")
    ap.add_argument("--min-support", type=int, default=20)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--use-fraction-head", action="store_true",
                       help="require eligible fractions (NFI collapse or LUCAS L2a)")
    modes.add_argument("--auto-fraction", action="store_true",
                       help="use fractions only for eligible entries in the pinned policy")
    ap.add_argument("--fraction-policy",
                    help="reviewed fraction-supervision policy JSON")
    ap.add_argument("--fraction-policy-sha256",
                    help="expected full policy SHA-256 from the reviewed run specification")
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
