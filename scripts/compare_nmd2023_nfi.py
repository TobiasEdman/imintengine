#!/usr/bin/env python3
"""Same-plots comparison of NMD2023, NMD2018 and (optionally) the model vs NFI.

NMD2023 v2.1 basskikt (10 m, EPSG:3006, uint16) is sampled at each NFI plot's
Easting/Northing and scored against the same species-derived NFI forest truth
the model is validated on (``validate_against_nfi``). NMD2018 is scored at the
identical plots so the two label generations are directly comparable, and — when
a per-plot model dump (``validate_against_nfi.py --dump-per-plot``) is supplied —
the model is folded in on exactly the same plot set.

Fairness rule: NMD2023 v2.x is a rolling delivery, so plots outside its produced
extent read raster value 0 (no class). Those plots are **excluded** from every
source's score (not counted as an NMD "non-forest" miss). The comparison is
reported on the intersection where NMD2023 has data — the "same ytor".

Local, no GPU: needs only the two NMD rasters + the NFI plot table. The model
column is filled from the per-plot parquet produced on the cluster.

    python scripts/compare_nmd2023_nfi.py \
        --nmd2023 data/nmd2023/NMD2023bas_v2_1.tif \
        --nmd2018 data/nmd/nmd2018bas_ogeneraliserad_v1_1.tif \
        --plots data/nfi/nfi_plots.parquet \
        --model-per-plot data/nfi/v8b_per_plot.parquet \
        --out docs/data/compare-nmd2023-nfi.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_against_nfi import (  # noqa: E402
    derive_nfi_forest_class, accuracy_suite,
)
from imint.training.class_schema import nmd_raster_to_lulc  # noqa: E402
from imint.training.unified_schema import nmd19_to_unified  # noqa: E402


def sample_nmd_unified(tif: str, easting, northing) -> np.ndarray:
    """Sample an NMD raster at points and map raw codes → unified classes.

    Forest codes 111–128 are identical in NMD2018 and NMD2023, so the repo's
    NMD2018 collapse chain reproduces the exact forest-type buckets for both
    (the property that makes this comparison consistent). NMD2023's 4-digit
    open-land codes exceed uint8 and clip to background — correct, they are
    non-forest. Returns unified classes plus the raw sampled code (0 = the
    raster has no data at that point / outside produced extent).
    """
    import rasterio

    with rasterio.open(tif) as src:
        raw = np.array([v[0] for v in src.sample(zip(map(float, easting),
                                                     map(float, northing)))],
                       dtype=np.int64)
    seq = nmd_raster_to_lulc(np.clip(raw, 0, 255).astype(np.uint8))
    unified = nmd19_to_unified(seq.astype(np.uint8)).astype(int)
    return unified, raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nmd2023", required=True)
    ap.add_argument("--nmd2018", default=None)
    ap.add_argument("--plots", required=True, help="nfi_plots.parquet (E/N + species)")
    ap.add_argument("--model-per-plot", default=None,
                    help="parquet from validate_against_nfi --dump-per-plot "
                         "(TractID/PlotID + model_pred); restricts the compared "
                         "plot set to the model's plots")
    ap.add_argument("--dominant-frac", type=float, default=0.7)
    ap.add_argument("--out", default="docs/data/compare-nmd2023-nfi.json")
    a = ap.parse_args()

    plots = pd.read_parquet(a.plots)
    print(f"NFI plots: {len(plots):,}")

    # Restrict to the model's plots (same ytor as the model) when a per-plot
    # dump is given — join by the unique (TractID, PlotID) key.
    model_pred_col = None
    if a.model_per_plot:
        mp = pd.read_parquet(a.model_per_plot)
        key = ["TractID", "PlotID"]
        keep = [c for c in key + ["model_pred"] if c in mp.columns]
        plots = plots.merge(mp[keep].drop_duplicates(key), on=key, how="inner")
        model_pred_col = "model_pred" if "model_pred" in plots.columns else None
        print(f"  ∩ model per-plot dump: {len(plots):,} plots"
              + (" (with model_pred)" if model_pred_col else ""))

    # NFI forest truth (1-4) or 0 for non-forest / treeless.
    truth = np.array([
        (c if (c := derive_nfi_forest_class(r, dominant_frac=a.dominant_frac))
         is not None else 0)
        for _, r in plots.iterrows()
    ])

    nmd23, raw23 = sample_nmd_unified(a.nmd2023, plots.Easting, plots.Northing)
    covered = raw23 != 0
    print(f"NMD2023 covered: {covered.sum():,}/{len(plots):,} "
          f"({100 * covered.mean():.1f}%) — comparison runs on covered plots")

    sources = {"NMD2023_v2.1": nmd23}
    if a.nmd2018:
        nmd18, _ = sample_nmd_unified(a.nmd2018, plots.Easting, plots.Northing)
        sources["NMD2018_v1.1"] = nmd18
    if model_pred_col:
        sources["v8b"] = plots[model_pred_col].to_numpy(dtype=int)

    result = {
        "n_plots_compared": int(covered.sum()),
        "n_plots_total": int(len(plots)),
        "nmd2023_coverage_frac": round(float(covered.mean()), 4),
        "restricted_to_model_plots": bool(model_pred_col),
        "sources": {
            name: accuracy_suite(list(truth[covered]), list(pred[covered]))
            for name, pred in sources.items()
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
