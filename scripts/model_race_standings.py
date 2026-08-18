"""scripts/model_race_standings.py — race the trained backbones, honestly.

Ranks every model that has an NFI per-plot dump by its NFI-209 held-out
forest-type accuracy (the honest grouped-tile split — no model's held-out
plots were seen in training/calibration). Each model is scored at its best
available config: hard members by the 28→5-class forest collapse of their
argmax; the Trädslag fraction member by its kappa-calibrated fraction collapse
(floor=0.05/dom=0.6), which is its native best.

This is the "model race" view — best SINGLE backbone, not an ensemble. A LUCAS
column is filled in when the L2 dumps land (higher-power cross-check).

    python scripts/model_race_standings.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble_stack import (  # noqa: E402
    suite_space_truth, tradslag_reference_correct,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_against_nfi import accuracy_suite  # noqa: E402

# model → (per-plot dump stem, backbone, collapse). "hard" = 28→5 argmax
# collapse; "fraction" = the calibrated Trädslag collapse.
MODELS = [
    ("Tessera · gated", "tessera_gated", "Tessera", "fraction"),
    ("Tessera · frac", "tessera_frac", "Tessera", "fraction"),
    ("Prithvi-600M · tradslag", "tradslag", "Prithvi-600M", "fraction"),
    ("Prithvi-300M · frac", "prithvi300m", "Prithvi-300M", "fraction"),
    ("Clay · frac", "clay", "Clay", "fraction"),
    ("Prithvi-600M · distill", "distill", "Prithvi-600M", "hard"),
    ("Tessera · distill (hard)", "tessera", "Tessera", "hard"),
    ("Prithvi-600M · v8b_nmd2023_long", "v8b_nmd2023_long", "Prithvi-600M", "hard"),
    ("Prithvi-600M · v8b_markfukt", "v8b_markfukt", "Prithvi-600M", "hard"),
    ("Prithvi-600M · v8b", "v8b", "Prithvi-600M", "hard"),
]

# Foundation-model race for the resolution ablation (C6). Each backbone's
# representative fraction model, its ViT patch size (the resolution axis),
# and the per-plot dump stem. Tessera is represented by its frac member for
# continuity with the published 0.589 (its gated member is in MODELS above).
# CROMA/TerraMind retrain on S1 v3 (2026-08-18) — dumps pending validation.
FM_RACE = [
    ("Tessera",      "tessera_frac", 1),
    ("Prithvi-600M", "tradslag",     14),
    ("Prithvi-300M", "prithvi300m",  16),
    ("Clay",         "clay",         8),
    ("CROMA",        None,           8),   # pending S1-v3 retrain
    ("TerraMind",    None,           16),  # pending S1-v3 retrain
]

FOREST = (1, 2, 3, 4)


def collapse_hard_28_to_5(model_pred: np.ndarray) -> np.ndarray:
    """28-class argmax → 5-class forest space: forest classes 1-4 stay, every
    other class (background, crops, water, …) is non-forest → 0."""
    return np.where(np.isin(model_pred, FOREST), model_pred, 0)


def score_209(stem: str, collapse: str, test_tiles: set) -> dict | None:
    path = Path(f"data/nfi/{stem}_per_plot.parquet")
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    ho = d[d["tile_name"].astype(str).isin(test_tiles)].reset_index(drop=True)
    y = suite_space_truth(ho["nfi_forest"].to_numpy())
    if collapse == "fraction":
        tr = ho.rename(columns={c: f"tradslag__{c}" for c in ("p1", "p2", "p3", "p4")})
        correct, _ = tradslag_reference_correct(tr, np.ones(len(tr), bool), y)
        oa = float(correct.mean())
        # kappa via the suite on the reconstructed prediction not needed here;
        # report OA (the gate metric) — kappa comes from the suite for hard.
        suite = {"overall_accuracy_5class": round(oa, 4), "cohen_kappa": None}
    else:
        pred = collapse_hard_28_to_5(ho["model_pred"].to_numpy())
        suite = accuracy_suite(y, pred)
    return {"n": int(len(ho)),
            "oa": round(float(suite["overall_accuracy_5class"]), 4),
            "kappa": suite.get("cohen_kappa")}


def main() -> None:
    split = json.loads(Path("data/distill/distill_split.json").read_text())
    test_tiles = {str(t) for t in split["test_tiles"]}

    rows = []
    for label, stem, backbone, collapse in MODELS:
        s = score_209(stem, collapse, test_tiles)
        if s is None:
            print(f"  (skip {label}: no dump)")
            continue
        rows.append({"model": label, "backbone": backbone,
                     "collapse": collapse, **s})
    rows.sort(key=lambda r: r["oa"], reverse=True)

    print(f"\n=== MODEL RACE — NFI-209 held-out forest-type ===")
    print(f"{'#':>2} {'model':34s} {'OA':>7s} {'kappa':>7s} {'collapse':>9s}")
    for i, r in enumerate(rows, 1):
        k = f"{r['kappa']:.3f}" if r["kappa"] is not None else "  —  "
        print(f"{i:>2} {r['model']:34s} {r['oa']:>7.4f} {k:>7s} {r['collapse']:>9s}")

    # Best per backbone (the actual "which model" answer).
    best = {}
    for r in rows:
        if r["backbone"] not in best or r["oa"] > best[r["backbone"]]["oa"]:
            best[r["backbone"]] = r
    print(f"\n=== best per backbone ===")
    for bb, r in sorted(best.items(), key=lambda kv: -kv[1]["oa"]):
        print(f"  {bb:14s} {r['oa']:.4f}  ({r['model']})")

    # Foundation-model race — the resolution ablation (C6). Same fraction
    # collapse and test tiles as above, so these numbers are directly
    # comparable to the nfi_209 rows (tessera_frac/tradslag reproduce the
    # published 0.5885/0.5789 exactly).
    fm_race = []
    for backbone, stem, patch in FM_RACE:
        if stem is None:
            fm_race.append({"backbone": backbone, "patch_size": patch,
                            "nfi_oa": None, "n": None, "status": "pending"})
            continue
        s = score_209(stem, "fraction", test_tiles)
        fm_race.append({"backbone": backbone, "patch_size": patch,
                        "nfi_oa": s["oa"] if s else None,
                        "n": s["n"] if s else None,
                        "status": "done" if s else "missing"})
    print(f"\n=== FOUNDATION-MODEL RACE (C6 resolution axis) ===")
    print(f"{'backbone':14s} {'patch':>5s} {'NFI OA':>7s} {'n':>4s}")
    for r in sorted(fm_race, key=lambda r: (r["nfi_oa"] is None, -(r["nfi_oa"] or 0))):
        oa = f"{r['nfi_oa']:.4f}" if r["nfi_oa"] is not None else "pending"
        n = str(r["n"]) if r["n"] else "—"
        print(f"{r['backbone']:14s} {r['patch_size']:>5d} {oa:>7s} {n:>4s}")

    out = Path("data/distill/model_race_standings.json")
    out.write_text(json.dumps({"nfi_209": rows,
                               "best_per_backbone": best,
                               "fm_race": fm_race,
                               "scoring": "5-class suite {0 non-forest,1-4 "
                               "forest}; fraction=calibrated collapse "
                               "floor0.05/dom0.6, hard=28→5 argmax; test "
                               "tiles from distill_split.json",
                               "note": "LUCAS column pending L2 dumps"},
                              indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
