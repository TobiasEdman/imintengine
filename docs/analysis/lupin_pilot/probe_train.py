#!/usr/bin/env python3
"""Probe v2: lupine vs target-group background on TESSERA embeddings.

Review fixes (savant W3-W5 + domain-reviewer 1, 3-5):
    W4  pooled out-of-fold AUC/AP (fold-size-independent; AP interpretable)
        + per-fold AUCs for spread.
    W3  non-linear geography control: MLP on lat/lon (not just logreg).
    D1  year-only control (one-hot logreg): kills the year-confound read.
    D3  veto sensitivity sweep: AUC at dist_lupin > 100/250/500 m.
    D2  habitat-matched subset: both classes within 30 m of an OSM road.
    D5  month stratification: June/July vs August positives.
    W5  duplicate-row assert + skipped-fold logging.

Verdict rule (per domain review): embeddings must beat BOTH geography
controls AND the year control with clear margin, and survive the
road-matched subset, before AUC>=0.8 counts as a go.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).parent / "data"

_ap = argparse.ArgumentParser()
_ap.add_argument("--features", default="features.npz",
                 help="features.npz (final) or features_partial.npz (interim)")
_ap.add_argument("--out", default="probe_results.json")
_ap.add_argument("--min-per-class", type=int, default=200)
_args = _ap.parse_args()

print(f"features: {_args.features}")
d = np.load(DATA / _args.features, allow_pickle=True)
X = d["X"].astype(np.float32)
y = d["y"].astype(int)
lat, lon = d["lat"].astype(float), d["lon"].astype(float)
year = d["year"].astype(int)
month = d["month"].astype(int)
block = d["block"]
dist_lupin = d["dist_lupin"].astype(float)
dist_road = d["dist_road"].astype(float)

# plan B: post-hoc road distances from compute_road_dist.py override NaN
_rd_path = DATA / "road_dist.json"
if _rd_path.exists():
    import json as _json
    _rd = _json.loads(_rd_path.read_text())
    _hits = 0
    for _i in range(len(dist_road)):
        _v = _rd.get(f"{lon[_i]:.6f},{lat[_i]:.6f}")
        if _v is not None:
            dist_road[_i] = _v
            _hits += 1
    print(f"road_dist.json: {_hits}/{len(dist_road)} points resolved")

# --- data hygiene, before the guards that assert it worked ---
# 1) All-zero embeddings are geotessera's no-coverage fill (water, outside
#    the landmask), not measurements: 60 points on 60 distinct coordinates
#    shared one all-zero vector, split across both classes. They are NOT
#    NaN, so the NaN guard below cannot see them.
# 2) Exact duplicate vectors mean two points hit the same TESSERA pixel.
#    Pixel dedup at extraction runs on the 10 m SWEREF99 grid, which is not
#    the TESSERA grid, so a few slip through; dedup here on the vector
#    itself, which is the pixel identity that actually matters. A duplicate
#    group carrying both labels is unusable — drop every copy, not one.
_live = ~np.all(X == 0, axis=1)
_n_zero = int((~_live).sum())
_uniq, _inv = np.unique(X[_live], axis=0, return_inverse=True)
_idx_live = np.flatnonzero(_live)
_by_vec: dict[int, list[int]] = {}
for _i, _g in zip(_idx_live, _inv):
    _by_vec.setdefault(int(_g), []).append(int(_i))
_keep = sorted(g[0] for g in _by_vec.values()
               if len({int(y[i]) for i in g}) == 1)
_n_dup = len(_idx_live) - len(_keep)
print(f"hygiene: dropped {_n_zero} all-zero (no-coverage) rows, "
      f"{_n_dup} duplicate/ambiguous rows -> {len(_keep)} usable")
X, y, lat, lon = X[_keep], y[_keep], lat[_keep], lon[_keep]
year, month, block = year[_keep], month[_keep], block[_keep]
dist_lupin, dist_road = dist_lupin[_keep], dist_road[_keep]

# --- sanity checks BEFORE any result (CLAUDE.md p.6 + savant W5) ---
assert X.ndim == 2 and X.shape[1] == 128, f"bad X shape {X.shape}"
n = len(y)
assert all(len(a) == n for a in (lat, lon, year, month, block,
                                 dist_lupin, dist_road))
assert not np.isnan(X).any(), "NaN in features"
assert len(np.unique(X, axis=0)) == n, "duplicate feature rows (B1/B2 broke)"
n_pos, n_neg = int(y.sum()), int((y == 0).sum())
assert n_pos >= _args.min_per_class and n_neg >= _args.min_per_class, \
    f"too few points: {n_pos}/{n_neg} (need {_args.min_per_class} each)"
assert (X.var(axis=0) > 0).sum() >= 100, "degenerate features"
print(f"sanity OK: N={n}, pos={n_pos}, neg={n_neg}, "
      f"blocks={len(set(block.tolist()))}")

groups_all = np.array([f"{int(np.floor(lo))}_{int(np.floor(la))}"
                       for lo, la in zip(lon, lat)])

LATLON = np.column_stack([lat, lon]).astype(np.float32)
YEAR_OH = OneHotEncoder(sparse_output=False).fit_transform(
    year.reshape(-1, 1)).astype(np.float32)


def make_models() -> dict:
    return {
        "logreg_embed": (make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)), X),
        "mlp_embed": (make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                          early_stopping=True, random_state=0)), X),
        "logreg_latlon_CTRL": (make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)), LATLON),
        "mlp_latlon_CTRL": (make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                          early_stopping=True, random_state=0)), LATLON),
        "logreg_year_CTRL": (make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)), YEAR_OH),
    }


def pooled_cv(model, feats, yy, groups, tag="") -> dict | None:
    """Grouped CV -> pooled out-of-fold AUC/AP + per-fold AUC spread."""
    n_groups = len(set(groups.tolist()))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        return None
    oof_p = np.full(len(yy), np.nan)
    fold_aucs, skipped = [], 0
    for tr, te in GroupKFold(n_splits=n_splits).split(feats, yy, groups):
        if len(set(yy[tr].tolist())) < 2:
            skipped += 1
            continue
        model.fit(feats[tr], yy[tr])
        p = model.predict_proba(feats[te])[:, 1]
        oof_p[te] = p
        if len(set(yy[te].tolist())) == 2:
            fold_aucs.append(float(roc_auc_score(yy[te], p)))
        else:
            skipped += 1
    m = ~np.isnan(oof_p)
    if m.sum() < 50 or len(set(yy[m].tolist())) < 2:
        return None
    res = {
        "auc_pooled": round(float(roc_auc_score(yy[m], oof_p[m])), 4),
        "ap_pooled": round(float(average_precision_score(yy[m], oof_p[m])), 4),
        "auc_folds": [round(a, 4) for a in fold_aucs],
        "n": int(m.sum()), "n_pos": int(yy[m].sum()),
        "n_groups": n_groups, "skipped_folds": skipped,
    }
    if tag:
        print(f"  {tag:34s} AUC {res['auc_pooled']:.3f} "
              f"AP {res['ap_pooled']:.3f} (n={res['n']}, "
              f"grp={n_groups}, skip={skipped})")
    return res


results: dict = {"main": {}, "veto_sweep": {}, "road_matched": {},
                 "month_strata": {}}

print("== huvudkörning (alla punkter, veto 100 m) ==")
for name, (model, feats) in make_models().items():
    results["main"][name] = pooled_cv(model, feats, y, groups_all, tag=name)

print("== vetosvep (D3): negativer med dist_lupin > tröskel ==")
for thr in (100, 250, 500):
    m = (y == 1) | (dist_lupin > thr)
    mdl, feats = make_models()["logreg_embed"]
    results["veto_sweep"][str(thr)] = pooled_cv(
        mdl, feats[m], y[m], groups_all[m], tag=f"veto>{thr}m")

print("== habitat-matchat subset (D2): båda klasser <=30 m från väg ==")
road_ok = ~np.isnan(dist_road)
m = road_ok & (dist_road <= 30)
if m.sum() >= 200 and len(set(y[m].tolist())) == 2:
    for name in ("logreg_embed", "mlp_latlon_CTRL", "logreg_year_CTRL"):
        mdl, feats = make_models()[name]
        results["road_matched"][name] = pooled_cv(
            mdl, feats[m], y[m], groups_all[m], tag=f"road30:{name}")
else:
    print(f"  otillräckligt: {int(m.sum())} punkter inom 30 m "
          f"(OSM-täckning {int(road_ok.sum())}/{n})")

print("== månadsstratifiering (D5): positiver juni/juli vs augusti ==")
for label, months in (("jun_jul", (6, 7)), ("aug", (8,))):
    m = (y == 0) | np.isin(month, months)
    mdl, feats = make_models()["logreg_embed"]
    results["month_strata"][label] = pooled_cv(
        mdl, feats[m], y[m], groups_all[m], tag=f"pos_{label}")

meta = {"features_file": _args.features, "n": n, "n_pos": n_pos,
        "n_neg": n_neg, "n_road_covered": int(road_ok.sum()),
        "results": results}
(DATA / _args.out).write_text(json.dumps(meta, indent=2))
print(f"saved {DATA/_args.out}")
