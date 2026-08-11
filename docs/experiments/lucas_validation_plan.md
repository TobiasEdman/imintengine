# Plan — LUCAS independent validation + head-refit QA (full-schema)

**Status:** plan, ready to execute · **Created:** 2026-08-11
**Branch:** `agent/te/opus/nfi-nmd2023-benchmark`
**Motivates:** the ensemble campaign's QA validates only the **5-class forest
collapse on 209 NFI plots**. That leaves ~23 of the 28 unified classes with no
independent field-truth check, and rests the whole honesty story on a thin
(R2) sample. LUCAS is a second, independent, far larger ground-truth that
touches most of the schema.

## Why LUCAS — the grounding fact

Swedish LUCAS 2022 = **34,480 points** (`data/lucas/EU_LUCAS_2022.csv`,
`POINT_NUTS0` starts `SE`), plus the 2018 Copernicus module
(`LUCAS_2018_Copernicus_attributes.csv`). `SURVEY_LC1` has genuine
tree-species detail — it resolves the exact tall/gran/löv/bland split the
campaign gates on, which NFI does at 209 plots and LUCAS does at ~12k
woodland points:

| LC1 | LUCAS meaning | SE-2022 n | → unified class |
|---|---|---|---|
| C22 | pine dominated | 3,413 | 1 tallskog |
| C21 | spruce dominated | 2,193 | 2 granskog |
| C10 | broadleaved | 1,672 | 3 lövskog |
| C31/C32/C33 | mixed woodland | 3,463 | 4 blandskog |
| C23 | other coniferous | 514 | (conifer; excluded from tall/gran) |
| G11/G21 | inland water | 7,067 | 10 vatten |
| H11/H12 | marsh / peatbog | 3,129 | 7 våtmark |
| E10/E20/E30 | grassland | 3,412 | 8 öppen mark (LU-split → 16 bete) |
| D10/D20 | shrubland / heath | 3,901 | 24 busk / 25 ris |
| F10/F30 | bare / lichen-moss | 1,482 | 27 öppen mark u. veg. |
| A** | artificial | 860 | 9 bebyggelse |
| B** | cropland | 1,870 | 11–21 crops (reuse `LUCAS_TO_CROP`) |

**~20 of 28 classes become independently checkable, at ~160× the NFI sample.**
Not LUCAS-observable (documented exclusions, not silent drops): 5 sumpskog,
6 tillfälligt ej skog, 22 hygge, 23 torvtäkt, 26 gräsdominerad (overlaps E).

## Two deliverables (user request)

1. **L-direct** — sample each member's dense prediction at each LUCAS point,
   score per-class over the full observable set. Independent QA breadth.
2. **L-head** — **repeat the head-refit** (the NFI arc's key move): fit a light
   head (logreg/MLP) on frozen pre-classifier features → LUCAS labels via
   grouped-tile CV. Measures what the representation *carries* per class, not
   what the hard-label training target permitted. Mirrors
   `scripts/nfi_head_cv.py` exactly.

## Reuse map (repo rule — do not reinvent)

| Step | Existing NFI asset to mirror |
|---|---|
| point → (tile,row,col) index | `scripts/nfi_tile_coverage.py` (TileConfig, EPSG:3006, `bbox_3006`) |
| pre-classifier feature dump | `scripts/extract_plot_features.py` |
| head CV (OOF, grouped, scaler-on-train) | `scripts/nfi_head_cv.py` |
| accuracy suite + McNemar/bootstrap | `scripts/validate_against_nfi.py`, `scripts/build_ensemble_stack.py` |

## Phases

### L0 — LUCAS truth set (local CPU, ~1 subagent pass)
`scripts/build_lucas_truth.py` (new):
- Parse both CSVs, filter `POINT_NUTS0` = SE.
- Map `SURVEY_LC1` → unified 28-class via an explicit, reviewed
  `LUCAS_LC1_TO_UNIFIED` table (crop codes reuse `crop_schema.LUCAS_TO_CROP`).
  Ambiguous/excluded codes are listed, never silently 0-mapped (§ honesty).
- **Purity filter:** keep points with `SURVEY_LC1_PERC` high (homogeneous
  10 m pixel) and/or `POINT_COPERNICUS` flag; drop mixed footprints. Report
  how many survive per class.
- **Year match (campaign rule):** 2022 points ↔ 2022 spectral, 2018 points ↔
  2018 spectral. Never cross years.
- Write `data/lucas/lucas_truth_sweden.parquet` (point_id, lat/lon, E/N in
  3006, unified_class, lc1, year, purity).

### L1 — coverage + point→pixel index (local, then PVC)
Run `nfi_tile_coverage.py`-equivalent on the LUCAS truth set against
`unified_v2_512` → `data/lucas/lucas_tile_index.parquet`
(point → tile, row, col, tile_year). Report coverage: how many of the 34k
land on built tiles, by class and year. **Leakage guard:** tag each point
train/test by the SAME `distill_split.json` tile grouping so L-head's held-out
never shares a tile with training.

### L2 — L-direct validation (GPU, test-split tiles)
Split by what the model actually outputs, matched to LUCAS's granularity.

**Data reality (verified 2026-08-11) — two tiers:**
- **2022 EU** (34,480 SE pts, 12,447 woodland): `SURVEY_LC1_PERC` empty, no
  usable LC2 → **argmax + mixedness only** (no magnitude). The LC1 code is
  ordinal: `C22/C21/C10` = pine/spruce/broadleaved **dominated**, `C31/C32/C33`
  = **mixed** (no dominant).
- **2018 Copernicus** (3,360 SE pts, 2,234 woodland): `LC1_PERC` populated on
  **all** woodland points + directional surrounding cover (`CPRN_LC1N/E/S/W`)
  → supports fraction **magnitude/ordinal** validation on this richer subset.
  Smaller n, so magnitude is a secondary corroboration, argmax the headline.

**L2a — forest (classes 1–4) → validate the FRACTION head, threshold-free.**
The fraction head is the model's real forest product (0.579 > the 0.502 hard
head), and its hard-collapsed 0.579 conflates head quality with the calibrated
floor=0.05/dom=0.6 collapse. LUCAS lets us decouple them:
- **Dominant-species argmax agreement** — LUCAS dominant (C21/C22/C10) vs the
  argmax of the model's {tall, gran, löv} fraction channels. **No collapse
  threshold** → tests the head, not the threshold.
- **Mixedness AUC** — LUCAS mixed (C3x) vs dominated (C2x/C10) is a binary
  label; the model's fraction concentration (max-fraction, or 1−entropy) is a
  continuous score. ROC-AUC tests the head's continuous output — the one thing
  hard classes cannot check.

**L2b — non-forest (water, wetland, grassland, shrub, bare, artificial, crops)
→ validate the HARD 28-class head.** No fraction channels exist here; LUCAS
dominant LC1 is the hard label. ~15 classes gain independent QA for the first
time. Per-class producer/user accuracy + confusion, held-out tiles only.

Run per member (v8b, markfukt, nmd2023_long, distill, tradslag, tessera); the
tradslag/tessera-frac members are the ones L2a can score.

### L3 — L-head refit (local, mirrors nfi_head_cv)
`extract_plot_features.py` at LUCAS pixels → 256-dim features per member →
`nfi_head_cv.py`-pattern grouped-tile CV (StandardScaler on train folds only,
OOF predictions, logreg + MLP). Two head targets, matching L2's split:
- **Classification head** over the non-forest + forest-dominant LUCAS labels —
  per-class OOF accuracy: does the representation carry class X even where the
  hard 28-class head collapsed it? (the NFI arc's 0.46→0.63 story, per class).
- **Fraction/ordinal head** for forest — a small regressor/ordinal head on the
  features predicting dominant-species + mixedness, scored by the L2a metrics
  (argmax agreement, mixedness AUC). Tests whether the *features* carry the
  fraction signal independent of the trained fraction head.
Both answer: does the frozen representation beat the trained head on LUCAS?

### L4 — cross-truth honesty (local)
- Same discipline as the ensemble P7: per-class support caveats (thin classes
  flagged), McNemar + bootstrap, denominator stated.
- **Agreement check:** does LUCAS corroborate the NFI "within-noise" G1
  verdict on the forest classes? Agree → strong. Disagree → investigate
  (LUCAS geolocation vs NFI tract design, purity, year drift).

## Gates
- **GL0:** ≥ N usable points/class after purity+coverage (set floor per class;
  classes under floor reported as "insufficient support", not scored).
- **GL1:** L2a forest dominant-species argmax agreement on LUCAS vs the NFI
  fraction member's argmax agreement — within CI = corroboration (both
  threshold-free, so a like-for-like cross-truth check).
- **GL2:** L-head per-class lift over the trained head — quantifies
  representation-vs-target gap across the full schema (the NFI insight,
  generalised).

## Risks
- **RL1 geolocation** — LUCAS point GPS ± a few m; a 10 m pixel may straddle a
  boundary. Purity filter + optional 3×3 majority mitigates; report both.
- **RL2 class-definition mismatch** — LUCAS LC1 ≠ NMD/unified semantics
  exactly (e.g. LUCAS grassland vs unified öppen mark/bete/slåttervall). The
  mapping table is the load-bearing artifact; it gets a review pass and the
  ambiguous codes are excluded, not forced.
- **RL3 leakage** — LUCAS points on training tiles inflate L-direct. Held-out
  tile restriction (distill_split grouping) is mandatory for the headline;
  all-tile numbers reported separately and labelled contaminated.
- **RL4 year coverage** — 2018 spectral tiles lack SKS/crop-year truth; LUCAS
  2018 mostly validates land-cover, not crops. Stated per class×year.

## Compute
| Phase | Where | Cost |
|---|---|---|
| L0 truth set | local CPU | minutes |
| L1 coverage/index | local + PVC scan | minutes |
| L2 L-direct | GPU (held-out tiles) | ~1 GPU-h × members |
| L3 L-head | local (features dump is the GPU part) | minutes CV |
| L4 honesty | local | minutes |

## References
NFI pattern: `scripts/{nfi_tile_coverage,extract_plot_features,nfi_head_cv,
validate_against_nfi}.py` · crop mapping: `imint/training/crop_schema.py`
(`LUCAS_TO_CROP`) · schema: `imint/training/unified_schema.py` (28 classes) ·
split: `data/distill/distill_split.json` · LUCAS raw:
`data/lucas/{EU_LUCAS_2022.csv,LUCAS_2018_Copernicus_attributes.csv}`.
