# Multi-year NFI control tiles, step 1: census of the legacy `holdout_val_512`

Measured 2026-10-12 on the data volume, read-only. This is the readiness
check decision D3 asked for before the August-2026 off-footprint holdout may
be reused as evaluation imagery. It is a planning result: no freeze, no
fetch, no GPU, no byte written under `/data`.

## Verdict

**Never trained on, never used for model selection: PASS.** No holdout tile
name appears in the training cohort directory, in the 2024 campaign staging,
in any of the fourteen split files on the volume (the seven teacher
`*_r2_split.json`, the LUCAS crop splits, the pinned plot list, the cohort
manifest), in any of the seven teacher feature tables, or in the pinned or v4
plot index. Student models draw their 10 % validation split by MD5 of the tile
name *inside the directories a training job lists*, and no training, ladder or
distill manifest in the repository lists `/data/holdout_val_512`; the only
reader outside enrichment is the dashboard's inference matrix, which renders
ten frozen holdout tiles for every finished checkpoint and selects nothing.
The seven text files on the volume that name holdout tiles are the Phase-A
manifest, three CDSE probes, the sen2cor plan and the two inference-matrix
files.

**The holdout covers the 1,000-per-year target for 2020, 2022, 2023 and 2024
on its own.** 2018, 2019 and 2021 fall short by 100, 33 and 102 eligible
plot-years if all 1,463 files count, and by 263, 62 and 124 if the 84
root-only files are excluded. Option A of the plan (top-up only) holds.

**The tiles are not at the 2024 bar yet.** Every one of the 1,463 carries
TESSERA from the pre-promotion source with no `tessera_source` stamp; 255
lack `frame_2016` and 261 lack `s1_vv_vh_2016`; four `frame_2016` and one
`tessera` array are all zero behind a positive flag; twelve tiles carry
all-zero forestry channels; 306 tiles have no autumn frame. Step 2 of the
plan (re-source, fill, stamp, into an evaluation-only copy) stands.

## Eligible plot-years per year

Distinct `(TractID, PlotID, Year)`, year-exact colocation as index v4, inside
the 496 × 496 common crop, outside the 841-plot-year teacher union of the
task-0103 sizing, finite non-negative volumes. "Root-only" are plot-years
whose tile is one of the 84 files with mode 0600 owned by root (unreadable to
the non-root sizing Job); "bar" are plot-years on tiles that already carry
every modality of the 2024 campaign (promoted TESSERA stamp excepted, which
no tile has).

| Year | Tiles with plots | On tiles | In crop | Teacher pool | Eligible | …world-readable | …root-only | …on bar tiles | …also on a v4 tile | Gap to 1,000 (all / world-readable) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2018 | 134 | 949 | 918 | 18 | **900** | 737 | 163 | 724 | 51 | 100 / 263 |
| 2019 | 144 | 998 | 967 | 0 | **967** | 938 | 29 | 816 | 0 | 33 / 62 |
| 2020 | 151 | 1,055 | 1,030 | 0 | **1,030** | 1,003 | 27 | 842 | 0 | 0 / 0 |
| 2021 | 127 | 919 | 898 | 0 | **898** | 876 | 22 | 692 | 0 | 102 / 124 |
| 2022 | 247 | 1,337 | 1,265 | 22 | **1,243** | 1,152 | 91 | 1,002 | 9 | 0 / 0 |
| 2023 | 175 | 1,280 | 1,245 | 0 | **1,245** | 1,181 | 64 | 1,014 | 0 | 0 / 0 |
| 2024 | 197 | 1,433 | 1,402 | 0 | **1,402** | 1,282 | 120 | 1,197 | 1,326 | 0 / 0 |
| **Total** | 1,175 | 7,971 | 7,725 | 40 | **7,685** | 7,169 | 516 | 6,287 | 1,386 | 235 / 449 |

No plot-year failed the value gate. The "also on a v4 tile" column is the
overlap with plots already indexed in-crop on cohort or 2024-campaign tiles:
for 2024 these are the campaign's own evaluation tiles (1,326 of 1,402), for
2018 and 2022 they sit on *training* tiles of the same year (51 and 9).

Cross-check against the task-0103 sizing: the world-readable column
reproduces its `geometric_reuse_all_roots` exactly for 2019 (938), 2020
(1,003) and 2023 (1,181), the years with no v4 coverage. For 2021 and 2022
it reproduces it as the union with the v4 in-crop plots (876 + 9 = 885;
1,152 + 24 − 9 = 1,167). For 2018 the sizing's 779 is 737 + 81 minus the
overlap among world-readable plots, which this census does not split out
(the 51 overlapping plot-years above are counted across all files). The two
measurements share the same inputs by hash and the same identity module by
sha256.

## Files

| | |
|---|---|
| Files present | 1,463 of 1,650 planned; 187 never built (2018 2, 2019 17, 2020 13, 2021 4, 2022 124, 2023 13, 2024 14) |
| Readable as root | 1,463; unreadable 0; none outside the Phase-A manifest |
| Not world-readable | 84, all mode `0600 uid=0` (2018 22, 2019 4, 2020 4, 2021 3, 2022 26, 2023 9, 2024 16); names in the JSON sidecar |
| Spectral `(24, 512, 512)`, bbox on the 10 m lattice, bbox equal to the manifest | 1,463 / 1,463 / 1,463 |
| Tile year equals manifest year (lenient and strict readers agree) | 1,463; 0 conflicts |

The 84 are readable by any root process, which every census and enrichment
Job in this repository is. Whether they count is a decision, not a technical
limit: a `chmod 644` on the volume would make them visible to the sizing Job
as well. Nothing was changed.

## Key census against the 2024 campaign bar

Keys where the holdout is behind the 475 campaign tiles by more than two
percentage points:

| Key | Holdout | Campaign |
|---|---:|---:|
| `frame_2016` (+ `_bands`, `_date`, `_scene`, `_year`, `has_frame_2016`) | 1,208 / 1,463 (82.6 %) | 98.3 % |
| `s1_vv_vh_2016`, `s1_dates_2016` | 1,202 (82.2 %) | 97.7 % |
| `tessera_source` | 0 | 99.8 % |
| `frame_2016_gap_reason` | 0 (gaps are unmarked) | 2.1 % |
| label sidecars `nmd_label_raw`, `nmd_area_ha`, `label_mask`, `n_parcels`, `parcel_area_ha`, `harvest_mask`, `n_harvest_polygons`, `n_mature_polygons` | 0 | 74–100 % |

The holdout labels were written by `build-labels-holdout-nmd2018`, which
stores the `label` array (1,463 / 1,463) but none of the per-tile label
statistics `build_labels.py` writes for the campaign. Nothing in the NFI
evaluation reads those sidecars; the difference is provenance, not support.
Keys present on every tile: spectral, label, the 11 aux channels, `s1_vv_vh`
(`s1_enrich_v` 4 on 1,462, 3 on 1), `tessera` + `tessera_year` (equal to the
tile year on all 1,463), `vpp_year` (equal to the tile year on all 1,463).

All-zero arrays and flags (the census2 checks):

| | Count | Note |
|---|---:|---|
| `tessera` all zero with `has_tessera = 1` | 1 | same defect class as the two campaign holes, unmarked here |
| `frame_2016` all zero with `has_frame_2016 = 1` | 4 | same mechanism as the two campaign frames demoted on 2026-10-05 (granule no-data passes the COT gate) |
| `height`, `volume`, `basal_area`, `diameter` all zero | 12 tiles | forestry channels written as zeros; the repo forbids zero fills, so these need a reason or a re-fetch |
| `frame_2016_year` | 2016 on 1,127, 2015 on 81, absent on 255 | |
| `temporal_mask` | `1111` on 1,157, `0111` on 306 | 21 % without the autumn frame |

Per year, tiles carrying every modality of the bar (promoted stamp
excepted): 2018 105 / 134, 2019 121 / 144, 2020 123 / 151, 2021 99 / 127,
2022 429 / 535, 2023 144 / 175, 2024 165 / 197.

## Footprint

Lattice tiles, no overlap among themselves. Against the 7,882 training
tiles: 1,370 holdout tiles overlap a training tile of *some* year, which
D6 allows; 212 overlap a training tile of the *same* year (2018 63, 2021 8,
2022 141; 270 training tiles involved). The "off-footprint" definition of
August 2026 was plot-based, not spatial, so this was expected. D6 covers
other-year overlap only; same-year pixel overlap for 2018, 2021 and 2022 is
reported here for Tobias to rule on. At plot level the same-year overlap is
the "also on a v4 tile" column above: 51 (2018) and 9 (2022) eligible
plot-years sit in-crop on a training tile.

## Reproduction and verification

- Driver `scripts/census_holdout_val_512.py` at
  `54c526e4380dc6b271f44b47bf818e53e82bfc7b`; Job
  `k8s/nfi-holdout-census-job.yaml`.
- Identity: `imint/eval/fieldtruth.py` at `ba31235c…` (sha256 `3f7143f2…`),
  the process commit of the task-0103 sizing; the script refuses any other
  version. Exclusion union 841, equal to the sizing. All eleven input hashes
  equal the sizing's (`nfi_fieldtruth_0103_tile_sizing_20261008.json` at
  `94b26bb…`).
- Job `nfi-holdout-census` (uid `b1bcbfbf-ba84-45b6-9f51-dd42d60b22f9`), pod
  `nfi-holdout-census-nwgfg` on p02r09srv06, image
  `python:3.11-slim@sha256:5be45dba…`, started 05:57:38 UTC, exit 0 at
  06:38:45 UTC (41 min). PVC mounted `readOnly` at volume and mount level.
  CPU only. Log 31,226 bytes, sha256 `e9b111c902229173a1a30657cef2a2177ee13cf80a27bf2e815051e4c211e96f`.
- Eight unit tests on synthetic tiles (crop border, teacher exclusion,
  invalid volume, root-only file, lying flag, split-file leak, identity
  leak guard, wrong identity module refused): `pytest
  tests/test_census_holdout_val_512.py`, 8 passed.
- Raw plot identities and coordinates stayed in the data environment; the
  report and the [JSON sidecar](nfi_multiyear_holdout_census_20261012.json)
  carry counts, year tables, tile names and file paths only.

## What this does not decide

Whether the 84 root-only files count, whether same-year footprint overlap is
acceptable for 2018/2021/2022, and whether step 2 (re-source TESSERA to the
promoted source, fill the 255 missing and 4 empty `frame_2016`, the 261
missing `s1_vv_vh_2016`, resolve the 12 zero forestry tiles, stamp, into an
evaluation-only copy) and step 3 (top-up fetch for 2018, 2019 and 2021) run.
Those are Tobias's calls; nothing here authorises them.
