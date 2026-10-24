# Multi-year NFI control tiles, step 3k: aux pass and census on the 230 top-up tiles

Run 2026-10-24 on Tobias's go, after a read-only dry run and two probes. This
follows [step 3](nfi_multiyear_topup_2026-10-19.md), where all 230 top-up
tiles were left `aux_complete = 0` because forestry, markfukt and tree height
were not yet in the open-data mirror (IM-022). Every count below is a Job's
own result line. Each Job's pod, node, times and log hash are in the
[JSON record](nfi_multiyear_topup_aux_census_20261024.json). No raw plot
identities or tile names are in either file.

## Verdict

**The aux pass completes 215 of 230 tiles. The 15 that remain are genuine
source gaps, recorded and not zero-filled.** The census then finds 714 of the
900 eligible top-up plot-years on tiles that meet the full modality bar.
**`frame_2016` is the main loss**: 38 tiles lack it. 35 of them miss nothing
else, and they carry 155 plot-years, 102 of them in 2018.

**2018 sets the quota, and on the full bar it is at most 704 plot-years.** The
evaluation copy has 381 on the bar (census of 2026-10-12, after same-year
exclusion) and the top-up 323. Their deduplicated union lies between 381 and
704, below the 826 counted by plot-years alone in step 3. The exact figure
needs the combined census of step 4. 2021 is at most 838 and 2022 at most
854.

## Aux pass

`k8s/nfi-topup-aux-pass-job.yaml` from main `b80658c`. The pod ran pin
`0f3fd26` and checked its script hash. The dry run at the same pin, with the
PVC read-only, gave the same result in every count. Height nodata is NaN, and
a tile whose height is NaN over more than 25 % of its in-county cells gets a
height gap (Tobias, 2026-10-22).

| Year | Tiles | Complete | Incomplete | Height written |
|---|---|---|---|---|
| 2018 | 133 | 120 | 13 | 123 |
| 2019 | 7 | 6 | 1 | 7 |
| 2021 | 47 | 47 | 0 | 47 |
| 2022 | 43 | 42 | 1 | 43 |
| **All** | **230** | **215** | **15** | **220** |

The 15 gaps:
- **Height, 10 tiles, all in county 25.** 9 have no valid pixel; one has 60.6 %
  of its in-county cells without data. The mirror's county 25 lacks height
  over 32 % of the county, rising northwards, which looks like upstream
  coverage.
- **Markfukt, 4 tiles** (2018 2, 2019 1, 2022 1): no valid pixel in the mirror.
- **Volume, basal area and diameter, 1 tile** (2018): no valid pixel.

8 more tiles are written with some in-county NaN below the limit, mostly
coastal ones whose county polygon includes sea. Every root kept its tile
count.

Before the pass, two probes at `b80658c` (`k8s/nfi-topup-mosaic-probe-job.yaml`,
`k8s/nfi-topup-impact-probe-job.yaml`) confirmed the county masks of #96:
- **The overwrite is gone.** In step 3's dry run, county 23's zeros covered
  counties 20 to 22: 16 tiles differed from their own county's raster on
  24–99 % of their cells. Now none differs on more than 0.22 %, and those
  cells are at county borders.
- **Every county raster is nodata outside its mask.**
- **County 24 is clean.** It no longer shows the retired all-zero rule: 0.1 %
  of its cells inside the county lack height, down from 20.5 %.

## Census

`k8s/nfi-topup-census-job.yaml`, renamed `nfi-topup-census-r2`, census pin
`7317d7e`, PVC read-only. It found no all-zero array, no positive flag on an
empty array, no year conflict and no named-tile use. All eleven sizing input
hashes match.

| Year | Tiles | All aux keys | Usable `frame_2016` | Tiles on the bar | Eligible plot-years | On the bar |
|---|---|---|---|---|---|---|
| 2018 | 133 | 120 | 110 | 99 | 449 | 323 |
| 2019 | 7 | 6 | 5 | 5 | 46 | 35 |
| 2021 | 47 | 47 | 42 | 41 | 210 | 186 |
| 2022 | 43 | 42 | 35 | 33 | 195 | 170 |
| **All** | **230** | **215** | **192** | **178** | **900** | **714** |

The bar is the census's `bar_modalities`:
- the 11 aux keys, forestry, `frame_2016` and the 2016 S1 layer;
- S1 and tessera;
- a VPP year matching the tile year;
- spectral and label.

Eligible plot-years are unchanged from step 3. The aux pass changes which
tiles meet the bar, not which plots they hold.

## Why tiles miss the bar

`k8s/nfi-topup-bar-attrib-probe-job.yaml` reuses the census's functions at
its pin, and asserts that its eligible and on-bar totals equal the census's.
A plot-year on two tiles counts under each tile's cause, so causes can
overlap; the per-year totals are exact.

| Cause (tile fails) | Tiles | Plot-years on them |
|---|---|---|
| `frame_2016` and S1 2016 only | 35 (2018 21, 2019 1, 2021 5, 2022 8) | 155 (102, 10, 19, 24) |
| aux only | 11 (2018 10, 2022 1) | 17 (16, 1) |
| `frame_2016`, S1 2016 and aux | 3 (2018 2, 2019 1) | 8 (7, 1) |
| aux and tessera | 1 (2018) | 1 |
| S1 2016 only | 1 (2021) | 5 |
| S1 and S1 2016 | 1 (2022) | 7 |

The S1 2016 layer is built on the `frame_2016` composite, so the two fail
together. The S1 tile in 2022 is the one with no descending-orbit composite
for its year (step 3).

## What this does not decide

- **A sen2cor fallback for the 38 `frame_2016` gaps.** It could recover at
  most the 155 plot-years on the 35 tiles that miss nothing else. This
  repo's rule is that a campaign is not complete until `has_frame_2016 == 1`.
  The GPU cost is Tobias's decision.
- **The combined census and the exact quota** (step 4, task 0103). Index v5
  over the evaluation copy and the four top-up roots, with same-year
  exclusion and deduplication, gives the binding per-year number.
- **Whether the 2018 shortfall is accepted at that number**, and the
  prespecified, seeded selection of the primary set. The freeze needs
  Tobias's go.

## Reproduction and verification

| Job | Source | Result |
|---|---|---|
| `nfi-topup-aux-pass-dryrun4` | aux-pass yaml + `--dry-run`, PVC read-only | exit 0, nothing written |
| `nfi-topup-mosaic-probe-r2` | `k8s/nfi-topup-mosaic-probe-job.yaml` | exit 0 |
| `nfi-topup-impact-probe-r2` | `k8s/nfi-topup-impact-probe-job.yaml` | exit 0 |
| `nfi-topup-aux-pass` | `k8s/nfi-topup-aux-pass-job.yaml` | exit 0, 230 tiles, counts unchanged |
| `nfi-topup-census-r2` | `k8s/nfi-topup-census-job.yaml`, renamed | exit 0 |
| `nfi-topup-bar-attrib-probe` | `k8s/nfi-topup-bar-attrib-probe-job.yaml` | exit 0, totals asserted |

All ran `python:3.11-slim@sha256:5be45dba…` on icekube. The aux pass was
the only writer and ran alone. The three read-only Jobs before it ran in
parallel, and the census and the probe after it. Logs are saved
under `~/.agents/continuity/evidence/ImintEngine/nfi-topup-aux-pass-dryrun4/`
and `…/nfi-topup-aux-pass-20261024/`; their hashes are in the JSON record.
