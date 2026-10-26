# Multi-year NFI held-out evaluation, step 4: per-year support and the quota

Task 0117, run 2026-10-26. This follows
[step 3k](nfi_multiyear_topup_aux_census_2026-10-24.md). Every count below
is a Job's own output, and each Job's pod, node, times and log hash are in
the [JSON record](nfi_multiyear_support_20261026.json). No plot identities,
tile names or coordinates are in either file.

## Verdict

**The quota is 742 plot-years per year, 2018–2024, under the freeze's own
rules** (Tobias, 2026-10-26). 2018 is the weakest year in every variant. The
primary set is 7 × 742 = 5,194 plot-years, chosen by the seeded selection at
the freeze; the other 2,731 eligible plot-years form the extended set.

| Year | Freeze rules | Absent height or markfukt mean-filled | Also requiring `frame_2016` |
|---|---|---|---|
| **2018** | **742** | **745** | **607** |
| 2019 | 962 | 963 | 826 |
| 2020 | 969 | 969 | 841 |
| 2021 | 1,007 | 1,007 | 850 |
| 2022 | 949 | 950 | 835 |
| 2023 | 1,180 | 1,180 | 1,017 |
| 2024 | 2,116 | 2,116 | 2,078 |
| **Total** | **7,925** | **7,930** | **7,054** |

Under the freeze rules, 596 of 2018's 742 plot-years sit on a tile with the
2016 frame and 2016 SAR, and 146 do not. Those 146 stay in, as decided on
2026-10-24, and results are to be split by a per-plot-year `has_frame_2016`
field built before the freeze. The `frame_2016` variant's 607 for 2018 falls
inside the 600–712 that step 3k derived from the census.

## How the counts were made

`scripts/prepare_nfi_holdout.py --report-only` (#106, #107, #109) applies the
freeze's own rules, then reports instead of freezing:
- **teacher exposure**: 949 index rows on a teacher's recorded feature
  plot-years are dropped;
- **cohort overlap**: 156 rows whose plot-year also lies on a training tile;
- **same-year training footprint**: none;
- **readiness and the aux audit** of every evaluation tile;
- **deduplication** to one tile per plot-year across all six evaluation
  roots: 2,026 duplicate rows;
- **NMD coverage**: 470 plot-years dropped, all from NMD2023's coverage gaps
  (80 in 2018); NMD2018 dropped none.

It read index v5 (`/data/nfi/nfi_index_v5_2026-10-24.parquet`, 11,809 rows,
9,576 plot-years over the cohort and six evaluation roots), built by
`k8s/nfi-index-v5-job.yaml`. It ran in the sealed `ladder-crop-distill` image
built at main `b2ebf3c`, as UID 2000 with the PVC read-only.

Tiles the freeze would not admit, per evaluation root:

| Root | Tiles | Not admitted | Why |
|---|---|---|---|
| nfi2024 campaign | 475 | 6 | markfukt all nodata 3, Tessera gap 2, Tessera flag on an empty array 1 |
| evaluation copy | 1,241 | 4 | markfukt all nodata 2, Tessera gap 1, old S1 version 1 |
| top-up 2018 | 133 | 13 | height absent 10, markfukt absent 2, Tessera gap 1 |
| top-up 2019 | 7 | 1 | markfukt absent |
| top-up 2021 | 47 | 0 | |
| top-up 2022 | 43 | 2 | no S1 1, markfukt absent 1 |

## Gates the report recorded

The freeze still refuses all three (Tobias, 2026-10-26: the report records
them, the freeze stays strict):
- **Tessera promotion**: 7,873 of 7,882 cohort tiles promoted; 9 have no v2.
- **Teacher sidecars**: CROMA and TerraMind each lack sidecars for the same
  10 cohort tiles; the other five teachers are complete.
- **Readiness**: one campaign tile has `has_tessera = 1` on an empty array.

## Getting there

Three report attempts stopped before the fourth completed:
1. At the promotion gate. Fix: #107.
2. On seven cohort tiles readable only by root, which the non-root report
   could not open. `k8s/nfi-cohort-chmod-job.yaml` set them to 0644 on
   Tobias's go, with their bytes unchanged (sha256 before and after).
3. At the sidecar inventory. A read-only diagnostic
   (`k8s/nfi-freeze-preflight-job.yaml`) then replayed every remaining gate
   and found only the sidecar shortfall and the one readiness error. Fix: #109.

`k8s/nfi-promotion-report-probe-job.yaml` and
`k8s/nfi-readability-probe-job.yaml` are the read-only probes behind steps 1
and 2.

## Before the freeze

The freeze needs Tobias's go, and these first:
- the 10 missing CROMA and TerraMind sidecars;
- the campaign tile's Tessera flag;
- the 9 cohort tiles without Tessera v2;
- the per-plot-year `has_frame_2016` field in `holdout.parquet`.
