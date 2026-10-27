# Multi-year NFI held-out evaluation, step 5: the freeze

Task 0117, run 2026-10-27 on Tobias's go ("start the freeze"). This follows
[step 4](nfi_multiyear_support_2026-10-26.md), whose quota it applies. Every
count below is the freeze Job's own output; the Job's pod, node, times, log
hash and the manifest digest are in the
[JSON record](nfi_multiyear_freeze_20261027.json). No plot identities, tile
names or coordinates are in either file. The frozen set itself
(`holdout.parquet`, `training.parquet`, `manifest.json`) stays on the PVC.

## Verdict

**Frozen: 5,194 plot-years, 742 in each of 2018–2024, on 1,269 evaluation
tiles. Manifest `nfi/nfi_holdout_freeze_2026-10-27/manifest.json`, sha256
`daaa428754aa4acfd832d9cc0fe5a0acb60e86ade2b511699f1181732697ff2c`.** The
Job ran 07:57–11:16Z on 2026-10-27 (3 h 19 min; most of it hashing every
input tile for the manifest), exit 0, and ended in state
`awaiting_user_go_no_go`. A read-only probe as UID 2000 then read all three
files back: the manifest digest equals the one the Job printed, both
parquets hash to what the manifest records, and the holdout has one row per
plot-year.

| Year | Frozen | With `frame_2016` | Without | Eligible (step 4) | Not selected |
|---|---|---|---|---|---|
| 2018 | 742 | 607 | 135 | 742 | 0 |
| 2019 | 742 | 642 | 100 | 962 | 220 |
| 2020 | 742 | 641 | 101 | 969 | 227 |
| 2021 | 742 | 627 | 115 | 1,007 | 265 |
| 2022 | 742 | 651 | 91 | 949 | 207 |
| 2023 | 742 | 642 | 100 | 1,180 | 438 |
| 2024 | 742 | 724 | 18 | 2,116 | 1,374 |
| **Total** | **5,194** | **4,534** | **660** | **7,925** | **2,731** |

The eligible counts are step 4's to the plot-year: the freeze's exclusion
chain produced the same 7,925 (teacher exposure 949 rows, cohort overlap 156,
same-year footprint 0, readiness and aux audit, 2,026 duplicate rows removed
across the six roots, 470 plot-years outside NMD2023's coverage, none
outside NMD2018's). The seeded rank (`sha256_plot_year_rank_v1`, seed
20261008) then took 742 per year. The 2,731 eligible plot-years it did not
take are not written anywhere: they are reproducible from the manifest's
inputs and method, but no file holds them.

By tile role: evaluation copy 3,823, nfi2024 campaign 718, top-up 2018 325,
2021 161, 2022 125, 2019 42. By NFI truth (`derive_nfi_forest_class`,
dominant fraction 0.7): no standing volume 281, pine 2,380, spruce 1,308,
deciduous 537, mixed 688. `training.parquet` holds the 841 teacher-exposed
plot-years the freeze excluded, for the record.

The 2018 frame split, 607 with and 135 without, is what r5 predicted once
the tie-break prefers a tile with the frame (#111).

## What was frozen

`scripts/prepare_nfi_holdout.py` (`k8s/nfi-holdout-freeze-job.yaml`) ran the
same rules as the step-4 report, then selected instead of reporting:
- the prespecified balanced selection, `--population balanced
  --inventory-years 2018 … 2024 --observations-per-year 742`, seeded rank
  per plot-year (`NFI_YEAR_SAMPLING`, seed 20261008);
- one tile per plot-year: the campaign tile first, then a tile with the 2016
  frame and 2016 SAR, then by name (#111);
- `has_frame_2016` per plot-year, so results can be stratified as decided on
  2026-10-24;
- the reviewed exceptions, `config/nfi_freeze_exceptions.json` (#111): the 9
  cohort tiles without Tessera v2 and the 10 cohort tiles without a CROMA or
  TerraMind sidecar. All are training-cohort tiles.

Inputs were those of report r5: index v5
(`/data/nfi/nfi_index_v5_2026-10-24.parquet`), the training cohort for
exclusion only, the nfi2024 campaign, the evaluation copy and the four
top-up roots, NMD2023 and NMD2018 as baselines, and the promotion Job's
JSON. It ran in the sealed `ladder-crop-distill` image built at main
`4cb93b4`, the sha that merged the prerequisites, and verified the image's
provenance against that sha before reading anything.

The manifest records every input's identity, the admitted tiles' identities
and geometry, the selection counts, the excluded tiles with reasons, the
aux audit, the teacher provenance and the exceptions, and ends in state
`awaiting_user_go_no_go`: inference and NMD scoring need their own go. The
scorer (`scripts/score_nfi_holdout.py --expected-manifest-sha256`) binds to
the manifest digest approved at that go/no-go.

## Writing to the PVC

The only write is the new directory `nfi/nfi_holdout_freeze_2026-10-27/`,
created exclusively after every check (the script refuses an existing
`--out-dir`). The PVC was mounted read-only except the `nfi/` subPath, as
for `k8s/nfi-index-v5-job.yaml`. The Job ran as root for the same reason
that Job did: a read-only probe as UID 2000 (`nfi-freeze-dir-probe`) found
`nfi/` root-owned 0755, so the image's own user could not create the
directory. Every file in `nfi/` is root-owned 0644 and read as UID 2000 in
r5; the freeze's files are the same.

## Gates at the freeze

All strict, all passed, as report r5 had shown after #111 and the campaign
tile's Tessera demotion:
- **Tessera promotion**: 7,873 of 7,882 cohort tiles promoted, exactly the 9
  listed exceptions unpromoted.
- **Teacher sidecars**: complete, with exactly the 10 listed exceptions for
  CROMA and TerraMind.
- **Readiness**: no error; the demoted campaign tile is an ordinary Tessera
  gap.

## Next

The go/no-go for inference and NMD scoring, against the manifest digest
above. Not in this step: inference, scoring, or any change to the frozen
files.
