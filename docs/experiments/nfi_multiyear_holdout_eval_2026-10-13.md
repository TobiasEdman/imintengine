# Multi-year NFI control tiles, step 2: the evaluation copy `holdout_eval_512`

Built 2026-10-12 19:13 to 2026-10-13 06:11 UTC on the data volume by the
eight Jobs of PR #75, every one submitted from a Codex-approved head, in the
order the plan prescribes and never two writers at once. Every count below is
the final census's (read-only, `holdout-eval-census`, 05:34 to 06:11 UTC) or
a Job's own result line; all of them, with the Job, pod, image, time, exit
code and log hash of each step, are in the
[JSON record](nfi_multiyear_holdout_eval_20261013.json). The legacy
`holdout_val_512` was not touched; the copy is the only thing evaluation may
read. Raw plot identities and coordinates stayed in the data environment.

## Verdict

**The copy exists and is at the promoted TESSERA source on every tile.**
1,241 tiles under `/data/holdout_eval_512`: the 1,463 legacy files minus the
212 that overlap a same-year training tile (decision 2026-10-12) and minus
the 10 whose forestry channels are all zero (moved to
`/data/holdout_eval_512_excluded/forestry_zero/`, not deleted). All 1,241
carry `tessera_source = geotessera-0.10.2` with `tessera_year` equal to the
tile year; 1,240 carry a usable embedding and 1 is a confirmed coverage hole
(`has_tessera = 0`, reason recorded). No tile has a contradictory year.

**Named-tile evidence is unchanged: no use detected, scans complete.** Every
split file on the volume (64, all parsed), the seven teacher feature tables
and the two plot indexes were scanned; none names a tile of the copy. The
twelve text files that mention the holdout are provenance records (the
Phase-A manifest, the CDSE probes, the two sen2cor plans, the copy's own
reports) or the dashboard's inference matrix. Spatial independence is settled
by exclusion for same-year overlap (0 such tiles remain) and remains allowed,
per D6, for other-year overlap.

**The copy is not at the 2024 bar on every tile, and three defects remain.**
163 tiles have no `frame_2016` after one sen2cor pass (gap flagged with a
dated reason); 2 tiles with a frame have no 2016 S1 composite; 1 tile keeps
its seasonal S1 at version 3 because no composite exists for its orbit and
year; and **4 tiles carry `has_frame_2016 = 1` over an all-zero array**, the
one flag-versus-array inconsistency the census found. The eligibility "bar"
counts below already treat those 4 frames as unusable. Fixing them means a
demotion step (flag to 0, reason), which is a new manifest and is not run
here.

## Eligible plot-years per year

Distinct `(TractID, PlotID, Year)`, year-exact colocation, inside the
496 × 496 common crop, outside the 841-plot-year teacher union, finite
non-negative volumes, on the 1,241 tiles of the copy. "Bar" are plot-years on
tiles that carry every modality of the 2024 campaign including the promoted
TESSERA stamp, a usable `frame_2016` and a 2016 S1 composite. Gap is to
1,000 plot-years per year.

| Year | Tiles with plots | Eligible | Eligible on bar | Gap | Gap on bar |
|---|---|---|---|---|---|
| 2018 | 71 | 489 | 389 | 511 | 611 |
| 2019 | 144 | 967 | 835 | 33 | 165 |
| 2020 | 151 | 1,030 | 888 | 0 | 112 |
| 2021 | 119 | 848 | 700 | 152 | 300 |
| 2022 | 162 | 838 | 735 | 162 | 265 |
| 2023 | 174 | 1,234 | 1,062 | 0 | 0 |
| 2024 | 196 | 1,396 | 1,275 | 0 | 0 |

Against the step-1 table after the same-year exclusion (489 / 967 / 1,030 /
848 / 840 / 1,245 / 1,402) the forestry exclusion cost 2 plot-years in 2022,
11 in 2023 and 6 in 2024. The top-up years are unchanged: 2018, 2019, 2021
and 2022, 858 plot-years in all; 2018 is the far one. 2024 keeps 1,326
plot-years in common with the v4 index crop and 70 new.

## What each step did

| Step | Job (head) | Result |
|---|---|---|
| Copy | `holdout-eval-copy` (938eff9, source commit 7317d7e) | 1,251 tiles copied, 0 skipped; same-year tiles not copied (212); demoted in the copy: 4 all-zero `frame_2016`, 1 all-zero `tessera` |
| TESSERA re-source, 4 workers | `holdout-eval-tessera` (3481550) | 1,250 resourced, 1 hole; 24 downloads swallowed by geotessera 0.10.2 and logged; 4 h 37 min |
| TESSERA re-probe, 1 worker | `holdout-eval-tessera-retry` (f11702f) | whole-copy pre-scan found 54 usable flags on partially delivered, non-finite or wrong-year embeddings; 55 targets: 54 recovered whole, 1 confirmed coverage hole, 0 unresolved; whole-copy final check clean |
| sen2cor `frame_2016`, GPU | `sen2cor-frame2016-holdout-eval` (3481550) | plan 29 scenes / 223 tiles / 0 unassigned; runner ok 59, deferred 84, cloud-gate rejected 80, sen2cor failed 6 scenes; `has_frame_2016 = 1` from 1,028 to 1,087 of 1,251; **56 min 20 s on one RTX 2080 Ti = 0.94 GPU-h** of the ~30 in D4 |
| Mark frame gaps | `holdout-eval-mark-frame-gaps` (3481550) | 164 tiles flagged (2018 13, 2019 18, 2020 20, 2021 20, 2022 50, 2023 24, 2024 19); 0 at a relaxed gate |
| S1 seasonal + 2016 | `holdout-eval-s1-2016` (3481550) | 1,245 ok, 6 failed (`no_composite_<orbit>_<year>`); verify: version 4 on 1,250 of 1,251, non-zero on 1,251, 2016 composite on 1,085 of 1,087 frame tiles; **Job exit 1** by its own post-check (one tile not at version 4) |
| Forestry | `holdout-eval-forestry` (3481550) | 10 all-zero tiles moved out, none deleted; 1,241 remain |
| Census | `holdout-eval-census` (3481550; census script 7317d7e) | this record; 36 min, exit 0 |

The six sen2cor scene failures are the Google Cloud archive's 2016 July
products without product metadata, first met by the 2024 campaign on
2026-10-05; no threshold changes them. A fallback pass over alternate scenes
at the same gate is the plan's next move for the 163 gaps and is not run
here.

The S1 Job's failure is a persistent gap, not a fault of this run: the
step-1 census already recorded one tile at seasonal version 3 and 1,462 at
version 4, the enricher returns before writing when no composite exists, and
the five other failed tiles kept their earlier version-4 data. It was not
re-run; the copy was carried forward with the gap recorded.

## Files

| | |
|---|---|
| Files in the copy | 1,241 of 1,650 planned; planned but absent 2018 65, 2019 17, 2020 13, 2021 12, 2022 273, 2023 14, 2024 15 (187 never built + 212 same-year + 10 forestry) |
| Readable, admitted, quarantined | 1,241 / 1,241 / 0; none outside the Phase-A manifest; none root-only |
| Spectral `(24, 512, 512)`, bbox on the 10 m lattice and equal to the manifest | 1,241 / 1,241 / 1,241 |
| Autumn frame present | 1,021 of 1,241 (220 with mask `0111`) |
| `frame_2016` | present 1,078 (year 2016: 1,007; 2015: 71), all-zero behind a positive flag 4, absent 163 |
| `s1_vv_vh_2016` usable | 1,076 |
| Seasonal S1 version | 4 on 1,240, 3 on 1 |
| `vpp_year` present and equal to tile year, 11 aux channels complete, label present | 1,241 each |

## Reproduction and verification

- Manifests `k8s/holdout-eval-*.yaml` and
  `k8s/sen2cor-frame2016-holdout-eval-job.yaml` on PR #75; each ran from the
  head Codex had approved for it (3481550, then f11702f for the re-probe
  Job), byte-identical to the branch tip f11702f. SHA-256 of each manifest is
  in the JSON record.
- Source binding: the tessera Jobs clone main at
  `d4d24f9ab1581ecdc8db739d4271905806401b6a` and check the helper's SHA-256
  before import; the census Job checks out `7317d7e` and verifies the census
  script's SHA-256, the identity module `imint/eval/fieldtruth.py` at
  `ba31235c…` (sha256 `3f7143f2…`) and the task-0103 sizing (sha256
  `4a7102cf…`); all eleven input hashes equal the sizing's, teacher union
  841.
- Execution evidence per Job (uid, pod, node, image digest, container
  start and end, exit code, log bytes and SHA-256) is in the JSON record.
  Seven pod logs were read from the namespace. The sen2cor Job and its pod
  had already been deleted by the namespace's `gpu-reaper` CronJob at
  05:21:31 UTC as a finished GPU Job; its evidence comes from the reaper's
  archive `/cephfs/ops/reaper_archive/sen2cor-frame2016-holdout-eval-20261013T052131Z/`
  (Job spec, pod record, pod log of 17,048 bytes, sha256 `f3ff0db1…`),
  read through the running census pod.
- The re-probe Job's behaviour is covered by eighteen tests that execute its
  heredoc offline: `pytest tests/test_holdout_eval_tessera_retry.py`.
- The copy's reports are on the volume under `/data/manifests/`:
  `holdout_eval_512_copy.json`, `holdout_eval_512_tessera.json`,
  `holdout_eval_512_tessera_retry.json` (per-tile partial-coverage fractions
  after the probe), `holdout_eval_512_frame2016_gaps.json` and
  `holdout_eval_512_forestry.json`.

## What this does not decide

Whether the 4 flag-on-empty frames are demoted, whether a sen2cor fallback
pass runs for the 163 gaps, and whether step 3 (top-up fetch for 2018, 2019,
2021 and 2022) runs. The first two need a manifest that does not exist yet;
the third is the plan's next step and starts with the planner, not a fetch.
