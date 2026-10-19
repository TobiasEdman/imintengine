# Multi-year NFI control tiles, step 3: the top-up tiles for 2018, 2019, 2021 and 2022

Fetched and enriched 2026-10-13 to 2026-10-19 by the step-3 Jobs, each run
from the head approved for it. The exceptions are listed under "Reproduction
and verification". Most came from #80, integrated as `09720dd`.
The re-probe came from #83 at `2fb697d`, integrated as `4252b61`. Steps ran
one at a time, never two writers on a staging root. Every count below is a
Job's own result line or the read-only top-up census (2026-10-19 07:23 to
07:58 UTC). Each step's Job, pod, node, image, times, exit code and log hash
are in the [JSON record](nfi_multiyear_topup_20261019.json). The tiles sit in
one staging root per year, `/data/unified_v2_512_nfi{Y}_staging`. None is in
the training cohort or the evaluation copy. Raw plot identities and
coordinates stayed in the data environment.

## Verdict

**By count, the top-up closes 2019 and 2021 and leaves 2022 and 2018
short.** "By count" means plot-years only: no top-up tile is evaluable yet,
see below. The planner counted each top-up plot-year only if no tile of the
evaluation copy already covered it. Its count of plot-years gained on the
written tiles is therefore added to the copy's eligible count. That count is
the sum of each manifest tile's plots over the tiles actually written:

| Year | Eligible on the evaluation copy | Gained by the top-up | Total | Short of 1,000 |
|---|---|---|---|---|
| 2018 | 489 | 337 | 826 | 174 |
| 2019 | 967 | 34 | 1,001 | 0 |
| 2021 | 848 | 153 | 1,001 | 0 |
| 2022 | 838 | 141 | 979 | 21 |

2019 clears by one plot-year, and 2021 too, before any deduplication a
combined census over the copy and the top-up roots might apply.

**2022** planned 171 plot-years against a need of 162. 6 of its 49 tiles
failed the DES date or cloud gate twice. They carry 30 plot-years, the
reviewer's sum from `/data/manifests/nfi2022_topup_512.json`. That matches
the retry log: 7 tiles with 33 plots were retried and one tile with 3 plots
was recovered.

**2018** was accepted below 1,000 on 2026-10-13. Its shortfall of 174 has
three parts:

| Cause | Plot-years |
|---|---|
| Ceiling of the first plan: 410 feasible against a need of 511 | 101 |
| Tiles touching a cell with no HR-VPP product (13 tiles) | 33 |
| Tiles that failed the DES gate twice (12 tiles) | 40 |

The ceiling comes mostly from the same-year rule: 262 candidate tiles,
carrying 912 plot-years, would overlap a same-year training tile. Before
tiling, 446 plot-years were already excluded as teacher-pool and 91 as lying
inside a same-year training footprint.

**No tile is at the full 2024 bar yet.** Following option (a), all 230 tiles
carry `aux_complete = 0`. Forestry and markfukt wait for the open GeoTIFF
source (IM-022), and nothing was zero-filled. A tile with `aux_complete = 0`
must not be read by evaluation until that pass has run.

**Named-tile evidence: no use detected in any year.** No top-up tile is
quarantined, has a contradictory year, carries a flag over an empty array, or
overlaps a same-year training tile.

## What each step did

| Step | Result |
|---|---|
| Plan, first run | 2018 alone could not reach 1,000: 410 feasible plot-years against a need of 511 |
| VPP prefetch, two runs (WEkEO, no processing units) | 2019 and 2022 complete. Seven (cell, year) pairs stayed empty twice: 2018 in 33UVC, 34WCA, 34WCU, 34WCV, 34WDR and 35WLP, and 2021 in 34WCT |
| Re-plan with substitutes (#80) | Excluded the five cells proven to have no product in any year, plus 2018:34WCA, 34WDR and 35WLP from the ledger (`/data/manifests/nfi_topup_no_vpp_confirmed.json`). Planned 250 tiles; every MGRS cell they need is cached |
| DES fetch, `--workers 6`, one year at a time | Written: 2018 133/145, 2019 7/9, 2021 46/47, 2022 42/49. No throttling, no processing units spent |
| Retry at the same gate | 2 of 22 recovered (one tile each in 2021 and 2022) |
| Labels | 230/230 |
| S1 (Planetary Computer) | 229/230. One 2022 tile has no descending-orbit composite for its year, so the Job exits 1 by its own check. The gap is persistent and was not re-run |
| VPP stamp | 230/230 |
| sen2cor `frame_2016` on one RTX 2080 Ti | 192/230 (2018 110, 2019 5, 2021 42, 2022 35). **7.41 GPU-h**, 8.35 with step 2, of the ~30 in D4 |
| Frame gaps + S1 2016 | 38 gaps flagged. Two frame tiles have no 2016 composite. The Job exits 1 on the same S1 tile |
| TESSERA (4 workers) | 229 stamped, 1 hole. The log shows 8 downloads swallowed by geotessera |
| TESSERA re-probe, one worker (#83) | 31 usable flags were on partial or invalid embeddings (2018 21, 2019 1, 2021 3, 2022 6). All 31 were recovered to the usable contract, and the 2018 hole was confirmed. 0 unresolved. Originals change only on an accepted outcome |
| Census (read-only) | Eligible plot-years on the top-up tiles: 2018 449, 2019 46, 2021 210, 2022 195. These overlap the evaluation copy, so do not add them to the copy's figures |

## Reproduction and verification

- **Submission.** How each Job was submitted, from what, and how its source
  was bound:

  | Job | Submitted with | Source the pod ran |
  |---|---|---|
  | first `nfi-topup-plan` | `kubectl apply` of the `3481550` manifest | checkout pinned to `7317d7e`, planner SHA-256 checked |
  | `nfi-topup-vpp-prefetch-w1` | `kubectl apply` of the `3481550` manifest with the Job name and `YEARS` edited (`"2018 2021"`), the split that manifest prescribes when over budget; applied bytes sha256 `bd34ad25…` | clone of main's tip, unpinned; HEAD logged `d4d24f9` |
  | `nfi-topup-vpp-prefetch-fbdgt`, `nfi-topup-plan-949bk` | `kubectl create -f` of the `66aeb25` manifest (`generateName`) | prefetch checkout pinned to `d4d24f9`; planner pinned to `639caf2`, SHA-256 checked |
  | fetch, retry, labels, enrich, VPP stamp, sen2cor, gap marking, TESSERA | `kubectl apply` of the `66aeb25` bytes, from the worktree at that commit or via `git show 66aeb25:<file>` | checkout pinned to `d4d24f9`, HEAD checked, SHA-256 of every script it runs checked before it starts |
  | `nfi-topup-tessera-retry` | `git show 2fb697d:<file> \| kubectl apply -f -` | checkout pinned to `d4d24f9`, helper SHA-256 checked |
  | `nfi-topup-census` | `git show 66aeb25:<file> \| kubectl apply -f -` | checkout pinned to `7317d7e`, census script SHA-256 checked |
- **Logs.** 19 step-3 logs are saved with their pod records under
  `~/.agents/continuity/evidence/ImintEngine/nfi-step3-20261019/`. After the
  session pause, each was re-read from its pod and matched the hash recorded
  at completion. The 20th, sen2cor, is not in that folder. The gpu-reaper
  removed the finished GPU Job; its log hash (`35183351…`) was recorded at
  completion, and the log itself is in `/cephfs/ops/reaper_archive/`.
- The census used the identity module at `ba31235c…` and the task-0103
  sizing. All eleven input hashes match, and the teacher union is 841.

## What this does not decide

- Whether forestry and markfukt (IM-022) are filled in. Until they are, no
  top-up tile reaches the bar.
- Whether a combined census is run over the copy plus the four roots, for
  exact deduplicated totals.
- Whether the shortfalls are accepted as final: 174 for 2018 and 21 for
  2022. Tobias accepted a 2018 shortfall on 2026-10-13, when the plan
  expected 899. 2022 was planned to close, and a further top-up for it
  would be a new plan.
- Whether the 38 missing `frame_2016` get a sen2cor fallback pass.
