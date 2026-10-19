# Multi-year NFI control tiles, step 3: the top-up tiles for 2018, 2019, 2021 and 2022

Fetched and enriched 2026-10-13 to 2026-10-19 by the step-3 Jobs, each applied
from the head approved for it. Most came from #80, integrated as `09720dd`.
The re-probe came from #83 at `2fb697d`, integrated as `4252b61`. Steps ran
one at a time, never two writers on a staging root. Every count below is a
Job's own result line or the read-only top-up census (2026-10-19 07:23 to
07:58 UTC). Each step's Job, pod, node, image, times, exit code and log hash
are in the [JSON record](nfi_multiyear_topup_20261019.json). The tiles sit in
one staging root per year, `/data/unified_v2_512_nfi{Y}_staging`. None is in
the training cohort or the evaluation copy. Raw plot identities and
coordinates stayed in the data environment.

## Verdict

**The top-up closes 2019 and 2021, nearly closes 2022, and leaves 2018 well
short.** The planner counted each top-up plot-year only if no tile of the
evaluation copy already covered it. Its count of plot-years gained on the
written tiles therefore adds directly to the copy's eligible count:

| Year | Eligible on the evaluation copy | Gained by the top-up | Total | Short of 1,000 |
|---|---|---|---|---|
| 2018 | 489 | 337 | 826 | 174 |
| 2019 | 967 | 34 | 1,001 | 0 |
| 2021 | 848 | 153 | 1,001 | 0 |
| 2022 | 838 | 138 to 171 | 976 to 1,009 | 0 to 24 |

2022 is a range. The retry recovered one tile, and its plot count was not
printed. A census over the copy and the four staging roots together would
give the exact figure. 2018 was accepted below 1,000 on 2026-10-13. Every
feasible tile was already planned, and the cells with no HR-VPP product
removed 13 more candidate tiles that year.

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

- Each step-3 Job was applied with
  `git show <approved commit>:k8s/<job>.yaml | kubectl apply -f -`.
  Applying the bytes of the approved commit does not depend on the local
  checkout. Each Job ran with `SCRIPTS_COMMIT = d4d24f9`, checked HEAD, and
  verified the SHA-256 of every script it runs before starting them. The
  first plan and VPP window ran from `3481550`, before #80.
- All 20 step-3 logs are saved, under
  `~/.agents/continuity/evidence/ImintEngine/nfi-step3-20261019/`, with
  their pod records. After the session pause, each log was re-read from its
  pod and matched the hash recorded at completion. The exception is sen2cor:
  the gpu-reaper removed that GPU Job after it finished. Its log hash was
  recorded at completion and a copy is in the reaper archive.
- The census used the identity module at `ba31235c…` and the task-0103
  sizing. All eleven input hashes match, and the teacher union is 841.

## What this does not decide

- Whether forestry and markfukt (IM-022) are filled in. Until they are, no
  top-up tile reaches the bar.
- Whether a combined census is run over the copy plus the four roots, for
  exact deduplicated totals.
- Whether 2018's shortfall of 174 is accepted as final. Tobias accepted a
  shortfall on 2026-10-13, at the expected 899.
- Whether the 38 missing `frame_2016` get a sen2cor fallback pass.
