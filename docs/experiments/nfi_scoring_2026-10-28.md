# Multi-year NFI held-out evaluation, step 6: scores against NMD

Task 0118, scored 2026-10-28. The 28 frozen-set prediction files
([inference record](nfi_inference_run_2026-10-27.md)) scored against NMD2023 and NMD2018 on
the frozen set ([freeze](nfi_multiyear_freeze_2026-10-27.md), manifest `daaa4287…ff2c`), on
Tobias's go. Aggregates only; the full report stays on the PVC
(`nfi/nfi_scoring_2026-10-28/scores.json`, sha256 `cabff93f…4f42`), summarised in the
[JSON companion](nfi_scoring_20261028.json). No plot identity is in either file.

## Verdict

**Every model's rung 3 and 4 beat NMD2023 except clay's and prithvi600m_r3; no rung 1 or 2
does.** 11 model cells agree with NFI forest type significantly more often than
NMD2023, 10 significantly less often, 3 are equivalent
(TOST) and 4 are undecided. The highest point estimate is **tessera_r4**
(53.4 % exact, κ 0.375) against NMD2023's
44.8 % (κ 0.304). Tessera's
rungs 3 and 4 are equivalent to each other; tessera_r4 is ahead of every other backbone's best cell
by 4.3 to 8.5 points, each difference supported.

Support: 5,194 plot-years, 742 per year 2018–2024 (balanced primary
population, seeded selection at the freeze), none excluded for NMD coverage;
182 spatial blocks of 50 km. Statistics as prespecified at the freeze:
spatial-block bootstrap intervals and an exact block sign-flip test, Holm-adjusted over the
whole family of 434 model–model and model–NMD pairs (311 supported
differences, 73 marginal equivalences). "Better" and "worse" below
need a supported difference: a nonzero interval and Holm p < 0.05.

| Source | Exact agreement | κ | vs NMD2023 (pp) | 95 % interval (pp) | Verdict |
|---|---|---|---|---|---|
| tessera_r4 | 53.4 % | 0.375 | +8.6 | +7.0 to +10.2 | better |
| tessera_r3 | 53.3 % | 0.374 | +8.5 | +6.9 to +10.1 | better |
| croma_r4 | 49.1 % | 0.326 | +4.3 | +2.9 to +5.8 | better |
| croma_r3 | 49.0 % | 0.324 | +4.2 | +2.7 to +5.8 | better |
| prithvi300m4f_r4 | 48.9 % | 0.321 | +4.1 | +2.5 to +5.7 | better |
| prithvi300m_r3 | 48.8 % | 0.323 | +4.1 | +2.4 to +5.8 | better |
| terramind_r3 | 48.7 % | 0.322 | +4.0 | +2.3 to +5.7 | better |
| prithvi300m4f_r3 | 48.6 % | 0.319 | +3.9 | +2.3 to +5.5 | better |
| terramind_r4 | 48.5 % | 0.319 | +3.8 | +2.1 to +5.5 | better |
| prithvi300m_r4 | 48.2 % | 0.317 | +3.4 | +1.7 to +5.2 | better |
| prithvi600m_r4 | 47.6 % | 0.317 | +2.8 | +1.4 to +4.3 | better |
| prithvi600m_r3 | 47.6 % | 0.319 | +2.8 | +1.3 to +4.3 | undecided |
| tessera_r2 | 45.4 % | 0.318 | +0.6 | -0.9 to +2.1 | equivalent |
| clay_r3 | 44.9 % | 0.276 | +0.1 | -1.6 to +1.9 | equivalent |
| clay_r4 | 44.8 % | 0.275 | +0.0 | -1.7 to +1.7 | equivalent |
| **NMD2023** | **44.8 %** | 0.304 | — | — | baseline |
| tessera_r1 | 43.6 % | 0.289 | -1.2 | -2.7 to +0.3 | undecided |
| prithvi600m_r1 | 43.2 % | 0.285 | -1.6 | -3.1 to -0.0 | undecided |
| prithvi300m4f_r1 | 42.1 % | 0.273 | -2.7 | -4.2 to -1.1 | undecided |
| prithvi300m4f_r2 | 41.9 % | 0.280 | -2.9 | -4.3 to -1.5 | worse |
| terramind_r1 | 41.4 % | 0.267 | -3.4 | -4.9 to -1.8 | worse |
| croma_r1 | 41.2 % | 0.262 | -3.6 | -5.2 to -2.0 | worse |
| croma_r2 | 40.9 % | 0.268 | -3.9 | -5.3 to -2.5 | worse |
| prithvi300m_r1 | 40.8 % | 0.259 | -3.9 | -5.7 to -2.1 | worse |
| terramind_r2 | 40.7 % | 0.270 | -4.1 | -5.4 to -2.7 | worse |
| prithvi300m_r2 | 40.4 % | 0.267 | -4.3 | -5.7 to -3.0 | worse |
| prithvi600m_r2 | 40.3 % | 0.268 | -4.5 | -5.9 to -3.0 | worse |
| clay_r1 | 38.8 % | 0.231 | -6.0 | -7.6 to -4.4 | worse |
| clay_r2 | 38.5 % | 0.238 | -6.3 | -7.7 to -4.9 | worse |
| **NMD2018** | **38.1 %** | 0.218 | — | — | baseline |

Rung 1 learned from NMD2018, rung 2 from NMD2023 (sidecar, 28 classes), rung 3 adds the
self-distilled NFI teacher, rung 4 the tree-species fraction head
([ladder](label_source_ladder.md)). The NFI plot-years whose features trained the teachers
were excluded at the freeze, so rungs 3–4 are scored on plot-years they never saw.

## The top model

| tessera_r4 vs | pp | 95 % interval | Result |
|---|---|---|---|
| tessera_r3 | +0.1 | -0.2 to +0.5 | equivalent |
| croma_r4 | +4.3 | +3.1 to +5.5 | supported |
| croma_r3 | +4.4 | +3.2 to +5.6 | supported |
| prithvi300m4f_r4 | +4.5 | +3.3 to +5.8 | supported |
| prithvi300m_r3 | +4.5 | +3.4 to +5.8 | supported |
| terramind_r3 | +4.7 | +3.6 to +5.8 | supported |
| prithvi300m4f_r3 | +4.7 | +3.5 to +6.0 | supported |

## By inventory year (exact agreement, %)

| Year | tessera_r4 | croma_r4 | prithvi300m_r3 | NMD2023 | NMD2018 |
|---|---|---|---|---|---|
| 2018 | 52.4 | 50.9 | 51.1 | 46.8 | 39.0 |
| 2019 | 50.9 | 46.2 | 46.6 | 42.4 | 38.8 |
| 2020 | 54.3 | 49.6 | 50.3 | 43.8 | 41.9 |
| 2021 | 57.0 | 51.6 | 52.0 | 42.4 | 38.4 |
| 2022 | 52.4 | 46.8 | 46.2 | 43.4 | 34.8 |
| 2023 | 53.0 | 48.9 | 46.1 | 46.6 | 38.0 |
| 2024 | 53.6 | 49.5 | 49.6 | 47.8 | 35.9 |

Per-year and per-tile-role scores for every source are in the JSON companion; they are
descriptive strata, and confirmatory inference applies only to the primary population.

## Run

`k8s/nfi-scoring-job.yaml` at main `0f1c394` (identical to the reviewed `36e30cb`): the
frozen `score_nfi_holdout.py`, unchanged, in the sealed image (sha-4cb93b4) under the model
interpreter. It re-verified the manifest, the sealed source and runtime, and every prediction
file before scoring. Pod `nfi-scoring-20261028-zcw59` on `p02r02srv01`, 07:44:11–07:45:49Z,
exit 0. The report was read back as uid 2000 and its sha256 matched the Job's.

## Not done

- A split by `has_frame_2016`: the frozen scorer has none, and Tobias declined a post-hoc
  split (2026-10-28).
- Any other population (all eligible plot-years, campaign only): the freeze prespecified the
  balanced population as primary.
