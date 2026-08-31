# Plan — Model race (best single backbone, not an ensemble)

**Status:** plan · **Created:** 2026-08-12 · **Supersedes the ensemble as the
headline strategy** (ensemble came back within-noise of its best member — see
`ensemble_g1g2_finding.md`).
**Thesis (user, 2026-08-12):** if combining doesn't beat the best member, stop
combining. Fine-tune every fine-tunable backbone to its best, race them on the
same honest held-out, ship the winner.

## Update — 2026-08-13 (decisions + state)

- **Race the whole field:** CROMA + Clay + TerraMind + **Prithvi-300M** (user,
  2026-08-13, "do them all"). TerraMind and 300M are kept **deliberately as the
  two patch-16 coarse anchors** for the resolution figure in the paper — they
  confirm the patch-size claim rather than needing to win.
- **thor is dropped** — `thor_v1_base` is a registry stub with **no seg model**
  (`imint/fm/thor_seg.py` does not exist); it can't be trained without
  implementing the model first. Out of scope.
- **The D-wiring does NOT actually work** (verified 2026-08-13 by ml-engineer —
  correcting an earlier optimistic note): the builder *functions*
  (`build_s2_{croma,clay}_tensor`) exist in `unified_dataset.py`, but the code
  paths are **never invoked** — `train_unified.py` never sets `model_keys`, so
  `s2_croma`/`s2_clay` are never built; `s1_vv_vh` (SAR, needed by CROMA+TerraMind)
  is **never emitted at all**; `trainer.py`'s forward reads only `batch["spectral"]`
  (no per-family branch); and the 3 seg models' `forward()` reject
  `return_fractions`/`temporal_coords`, so the frac-head target can't attach.
  **Result: CROMA/Clay/TerraMind crash on the first batch** (`TypeError`) — they
  need real model↔trainer glue, not a config tweak. Clay also has no cached
  weights (only a 404 marker). **Only Prithvi-300M is trainer-compatible today.**
- **Patch-16 crop constraint (likely 300M's "mismatch"):** 504 ÷ 16 = 31.5 →
  patch-16 backbones (**Prithvi-300M, TerraMind**) can't use the 504 crop; use a
  crop divisible by 16 (496 or 512). Clay/CROMA are patch-8 (504 fine).
- **Gated fusion result (this session):** Tessera + gated aux-fusion **ties**
  concat — NFI 0.588 vs 0.589, LUCAS 0.507/0.825 vs 0.499/0.809. Another
  "complexity doesn't pay" data point.
- **Validation is now cheap:** the two-stage cached eval (`infer_tiles.py` +
  `score_against_truth.py`, `docs/plans/faster_validation_architecture.md`) scores
  a trained checkpoint on NFI + LUCAS in minutes from one cached inference pass.
- **Status:** an ml-engineer is prepping the 4 training jobs (coverage check +
  per-model smoke + manifests); H100 launches are gated on its go/no-go. Both
  H100 nodes free → 2 parallel + queue.
- **Feeds:** `docs/publication/publication_plan.md` (this race closes claim C9
  and completes the resolution claim C6).

## Update — 2026-08-14 (adversarial review → test ALL; earn every "tie")

Three adversarial reviewers (RSE peer-review lens, statistical skeptic,
field-norms research) were tasked with challenging an earlier "drop TerraMind /
defer CROMA-Clay" idea. They converged unanimously; the idea is **reversed**.

- **Test ALL of CROMA, Clay, TerraMind** (+ 300M already training). Dropping any
  is scientifically indefensible: Clay/CROMA are the *only* patch-8 models, so
  omitting them **guts claim C6** (deletes the middle of the resolution axis);
  TerraMind is the within-stratum control for patch-16 (300M alone = n=1).
- **The "ties" are underpowered non-differences, not equivalences.** McNemar
  p=0.88 = "failed to reject", NOT "equivalent". At ~0.58 on n=209 the paired
  difference CI is **±8–10 pts** — a real 5-pt gap hides inside every "tie".
  Relabel all current head-to-heads (gated≈concat, Tessera≈Prithvi,
  ensemble-within-noise) as **INCONCLUSIVE** until equivalence-tested.
- **The optical/same-resolution base rate does NOT transfer** to radar (CROMA)
  or patch-8 (Clay) — expected-null is exactly when the test is most informative.
- **External norm:** GEO-Bench-2 (Nov 2025) already benchmarks TerraMind + Clay;
  Clay's patch-8 is a *documented* dense-prediction advantage; CROMA's SAR
  fusion = +6.4 % mIoU on segmentation (but PANGAEA found SAR sometimes *hurts*
  — only knowable by running it). A 2026 audit ("No One Knows the SotA in GFMs")
  targets exactly this incomplete-comparison failure mode.
- **Eval-rigor upgrades folded into the protocol** (§Protocol below is superseded
  by these): move off the 209-plot holdout to the **944 NFI + 10 k LUCAS**;
  **spatial-block CV** (adjacent plots aren't independent); **equivalence testing
  (TOST/ROPE)** against a pre-set SESOI before any "it ties" claim; Holm/BH
  multiplicity correction across the campaign.
- **Fallback if H100/eng time is tight:** linear-probe / frozen-encoder all three
  + disclose the tiered eval — NEVER silent omission.
- **Decision (user, 2026-08-14): build the glue for all three; adopt the rigor
  upgrades.** Glue = dataset `model_keys`+S1 emission (filter to S1-complete tiles
  for CROMA/TerraMind — only 6011/7882 have `s1_vv_vh`), per-family `trainer.py`
  forward branch, `enable_tradslag_head`+`(logits,frac_logits)` in the 3 seg
  models, Clay weights source fix.

## The race is already half-run
All contenders share ONE target (the winning combo: distill 28-class labels +
Trädslag fraction head) and ONE eval (NFI 209 grouped-tile held-out + LUCAS).

Status as of 2026-08-13 (wiring column corrected — it exists; see the update note above):

Status as of 2026-08-13 (verified by ml-engineer smoke pass):

| Backbone | patch | seg model | trainer-compatible? | trained | NFI forest | LUCAS (L2b/L2a) |
|---|---|---|---|---|---|---|
| **Prithvi-600M** | 14 | ✓ | ✓ | ✓ | **0.579** (leader) | — |
| **Tessera** | 1 | ✓ | ✓ (baked) | ✓ | **0.589** concat / 0.588 gated | 0.507 / 0.825 |
| **Prithvi-300M** | 16 | ✓ | ✓ (496 crop) | ✓ **field-validated 08-17** (coarse anchor) | **0.558** (913 plots) | 0.438 / 0.746 |
| Clay | 8 | ✓ | ✓ (glue + weights landed) | ✓ **field-validated 08-17** | **0.483** (last) | 0.404 / 0.698 (last) |
| CROMA | 8 | ✓ | ✓ glue landed | awaits S1 v2 re-enrich → retrain | — | — |
| TerraMind | 16 | ✓ | ✓ glue landed | awaits S1 v2 re-enrich → retrain | — | — |
| ~~thor~~ | — | ✗ no model | — | dropped | — | — |

**Data coverage** (7882 tiles): `spectral`/`b08`/`rededge` 7882, `tessera` 7874,
`s1_vv_vh` **6011** (1871 tiles lack SAR). Clay/CROMA optical inputs are fully
covered; the SAR gap is currently moot because `s1_vv_vh` is never emitted anyway.

**Glue needed to unblock CROMA/Clay/TerraMind** (model-side code, ~a focused pass):
(1) `train_unified.py` set `model_keys` + emit `s1_vv_vh` in the dataset;
(2) a per-family forward branch in `trainer.py` (CROMA `sar/optical`, Clay
`chips/timestamps/wavelengths`, TerraMind `{S2L2A,S1GRD}` dict);
(3) `enable_tradslag_head` + `(logits, frac_logits)` return in the 3 `*_seg.py`;
(4) fix Clay's weights source (only a 404 marker cached).
Manifests `k8s/train-{croma,clay,terramind}-job.yaml` exist with fail-loud
preflight guards, so none can burn an H100 until the glue lands.

### Clay is IN the field (correcting the old "ej segmenterbar" note)
Verified 2026-08-12: `clay_seg.py` is functional — Clay's default `encoder()`
returns a pooled `(B,1024)` image embedding (the source of the "not segmentable"
impression), but the wrapper hooks the **pre-pool tokens** → `(B,1024,32,32)`
dense map → per-pixel logits. Crucially Clay is **`patch_size=8`** vs
Prithvi-600M's 14 — ~1.75× finer patch grid. NMD2023 is a **per-pixel raster
target**, and finer patches mean less to reconstruct when decoding to pixels, so
Clay is a genuine contender for this dense task, not a write-off (user point,
2026-08-12). Wiring cost = the usual D-pass (dataset must emit Clay's S2 band
order + `wavelengths` + `timestamps` [week,hour,lat,lon]) + frac head + a fine-
tune; weights cached (`models--made-with-clay--Clay`).

So **Prithvi-600M + fractions (0.579) is the current champion**; Tessera is a
distant second. LUCAS numbers for both land shortly (L2 jobs running).

## What each new contender needs (per-model "D-wiring", ~the D1 tessera pass)
Each is: dataset input-routing on `spec.family` + frac-head on `<fam>_seg` +
inference routing — then a 1-epoch local smoke, then one H100 fine-tune, then
eval on the shared harness (`validate_against_nfi` + `validate_against_lucas`,
both exist).
- **CROMA** — route S1 `s1_vv_vh` (in the tiles) + assemble the 12-band S2
  (`build_s2_croma_tensor` exists). The ONLY contender with a different
  *modality* (radar) → the highest-value bet, because G2 showed *optical*
  encoder diversity ≈0 but says nothing about radar.
- **Clay** — the *spatial-resolution* bet (patch_size 8 vs 14): the orthogonal
  axis to CROMA's modality bet, and the one most aligned with a per-pixel raster
  target. Route S2 + wavelengths + timestamps.
- **TerraMind** — route spectral + S1. Optical-family, coarse patches; lower
  expected value (Tessera, also optical, didn't separate) — the weakest bet.

## Protocol (fair race — no silent advantages)
- Identical target (distill labels + `--enable-tradslag-head`), split
  (`distill_split.json` 735/209), and regime knobs where the backbone allows.
- Each model gets the frac head (fractions drove 0.579 — every model's best shot).
- Report per model: NFI-209 forest 5-class (fraction argmax + mixedness) AND
  LUCAS (28-class + forest-fraction). Head-to-head McNemar + bootstrap between
  the top two; §6 — a within-noise lead is not a win.
- Winner = best on the honest held-out with a defensible margin; ties broken by
  LUCAS (higher power) and by cost (Tessera is far cheaper to serve — no encoder
  forward).

## Cost
Per new contender: 1 wiring pass (subagent) + 1 H100 fine-tune (few h) + eval
(minutes GPU). CROMA+TerraMind = 2 wirings + 2 H100 runs. Clay adds a 3rd iff
its blocker clears.

## Decision — RESOLVED 2026-08-13
Race the **full field of four**: CROMA + Clay + TerraMind + Prithvi-300M.
- **CROMA / Clay** — the two *win-candidate* bets: CROMA = modality diversity
  (S1 radar, untested), Clay = spatial resolution (patch 8, suits the per-pixel
  target).
- **TerraMind / Prithvi-300M** — kept as the two **patch-16 coarse anchors** for
  the paper's resolution figure. Not expected to win (both share the coarse patch
  that made 300M's segments blocky); their job is to *confirm* patch size drives
  crispness, giving two independent coarse points rather than one.

Expectation (base rate): everything so far has tied, so the likely outcome is all
four land near the pack — but CROMA (radar) and Clay (resolution) are the only
untested axes, so they're the best remaining shot at a real separation, and
TerraMind/300M pay for themselves as controls regardless of where they land.

## Update — 2026-08-17 (300M complete: the coarse anchor lands below the pack)
Prithvi-300M field-validated (single-frame input fix `d958653`): **NFI 0.558 ·
LUCAS-28 0.438 · LUCAS-frac 0.746** — consistently lowest on BOTH truths
(pack: 0.579–0.589 / 0.477–0.507 / 0.784–0.825). The resolution axis now has
three measured points falling monotonically with coarser patch: p1 0.589 →
p14 0.579 → p16 0.558 — first real supporting data for C6 (equivalence-test
caveats still apply; 496-crop → 913 plots, not the exact 944 set). Per-species
spruce 0.457 — the shared weak axis again. Clay trained to val mIoU **0.319**
(surprisingly weak for the resolution bet — if its field numbers confirm,
that NUANCES C6: fine patch alone doesn't win; encoder quality gates it).
S1 v2 season-composite re-enrichment running (~41–48 tiles/h cold-cache,
watch rate); validation routing for clay/croma/terramind landed `4b9b785`
(82 tests). Clay NFI+LUCAS validation launched 08-17.

## Update — 2026-08-17 (Clay verdict: the interior point BREAKS the resolution curve)
Clay (patch-8) field-validated: **NFI 0.483 · LUCAS-28 0.404 · LUCAS-frac
0.698** — last on ALL three truths, and **below the patch-16 anchor**
(300M 0.558). The monotone patch-size story (0.589→0.579→0.558) is refuted
by its own interior point: **fine patch alone doesn't win — encoder quality
gates the outcome.** What survives: tiny-but-strong Tessera (p1) beats the
600M giant ("cheap-and-strong beats big"). C6 is REFORMULATED accordingly in
`docs/publication/publication_plan.md`. CROMA (p8, different encoder,
+radar) is now the decisive second patch-8 point. Label-ceiling holds for
every model: even Clay (0.483) beats NMD2023 (0.390) by +0.09.
Ops note: the S1 v2 enrichment filled the 1.6T PVC (317 GB product cache,
tile 165/7882) — cache now LRU-capped at 150 GB (`8f8624c`), job relaunched.
