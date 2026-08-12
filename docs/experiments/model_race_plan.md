# Plan — Model race (best single backbone, not an ensemble)

**Status:** plan · **Created:** 2026-08-12 · **Supersedes the ensemble as the
headline strategy** (ensemble came back within-noise of its best member — see
`ensemble_g1g2_finding.md`).
**Thesis (user, 2026-08-12):** if combining doesn't beat the best member, stop
combining. Fine-tune every fine-tunable backbone to its best, race them on the
same honest held-out, ship the winner.

## The race is already half-run
All contenders share ONE target (the winning combo: distill 28-class labels +
Trädslag fraction head) and ONE eval (NFI 209 grouped-tile held-out + LUCAS).

| Backbone | seg model | dataset wiring | weights | trained | best NFI-209 |
|---|---|---|---|---|---|
| **Prithvi-600M** | ✓ | ✓ | ✓ | ✓ | **tradslag 0.579** (leader) |
| **Tessera** | ✓ | ✓ | baked | ✓ | 0.53 |
| CROMA | ✓ (croma_seg) | ✗ S2-12band + S1 | cached | ✗ | — |
| TerraMind | ✓ (terramind_seg) | ✗ spectral + S1 | cached | ✗ | — |
| Clay | ✓ (clay_seg) | ✗ S2 + wavelengths + timestamps | cached | ✗ | **viable** (see below) |
| Prithvi-300M | ✓ | ✓ | ✓ | ✗ | schema/tile-mismatch, low priority |

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

## Open decision
How wide to make the field (scope vs H100 spend). The two strongest bets attack
Prithvi-600M's lead on **orthogonal axes**, and both are untested:
- **CROMA** — modality diversity (S1 radar; G2 only tested *optical* diversity).
- **Clay** — spatial resolution (patch 8 vs 14; suits the per-pixel NMD2023 target).

Recommendation: race **CROMA + Clay** as the two orthogonal contenders; hold
TerraMind (optical + coarse — the weakest bet) unless one of the two pays.
Current step (user choice): formalize standings on the trained models first,
then commit the H100 for CROMA/Clay.
