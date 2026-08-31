# Finding — Ensemble G1/G2: tessera lands, but the ensemble stays within noise

**Date:** 2026-08-11 · **Branch:** `agent/te/opus/nfi-nmd2023-benchmark`
**Inputs:** 6 members' per-plot dumps (v8b, +markfukt, nmd2023_long, distill,
tradslag, **tessera**) on the 209 grouped-tile held-out NFI plots.
**Discipline:** config chosen on train-OOF (735) only; the 209 read once.

## G0 — tessera is a valid member ✓
Tessera (distill labels + frac head on frozen Tessera embeddings, 30 epochs,
best dense val mIoU 0.4650) scores **0.5339 / κ0.4065** 5-class OA on the 944
plots — above the ~0.40 floor and marginally above the distill Prithvi member
(0.5265). Non-degenerate; the `--backbone-name tessera_v1` load path is correct.

## G1 — the combiner does NOT significantly beat the 0.579 gate
OOF-locked config = **all5_tessera / plus_fractions / logreg** (OOF 0.6014):
- held-out OA **0.6172**, ΔG1 **+0.0382** vs the Trädslag gate (0.5789)
- McNemar p = **0.374**, bootstrap CIΔ = **[−0.038, +0.110]** (spans 0)
- **verdict: within_noise** — numerically ahead, not a statistically significant win.

R3: MLP not robustly better than logreg on OOF (CI [−0.010, +0.053], p=0.20) →
logreg reported. (The lone `beats_gate_significant` rows — all5/hard_p/mlp 0.660
and all5_tessera/plus_fractions/mlp 0.651 — are selection-biased sweep entries,
not the OOF-locked headline; they fail R3.)

## G2 — encoder diversity (tessera) does not clearly pay
tessera-in vs tessera-out at matched configs (held-out OA):

| config | no tessera | +tessera | Δ |
|---|--:|--:|--:|
| all5 / hard_p / logreg | 0.6124 | 0.6268 | +0.014 |
| all5 / plus_fractions / logreg | 0.5933 | 0.6172 | +0.024 |
| all5 / plus_fractions / mlp | 0.6411 | 0.6507 | +0.010 |
| baseline A (softmax-mean) | 0.6316 | 0.6268 | −0.005 |

On **train-OOF** tessera lifts the logreg combiner 0.5782 → 0.6014 (+0.023) and
the OOF-selection now prefers the tessera-in set — a **weak positive**. But every
held-out delta is within ±0.025 and none is significant. **G2 ≈ 0.**

## Bottom line & implications
- The single **Trädslag fraction member (0.579)** remains the best *defensible*
  forest product; the 6-member ensemble is **within noise** of it on the 209.
- Consistent with the NFI arc: the hard-label training **target** was the ceiling;
  the fraction head already extracted most of the available signal, so stacking
  more members/encoders on the same target doesn't break through.
- **209 is thin (R2)** — "within noise" is partly a power limit, not proof of
  no effect. LUCAS validation (10,329 pts) is the higher-power cross-check.
- **P8/CROMA recommendation: do NOT launch the H100 campaign.** The plan gated
  P8 on G2 paying; a genuinely different encoder (Tessera) moved the needle
  <0.03 and not significantly, so more foundation models are unlikely to. CROMA
  prep (weights cached, draft yaml) stays on the shelf unless LUCAS changes the
  picture.
