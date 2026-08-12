# Finding — LUCAS L2 model validation (independent field truth)

**Date:** 2026-08-12 · 10,108 in-crop LUCAS points, year-matched.
**Status:** distill + tessera done; tradslag (Prithvi-600M frac) still running.

## L2b — 28-class breadth (all points, the QA the 209 couldn't give)
| model | overall | κ |
|---|--:|--:|
| distill (Prithvi-600M, hard) | 0.484 | 0.458 |
| **tessera (hard)** | **0.499** | **0.473** |

Tessera edges distill on independent LUCAS truth too (0.499 vs 0.484) —
**corroborates the NFI picture** (tessera ≈ the best Prithvi, nominally ahead).

**Per-class (both models similar) — the breadth payoff:**
- **Strong (F1):** vatten 0.96, majs/sockerbetor/potatis 0.79–0.94, vete 0.78,
  oljeväxter 0.78. Water and spectrally-distinct crops are solid.
- **Weak (F1):** öppen mark 0.01–0.02, buskdominerad 0.07, öppen mark u. veg
  0.15–0.19, blandskog 0.21, bete 0.29. The open/shrub/mixed classes are where
  the model is genuinely poor — first time quantified against independent truth.

## L2a — forest fraction head vs LUCAS dominant species (tessera, threshold-free)
n_forest 3080, n_dominated 2090. **Dominant-species argmax agreement 0.81:**
- tall 0.83 · **gran 0.55** · löv 0.95

The fraction head's dominant call agrees with independent LUCAS 81% of the time —
strong for pine/broadleaf, but **gran (spruce) is the weak axis (0.55)**: spruce
is the hard forest class, confused with the others. (Prithvi-600M/tradslag's L2a
lands when its job finishes — the direct fraction-head comparison to tessera.)

## Crop year-matching (user-flagged, enforced)
Every crop point (11-21) asserted year-exact to its tile's spectral year at
runtime (fail-loud). Crops scored well (vete 0.78, majs 0.94, sockerbetor 0.93),
consistent with the LUCAS×LPIS label crosscheck.

## Reading
- LUCAS confirms tessera and the best Prithvi are **the same tier** on
  independent truth — reinforcing the model-race "statistical tie" (and tessera
  the cheaper of the two).
- The model is water/crop-strong, open/shrub/mixed-weak, gran-weak — a clear map
  of where accuracy actually lives, from a truth source the models never trained on.
