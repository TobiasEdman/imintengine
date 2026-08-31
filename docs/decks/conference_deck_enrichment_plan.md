# Plan — enrich the conference deck (v8b_nfi_conference)

**Status:** plan, ready to execute · **Created:** 2026-08-12
**Edit in place:** `docs/decks/build_conference_deck.js` (ENGLISH only; already
has the 16-slide arc + the 5-panel "Five views" slide with the Tessera frame).
Keep the DES brand. Regenerate `docs/decks/v8b_nfi_conference.pptx`.

## Goal (user, 2026-08-12)
1. Land the **"student beats teacher"** story explicitly (a named scientific hook).
2. Make **Prithvi vs Tessera a real head-to-head** across every truth.
3. Overall: **more information, richer/annotated figures, better explanations** —
   every results slide earns 1-2 sentences of "what it means / why it matters",
   grouped+annotated charts over single bars, per-category breakdowns where we
   have them, richer speaker notes.

## NEW slide — "The student beats the teacher" (after the held-out benchmark, before "Five views")
Hook: in distillation a student normally can't exceed its teacher — here it does.
- Trained ONLY on NMD2023 labels (teacher). On independent NFI field truth the
  student scores **50.2%** vs the teacher's **43.1%** (**+7.1 pp**); the forest-type
  layer **57.9%** (**+14.8 pp**). Kappa 0.298 → 0.371 → 0.420.
- Figure: grouped/annotated bar — Teacher 43.1 · Student 28-class 50.2 · Student
  forest-type 57.9 — deltas annotated above the student bars, faint reference
  marker at the teacher level.
- Why (3 cards): (1) **label noise averages out** — random per-pixel teacher errors
  regularize away over 7,882 tiles ("denoising you can see"); (2) **field truth
  exposes the teacher's own errors** — invisible when NMD is scored against itself;
  (3) **a richer representation than the label** — 4 temporal frames + 11 aux
  channels carry what a single-date NMD label can't.

## REWORK slide — "Prithvi vs Tessera: head to head" (replace the 2-bar model-race slide)
Two backbones, same field-calibrated target, raced on every truth.
- **Grouped bar** — 3 metric groups × 2 bars (Prithvi-600M vs Tessera):
  - NFI-209 forest type: 0.579 vs **0.589** (annotate: statistical tie, McNemar p=0.88)
  - LUCAS 28-class (independent): 0.477 vs **0.499**
  - LUCAS forest fraction (independent): 0.784 vs **0.809**
  Tessera ≥ Prithvi on all three.
- **Versus mini-table** (winner highlighted per row):
  backbone (600M full encoder forward | frozen precomputed 128-d embedding) ·
  inference compute (heavy | ~free) · NFI 0.579|0.589 · LUCAS-28 0.477|0.499 ·
  LUCAS-frac 0.784|0.809.
- **Verdict callout:** "Statistical tie on accuracy across three independent
  truths — Tessera wins decisively on compute. Deployment winner: Tessera."

## ADD if it fits — per-species forest breakdown (on/near the LUCAS slide)
LUCAS L2a dominant-species agreement, both models:
Pine 0.83/0.81 · Spruce **0.55/0.53** · Deciduous 0.95/0.92. Point: spruce is the
hard axis for BOTH; pine/deciduous strong. Small grouped bar or annotated row.

## Exact numbers (authoritative — do not alter)
Held-out 209: NMD2023 43.1% / distilled 50.2% / fraction 57.9%; kappa 0.298/0.371/0.420.
Ceiling journey (944): 46.5 / 46.0 / 49.3 / 52.7 / 63.7.
Per-class F1 (NMD/distill/frac): pine .59/.61/**.74** · spruce .56/**.63**/.55 ·
decid .30/.43/**.59** · mixed .28/**.30**/.29 · non-forest .24/**.32**/.00.
Ensemble: best combiner 0.617, within noise (McNemar p=0.37, CI spans 0);
baseline softmax-mean 0.632; encoder-diversity Δ <0.03 (not significant).
Model race NFI: Tessera 0.589 vs Prithvi 0.579 (Δ+0.010, p=0.88, tie).
LUCAS L2b: Tessera 0.499 · distill 0.484 · Prithvi/tradslag 0.477.
LUCAS L2a: Tessera 0.809 (pine .83/spruce .55/decid .95); Prithvi 0.784 (.81/.53/.92).
LUCAS×LPIS crops: 79.8% on-parcel, agreement 0.81 excl. pasture (cereals/roots .69-.91).
Data facts: 28 classes · 7,882 tiles · 944 NFI plots (209 held out) · 34,480 LUCAS pts.

## Execute + QA
1. Edit `docs/decks/build_conference_deck.js`; `NODE_PATH=$(npm root -g) node docs/decks/build_conference_deck.js`.
2. **Use `.venv/bin/python` for the skill scripts** (system python3 is 3.9; scripts need 3.10+; defusedxml is installed in .venv):
   `.venv/bin/python "<pptx-skill>/scripts/office/validate.py" docs/decks/v8b_nfi_conference.pptx` — fix faults in the generator.
   render: `.venv/bin/python "<pptx-skill>/scripts/office/soffice.py" --headless --convert-to pdf --outdir /tmp docs/decks/v8b_nfi_conference.pptx` → `pdftoppm -jpeg -r 120 /tmp/v8b_nfi_conference.pdf /tmp/cv` → inspect every slide for overflow/overlap/contrast; iterate.
3. pptxgenjs footguns: no `#` hex, fresh option obj per add* call, grouped charts pass an array of series, chartColors from the DES palette, dataLabelPosition ctr/inEnd/inBase on stacked (outEnd corrupts). Space Grotesk is QA-unreliable in LibreOffice — leave ~10% slack.
