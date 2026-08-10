# Plan — ensemblen: NFI-stackad multi-modell (fas efter NFI-arcen)

**Status:** plan (ej påbörjad) · **Skapad:** 2026-08-10 · **Ägare:** Tobias
**Förutsättning:** NFI-arcen komplett (docs/data/*_finding.md, PR #27). Eval-ramverket
`imint/eval/` (Fas 1–5, maj 2026) har väntat på "real ensemble checkpoint" — nu finns
medlemmarna.

## Varför nu

Ensembler vinner på **diversitet**, och NFI-arcen skapade den gratis: samma backbone,
olika *superviseringsmål* (NMD2018, NMD2023-hard, distillerade NFI-pseudolabels,
Trädslag-fraktioner). Deras fel är bevisat olika (per-klass-vinnarna skiljer sig:
fraktioner vinner tall/löv, distill vinner gran/icke-skog, v8b är bäst på bland).
Dessutom finns hybrid-head-maskineriet (per-plot-features/-dumps + train/test-split +
kalibrering) som generaliserar direkt till **stacking kalibrerad på fältdata**.

## Medlemsinventering

| Medlem | Checkpoint | Status |
|---|---|---|
| v8b (NMD2018, 23-kl) | /cephfs/nfi_eval/v8b_best_model.pt | ✅ klar, per-plot-dump finns |
| v8b+markfukt (11-aux) | /cephfs/checkpoints/v8b_markfukt/ | ✅ klar |
| v8b-NMD2023-long (28-kl) | /cephfs/checkpoints/v8b_nmd2023_long/ | ✅ klar |
| Distillerad (28-kl) | /cephfs/checkpoints/v8b_nmd2023_distill/ | ✅ klar, per-plot-dump finns |
| Trädslag (28-kl + frac-head) | /cephfs/checkpoints/v8b_nmd2023_tradslag/ | ✅ klar, per-plot-fraktioner finns |
| Prithvi-300M/256 | /checkpoints (train-prithvi-300m-256, maj) | ⚠ verifiera format/schema |
| clay/terramind/tessera/croma seg | imint/fm/*_seg.py + cachade FM-vikter | ❌ otränade — DYRT, ej i steg 1 |

## Arkitektur — två nivåer

**A. Baseline-ensemble (billig sanity):** softmax-medel över medlemmarnas 5-klass-
kollapsade sannolikheter vid NFI-plottarna. Ingen träning; sätter golvet.

**B. NFI-stackad combiner (huvudspåret):** per plott, konkatenera medlemmarnas
utsignaler (5-klass-sannolikheter + Trädslags 4 fraktioner + ev. 256-features) →
träna en liten combiner (logreg/MLP, som hybrid-head) **enbart på train-splittens
735 plottar** → utvärdera på de orörda 209. Detta är stacking med exakt samma
ärliga håll-ut-protokoll som distill-experimentet (distill_split.json ÅTERANVÄNDS —
splitten är redan "förbrukad" åt rätt håll: inga av de 209 har sett någon medlem).

## Steg

1. **Per-plot-dumpar för saknade medlemmar** (GPU, ~15 min/st på 2080ti):
   v8b+markfukt (finns: aggregat men ej dump med p-kanaler), v8b_nmd2023_long,
   ev. 300M/256. Kör `validate_against_nfi.py --dump-per-plot` (p1–p4 finns redan
   i dumpen sedan kalibreringsfixen). Trädslag + distill + v8b: klara.
2. **Stack-matris lokalt:** join:a dumparna på TractID/PlotID → X = medlemmarnas
   kanaler, y = NFI-truth. Baseline A (softmax-medel) + combiner B med
   train/test-split. Rapportera håll-ut-209 vs bästa enskilda (fraktioner 0,579).
3. **Beslutsgrind:** slår stacken 0,579 på håll-ut? Om nej → stanna, dokumentera
   (medlemmarna kan vara för korrelerade — samma backbone). Om ja → steg 4.
4. **Dense ensemble-artefakt:** exportera combinern (npz, som distill-head) +
   ett `ensemble_predict`-skript som kör medlemmarna + combinern per tile →
   det "ensemble checkpoint" som `run_full_eval.py` väntat på sedan maj.
5. **Full eval:** `run_full_eval.py` (Fas 1 in-distribution) + `compare_eval_runs.py`
   mot v8b- och distill-baselines; NFI-harnessen på 209 som huvudmått.
6. **Ev. utökning:** träna 1 diversare medlem (clay eller terramind seg — annan
   backbone = äkta diversitet) om steg 3 visar korrelationstak. Egen H100-kampanj.

## Kostnad

Steg 1–3: ~1 h GPU (2080ti) + lokal CPU — billigt. Steg 4–5: inferens-jobb över
tiles för dense eval (~timmar, ingen träning). Steg 6 (om det behövs): full
träningskampanj — eget beslut.

## Risker / beslutspunkter

- **Korrelerade medlemmar:** alla fem är Prithvi-600M-varianter — vinsten kan bli
  liten. Steg 3-grinden skyddar mot att bygga vidare på en död idé. Äkta diversitet
  (annan backbone, S1, tessera-embeddings som combiner-features) är steg 6-frågor.
- **209 plottar är tunt** för att särskilja stack vs bästa medlem — rapportera
  parad jämförelse (McNemar) och var ärlig om osäkerheten.
- **Beslut för användaren:** (a) medlemsmängd i steg 1 (alla 5 vs topp-3),
  (b) combiner-features (bara 5-klass-p vs +fraktioner vs +256-features),
  (c) om steg 6 (ny backbone) ska förberedas parallellt.

## Referenser

Hybrid-maskineri: scripts/{extract_plot_features,nfi_head_cv,train_distill_head}.py ·
split: data/distill/distill_split.json · eval-ramverk: imint/eval/, scripts/run_full_eval.py ·
findings: docs/data/{hybrid_nfi_head,tradslag_fraction,distill}_finding.md
