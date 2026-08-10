# Kampanjplan — NFI-stackad multi-modell-ensemble (full)

**Status:** plan, redo att exekvera · **Skapad:** 2026-08-10 (arkitekt- + gap-agent-syntes)
**Ersätter:** ensemble_plan.md (rev 2, behålls som historik/seed)
**Branch:** `agent/te/opus/nfi-nmd2023-benchmark`

## Mål & tvåspårsstruktur

Ensemblen byggs på **modelldiversitet** (user-beslut 2026-08-10): Prithvi-varianterna
ger mål-diversitet, **Tessera** ger encoder-diversitet nu, CROMA/TerraMind senare.
Träningsmål för nya medlemmar: **2023-materialet multi-task = distillerade hårda
labels (28 kl) + Trädslag-fraktioner** (kombinationen bakom våra två bästa resultat).

Två spår med olika sanningar — blandas inte ihop:
- **Track A (huvudresultatet):** NFI-stackad combiner, 5-klass skogstyp på de
  **209 håll-ut-plottarna**. Billig (~1 GPU-h + lokal CPU). Grind: slå **0,579**.
- **Track B (dense produkt):** 28-klass mIoU över tiles via `run_full_eval.py`.
  OBS: `run_full_eval.py` och `compare_eval_runs.py` är **skelett**
  (`NotImplementedError`) — Track B är riktig implementation och körs bara om
  Track A klarar grinden.

## D1 — Tessera-wiring (gap-agentens dom: M+S, inga blockers)

~70 % finns: registry (`tessera_v1` — OBS nyckeln, inte `tessera`), loader,
`TesseraSegmentationModel` (num_classes-parameteriserad, aux-fusion),
`build_segmentation_from_spec`-routing, `_LabelOverlay`/frac-sidecars, tester.
Luckor (fil:rad i gap-rapporten):
1. **Dataset/trainer-routing (M, ~30 rader):** UnifiedDataset läser aldrig
   `tessera`-nyckeln (hårdkodad `spectral`); trainer.py:501 hårdkodar
   `batch["spectral"]`. Fix: branch på `spec.family` i unified_dataset
   (~454/484–495/606) + input-routing i trainern (4D, ingen 5D-reshape).
2. **Frac-head på tessera_seg (S, ~25 rader):** kopiera Prithvi-mönstret —
   `enable_tradslag_head` + `Conv2d(final_in, 4, 1)` + `return_fractions` i
   forward. Trainern behöver noll ändringar (frac-loss-wiring är klar).
3. **Inferens-routing (S, ~10 rader/script):** `_build_inference_inputs` +
   `make_model_predict_fn` — läs tessera-nyckeln, hoppa över Prithvi-norm
   (embeddings redan normerade).

## Faser

### P0 — Tessera-medlemmen (H100; beroende: D1-fixarna ovan)
- Implementera D1 (subagent-pass), **1-epok lokal smoke FÖRE k8s-submit**
  (repo-regel) — verifierar tessera-tensor → tessera_seg end-to-end.
- Träna: `k8s/train-tessera-distill-job.yaml` (klon av train-v8b-distill) med
  `--backbone-name tessera_v1 --num-classes 28 --label-dir /cephfs/nmd2023_distill_labels
  --frac-dir /cephfs/tradslag_fracs --enable-tradslag-head` — INGEN
  multitemporal (tessera är single-frame), ingen warm-start, större batch
  (ingen encoder-forward → få H100-timmar).
- NFI-validate + per-plot-dump (→ P1).
- **G0:** icke-degenererad på 209 (≥ ~0,40)? Annars Prithvi-only och G2 utgår.

### P1 — Per-plot-dumpar (2080ti, ~15 min/st, parallellt med P0)
Saknas med p-kanaler: v8b (23 kl), v8b+markfukt (`--enable-markfukt`),
v8b_nmd2023_long (28 kl) + tessera efter P0. distill + trädslag klara.
Verifiering: varje dumps `accuracy_suite` reproducerar publicerat tal
(0,465 / ~0,466 / 0,460 / 0,502 / 0,579) — regressionskoll på dumpen.

### P2 — Stack-matris + combiner (lokalt)
`scripts/build_ensemble_stack.py`: join på TractID+PlotID; **tagga
kanalsemantik per medlem** (trädslag = fraktioner, övriga = softmax — får inte
blandas otaggat, R6). Baseline A = softmax-medel (golv). Combiner B = logreg +
MLP(128), tränad ENBART på train-splittens 735 (StandardScaler på train,
`nfi_head_cv`-mönstret), utvärderad på 209; npz-export med port-check
(`train_distill_head`-mönstret). Tre feature-varianter: (i) 5-klass-p,
(ii) +fraktioner, (iii) +tessera — (iii) vs (i/ii) ÄR G2-ablationen.

### P3 — Beslutsgrindar
- **G1:** combiner-B > 0,579 på 209? Nej → stoppa Track B, skriv negativ
  finding, skeppa kompositionen (distill-bas + fraktionslager). Ja → P4.
- **G2:** tessera-ablation Δ — betalar encoder-diversiteten? Om Δ≈0 höjs
  ribban kraftigt för P8 (CROMA/TerraMind lär inte heller betala).
- **G3:** parad **McNemar** (combiner vs trädslag på samma 209) + bootstrap-CI.
  0,60 vs 0,579 ≈ 4 plottar — utan signifikans rapporteras det som "inom brus",
  inte som vinst.

### P4 — Dense ensemble-artefakt (efter G1-pass)
`ensemble.pt` = **manifest-bundle** (medlemslista m. checkpoint/num_classes/
mode[hard|fraction]/aux + inline combiner-npz) + nytt `imint/eval/
ensemble_predict.py::make_ensemble_factory` → `predict_fn(npz)→(pred, probs)`
(exakt kontraktet i `eval_in_distribution.py:60`). Återanvänd
`make_model_predict_fn`/`make_fraction_predict_fn`.
**Output-space (beslut 3):** rekommenderat = **skogstyps-raffinering ovanpå
distillerade 28-klassbasen** (combinern styr bara klass 1–4; hårda basen ger
mask + övriga klasser) — enda varianten där Phase-1-mIoU är jämförbar med
distill-baselinen.

### P5–P6 — Implementera eval-skeletten + jämför (Track B)
Fyll `run_full_eval.py` (endast Fas 1 nu; Fas 2–4:s split-byggare/baselines är
stubbar — blockera inte på dem; `trivial_majority` är enda implementerade
baselinen, inkludera den) och `compare_eval_runs.py` → delta-tabell ensemble
vs distill/v8b. Kör på test-splitten (~10 % av tiles), inte alla 7 882.
Kostnad: timmar GPU (N medlemmar × dense inferens).

### P7 — Statistisk ärlighetspass
McNemar (b,c + p), bootstrap-95%-CI (10k), per-klass-supportcaveats (bland/
icke-skog tunna på 209). §6-disciplin: inom-brus-resultat får inte rubriceras
som vinst.

### P8 — CROMA / TerraMind (grindad; egna H100-kampanjer)
Trigger: G1 ✓ OCH G2 visade att encoder-diversitet betalar OCH parvis
medlems-disagreement (från P2-matrisen) visar korrelationstak. CROMA först
(S1 = `s1_vv_vh` finns i tiles → modalitetsdiversitet); vikter behöver hämtas
(ej i cachen). TerraMind-vikter cachade. Clay utgår (ej segmenterbar);
Thor overifierad; Prithvi-300M/256 tveksam (schema/tile-mismatch).

## Compute-karta

| Fas | Var | Kostnad |
|---|---|---|
| P0 wiring+smoke | lokalt | ~1 subagent-pass + minuter |
| P0 träning | H100 | få timmar (ingen encoder-fwd) |
| P1 dumpar ×4 | 2080ti | ~1 GPU-h (parallellt med P0) |
| P2–P3 | lokalt | minuter |
| P4 | lokalt + 1 tile-smoke | minuter |
| P5–P6 | GPU (test-split) | timmar |
| P8 | H100 | kampanj/medlem (grindad) |

## Beslut att ratificera före P2

1. **Medlemsmängd:** alla 5 Prithvi + tessera, eller topp-2 (distill+trädslag)
   + tessera? (Fler medlemmar = mer combiner-överfit på 735; rek: kör båda,
   rapportera.)
2. **Combiner-features:** kör alla tre varianterna (i/ii/iii) och rapportera —
   valet ÄR ablationen. (Rek: ja.)
3. **Output-space P4:** skogstyps-raffinering över distillerad bas (rek) vs
   ren 5-klassprodukt.
4. **P8:** vänta på G2 (rek) eller förbered CROMA-kampanjen parallellt?

## Risker
R1 korrelerade Prithvi-medlemmar (därför är tessera på kritiska vägen; parvis
disagreement rapporteras) · R2 209 är tunt (P7 obligatorisk) · R3 combiner-
överfit på 735 (logreg rapporteras om MLP inte slår logreg utanför CI) ·
R4 eval-skeletten är riktig implementation · R5 output-space-mismatch (P4
option 1 enda ärliga) · R6 fraktions-/prob-kanaler får inte blandas otaggat ·
R7 D1-wiring (låg risk per gap-rapporten, men smoke före submit).

## Nyckelreferenser
Kontrakt: `imint/eval/eval_in_distribution.py:60` · skelett:
`scripts/{run_full_eval.py:75,compare_eval_runs.py:51}` · stack-maskineri:
`scripts/{nfi_head_cv,train_distill_head,validate_against_nfi}.py` · registry:
`imint/fm/registry.py:229` (`tessera_v1`) · split:
`data/distill/distill_split.json` · jobbmallar:
`k8s/{train-v8b-distill,nfi-validate-distill}-job.yaml`
