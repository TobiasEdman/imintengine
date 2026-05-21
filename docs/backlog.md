# ImintEngine — Backlog

Lättviktig löpande lista över kända, ej-färdiga åtgärder. Kompletterar
`docs/conversation_log/` (kronologisk historik) med en *åtaganden-vy*
som ska gå att skanna på 30 sekunder.

**Inte ersättning för** stories/PR-discussions/agile-sprint-planering —
ImintEngine är ML-pipeline/forskning, inte produktutveckling. Det
finns ingen PO, ingen personas, inga sprintar. Den här filen är ett
*git-versionerat index* över småskaliga uppgifter som annars riskerar
glömmas mellan sessioner.

---

## Konventioner

**ID:** `IM-NNN` (ImintEngine, nollpaddad löpnummer). Välj nästa lediga
genom `grep -oE "IM-[0-9]+" docs/backlog.md | sort -u | tail -1`.

**Status:**
- `open` — inte påbörjad
- `in-progress` — aktiv branch eller pågående utredning
- `blocked` — väntar på extern händelse (annan PR, jobb-completion)
- `done` — landad i main; flyttas till "Done"-sektion nedan
- `wontfix` — beslutat att inte göra; flyttas till "Done" med motivering

**Typ:**
- `bug` — felaktigt beteende (kosmetiskt / funktionellt)
- `tech-debt` — kod fungerar men borde vara renare
- `infra` — k8s, PVC, scheduling, secrets
- `housekeeping` — committs, branch-merges, städning
- `investigation` — diagnostik utan färdig åtgärd
- `gate` — väntar på annan färdigställning

**Commit-referens:** prefixa commit-meddelande med ID, t.ex.
`IM-001: trainer writes tile_counts to training_log.json`. Då fungerar
`git log --grep="IM-001"` som leverans-historik per item.

**Status-uppdatering:** manuell. Den som mergar en PR (eller flyttar
en blocker) uppdaterar tabellen i samma PR där det är möjligt.

---

## Active

| ID | Titel | Typ | Status | Branch/PR | Anmärkning |
|---|---|---|---|---|---|
| IM-001 | Trainer skriver `train_tiles`/`val_tiles` till `training_log.json` | tech-debt | open | — | Förebyggande fix; just nu rekonstruerar dashboard-podden via MD5-split |
| IM-002 | Dashboard `<title>` + `<h1>` säger "19-Class" — uppdatera till "23-Class" | bug | open | — | Kosmetiskt; v5-schema är 23 klasser |
| IM-003 | Null-safe-fix för `train_tiles.toLocaleString()` till repo (just nu bara server-side) | tech-debt | in-progress | server-side i `k8s/training-dashboard-server.yaml` | Lyft till `dashboards/unified_training_dashboard.html` i nästa dashboard-PR |
| IM-004 | `nodeAffinity p02r08srv01` på dashboard-pod är skör — byt till `podAffinity` mot training-pod | infra | open | — | Bryts vid nästa H100-nodbyte. Bättre: colocate-by-label med aktiva training-pods |
| IM-005 | ~~Efter PR #13 merge: ta bort server-side patcher~~ | housekeeping | **done** 2026-05-21 | edit i `training-dashboard-server.yaml` | Patch-blocket borttaget; pod var crash-loop före (bakgrund-filter regex matchade 0). Bara sed null-safe (IM-003) kvar som server-side |
| IM-006 | Merga `feat/dataset-clay-croma-emission` | gate | blocked | blocked-on: Clay/CROMA training end-to-end validation | Branch innehåller `s2_clay`/`s2_croma`-keys i `UnifiedDataset`. Smoke PASS, men ej kört ett tränings-batch än |
| IM-007 | Råg → vete asymmetri 24.3% — diagnostisera och åtgärda | investigation | **findings-in** | — | 600M klar: råg-IoU 0.333 vs 300M:s 0.381 (REGRESSION). 600M bloat 1.71× vs 300M:s 1.25×. Slutsats: större modell hjälper INTE (allokerar params till dominerande klasser). Åtgärd: bumpa class weights / oversample → IM-017 |
| IM-008 | CLASS_NAMES single source of truth — just nu duplicerat i `imint/training/unified_schema.py`, `dashboards/unified_training_dashboard.html` | tech-debt | open | — | Föreslå: dashboard hämtar namn från `data.config.class_names` om trainern börjar exportera dem (kopplat till IM-001). Server-side dup borttagen sedan IM-005 |
| IM-009 | Otrackade k8s/-YAMLs (audit-rededge, probe-h100, train-prithvi-{300m,600m}, smoke-dataset, inspect-checkpoints, training-dashboard-server, inference-compare-raw-and-superpixel) | housekeeping | open | — | Alla körda + verifierade under sessionen 2026-05-20/21 |
| IM-010 | `trainer.py:1000-1009` sparar minimal config till checkpoint — saknar `backbone_name` | tech-debt | open | — | Tvingade inferens-scriptet att infera backbone från `pos_embed` embed_dim (1024=300M, 1280=600M). Fix: lägg till `backbone_name` + `backbone` + `patch_size` i checkpoint config-dict |
| IM-011 | `data/viz_tiles/` saknades i k8s git-clone (gitignore) — krävde `git add -f` på 18 MB fixtures | tech-debt | **partial-done** | commit `00b7cc7` på `fix/inference-comparison-patch-size` | Långsiktig fix: PVC-baserad fixture-distribution istället för repo-bloat |
| IM-012 | `/compare/` HTML: NIR-CIR / GT / NMD-kolumner visas som broken images | bug | open | — | viz_data5.json base64-strings rendererar inte korrekt. Modell-prediktioner OK |
| IM-013 | SLIC superpixel post-processing visuellt VÄRRE än raw | investigation | open | — | Förväntat: spektral-edge-snappning. Faktiskt: blockigare/grövre än raw. Möjliga orsaker: fel `--superpixel-segments`, fel input-format (6-band reflectance vs RGB), eller behöver `--cleanup` i kombination |
| IM-014 | Modell-meta i Compare-HTML: `mIoU ?` istället för faktiskt värde | bug | open | — | `_metric` field i prediction-JSON är `"?"` istället för 0.4716/0.4769. Fixa i `inference_comparison.py` eller `build_comparison_html` |
| IM-015 | Inference-fix-branch `fix/inference-comparison-patch-size` ej PR-öppnad | housekeeping | open | branch pushad, 3 commits (`c117ac3` + `00b7cc7` + `5e7860d`) | Öppna PR när compare verifierats end-to-end (görs nu) |
| IM-016 | Klassvikt-rebalansering: sänk över-viktade rare crops, höj lövskog | investigation | open | — | Direkt åtgärd för IM-007 findings. Föreslag: bete 2.10 → 1.5, potatis 3.02 → 2.0, majs 5.0 → 3.5, råg 4.11 → 3.5 för 600M. Switch till effective-number weighting |

---

## Done

*(Tom — flytta hit items när de landar. Behåll ID, status `done`,
länk till mergad PR / commit-SHA.)*

---

## Källor

Items i tabellen ovan kommer från:
- Live session-arbete 2026-05-20 / 2026-05-21 (initial seed)
- `docs/conversation_log/*.md` — ärver "## Next Steps"-sektioner när någon
  hittar tid att skörda dem hit
- Savant-review / `/review` output — säkerhetsbuggar och tech-debt-fynd
- PR-discussions där en sidoupptäckt inte ryms i nuvarande scope
