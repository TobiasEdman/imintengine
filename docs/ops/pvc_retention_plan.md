# PVC-retention — städplan för training-data-cephfs (REVIEW, ej körd)

**Datum:** 2026-09-03. **Frigör totalt ~333 G.** **Status:** godkänd av
Tobias i princip; körs EFTER #36-rollout. INGEN mutation gjord när detta
skrevs.

## Varför (kapacitetsbilden)

`training-data-cephfs` är 1.6 T och stod på **100 % full** 2026-09-03
(en eval-pred_cache på 83 G blev sista droppen; den raderades → 95 %).
Volymen är inte läckande — den är **dimensionerad precis över sin egen
arbetsmängd**:

| katalog | storlek | typ |
|---|---|---|
| `unified_v2_512` | 810 G | kanoniskt dataset (load-bearing) |
| `checkpoints` | 328 G | modeller (varav ~183 G mellanepoker) |
| `holdout_val_512` | 146 G | holdout-tiles (load-bearing) |
| `vpp_wekeo`+`nvv_aux` | 104 G | fetch-cachar (kalla) |
| `embeddings`+`tessera_cache` | 35 G | tessera-enrichmentcachar (kalla) |
| `model_cache` | 22 G | HF-vikter (AKTIV) |

Dataset + holdout (956 G) kan inte röras. Marginalen kommer från
checkpoint-retention + kalla bygg-tids-cachar. Alternativet är att utöka
volymen — separat beslut, se "Strukturellt" nedan.

## Vad som raderas
`/cephfs/checkpoints/ladder/*/epoch_*.pt` — de mellanliggande epok-
snapshotsen (epoch_005…epoch_030) i varje ladder-cell.

## Vad som BEVARAS (aldrig rört)
- `best_model.pt` — resultatet varje cell använder (distill, eval, tabell).
- `last_checkpoint.pt` — resume-punkt.
- `training_log.json` — hela träningshistoriken (mIoU-kurvor, config).
- ALLA icke-ladder-checkpoints: v8/v8b/prithvi_300m_512/produktions-dirs
  under /cephfs/checkpoints/ (utanför /ladder/) — ORÖRDA.

## Per-cell (störst återvinning först)
| grupp | celler | drop/cell | delsumma |
|---|---|---|---|
| prithvi600m | r1–r4 | ~15.3 G | ~61 G |
| clay | r1–r4 | ~14.5 G | ~58 G |
| prithvi300m | r1–r4 | ~7.2 G | ~29 G |
| croma | r1,r2 | ~4.7 G | ~9 G |
| terramind | r1,r2 | ~2.2 G | ~4 G |
| tessera | r1–r4 | ~0 G | ~0 G (head-only, försumbart) |
| _nomarkfukt-varianter | (historiska) | — | ~21 G (se nedan) |

Bevarat totalt (best+last): 73.7 G. **Återvinningsbart (epoch_*): 182.7 G.**

## Säkerhet / motivering
- Alla 24 ladder-celler är KLARA (30/30 epoker) → best_model.pt ÄR
  resultatet; mellanepokerna behövs bara för att inspektera en avbruten
  körning, vilket ingen är. Ingen resumability offras (last_checkpoint.pt
  behålls som extra försäkring).
- Rör inte best/last/log → distill-OOF, eval och tabellen är oförändrade.

## _nomarkfukt-dirs: HELA borttagningar [user-bekräftat 2026-09-03: behövs ej]
`*_r1_nomarkfukt`-dirs (croma/prithvi300m/prithvi600m/tessera) är
SUPERSEDERADE pre-ladder-ablationer (markfukt av). Hela dirs raderas —
best+last+epoch. Deras epoch_* ingår redan i 182.7 G ovan; det EXTRA från
att även ta best/last är ~11 G. Full dir-storlek ~32 G.

## Fetch-cachar: återvinningsbara [per user no-tiles-bekräftelse 2026-09-03]
Inga nya tiles byggs → bygg-/fetch-tids-cachar är kalla (träning/eval/
distill läser aux FRÅN tiles, inte härifrån):
- `/cephfs/vpp_wekeo` (68 G) + `/cephfs/nvv_aux` (36 G) = **+104 G**.
- Förbehåll: en framtida S1/VPP-enrichment bygger om vpp_wekeo till PU-
  kostnad — men #36-containment tar bort behovet, dataset fryst.
- `/cephfs/embeddings` (19 G) + `/cephfs/tessera_cache` (16 G) = **+35 G**
  [verifierat 2026-09-03: redundanta för runtime]. embeddings = Tesseras
  globala 0.1°-representation (källa); tessera_cache = per-tile-cache;
  ENDAST enrich_tiles_tessera.py läser dem — träning/eval/distill läser
  bakade `tessera`-nyckeln från tiles. Kalla med dataset fryst. Förbehåll:
  re-enrichment bygger om från extern Tessera-produkt (embeddings = källan,
  något mer värd än den deriverade tessera_cache).
- EJ i denna städning: `model_cache` (22 G) — MODELLVIKTER (HF), inte
  tessera/tile-data; alla tränings-/eval-/distill-jobb laddar den. BEHÅLL.

## KONSOLIDERAD SÄKER ÅTERVINNING (post-#36, en godkänd körning)
| post | frigör |
|---|---|
| ladder epoch_*.pt (alla celler, behåll best/last/log) | 182.7 G |
| _nomarkfukt-dirs helt (extra best/last utöver epoch) | +11 G |
| vpp_wekeo + nvv_aux | +104 G |
| embeddings + tessera_cache | +35 G |
| **TOTALT** | **~333 G** |
83 G ledigt → ~416 G (26%). ENDAST model_cache (22 G) + icke-ladder
(v8/v8b/produktion) orörda.

## Körning (efter godkännande + #36-rollout)
Reviewbart jobb, dry-run först:
1. DRY: `find /cephfs/checkpoints/ladder/*/epoch_*.pt` → lista + summa,
   assert INGEN best_model.pt/last_checkpoint.pt/training_log.json i listan.
2. df före.
3. `rm` exakt den listan (inget glob utanför epoch_*.pt).
4. df efter + assert best/last/log intakta i varje cell.
5. Evidens (fillista + before/after df) arkiveras.

## Sekvens-constraint
Kör INTE under #36:s apply-fönster (PVC-kontention). Kör efter rollout,
före prithvi300m4f-omträning + eval-omkörning (som behöver marginalen).

## Hur redundansen verifierades (2026-09-03)
- **epoch_\*.pt:** alla 24 ladder-celler är klara (30/30 epoker,
  `training_log.json: status=completed`) → `best_model.pt` ÄR resultatet;
  mellanepoker behövs bara för att inspektera en AVBRUTEN körning.
  `last_checkpoint.pt` behålls ändå som resume-försäkring.
- **vpp_wekeo / nvv_aux:** fetch-tids-cachar. Träning/eval/distill läser
  aux FRÅN tiles (npz-nycklar), inte från cacharna. Kalla när inga nya
  tiles byggs [user-bekräftat: inga tile-planer].
- **embeddings / tessera_cache:** `grep` över `scripts/`, `imint/`, `k8s/`
  ger NOLL runtime-referenser till `/cephfs/embeddings`; `tessera_cache`
  refereras ENDAST av `scripts/enrich_tiles_tessera.py`
  (`--embeddings-dir /data/tessera_cache/embeddings`) — dvs
  tile-enrichment, inte inferens. Tiles bär den bakade `tessera`-nyckeln.
- **model_cache:** HF-modellvikter, laddas av varje tränings-/eval-/
  distill-jobb → BEHÅLL (ingår inte i städningen).

## Strukturellt (separat beslut, ej denna plan)
Även efter ~333 G står dataset+holdout för 956 G av 1.6 T. Nya
checkpoints (prithvi300m4f-omträning = 4 celler) och pred-cachar äter
marginalen igen. Om PVC:n ska bära fler experiment-vågor är **utökning av
volymen** den strukturella lösningen; städning är den återkommande.

## Framtida kostnad om cacharna raderas
- `vpp_wekeo`: en framtida S1/VPP-enrichment bygger om den — PU-fri via
  WEkEO HDA men tidskrävande. #36-containment tar bort behovet för
  crop-spåret; dataset fryst ⇒ risken teoretisk.
- `embeddings` (Tesseras globala 0.1°-produkt) är KÄLLAN som
  `tessera_cache` deriveras ur — något mer värd att behålla om man vill
  kunna re-enricha utan extern nedladdning. Bägge är dock reproducerbara.
