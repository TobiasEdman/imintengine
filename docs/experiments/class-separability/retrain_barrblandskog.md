# Omträning med barrblandskog som egen klass — TESSERA först

IM-023, 2026-10-30. **Beslut (Tobias):** barrblandskog (NMD 113) blir egen
unified-klass 28 och modellen tränas om, TESSERA-familjen först. Underlag:
`forest5`-provet i [README.md](README.md) — barrblandskog skiljs ut med 58 %
recall (slump 20 %) och förväxlas med lövskog i 0,1 % av fallen; R4:s frysta
representation och råa TESSERA är lika bra (0,77), så det som saknades var
målklassen.

Den här planen ändrar ingen förseglad sanning, ingen befintlig checkpoint och
ingen etikettkatalog som R4 bygger på. Varje klustersteg kräver Tobias
godkännande och `kubectl create --dry-run=server` först.

## Vad som redan är gjort (gren `agent/te/claude/im023-barrblandskog`)

- Schemat har 29 klasser: 28 = barrblandskog (NMD-färg `#4e6f00`), NMD2023
  113 → 28, 123 (på våtmark) stannar i sumpskog. ID 0–27 är oförändrade, så
  R4 och äldre etiketter betyder samma sak som förut.
- `FOREST_CLASSES` (1–6, 28) är den enda skogsmängden; SKS-hygge och
  aux-QC läser den. Destillationshuvudet saknar barrblandskog, så NMD 113
  behåller klass 28 i stället för att skrivas om till tall eller gran.
- Den serverade R4 är låst till 28 klasser (kontrakt, klassräkning, legend,
  färgtabell) och `compute_miou` behåller namnen för 28-klassmodeller.

## Förutsättningar före träning

1. **Fraktionsförlusten måste rättas först.** R4 tränades med
   `--enable-tradslag-head --lambda-frac 1.0`, och `FractionLoss` på main tolkar
   datasetets positiva mask omvänt (1 = övervaka lästes som 1 = uteslut).
   Rättelsen finns på `agent/te/claude/study-fm-experiments` (`fec5a7d9`,
   `imint/training/losses.py` + `tests/test_fraction_loss_mask.py`, 15/15 mot
   9 fel utan). Den går rent på main och ska in i träningsgrenen.
2. **En träningsgren** med schemaändringen, fraktionsrättelsen och
   etikettomvandlingen nedan. Träningsjobbet klonar en namngiven gren; den
   ska pinnas till en commit, inte en rörlig gren.

## Steg

### 1. Etiketter (CPU, minuter–en timme)

Dagens kedja är `build-labels-nmd2023` → `/data/nmd2023_labels` →
`ladder/distill-tessera-job.yaml` (rung-2-checkpoint + `tessera_r2_head.npz`)
→ `/cephfs/distill/tessera_r2`, som R4 tränades på.

Med schemaändringen blir de nya etiketterna exakt:

- NMD-etikett: `where(nmd_label_raw == 113 & label == 3, 28, label)`. LPIS
  skriver bara över jordbrukskoder och hygge appliceras efter basen, så inga
  andra pixlar ändras.
- Destillerad etikett: `where(ny_nmd_etikett == 28, 28, tessera_r2_etikett)`.
  Destillationen skriver bara över där både original och huvud är 1–4 och
  huvudet beror inte på etiketten, så alla andra pixlar är identiska.

Plan: ett litet skript som skriver till **nya** kataloger
(`/data/nmd2023_labels_b28`, `/cephfs/distill/tessera_r2_b28`), med
sidecar-metadata om källa och regel. **Verifiering före full körning:** på ett
urval tiles (minst tio, deterministiskt valda) jämförs omvandlingen bit för bit
mot den kanoniska vägen — `build_labels.py --nmd-version 2023` plus
`distill_forest_labels.py` med samma checkpoint och huvud. Avviker en enda
pixel används den kanoniska vägen för allt (≈ 6 h CPU + ≈ 1 dygn GPU).

### 2. TESSERA-träning (1× H100, ≤ 12 h)

Samma recept som `k8s/ladder/ladder-r4-tessera-job.yaml` (30 epoker, batch 8,
lr 1e-4, focal_dice, patch 504, gated aux, fraktionshuvud) med två ändringar:
`--num-classes 29` och `--label-dir /cephfs/distill/tessera_r2_b28`, plus den
rättade fraktionsförlusten i koden. Ny checkpointkatalog; R4 rörs inte.

Två saker ändras samtidigt (klass 28 och fraktionsrättelsen). **Beslut
(Tobias 2026-10-30):** en kontrollkörning görs också — R4-receptet med 28
klasser, dagens `tessera_r2`-etiketter och bara fraktionsrättelsen, ≤ 12 h
H100, parallellt med huvudkörningen — så att effekterna går att skilja åt.

Först en 1-epoks rökkörning (repots regel för träningspipelinen), sedan full
körning.

### 3. Utvärdering (förregistreras innan resultat ses)

1. **NMD-överensstämmelse på holdout-tiles** (`ladder_holdout_eval.py`):
   mIoU och per-klass-IoU för 1–4 och 28 mot R4; övriga klasser får inte
   försämras utanför bootstrap-intervallet.
2. **forest5-provet** med den nya checkpointen: andelen NMD 113 som modellen
   ger klass 28, och att lövskog fortsatt är ≈ 0 %.
3. **NFI-holdout (förseglad, sanning v1):** NFI-regeln saknar barrblandskog.
   **Beslut (Tobias 2026-10-30):** primärt slås 28 ihop till tall eller gran
   efter modellens fraktionshuvud (tall- mot granandel), samma dominansregel
   som `collapse_fractions_to_nfi_class`, så att jämförelsen med R4 är
   rättvis; sekundärt redovisas 28 som egen kolumn i förväxlingsmatrisen.
   Sanningen ändras inte. Kontrollkörningen och R4 poängsätts med samma regel.
4. **LUCAS** på samma sätt, med 28 räknad som barrskog.

### 4. Demon (efter godkänd utvärdering)

Tjänsten är låst till R4:s 28 klasser. En efterträdare behöver ett nytt
serveringskontrakt (29 klasser), legendpost för 28 med NMD 113-färgen och
NMD-noten, samt nya kvitton (paritet, pixelinspektion). Eget steg och egen
driftsättning.

## Sumpskog — mäts parallellt, tränas inte samtidigt

`sumpskog5` (121–127) är förregistrerad i README.md och körs med
`bash scripts/run_forest_separability.sh sumpskog5 2022`. Visar den att
trädslagen går att skilja på våtmark prövas en uppdelning i en **egen**
omträning efter denna, så att effekten av varje schemaändring går att avgöra.
Ett alternativ att väga in då: trädslag för all skog plus våtmark som separat
attribut (markfukt finns redan som aux), i stället för fem nya klasser.

## Beslut och vad som återstår

Beslutat 2026-10-30: egen klass 28, TESSERA först, kontrollkörning, primär
NFI-regel via fraktionshuvudet, schemagrenen pushas separat från demogrenen.

Återstår att godkänna per klustersteg: etikettomvandlingen (CPU),
rökkörningarna, de två fulla träningarna och utvärderingsjobben.
