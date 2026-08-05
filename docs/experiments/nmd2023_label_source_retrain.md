# Experiment — träna om v8b med NMD2023 som labelkälla

**Status:** plan (ej påbörjad) · **Skapad:** 2026-08-05 · **Ägare:** Tobias
**Branch för koden:** `agent/te/opus/nfi-nmd2023-benchmark` (NFI-jämförelsen som motiverar detta)

## Syfte

Reproducera "studenten slår läraren"-effekten mot den *nuvarande* NMD-generationen.
v8b tränades på NMD2018-etiketter och slår ändå NMD2018 vid fältpunkterna
(multitemporal S2 + LiDAR-aux avbrusar per-pixel-felen). Frågan: om vi byter
labelkälla till NMD2023 — kan en NMD2023-tränad v8b slå även NMD2023?

## Hypotes

Samma denoising-mekanism överför. En modell tränad på NMD2023 (en bättre lärare:
laser + 2023 S2) bör landa på en **högre absolut** fält-accuracy än dagens v8b,
och sannolikt över NMD2023 självt vid fältpunkterna.

**Framgångskriterium:** NMD2023-tränad v8b overall accuracy (5-klass) mot NFI
på de gemensamma 912 plottarna **> 49,3 %** (NMD2023), helst med marginal över
dagens v8b (46,3 %).

## Utgångsläge (samma-ytor, 912 NFI-plottar)

| Källa | Overall | Kappa | tall F1 | gran F1 | löv F1 | bland F1 |
|---|---|---|---|---|---|---|
| NMD2023 v2.1 | 0,493 | 0,366 | 0,61 | 0,60 | 0,41 | 0,31 |
| v8b (nu) | 0,463 | 0,335 | 0,57 | 0,55 | 0,45 | 0,34 |
| NMD2018 v1.1 | 0,406 | 0,257 | 0,54 | 0,47 | 0,34 | 0,30 |

Källa: `docs/data/compare-nmd2023-nfi-sameytor.json`.

## Scope-beslut (fastställda)

- **Ingen ny spektralhämtning.** Träningssetet är *inte* 2018-spektral-only — det
  spänner över år (crop-tiles är årsmatchade mot sin LPIS-årgång). Vi återanvänder
  befintlig spektral som den är. Ingen refetch, ingen PU/DES-kostnad.
- **Bara NMD-basen byts.** `build_labels.py` tar `--nmd-raster` som en ren
  bas-swap; LPIS-gröd-overlay och SKS-hygge-overlay är årsspecifika och lämnas
  oförändrade (redan årsmatchade per tile). NMD2023 rör bara skog/våtmark/öppen
  mark/vatten-basen.

## Metod

### Steg 1 — full NMD2023→unified-mappning (den egentliga kodinsatsen)

NFI-benchmarken kunde återanvända NMD2018:s collapse-kedja **för att den bara
behövde de fyra skogstyperna + icke-skog-kollaps** (skogskoder 111–128 är
identiska mellan NMD2018 och NMD2023). Träningsetiketter behöver hela
nomenklaturen korrekt — och där skiljer sig NMD2023:

- **Skog 111–128** — identiska koder → återanvänd befintlig mappning.
- **Låg fjällskog (23, 43, 230)** — NY skogsklass i basen (tidigare tilläggsskikt).
  Måste mappas (lövskog/sumpskog beroende på fast-/våtmark), annars faller
  fjäll-tiles till bakgrund. Nordligt → påverkar norra tiles.
- **Öppen våtmark (200, 211–218, 221–230)** — finindelade myrklasser. NMD2018 hade
  bara kod 2. Alla → `open_wetland` (unified 7).
- **Öppen fastmark (411–413, 4211–4233)** — 4-siffriga koder. NMD2018 hade 41/42.
  Överskrider uint8 → kräver uint16-LUT (dagens `clip(0,255)` klipper dem till
  255 → bakgrund; **måste ersättas** för dessa koder).
  Beslut 2026-08-05: bryt ut på **struktur-nivå** som egna unified-klasser
  (busk/ris/gräs är spektralt/strukturellt separerbara — SLU 33VWC nivå 3:
  user's 73–98 %, prod. 74–96 %), men **kollapsa fuktnivån** (torr/frisk/
  frisk-fuktig härleds ur markfuktighetsindex, inte S2-reflektans — inte
  spektralt lärbart, prod. ned till 15 %):
    - `421, 4211, 4212, 4213` → **buskdominerad, ny klass 24**
    - `422, 4221, 4222, 4223` → **risdominerad, ny klass 25**
    - `423, 4231, 4232, 4233` → **gräsdominerad, ny klass 26**
    - `411` öppen mark utan vegetation → **ny klass 27** (bar mark/berg är
      spektralt mycket distinkt: ljus, låg NDVI, hög SWIR. Pipelinen skilde redan
      `open_land_bare` (19-klass 13) från `open_land_vegetated` (14) — unified
      kollapsade dem till 8; klass 27 återexponerar den. SLU 33VWC: prod. 96 %,
      user's 65 % — NMD överskattar bar mark, men spektralt separerbart.)
    - `412` glaciär + `413` snöfält → **ny sammanslagen klass 28 "snö/is"**.
      Snö/is är spektralt mycket distinkt (extremt ljus, hög synlig reflektans),
      men glaciär vs snöfält är inte spektralt separerbart från varandra → slås
      ihop till en klass. Sällsynt/alpin → låg support, class-weighting behövs.
    - Efter utbrytningen är generisk **öppen mark (8) nästan tom på NMD2023-
      fastmark** → residual-/legacy-hink (påverkar class-weighting).
- **Torvtäkt (54)** — **NY egen unified-klass 23**.
  Beslut 2026-08-05: torvtäkt är morfologiskt/spektralt distinkt och NMD2023
  klassar den nära perfekt (SLU 33VWC: user's 100 % / producer's 100 %, n=30),
  så den bryts ut i stället för att kollapsas.
- **Vatten 61/62, bebyggelse 51–53** — samma som NMD2018.

Konkret: skriv en NMD2023-specifik LUT (uint16, inte uint8-clip) i
`class_schema.py` / `unified_schema.py`, parametriserad så `build_labels` väljer
2018- eller 2023-mappning. Regressionstesta mot `tests/test_schema.py`.
Grid-koder + namn finns i `data/nmd2023/sidecar/NMD2023bas_v2_1.qml`.

### Steg 2 — bygg om etiketter

```
python scripts/build_labels.py \
  --nmd-raster data/nmd2023/NMD2023bas_v2_1.tif \
  --lpis-dir data/lpis --sks-dir data/sks --workers N
```

Kör `merge_all()` per tile: NMD2023-bas → LPIS-overlay (oförändrad) → SKS-hygge
(oförändrad). Skriv till en **separat** staging-katalog (t.ex. `unified_v2_nmd23`)
— skriv aldrig över NMD2018-labels; vi vill kunna A/B:a.

**Täckningsbeslut — ren NMD2023 (beslut 2, 2026-08-05):** NMD2023 v2.1 täcker
~94,5 % av landet (nordlig inland/fjäll saknas). Ren run: **ingen NMD2018-
fallback.** Pixlar där NMD2023-rastret är 0 (utanför utbredning) → bakgrund/
`ignore_index`; tiles som blir ~helt bakgrund fångas av nodata-filtret i QC och
faller bort. Konsekvent NMD2023-labelkälla överallt → ingen blandad-källa-
konflikt. **Kostnad:** ~5,5 % nordliga tiles/pixlar tränas inte → geografiskt
skevare set, svagare nordlig skogstyp. Accepterat för experimentet (renare
attribution + de 912 NFI-valideringsplottarna ligger mest i täckta områden).
Hybrid sparas för en ev. produktionsmodell.

### Steg 3 — QC på nya etiketterna

Samma QC som ordinarie pipeline: klass-histogram per tile, andel bakgrund,
frame-check. **Jämför klassfördelning NMD2018- vs NMD2023-labels** — stora skift
avslöjar mappningsbuggar (särskilt öppen mark / våtmark, se steg 1). Verifiera
mot en handfull tiles visuellt innan full retrain.

### Steg 4 — finetuna från v8b (beslut Q1)

Samma arkitektur, aux-kanaler och splits som v8b (`scripts/train_unified.py`,
`--enable-multitemporal`), varmstart från v8b-checkpointen.

**Head-expansion (pga sex nya klasser):** v8b:s klassificeringshuvud är
23-brett. Utöka till **29** — de 23 befintliga raderna ärvs från v8b, de sex nya
(23 torvtäkt, 24 busk, 25 ris, 26 gräs, 27 bar öppen mark, 28 snö/is) kallstartar.
Med finetune räcker några epoker för de nya raderna; backbone + 23 gamla rader är
varmstartade, så konvergensen är snabb.

### Steg 5 — validera med samma NFI-harness

```
# på ICE, samma jobbmönster som nfi-validate-v8b-perplot
python scripts/validate_against_nfi.py \
  --checkpoint <v8b-nmd23>.pt \
  --plot-index /data/nfi/nfi_index_unified_v2_512.parquet \
  --dump-per-plot /data/nfi_eval/v8b_nmd23_per_plot.parquet ...
# lokalt: fäll in i den tresidiga jämförelsen
python scripts/compare_nmd2023_nfi.py --model-per-plot ... 
```

Poängsätt på **exakt samma 912 covered plottar** → fyrsidig jämförelse:
v8b-2023 / NMD2023 / v8b-2018 / NMD2018.

## Risker & caveats

- **Mappningskorrekthet (steg 1) är den kritiska risken.** Fel på öppen
  mark/våtmark-koderna förgiftar en stor andel icke-skog-etiketter tyst. Gate:
  klassfördelnings-diff + visuell tile-check innan retrain.
- **Temporal mismatch.** NMD2023 = 2023 S2 + laser 2019–2022; en del tiles har
  äldre spektral. Accepterat för skogstyps-basen (långsamt föränderlig).
  Hygge/crop-overlays är fortsatt årsmatchade, så den dynamiska signalen är opåverkad.
- **Inte label-läckage i valideringen.** Modellen tränas på NMD2023 och jämförs
  mot NMD2023 — men *valideringen* sker mot oberoende NFI-fältdata, så jämförelsen
  är rättvis. (Att modellen och NMD2023 kan korrelera mer med varandra är just det
  vi mäter mot fält.)
- **Täckningsgap i norr** kan göra nordlig skogstyp sämre om (b) väljs.

## Kostnad

- **Fetch:** noll (ingen ny spektral, ingen aux — allt återanvänds).
- **Label-rebuild:** CPU, ~20 min–några h beroende på workers (som ordinarie
  build-labels-jobb).
- **Träning:** en H100-körning (finetune kortare än full 20-epoks retrain).
- **Validering:** en 2080ti-inferens (~minuter), PU-fri.

## Schema-ändring — 23 → 29 klasser

Sex nya unified-klasser, alla bara kodade i NMD2023:
`23 torvtäkt · 24 buskdominerad · 25 risdominerad · 26 gräsdominerad ·
27 öppen mark utan vegetation (bar) · 28 snö/is (glaciär + snöfält)`.
Touch points: `unified_schema.py` (mappning + `UNIFIED_CLASSES`-namn + colormap),
`num_classes` 23→29 (loss/head/dataset), `tests/test_schema.py`, dashboard-legend.

### Caveats för de nya klasserna

- **Konsekvent supervision (ren NMD2023, beslut 2).** Alla nya klasser (23–28)
  kommer bara ur NMD2023. Med den rena run:en (ingen NMD2018-fallback) är
  labelkällan enhetlig överallt → **ingen blandad-källa-konflikt** för öppen
  mark. Pris: ~5,5 % nordliga tiles/pixlar utan NMD2023 tränas inte (→ bakgrund).
- **Låg support (särskilt torvtäkt + snö/is).** Torvtäkt och snö/is är spatialt
  små → få pixlar. Busk/ris/gräs är vanligare men fortfarande minoritet. Kräver
  class-weighting (`compute_class_weights`, sqrt/inverse) eller riktad sampling;
  annars lärs de rara klasserna inte trots att de är separerbara.
- **Fuktnivån bortkollapsad i första run:en.** torr/frisk/frisk-fuktig (nivå 4)
  härleds ur markfuktighet, inte S2 — inte lärbart ur enbart reflektans (SLU
  prod. ned till 15 %). OBS: SLU-markfukt-datan är **redan hämtad och lagrad** i
  varje tile (`tile_fetch.py` → `aux["markfukt"]`), men **ingår inte** i v8b:s
  10-kanals `AUX_CHANNEL_NAMES`. Att lägga till den = input-sides-ändring
  (10→11 aux, ny kanal kallstartar) men **ingen refetch**. Nästa-steg-experiment:
  markfukt-aux + full fuktdelning; hålls utanför första run:en för att inte
  stapla för många rörliga delar.
- **Orthogonalt mot NFI-benchmarken.** Klass 23–28 är icke-skog → kollapsar till
  icke-skog i skogstyps-jämförelsen. Påverkar inte 912-plotts-siffrorna;
  utvärderas separat (mot NMD2023 eller visuellt).

## Beslut (fastställda 2026-08-05)

1. **Träning:** finetune från v8b (ej full retrain), head 23 → 29.
2. **Ren NMD2023-run** (ingen NMD2018-fallback). Pixlar utan NMD2023 → bakgrund/
   `ignore_index`; enhetlig labelkälla, ingen blandad-källa-konflikt. Pris:
   ~5,5 % nordliga tiles tränas inte. Hybrid sparas för produktionsmodell.
3. **Nya klasser (schema 23 → 29):** torvtäkt (54) → 23; busk (421/42x) → 24;
   ris (422/42x) → 25; gräs (423/42x) → 26; bar öppen mark (411) → 27;
   snö/is (glaciär 412 + snöfält 413, sammanslagna) → 28. Fuktnivån
   (torr/frisk/frisk-fuktig) bortkollapsad i första run:en — ej spektral.
4. **Låg fjällskog:** fastmark (43) → lövskog (3); våtmark (23, 230) → sumpskog (5).

Nästa-steg-run (ej i denna): **markfukt-aux** (redan hämtad, ej i modellstacken)
+ full fuktdelning av öppen mark.

Kvarvarande öppet (avgörs vid label-QC): class-weighting-metod för de rara
klasserna (torvtäkt, snö/is).

## Referenser

- Jämförelsekod: `scripts/compare_nmd2023_nfi.py`, `scripts/validate_against_nfi.py`
  (`--dump-per-plot`), branch `agent/te/opus/nfi-nmd2023-benchmark`.
- Resultat: `docs/data/compare-nmd2023-nfi-{sameytor,national,modelplots}.json`.
- NMD2023 koder/legend: `data/nmd2023/sidecar/NMD2023bas_v2_1.qml`.
- SLU:s egen QA av NMD2023 v2.0 (33VWC, sydöstra Sverige) mot Riksskogstaxeringen:
  `data/nmd2023/Kvalitetsutvardering_33VWC_v2_0.pdf`.
