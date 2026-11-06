# Omträning med barrblandskog som egen klass — TESSERA först

IM-023, 2026-10-30. **Beslut (Tobias):** barrblandskog (NMD 113) blir egen
unified-klass 28 och modellen tränas om, TESSERA-familjen först. Underlaget
är `forest5`-provet, förregistrerat i [README.md](README.md). Resultatet finns
i den privata evidenskatalogen `ImintEngine/class-separability-0127/forest5-2022/`
och hålls privat till publiceringen.

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
- `build_labels.py` tillåter klass 28 för NMD2023. Skrivkontrollen stoppade
  tidigare vid 27, så en kanonisk ombyggnad hade fallerat på varje tile med
  barrblandskog.
- Fraktionsförlustens rättelse (`fec5a7d9` på FM-grenen) är inplockad här som `dde7b037`.
- Etikettomvandlingen, receptet och jobbfilerna nedan finns och är testade
  lokalt, inklusive en CPU-rökning av hela kedjan. Inget har körts i klustret.

## Förutsättningar före träning

Båda är uppfyllda på grenen: fraktionsrättelsen (R4 tränades med
`--enable-tradslag-head --lambda-frac 1.0` och en `FractionLoss` som läste
datasetets positiva mask omvänt) och en pinnad källa. Jobben hämtar exakt
den commit som står som `SHA=` i `k8s/im023/` och kontrollerar den. Imagen är pinnad med digest. Ett test
larmar om omvandlings- eller receptskriptet ändras efter den commit som
jobben pekar på.

## Steg

### 1. Etiketter (1× 2080 Ti, ≤ 6 h)

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

Skript: `scripts/relabel_barrblandskog.py` (`sample`, `verify`, `run`).
Jobb: `k8s/im023/relabel-b28-job.yaml` (2080 Ti, samma GPU-typ som R4:s
destillation), i fyra steg:

1. **Urval.** 12 destillerade tiles med NMD 113 och 3 utan, valda i
   sha256-ordning på namnet, alltså deterministiskt.
2. **Kanonisk kedja på urvalet.** `build_labels.py` med samma argument som
   `build-labels-nmd2023` skapade etiketterna med. Därefter
   `distill_forest_labels.py` med samma argument, checkpoint och huvud som
   R4:s destillation, två gånger: från de lagrade NMD-etiketterna och från
   de ombyggda.
3. **Verifiering, bit för bit.** Två grindar, där minsta skillnad i värde,
   datatyp eller form stoppar jobbet:
   - ombyggd NMD-etikett = omvandlad lagrad NMD-etikett;
   - nydestillerat från ombyggd = omvandlat från nydestillerat från lagrad.

   Den andra grinden jämför två körningar i samma jobb. Den prövar alltså
   genvägen, inte GPU:ns reproducerbarhet. Om en ny destillation återskapar
   de lagrade `tessera_r2`-filerna redovisas separat i `report.json`, liksom
   hur många pixlar en omkörning av huvudet ändrar.

   Det är ett medvetet val att den raden inte stoppar jobbet. Genvägen
   utgår från R4:s egna etiketter, så huvud- och kontrollkörningen har
   identiska etiketter utanför klass 28. Det är just det kontrollkörningen
   behöver för att effekterna ska gå att skilja åt. Fallerar en grind
   används den kanoniska vägen för allt (≈ 6 h CPU + ≈ 1 dygn GPU).
4. **Omvandling.** Resultatet skrivs till nya kataloger:
   `/cephfs/nmd2023_labels_b28` (alla NMD-etiketter) och
   `/cephfs/distill/tessera_r2_b28` (R4:s kohort).
   - Skrivningen är atomisk, och varje fil stämplas med regel och
     källsumma.
   - En avbruten körning återupptas: aktuella filer hoppas över.
   - `relabel_manifest.json` skrivs sist och är klarsignalen.
   - Jobbet stoppar om antalet destillerade filer skiljer sig från R4:s.

Underlaget sparas i `/cephfs/im023/relabel_verify/<UTC-tid>/`.

### 2. TESSERA-träning (1× H100, ≤ 12 h per körning)

Receptet finns på ett ställe, `scripts/im023_retrain.py`: R4:s flaggor
(30 epoker, batch 8, lr 1e-4, focal_dice, patch 504, gated aux,
fraktionshuvud) plus en variantväxel. Ett test håller kontrollvarianten
flagga för flagga lik `k8s/ladder/ladder-r4-tessera-job.yaml` och visar att
huvudvarianten bara skiljer sig i `--label-dir` och `--num-classes`.

| Variant | Klasser | Etiketter | Checkpoint |
|---|---|---|---|
| huvud (`main`) | 29 | `/cephfs/distill/tessera_r2_b28` | `/cephfs/checkpoints/im023/tessera_b28` |
| kontroll (`control`) | 28 | `/cephfs/distill/tessera_r2` (R4:s) | `/cephfs/checkpoints/im023/tessera_control` |

Båda körs med den rättade fraktionsförlusten. Huvud minus kontroll ger alltså
effekten av klass 28, och kontroll minus R4 effekten av rättelsen
(**beslut Tobias 2026-10-30**). R4 rörs inte.

Före träning vägrar skriptet att starta om:
- checkpointkatalogen redan har en modell;
- R4:s `_GATE_OK` saknas;
- omvandlingsmanifestet saknas eller kommer från en annan regel;
- kohorten skiljer sig från R4:s;
- en stickprovsetikett ligger på eller över antalet klasser;
- huvudvarianten saknar `class_stats.json` med `pixel_counts` att härleda
  klass 28:s vikt ur.

Ordning:

1. `k8s/im023/train-tessera-smoke-job.yaml`: en epok av varje variant i
   följd. Detta är repots regel om rökkörning för träningspipelinen. Den
   lokala CPU-rökningen (`tests/test_im023_retrain.py`) kör samma kedja på
   syntetiska tiles.

   Den enda H100-noden har varit avstängd för schemaläggning sedan 2026-10-29,
   så `k8s/im023/train-tessera-smoke-2080ti-job.yaml` kör samma kedja på ett
   2080 Ti (Tobias 2026-11-03). Den följer inte R4:s recept: FP32, eftersom
   Turing saknar BF16 i hårdvara, batch 2, eftersom FP32-aktiveringarna för
   batch 8 inte ryms i 11 GB, och 6 workers. Den skriver till
   `smoke2080_{main,control}`, så den räknas inte som rökkörningen. Grinden
   väntar på H100-körningen.
2. `k8s/im023/train-tessera-b28-job.yaml` och
   `k8s/im023/train-tessera-control-job.yaml` parallellt. Båda vägrar starta
   innan rökkörningens två modeller finns.

**Klassvikter.** Tränaren hämtar klassvikterna från `class_stats.json` i
datakatalogen, inte från etikettkatalogen. Så var det också för R4. Filen
känner bara 23 klasser, så klass 28 räknades med antalet 1 och fick taket
`--max-class-weight` (5,0).

**Beslut (Tobias 2026-11-03): bara klass 28 rättas nu.** Huvudvarianten
tränar på en kopia, `class_stats_b28.json` i checkpointkatalogen. Där tar
klass 28 den andel av klass 3 som omvandlingen flyttade, och totalen är
oförändrad. Andelen räknas i samma domän som `class_stats.json`, alltså i de
kanoniska NMD-etiketterna som manifestet anger
(`/cephfs/nmd2023_labels_b28`). Där är klass 3 plus klass 28 exakt den gamla
klass 3, och hygge förblir undantaget. De destillerade etiketterna duger inte,
eftersom destillationen skriver om klass 1–4. Klass 28:s pixlar måste stämma
med manifestets `class_28_pixels`, annars avbryts körningen. Klass 28 får
alltså den vikt receptet ger dess frekvens. Klass 3 får sin egen vikt utan
barrblandskogen, och övriga vikter är desamma som R4:s. Kontrollkörningen
läser filen som R4 gjorde.

Klasserna 23–27 saknas också i filen och tränades med taket i R4. Det rättas
i en egen omträning, en ändring i taget.

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
5. **Blandklasserna: egen klass eller härledd ur andelar.** Förregistrerad
   2026-11-06 på Tobias begäran, under fullkörningarna och innan något
   utvärderingsresultat finns. Frågan gäller både barrblandskog (28) och
   blandskog (4). Blandning definieras på beståndsnivå, medan en 10 m-pixel
   bara rymmer några kronor. Är klasserna meningsfulla på pixelnivå, eller
   fångas blandningen bättre av fraktionshuvudets trädslagsandelar?
   - **Sanning:** NFI-provytornas volym per trädslag, samma fält som sanning
     v1 (`derive_nfi_forest_class`). Sanning v1 ändras inte.
     - Blandskog: sanning v1:s klass 4, där varken barr eller löv når 0,7 av
       volymen.
     - Barrblandskog: härleds ur samma fält. Barr når 0,7 av volymen, men
       varken tall (tall + contorta) eller gran når 0,7 av barrvolymen.
     - Ytor utan volym utesluts, som i v1.
   - **Prediktorer**, på samma pixelurval som den primära NFI-poängsättningen:
     - a) main:s hårda huvud (klass 4 respektive 28);
     - b) control:s hårda huvud (klass 4; 28 finns inte);
     - c) andelsregeln på control:s fraktionshuvud;
     - d) andelsregeln på main:s fraktionshuvud.

     Andelsregeln är `collapse_fractions_to_nfi_class` för blandskog. För
     barrblandskog tillämpas samma regel ett steg ner: barr når 0,7, men
     varken tall- eller granandelen når 0,7 av barrandelen. R4 poängsätts
     som b och c.
   - **Mått:** recall, precision och F1 för blandskog, barrblandskog, tall och
     gran, med blockbootstrap som i den primära poängsättningen.
   - **Läsning, bestämd i förväg:** en blandklass bedöms onödig som egen
     pixelklass om två villkor gäller samtidigt:
     1. andelsregeln (c eller d) ger minst lika hög F1 för klassen som det
        hårda huvudet (a), inom bootstrapintervallet;
     2. det hårda huvudet sänker tall eller gran utanför intervallet,
        jämfört med b.

     Då rekommenderas härledning ur andelar i stället. Annars behålls klassen.
     Inga trösklar läggs till eller ändras efter att resultat har setts.
   - Poängsättning mot den förseglade holdouten körs bara på Tobias
     klartecken, som tidigare.

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

Beslutat 2026-10-30:
- barrblandskog blir egen klass 28;
- TESSERA tränas först;
- en kontrollkörning görs;
- den primära NFI-regeln går via fraktionshuvudet;
- schemagrenen pushas separat från demogrenen.

Körordning i klustret. Varje steg kräver Tobias godkännande och
`kubectl create --dry-run=server -f <fil>` först:

1. `k8s/im023/relabel-b28-job.yaml`
2. `k8s/im023/train-tessera-smoke-job.yaml`
3. `k8s/im023/train-tessera-b28-job.yaml` och
   `k8s/im023/train-tessera-control-job.yaml`
4. Utvärderingsjobben enligt steg 3. De är inte skrivna ännu.
   `ladder_holdout_eval.py` är låst till 28 klasser och behöver en
   29-klassväg.
