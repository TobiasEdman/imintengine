# Barrblandskog och lövskog: historik och separerbarhet

IM-023, 2026-10-29. Körs parallellt med kommun- och årsutbyggnaden i
IM-024. Demon använder fortsatt sin verifierade R4-modell; detta arbete
ändrar inga servervikter, målklasser eller träningsetiketter.

## Vad historiken faktiskt visar

| Datum / commit | Belägg | Slutsats och begränsning |
|---|---|---|
| 2026-02-22, `63d8c19` | Första NMD-analyzern anger `forest_deciduous: {113}` och `forest_temp_non_forest: {115,116,117,118}`. | Fel benämning/mappning finns redan när NMD-stödet introduceras. Detta är inte ett belagt beslut att slå ihop en korrekt identifierad barrblandskog med lövskog. |
| 2026-02-24, `9e1c5ce` | Träningsschemat bygger sin LUT från analyzerns `NMD_LEVEL2`. | Samma klassbetydelse förs in i träningskoden. Det bevisar kodvägen, inte vilka historiska checkpointar som körde den. |
| 2026-03-23, `083459a` | En korrigering flyttar 115–117 till lövgruppen men lämnar 113 kvar. Kommentaren kallar dessutom 114 barrblandskog. | Den korrekta reparationen av lövskogstyperna kombineras med en kvarvarande förväxling av 113/114. Därmed grupperas 113 och 115–117 uttryckligen tillsammans. |
| 2026-08-05, `df45803` | NMD2023-mappningen behåller 113→3 och skriver uttryckligen ”for consistency, not correctness”. | Här finns ett dokumenterat kompatibilitetsval att bevara den befintliga skogsmålrymden vid finjustering från v8b. Det förklarar vidareförandet, inte det ursprungliga felet. |
| 2026-08-10, `c1ebf94` | `docs/data/distill_finding.md` nämner uttryckligen den äldre 113→löv-mappningen och beskriver ändrade skogsetiketter genom fältstyrd destillation. | Felet var beskrivet före dagens demo. Destillationen gör att den ursprungliga NMD-LUT:en ensam inte beskriver R4:s faktiska skogsmål. Dokumentets historiska prestandapåståenden har inte återvaliderats i denna audit. |

Producenternas sparade QML-legender anger 113 som barrblandskog både för
NMD2018 och NMD2023, 114 som lövblandad barrskog och 115–117 som
lövskogstyper. SHA-256 för källfiler och exakta gitrevisioner finns i
[history-evidence.json](history-evidence.json).

Git och beslutsdokument ger belägg för ett inledande kodfel, en ofullständig
reparation och senare medveten kompatibilitet. De fastställer inte vilken
kommunikation som fördes eller vem som borde ha upptäckt felet. Författar-
och medförfattarfält på commits är inte belägg för användarens avsikt eller
kontroll. Något verifierat beslut att slå ihop klasserna på grund av empiriskt
visad dålig separerbarhet har inte hittats i de granskade källorna.

## R4:s etikettkedja kräver en separat kontroll

Den versionerade R4-manifesten pekar på `/data/distill/tessera_r2`, inte direkt
på NMD2023-etiketterna. Producenten `distill_forest_labels.py` ersätter
skogstyp inom den befintliga skogsmasken med ett litet huvud tränat på
modellegenskaper och fältreferenser. Den behåller råa NMD-koder i sidecaren.

Fältklassificeringen i `derive_nfi_forest_class` använder artvolym, med
barr-/lövgräns 70 procent. Vid barrdominans väljer den tall eller gran efter
vilken volym som är störst. Den har **ingen separat barrblandskogsklass**.
Artvolym är dessutom inte samma referensmått som NMD:s krontäckningsandel.
Klassregeln finns oförändrad i betydelse sedan `4fa549b` (2026-06-11);
`a9d1ad9` (2026-10-08) tillförde enbart kontroller för ogiltiga artvolymer.
Den konservativa ersättningsregeln i destillationen är oförändrad sedan
`4378a6b` (2026-08-09). Detta verifierades med funktionsspecifik git-historik.
Nästa schema behöver samordna NMD-mappning, fältdefinition, destillationshuvud,
träning och utvärdering; att ändra en LUT eller legend räcker inte.

Read-only-inventeringen bekräftade en faktisk R4-checkpoint, epok 27,
28 klasser, 128-dimensionell TESSERA-ingång och 11 auxkanaler. Dess sparade
config beskriver arkitekturen men innehåller inte en fullständig pinnad
inventering av de etikettfiler som användes vid träningen. Den planerade
auditen binder därför dagens checkpoint och lästa NMD-/destillationssidecarer
med bytehashar. Den kan verifiera dagens artefakter, men kan inte retroaktivt
bevisa att varje sidecar var identisk under den historiska träningen.

**Viktigt:** `nmd_label_raw` i själva träningstilen är redan reducerade
19-klassers koder (`uint8`). Samma nyckel i NMD2023-sidecaren innehåller
ursprungliga NMD-koder (`uint16`). Provet kräver uttryckligen den senare.

## Det första avgränsade provet

- Frysta befintliga R4-egenskaper före klassificeraren jämförs med råa
  TESSERA-egenskaper på exakt samma pixlar. Huvuden och skalning tränas enbart
  inom respektive träningsfold; inga parametrar väljs efter utfall.
- Referens: rå NMD2023 113 mot 115–117 på fastmark. 114, våtmarksskog och låg
  fjällskog ingår inte i denna första binära fråga.
- Högst 24 historiska utvecklingstiles från `unified_v2_512`, gemensamt
  TESSERA-år **2022** mot fast NMD-referens **2023**. Den tidigare dokumenterade
  NMD2023-omträningen behöll uttryckligen äldre spektralår och bytte NMD-basen
  (se `../nmd2023_label_source_retrain.md`, Scope-beslut). Ett års skillnad
  skrivs i kontrakt och rapport; utfallet påstår inget om faktisk skogstyp
  under 2022 eller temporal generalisering. Inget NFI-index, inga
  NFI-observationer och inga förseglade
  sluttestfiler läses. Inga externa data hämtas.
- Deterministisk ordning före resultat: högst en tile per 50 km ruta, hela
  tilen inom samma ruta, och minst 15 km separation i minst en koordinataxel mellan tilecentrum.
  Högst 128 pixlar per klass och tile, på ett 100 m rutnät. En 3×3-erosion
  utesluter omedelbara klassgränser; den förenklingen ska synas i tolkningen.
- Fem geografiska foldar med 50 km grupper och minst 5 km avstånd mellan
  de nya huvudenas tränings- och utvärderingspunkter. Samma grupper används
  för båda representationerna. Dubbletter eller foldar utan båda klasserna
  avvisas. Majoritetsklass är baslinje.
- Klassvis precision, recall, IoU och förväxlingsmatris; balanserad träffsäkerhet
  med 500 bootstrap-omdragningar av hela geografiska block. Intervallet är
  villkorat på dessa modeller och detta urval, inte ett nationellt osäkerhetsmått.

Detta är ett **representationsdiagnostiskt utvecklingsprov** på historiska
träningsplatser. R4 kan ha sett platserna under sin tidigare träning; geografisk
separation av det nya huvudet gör inte R4 till en osedd modell. Utfallet mäter
överensstämmelse med NMD bland utvalda interiörpixlar, inte oberoende
fältprecision eller nationell generalisering. Svagt utfall kan bero på
etiketter, urval, skala eller indata och räcker inte som skäl för sammanslagning.

## Reproducerbar körning

Processen finns i `scripts/extract_forest_separability.py`,
`scripts/probe_forest_separability.py` och
`scripts/render_forest_separability_job.py`. Renderaren skapar ett arkiv med
exakta källfilshashar och en manifest med pinnad befintlig image; den skickar
inte själv något jobb till klustret.

```sh
python -m pytest tests/test_forest_separability.py -q
python scripts/render_forest_separability_job.py --output <ny-receptkatalog> \
  --scheme binary --feature-year 2022
kubectl --context icekube -n prithvi-training-default create --dry-run=server -f <ny-receptkatalog>/job.json
```

`--scheme` är obligatoriskt i extraktion, prov och rendering och skrivs in i
extraktionskvittot; provet vägrar ett kvitto med annat schema. Det första
binära provet kördes innan fältet fanns, så dess kvitto saknar schema och
kan inte provas om med den här versionen. Resultatet står kvar som det är.

Det godkända provets sammanlagda tak är 2 CPU, 4 GiB, ingen GPU och
30 minuters resursanvändning inklusive resultatinsamling. Första försöket
använde ett agentvalt 2023-filter och avslutades under metadataurvalet,
utan att någon modellegenskap eller klasspoäng producerats. Innan några
resultat sågs kvalificerades i stället 2022-kohorten mot det dokumenterade
fasta NMD2023-referensupplägget. Det andra försöket har ett hårt tak på
1 050 sekunder så den gemensamma resursbudgeten hålls. Träningsvolymen monteras skrivskyddad. Resultaten hamnar i
egen tillfällig volym och kopieras till privat lokal evidenskatalog innan
jobbet kvitteras och resurserna städas. Inga privata vikter, koordinater eller
provresultat ska läggas i Git.

## Beslut efter provet

1. Slutför artefaktkedjan så långt bevarade manifest och bytehashar tillåter.
2. Tolka separerbarhet och fel per geografisk grupp; håll isär kodfel,
   klassdefinition, etikettkvalitet och modellbegränsning.
3. Fastställ ett explicit nästa klasschema. Barrblandskog ska inte döljas i
   en lövklass; alternativ som separat fin klass med grövre valbar redovisning
   ska bedömas efter evidensen.
4. Ta fram korrigerade etiketter, fält-/destillationsmål och avgränsad
   träningsplan. Full omträning och sluttest sker i en senare kontrollerad
   körning; dagens diagnos ändrar inte den demomodell som används.

## Genomfört och vad som återstår

Det avgränsade CPU-provet genomfördes med exit 0. Integritet, antal,
dimensioner, ändliga värden och sidecarernas gemensamma huvudstämpel
verifierades på klustret. Tio fokuserade tester passerade. Processbeviset
finns i [verification.json](verification.json); resultaten ligger i den
privata, platsfria sammanställningen för task 0127. Modellfeatures,
koordinater, tilenamn och etikettarrayer exporterades inte.

Jämförelsen visar att frågan **barrblandskog kontra lövskog** kan undersökas
med befintliga representationer. Den avgör inte den svårare gränsdragningen
mot tallskog, granskog och lövblandad barrskog. Den jämför heller inte alla
modellfamiljer på ett oberoende, gemensamt underlag.

Fortsättningen bör därför börja med en förregistrerad flerklassdiagnos av
111, 112, 113, 114 och lövgruppen 115–117, med tydliga separata
tränings-/utvecklingsområden och temporal känslighetsanalys. Kontrollera
klassdefinitionerna för artvolym respektive krontäckning innan fältdata
används som referens. Ett nytt klasschema kräver också ett motsvarande
fält-/destillationsmål; ändra inte ett förseglat sluttests sanning retroaktivt.

R4:s träningslogg bekräftar epok och modellfamilj men sparar inte `label_dir`
eller en fullständig byteinventering. Dagens destillationshuvud har klass-ID
0–4 och dess hashprefix matchar samtliga granskade sidecarer. Det stärker den
aktuella artefaktkedjan men ersätter inte en saknad historisk träningsreceipt.

För demon är de säkra skogsetiketterna modellnamnen **Lövskog** och
**Blandskog**, med en gemensam förklaring: *Skogstyperna är modellklasser som
förfinats med fältdata och motsvarar inte NMD exakt. Barrblandskog redovisas
inte separat.* Destillationsregeln behåller originaletiketten när huvudet
säger icke-skog och lämnar ytterkanten orörd. Därför får ett stickprov inte
användas som belägg för att alla äldre 113→löv-etiketter har försvunnit.

Ingen full omträning har startats och inga servervikter eller numeriska
klasser har ändrats. Det återstår ett schema-/träningsbeslut efter denna
evidens, medan demoutbyggnaden fortsätter oberoende.

## Förregistrering: flerklassdiagnos `forest5` (2026-10-30)

Skriven innan någon körning gjorts. Ändras inte efter att resultat setts;
avvikelser redovisas som avvikelser.

**Fråga.** Var hamnar barrblandskog (NMD 113) när den måste skiljas från
*alla* skogstyper på fastmark: tallskog (111), granskog (112), lövblandad
barrskog (114) och lövskog (115–117)? Och vilken unified-klass ger
basmappningen, de destillerade etiketterna och R4:s utdata varje NMD-typ?

**Klasser.** Schemat `forest5` i `scripts/probe_forest_separability.py`:
111, 112, 113, 114 och 115–117 som en grupp. Utanför: 118, all skog på
våtmark (121–128) och låg fjällskog (43).

**Data och urval.** Samma protokoll som det binära provet: högst 24
historiska utvecklingstiles i `unified_v2_512`, en per 50 km-ruta och minst
15 km mellan tilecentrum, 3×3-interiörer på 100 m-rutnät, högst 128 pixlar
per klass och tile, seed 42. En tile kvalificeras bara om **alla fem**
klasserna har minst åtta sådana pixlar. Det utesluter områden där en klass
är sällsynt och ska stå i tolkningen. Referens är rå NMD2023.

- Primär körning: TESSERA 2022 mot NMD2023 (samma som det binära provet).
- Temporal känslighet: TESSERA 2023 mot NMD2023, samma år. Skillnaden mellan
  körningarna redovisas; ingen av dem väljs i efterhand som "den riktiga".

**Huvud och validering.** StandardScaler + LogisticRegression (C=1,
balanserade vikter, max_iter 2000), ingen hyperparametersökning. Fem
geografiska foldar med 50 km-grupper och 5 km buffert; varje fold måste
innehålla alla fem klasser i både träning och test, annars avbryts provet.
Samma foldar för frysta R4-egenskaper och råa TESSERA-egenskaper.

**Det som rapporteras, i den här ordningen.**

1. Raden för barrblandskog i 5×5-förväxlingsmatrisen, för R4 och TESSERA:
   till vilka klasser 113 förväxlas och hur mycket.
2. Recall, precision och IoU per klass; balanserad träffsäkerhet med 95 %
   intervall från 500 omdragningar av hela 50 km-block; majoritetsbaslinje.
3. `assigned_unified_class`: per NMD-klass, fördelningen över unified-klasser
   i basetiketten, den destillerade etiketten och R4:s utdata. Det visar hur
   R4 faktiskt behandlar barrblandskog, oberoende av de nya huvudena.

**Vad resultatet inte avgör.** Ingen tröskel avgör automatiskt
sammanslagning eller uppdelning. Utfallet mäter överensstämmelse med NMD på
interiörpixlar i historiska träningsområden, inte fältprecision, och R4 kan
ha sett platserna. Det är underlag för schemabeslutet i IM-023, som tas av
Tobias tillsammans med fältdefinitionen (artvolym kontra krontäckning) och
ett matchande destillationsmål.

**Körning.** Kräver godkännande och klusteråtkomst. Samma budget som det
binära provet: 2 CPU, 4 GiB, ingen GPU, högst 30 minuter per körning.

```sh
python scripts/render_forest_separability_job.py --output <katalog-2022> \
  --scheme forest5 --feature-year 2022
python scripts/render_forest_separability_job.py --output <katalog-2023> \
  --scheme forest5 --feature-year 2023
kubectl --context icekube -n prithvi-training-default create --dry-run=server -f <katalog-2022>/job.json
```

## Resultat `forest5`, TESSERA 2022 mot NMD2023 (2026-10-30)

Körningen följde förregistreringen ovan: 6 738 interiörpixlar från 24 tiles,
varav 787 NMD 113. Endast platsfria aggregat återges här; `probe.json` och
bundle-hashar ligger i den privata evidenskatalogen
`ImintEngine/class-separability-0127/forest5-2022/`.

1. **Raden för barrblandskog** (787 pixlar), nya huvuden i fem geografiska
   foldar: R4-egenskaper tallskog 16 %, granskog 16 %, **barrblandskog 58 %**,
   lövblandad barrskog 10 %, lövskog 0,1 %. Råa TESSERA: 15 %, 17 %, **57 %**,
   11 %, 0,1 %.
2. **Balanserad träffsäkerhet** över fem klasser: R4 0,770 (95 % blockbootstrap
   0,726–0,804), TESSERA 0,771 (0,728–0,804); slump 0,20.
3. **Tilldelad unified-klass för NMD 113:** destillerade etiketter tallskog
   74 %, granskog 22 %, blandskog 4 %, lövskog 0 %; R4:s utdata tallskog 71 %,
   granskog 22 %, blandskog 4 %, sumpskog 2 %, lövskog 0 %.

Tolkning inom förregistreringens gränser: barrblandskog förväxlas nästan
aldrig med lövskog och ligger mellan tall- och granskog. Den frysta
R4-representationen och råa TESSERA skiljer klasserna lika bra, så det som
saknas är en målklass, inte information i indata. **Beslut (Tobias
2026-10-30):** barrblandskog blir egen unified-klass 28 och modellen tränas om,
TESSERA först.

**Avvikelse:** körningen med TESSERA 2023 avbröts i extraktionen
(`OUTPUT_READY 1`) och gav inget resultat. Urvalsräkningen gick förlorad
eftersom podden kvitterades direkt; trolig orsak är att färre än fem
2023-tiles har alla fem klasser. `scripts/run_forest_separability.sh` sparar
nu urval och loggslut före kvitteringen. Den temporala känslighetsanalysen
är därmed inte genomförd.

## Förregistrering: `sumpskog5` (2026-10-30)

Skriven innan någon körning. Fråga: bör sumpskog (unified 5) delas upp efter
trädslag? Schemat `sumpskog5` jämför samma typer på våtmark: 121 tallskog,
122 granskog, 123 barrblandskog, 124 lövblandad barrskog och 125–127
lövskog. 128 (temporärt ej skog) och låg fjällskog (23/230) ingår inte.

Protokoll, rapporteringsordning och begränsningar är desamma som för
`forest5` (TESSERA 2022 mot NMD2023, högst 24 tiles, 3×3-interiörer, fem
geografiska foldar). Läs i den här ordningen:

1. förväxlingsmatrisen per klass, särskilt om barrblandskog på våtmark beter
   sig som på fastmark;
2. balanserad träffsäkerhet med blockbootstrap och majoritetsbaslinje;
3. `assigned_unified_class`: vilka klasser R4 och de destillerade etiketterna
   faktiskt ger våtmarksskogen.

Om färre än fem tiles har alla fem klasser avbryts körningen. Det är i sig ett
resultat: undertyperna är för glesa för att bära egna klasser, och urvalet
redovisas från `availability.json`. Ingen tröskel delar automatiskt upp
sumpskog. En uppdelning prövas i så fall i en egen omträning efter
barrblandskogen, inte i samma, så att effekterna går att skilja åt.

```sh
bash scripts/run_forest_separability.sh sumpskog5 2022
```
