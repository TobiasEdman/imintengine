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
python scripts/render_forest_separability_job.py --output <ny-receptkatalog>
kubectl --context icekube -n prithvi-training-default create --dry-run=server -f <ny-receptkatalog>/job.json
```

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
