# E01 — körunderlag för index- och geometriaudit

Status 2026-09-18: förberett, lokalt verifierat och oberoende granskat.
Serverdryrun godkänner ConfigMap, Job och en representativ Pod. **Det verkliga
jobbet har inte skapats. Användarens resursbeslut återstår.**

Bakgrunden är de 404 gemensamma LUCAS-punkt-ID:na mellan distill och den
frysta crop-holdouten; se [observationsrapporten](fm_e01_observation_roles.md).
Nästa kontroll räknar berörda rader, skiljer samma år från återbesök och
undersöker geometri. Den ändrar ingen split eller historisk modellpoäng.

## Körresultat 2026-09-24

Användaren godkände körningen. Jobbet är nu avslutat med `incomplete_geometry`
(exitkod 2): 386 av 404 gemensamma distill/holdout-ID:n förekommer samma år,
och nio NFI-rader på två tiles saknar geometri i det använda manifestet.
Resultatet är arkiverat; inget omförsök gjordes och inget scoringklartecken
har getts. Se [den daterade resultatrapporten](fm_e01_index_audit_result.md).
Statusen ovan och förberedelseavsnitten nedan beskriver läget 2026-09-18.

## Föreslagen resursram

- Ett CPU-jobb: **2 CPU, 4 GiB RAM, 1 GiB temporär disk, högst 600 sekunder**.
- Ingen GPU, ingen träning, inget nytt imagebygge eller paketinstallation.
- `backoffLimit=0`, `restartPolicy=Never`, termineringsfrist 0 sekunder.
- Full tidsram motsvarar högst 20 CPU-minuter och 40 GiB-minuter RAM.
  Detta är en resursram; inget pris i kronor har beräknats.
- Befintlig image:
  `ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:25919e5cc0bc5ccba770b80f5906d0721cf47a779e3b4591a07cf5c5f3dc3599`.
  Scoringmiljön `/opt/venvs/scoring/bin/python` har pinnad PyArrow 17.0.0.
- Jobbnamn: `fm-e01-index-audit-e80200a4d967`.
  Inga automatiska omförsök eller höjningar av resursramen.

## Exakt indata och dataskydd

Fem filer monteras var för sig med skrivskyddad `subPath`, sammanlagt
1 507 551 byte. Hela filernas bytes hashas; enbart kolumnerna nedan avkodas.
Det vore därför missvisande att säga att inga utfallsbärande bytes läses.
Inga utfallskolumner eller featurekolumner avkodas.

| Fil under `/cephfs/` | Avkodade uppgifter |
|---|---|
| `distill/crop_split/lucas_crop_split.MANIFEST.json` | Frysta artefakthashar och protokollidentitet |
| `distill/crop_split/lucas_crop_split.json` | Frysta distill-, holdout- och äldre testnycklar |
| `lucas/lucas_tile_index.parquet` | `tile_name`, `point_id`, `Easting`, `Northing`, `Year`, `row`, `col` |
| `nfi/nfi_index_unified_v2_512.parquet` | `tile_name`, `TractID`, `PlotID`, `Easting`, `Northing`, `Year`, `row`, `col` |
| `lucas/tile_bbox_manifest.parquet` | `tile_name`, `easting`, `northing`, `year` |

Validatorparquet, NPZ-tiles, featurefiler, vikter och resultattabeller monteras
inte. Validatorns kända filnamn och manifestbundna innehållshash avvisas före
öppning, även vid namnbyte. Logg och resultat innehåller räknare, årsfördelningar,
fil-/kodhashar och exekveringsmetadata; inga observationsrader, punkt-ID:n eller
koordinater exporteras.

Indataspecifikationen `e01-index-audit-inputs.json` har SHA-256
`8871e85e40daf7c9ea1667e9faa9630fdc517e19d61a23e2dda709818feddcf2`.
LUCAS-indexets hash `e3bb505c4de469d0436e0e91de27327f063083199c8ddc781ec6eaf7d42e9a41`
stämmer med den historiska källhashen i den frysta splitten. Detta kontrolleras
igen före indexläsning. Ändrade bytes stoppar körningen.

## Kontroller och tolkning

Auditen verifierar den frysta identiteten och behåller överlappsfyndet.
Samtliga frysta observationsnycklar måste finnas i det hashbundna källindexet.
Den räknar gemensamma punkt-ID:n, berörda rader på respektive sida, gemensamma
punkt-ID/år och ID:n som endast överlappar mellan olika år. NFI använder en
separat ID-rymd, `(TractID, PlotID)`.

Geometrin använder repoets `TileConfig`: EPSG:3006, 512 × 512 pixlar à 10 m,
med grid-snappade centra från det pinnade manifestet. Punktens år och pixelrad/
kolumn jämförs med denna geometri. Saknade eller motsägande uppgifter ger
`incomplete_geometry`; inga tysta antaganden ersätter dem. Överlapp mellan
tileytor kräver positiv area: enbart gemensam kant räknas inte.

Närhetsdiagnostiken räknar rollordnade ID-par inom 10 meter: `(vänster-ID,
höger-ID)`. Om både A och B finns i båda rollerna räknas `(A,B)` och `(B,A)`
separat. Samma LUCAS-ID undantas här eftersom identitetsöverlappet redan
redovisas separat. Tio meter är en diagnostisk radie, ingen fastställd gräns
för statistiskt oberoende. Varje närliggande kandidatjämförelse omfattas av
jämförelsebudgeten, även när den inte ger en träff.

Detta verifierar **manifestbaserad geometri**, inte dagens NPZ-innehåll.
Historiska foldtilldelningar, faktisk träningsmedverkan, parcelidentitet och
vidare spatialt beroende återstår. NFI-indexet är inte i sig en fastställd
utvecklings-/testpopulation. Ingen uppskattning av poängbias görs.

`diagnostic_audit_complete` och exitkod 0 betyder att just diagnostiken kunde
genomföras. `cleared_for_scoring` är alltid `false`, även vid noll träffar.
Ofullständig geometri ger exitkod 2; ogiltiga indata ger exitkod 1.

## Verifierat före kostnadsbeslut

48 tester passerar på 8,73 sekunder: 22 nya och 26 befintliga guardtester.
De omfattar riktiga Parquet-filer, explicit kolumnurval före avkodning, verklig
CLI med JSON på disk, hashförändring, skyddad validator, olika år, pixelriktning,
kantkontakt, exakt 10-metersgräns, symmetriska rollpar och jämförelsebudget.
Oberoende slutgranskning: **LGTM**, inga kvarvarande fynd.

Serverdryrun godkänner slutligt Job/ConfigMap och representativ Pod. Den
antagna Pod-definitionens image, kommando, argument, resurser, datamontage och
säkerhetsinställningar stämmer med förslaget. Inga riktiga resurser skapades.
Torrkörningen bevisar inte runtime-läsbehörighet eller verklig Parquet-körning;
det är det föreslagna jobbets uppgift.

Evidens i kampanjens artifactkatalog:
`e01-index-audit-input-inventory.json`, `e01-index-audit-inputs.json`,
`e01-index-audit-job.v2.json`, `e01-index-audit-tests.final.xml`,
`e01-index-audit-server-dryrun.v2.json`, `e01-index-audit-review.md` och
`e01-index-audit-verification.json`. Första renderingen och dess tester behålls
som historik; endast `v2` är aktuellt körunderlag.

## Efter ett uttryckligt resursgodkännande

1. Kontrollera att exakt detta manifest och kod fortfarande matchar kvittot.
   Kontrollera att jobbet inte redan finns; återanvänd inte en okänd Job UID
   och starta aldrig om redan slutfört arbete.
2. Registrera körningens kodlås och spara Job-/Pod-UID samt faktisk imageID.
   Skapa endast den granskade ConfigMap/Job-listan; Pod-dryrunfilen skickas
   aldrig som en verklig Pod.
3. Bevaka terminalstatus och spara logg, resultat, Job/Pod-status och faktiskt
   resursutfall i kampanjarkivet före sjudagarstidsgränsen. Om bevakningen
   behöver fortsätta efter sessionen ska den ha slutvillkor och hård sluttid.
4. Tolka även ofullständiga/negativa fynd, frigör kodlåset efter bekräftad
   terminalstatus och städa den egna ConfigMapen först efter arkivering.
   Inget resultat får automatiskt starta en pilot, omträning eller ny split.
