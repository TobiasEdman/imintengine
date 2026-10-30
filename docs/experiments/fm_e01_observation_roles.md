# E01 — observationsroller: verifierat överlapp i crop-splitten

Status 2026-09-18: **blockerad oberoendekontroll** för den befintliga
crop-splitten. Detta är ett negativt metodfynd från verifierade identitetsmetadata,
inte ett modellutfall. E01 och E04 är ofärdiga.

## Kompletterande indexaudit 2026-09-24

Den godkända CPU-auditen är nu genomförd: 960 distillrader och 616 holdoutrader
bär de 404 gemensamma ID:na; 386 ID:n förekommer i båda rollerna samma år.
Geometri kontrollerades mot ett pinnat tilemanifest. Nio NFI-rader på två
tiles saknar referens där, varför resultatet är `incomplete_geometry`.
[Resultatrapporten](fm_e01_index_audit_result.md) redovisar också bildytornas
överlapp och vad som ännu inte är verifierat. Väntande körstatus och öppna
indexfrågor i ursprungstexten nedan avser 2026-09-18 och ersätts av detta
resultat; historiska folds och träningsmedverkan är fortfarande outredda.

## Vad den faktiska kontrollen visar

| Kontroll | Resultat |
|---|---:|
| Distill-rader | 2 491 |
| Unika punkt-ID:n i distill | 1 521 |
| Crop-holdout-rader | 1 064 |
| Unika punkt-ID:n i crop-holdout | 804 |
| Gemensamma punkt-ID:n, distill ↔ crop-holdout | **404** |
| Gemensamma exakta (tile_name, point_id)-nycklar | 0 |
| Distill ↔ skyddade tile-namn | 0 |
| Distill ↔ äldre testurval, gemensamma punkt-ID:n | **8** |
| Äldre testurval | 71 rader, 65 punkt-ID:n, 53 tiles |

De åtta äldre test-ID:na ingår även i de 404 gemensamma crop-ID:na; antalet
unika distill-ID:n i hela den skyddade unionen är därför 404. Unionen omfattar
849 punkt-ID:n och 658 tile-namn.

Detta visar varför en split grupperad enbart efter tile-namn inte räcker.
Samma punkt-ID kan ha fler än en rad och ett annat tile-namn. Siffran 404
avser **unika punkt-ID:n**, inte antalet berörda holdoutrader. Antalet berörda
rader, deras år och deras geografiska relation behöver fortfarande kontrolleras
i indexmetadata. Kontrollen verifierar inte koordinater eller alias mellan
olika punkt-ID:n.

Den befintliga holdouten får därmed inte beskrivas som oberoende från
distill-populationen på punkt-ID-nivå. Detta säger inte i sig hur stora
modellpoängens eventuella bias är. Ingen poäng beräknades eller öppnades.

## Identitet och avgränsning

Kontrollen läste exakt två filer på ett befintligt, skrivskyddat montage:

- `lucas_crop_split.MANIFEST.json`, SHA-256
  `0d4e21d01a87de203eaa04408a11210e193dabbfa12bd440cdf0659bfc6179ed`.
- `lucas_crop_split.json`, SHA-256
  `8d5e65f9fcae8e8ca6ae5fd44cb3858c93052d81c04a19f89b05fbddbc58171e`.

Förväntad splitkälla är `c6ad69242e7239662461bf7ff0b6bcd4d072509a`,
hämtad från de gemensamma körmanifesten i
`docs/evidence/crop-distill/run-manifests.json`. Båda registrerade
körhistorikerna anger samma splitauktoritet.

Fulla filhashar, käll-SHA, räknare och samtliga sex kontrollerade
nyckel-/partitionsdigester stämmer. Metadata är alltså identifierade och
internt konsekventa, samtidigt som rollernas punkt-ID:n överlappar.
Resultatet behåller `not_verified`, exitkod 2 och
`cleared_for_scoring=false`.

Validatorparqueten, NPZ-tiles, featurefiler och utfall öppnades inte.
Inga punkt-ID:n eller tilerader exporterades. Inga jobb eller paketinstallationer
startades; befintlig dashboard användes. Den andra, diagnostiska läsningen
tog 2,702 sekunder inklusive kubectl-anropet. Den första stoppande kontrollen
och dess exakta kod är bevarade separat.

## Implementerad kontroll och verifiering

`scripts/crop_holdout_guard.py` kan läsa splitmetadata och granska en
kandidatmängds identiteter. Den skyddar både crop-holdout och äldre testurval.
Punkt-ID-kontrollen gäller LUCAS även när tile-namn eller år har ändrats;
NFI kontrolleras mot tile-namn och behandlas som en annan ID-rymd.

Kandidatläget läser endast Parquet-kolumnerna `tile_name` och för LUCAS
`point_id`, efter kontroll av hela filens förväntade bytehash.
Validatorns kända filnamn och dess manifestbundna hash avvisas före öppning,
även om samma fil har döpts om. Alla faktiskt lästa filer redovisas.
Tomma eller felaktiga identiteter avvisas. En identitetskontroll utan träff
ger fortfarande inte klartecken för scoring.

26 lokala tester passerar, inklusive faktisk Parquet-I/O och CLI,
samma digestformat som splitproducenten, punkt-ID-överlapp under olika tiles,
båda förbjudna validatorvägarna och stopp med räknarbaserad diagnostik.
Oberoende slutgranskning: **LGTM**. Inga blockerande kodfynd kvarstår.
Kontrollen är ännu inte integrerad i Stage A/B:s körprotokoll.

## Konsekvens för nästa steg

1. Behåll den frysta splitten och håll utfallen stängda. Ändra inte gamla
   körningar eller resultat, och kalla inte deras rollkontroll godkänd.
2. Kör en avgränsad identitetsaudit på NFI-/LUCAS-index och använda crop-roller:
   antal berörda rader, återbesök/år, koordinater, överlappande tileytor,
   foldkoppling och koppling till historisk träning. Läs endast nödvändiga
   identitets-/geometrikolumner.
3. Skriv ett metodförslag som hanterar de sammanlänkade punkt-/tilegrupperna
   och tydligt anger vilka tidigare utvecklingsresultat som kan återanvändas.
   En ändring av split/analys ska vara prospektiv och versionsbunden.
4. Frys därefter tillåtna pilot-/utvecklingslistor och integrera dem i
   körprotokollet. En modellpilot eller omträning kräver separat resursbeslut.

Dashboardmiljön saknar PyArrow, så den större indexauditen är inte genomförd.
Förbered en konkret CPU-audit med pinnad miljö och begränsade datamontage
innan användaren ombeds besluta om resursförbrukningen.

Körunderlaget är nu färdigt: [index- och geometriaudit](fm_e01_index_audit.md).
48 lokala tester och serverdryrun passerar; det verkliga CPU-jobbet väntar
på användarens resursbeslut.

Terminal evidens finns i kampanjens artifactkatalog:
`e01-crop-holdout-live-audit.v2.json`, originalet
`e01-crop-holdout-live-audit.json`, de två arkiverade
`crop_holdout_guard.executed*.py`, `e01-crop-holdout-tests.final.xml`
och `e01-crop-holdout-verification.json`.
