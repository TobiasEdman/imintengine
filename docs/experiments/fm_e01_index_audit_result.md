# E01 — resultat från index- och geometriauditen

Status 2026-09-24: det användargodkända CPU-jobbet är avslutat och resultatet
är arkiverat. Auditen gav `incomplete_geometry`, exitkod 2. Körningen gav
verifierade identitetsfynd men ingen rätt att starta scoring eller ändra splitten.

## Verifierade fynd

| Kontroll | Resultat |
|---|---:|
| Distill-rader / unika punkt-ID:n | 2 491 / 1 521 |
| Crop-holdout-rader / unika punkt-ID:n | 1 064 / 804 |
| Gemensamma punkt-ID:n | 404 |
| Gemensamma ID:n som finns på båda sidor samma år | 386 |
| Gemensamma ID:n som bara överlappar mellan olika år | 18 |
| Distill-rader med något av de gemensamma ID:na | 960 |
| Holdoutrader med något av de gemensamma ID:na | 616 |
| Exakta gemensamma `(tile_name, point_id)`-nycklar | 0 |
| NFI-rader som saknar geometri i det använda manifestet | 9 av 982, fördelade på 2 tiles |

Överlappet är alltså inte enbart återbesök under olika år. Samma punkt-ID
förekommer under olika tile-namn, och 386 ID:n förekommer i båda rollerna
under samma år. De 616 berörda holdoutraderna är inte 616 unika punkter.
Den frysta holdouten är inte punkt-ID-oberoende från distill-populationen.
Detta kvantifierar inte modellpoängens bias och bevisar inte i sig läckage
mellan de historiska OOF-foldarna; fold- och träningskopplingen återstår.

Års- och pixelpositioner stämde med manifestgeometrin för samtliga 2 491
distillrader, 1 064 holdoutrader och 71 äldre testrader. För NFI saknas
manifestgeometri för nio rader; inga år-/pixelfel hittades bland de övriga.
Det använda manifestet är `lucas/tile_bbox_manifest.parquet`. Bristen visar
att denna geometrikälla inte täcker hela NFI-indexet, inte att nio NFI-rader
har bevisats geometriskt felaktiga. Det är orsaken till `incomplete_geometry`.

Den skyddade unionen omfattar crop-holdout och det äldre testurvalets
**deklarerade** tile-listor: 658 tiles. Distill har 3 082 tilepar med positiv
överlappsyta mot unionen, varav 2 915 samma år och 167 olika år. För NFI
hittades 158 sådana par på täckt geometri; två NFI-tiles saknar geometri.
Detta är överlapp mellan manifestbaserade bildytor, inte en ny punktmatchning
eller ett mått på poängbias. De 71 äldre testraderna ligger på 24 tiles;
den äldre deklarerade skyddslistan omfattar 53 tiles och har behållits.

Tio-metersdiagnostiken hittade inga ytterligare par med olika LUCAS-ID:n,
och inga NFI–skyddade LUCAS-par inom tio meter. Samma LUCAS-ID undantas från
just denna diagnostik. Noll här ersätter inte vidare spatial kontroll och
upphäver inte de 404 identitetsöverlappen. Aktuella NPZ-filer, parcelidentitet
och faktisk historisk träningsmedverkan har inte kontrollerats.

## Körning och verifiering

- Jobb: `fm-e01-index-audit-e80200a4d967`, ICE / `prithvi-training-default`.
- Job UID: `05b862f5-bf4d-4c50-acef-5006068547dd`.
- Pod UID: `8255b20d-4103-4741-852d-9a166282630a`.
- Godkänd ram: 2 CPU, 4 GiB RAM, 1 GiB temporär disk, 600 sekunder,
  ingen GPU och inga omförsök.
- Processens uppmätta tid: 0,737 sekunder; CPU-tid 0,98 sekunder;
  maximal RSS 107 122 688 byte (cirka 102 MiB). Containerstart/slut:
  2026-09-24 14:47:29–14:47:30 UTC. Detta är inte fakturerad resurskostnad.
- Python 3.11.16, PyArrow 17.0.0; faktisk image-digest matchar den pinnade
  `25919e5cc0bc5ccba770b80f5906d0721cf47a779e3b4591a07cf5c5f3dc3599`.
- Exekverad kod, indataspecifikation, samtliga indatahashar, Pod-identitet
  och faktisk resursram stämmer med det granskade underlaget.
- Endast identitets-/geometrikolumner avkodades. Hela indatafiler hashas;
  validatorparquet och featurefiler öppnades inte. Inga råa observationsrader,
  ID:n, koordinater eller utfallskolumner exporterades.
- Resultatets populationer, årssummor och överlappskategorier kontrollerades
  före rapportering. Körningen avslutades utan omförsök. Kubernetes markerar
  Jobbet `Failed` eftersom den avsiktliga exitkoden 2 betyder ofullständig
  diagnostik; processen producerade ett fullständigt JSON-resultat.
- ConfigMap och kodlås är borttagna efter arkivering. Jobb och Pod behålls
  enligt befintlig sjudagars-TTL. Ingen historisk data eller modell ändrades.

## Konsekvens och nästa steg

Behåll den befintliga splitten som versionsbunden historik. Varken denna
audit eller den tidigare OOF-tabellen etablerar en oberoende sluttestpopulation.
Innan pilotlistor och utvärderingsprotokoll låses behöver två frågor lösas:
kompletterande geometriproveniens för de två NFI-tilesen samt faktisk koppling
mellan observationer, historiska folds och träning. Ett nytt splitförslag
måste hålla samman återbesök och relevanta geografiska grupper och skiljas
från tidigare exponering. Inga automatiska omkörningar, resplitter eller
fraktionsscoringbeslut följer av detta resultat.

## Arkiv

Kampanjarkivets `e01-index-audit-run-20260924/` innehåller `result.json`,
`audit.log`, `verification.json`, full Job-/Pod-status, arkiverad ConfigMap,
serverdryrun, startkvitto och städkvitto. `result.json` har SHA-256
`3af1b39871fe38a0dc3eafb27aec0747556858450412738848ac4ccb134b951f`.
Råloggens SHA-256 är
`863fd54c2d0636b1d2bc8e94b6baabe9d892fab0213c5a37bd138b11e9b85285`.
Se [körunderlaget](fm_e01_index_audit.md) och
[observationsrollerna](fm_e01_observation_roles.md).
