# E01 — innehållsbunden prediktionscache

Status 2026-09-18: lokal cacheintegritet och familjerouting är verifierade med
syntetiska modeller och riktiga filer. 61 tester passerar. E01 är fortfarande
ofärdigt: vetenskaplig behörighet, dataroller och verklig modellpilot återstår.

## Vad som ändrats

Stage A använder `imint-prediction-cache-v2`. Katalogens fulla SHA-256
identifierar ett oföränderligt kontrakt med checkpointens bytes, tile-inventering,
aktuell kod och konfiguration under `imint/` och `scripts/`, runtimeuppgifter,
modellfamilj, ramar, auxkanaler, crop, klassantal, batchstorlek och shardindelning.
Ändrad checkpoint, indata eller beräkningskonfiguration ger en annan nyckel.
Paketversioner ersätter inte verifiering av externa beroenden eller image-digest.

`MANIFEST.json` publiceras under lås. Varje NPZ-resultat får ett JSON-kvitto med
full hash och storlek. Innehållet kontrolleras före återanvändning och scoring:
nycklar, former, datatyper, finita värden, intervall och klass/probabilitetskonsistens.
Float16 kan skapa lika maximala sannolikheter; den sparade klassen måste vara
en av dessa maxima. Fraktionernas fyra oberoende sigmoidkanaler behöver inte
summera till ett.

Saknat kvitto, korruption eller identitetskonflikt stoppar körningen. Befintliga
resultat skrivs inte över automatiskt. Vid en partiell återstart beräknas hela
den ursprungliga batchen om, men bara saknade tiles publiceras. Det bevarar
batchsammanhanget och befintliga filer. Full cache krävs före scoring, även när
inferensen är fördelad över flera shards. Alla kvitton och filstorlekar kontrolleras
då; full output-hash och arrayinnehåll kontrolleras för varje tile som konsumeras.

Tile-geometri och år läses från samma hashkontrollerade bytes med
`allow_pickle=False`. Crop-offset beräknas separat för varje tiles höjd och
bredd. Ändrade eller saknade indata ger fel i stället för ett tyst populationsbyte.
Gränspunkter utanför den uttryckliga cropytan tas bort och antalet redovisas.

## Användning

`infer_tiles.py --cache-key-out PATH` skriver den fulla kontraktsnyckeln efter
lyckad inferens för den aktuella sharden. Stage B:s befintliga argumentnamn
`--ckpt-sha` tar nu denna 64-teckensnyckel. Äldre cachekataloger med en
16-teckens checkpointnyckel avvisas; att byta namn på dem skapar ingen evidens.

Jobbmallarna läser tillbaka nyckeln från Stage A. YAML och inbäddad Bash har
syntaxkontrollerats. Mallarnas föränderliga image-, käll- och beroendereferenser
är fortfarande ett hinder för kampanjkörning; ingen mall har skickats till klustret.

## Verifiering

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q -p no:cacheprovider \
  tests/test_prediction_cache.py \
  tests/test_cached_validation_parity.py \
  tests/test_inference_tile_identity.py
```

61/61 tester passerade på 123,90 sekunder. Testerna gör verklig disk-I/O,
preprocessing, routing och CPU-scoring med små syntetiska modeller. De täcker
alla sex modellfamiljerna, Prithvi 1f/4f, huvud med/utan fraktioner,
cacheinvalidering, korruption, ofullständig publicering, samtidiga disjunkta
skrivningar, metadata före hashkontroll, objektmetadata utan pickle,
rektangulär crop och oförändrad batch vid återstart.

Oberoende uppföljande granskning: LGTM with nits, inga kvarvarande BLOCKER eller
WARNING i rättningarna. En oanvänd `_tile_hw` är kvar som städpunkt.
Den tidigare routinggranskningens testbegränsning kvarstår: syntetiska
koordinatsummor fångar inte varje möjlig komponent- eller tidspermutation.

Maskinläsbar verifiering, JUnit, mallkontroller och granskningsanteckning finns
i kampanjens artifactkatalog som `e01-cache-verification.json`,
`e01-cache-tests.xml`, `e01-cache-template-checks.json` och
`e01-cache-review.md`.

## Kvar innan experimentkörning

1. Stage B har nu en hashbunden supervision-grind; se
   [fm_e01_fraction_gate.md](fm_e01_fraction_gate.md). Den nuvarande policyn
   frikänner ingen av de 28 cellerna för fraktionsscoring. Själva verifieringen
   av fraktionsmålens lineage återstår. Prithvi-600M R4 saknar avsedd behandling
   och får inte användas för det avsedda R4−R3-påståendet.
2. Bind historisk kod, runtime, targetdata och splitidentitet till evidens.
   Det nya cachekontraktet dokumenterar inferensen, inte historisk träning.
3. Utred det verifierade överlappet: **404 unika LUCAS-punkt-ID:n delas
   mellan distill och crop-holdout**. Se
   [observationsrollernas audit](fm_e01_observation_roles.md).
   De 1 064 frysta holdoutraderna förblir skyddade; deras oberoende är inte
   godkänt. Lås roller, läckagekontroller och datakvalitet innan pilotlistor
   eller sluttestprotokoll får klartecken.
4. Förbered och kostnadssätt en liten utvecklingspilot med verkliga checkpoints,
   strict-load, forward och direkt/cache-jämförelse. Användaren beslutar om
   kostnadsdrivande åtgärder innan start.

Ingen ny klusterkörning, träning eller sluttestutvärdering gjordes i detta steg.
Det separata tidigare E00-testets timeout är inte löst av cacheändringen.
Ingen commit är gjord.
