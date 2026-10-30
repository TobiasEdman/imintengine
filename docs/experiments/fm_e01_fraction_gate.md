# E01 — grind för fraktionssupervision

Status 2026-09-18: Stage B skiljer mellan ett befintligt fraktionshuvud och
verifierad supervision. Datapolicy, historisk lineage och verklig modellpilot
är fortfarande separata krav.

Ett konkret maskfel har nu reproducerats i dataset→förlust-kontraktet och
rättats lokalt; se [fraktionsmålens kodgranskning](fm_e01_fraction_lineage.md).
Det ger inte historiska checkpoints klartecken och ändrar inte beslutstabellen.

## Beslut för de 28 befintliga checkpointfilerna

Det versionsbundna underlaget är `fm_fraction_scoring_policy.json`:

- 24 celler är `unverified`: huvud finns enligt insamlade loggar.
  Sex R4-celler har registrerad fraktionsförlust, men målens lineage är
  fortfarande overifierad. För övriga 18 registreras ingen fraktionsförlust.
- Fyra Prithvi-600M-celler är `ineligible`: huvud är avstängt enligt logg.
  R4 uppfyller därmed inte den avsedda fraktionsbehandlingen.
- Ingen cell är `eligible`.

Varje rad binder checkpointens fulla SHA-256 och storlek, träningsloggens hash,
huvudflagga, förlustsignal, skäl och underlagsrapportens hash. Inga historiska
råstatusar har ändrats. Policyn härleds från den redan insamlade
`e01-inference-provenance.json` och `checkpoint-identities.jsonl`.
Generatorn finns i artifactkatalogen som `build_fraction_scoring_policy.py`.

## Hur scoring använder beslutet

Båda scoringvägarna kontrollerar beslutet innan observationsindex läses.

| Val | Beteende |
|---|---|
| Inga fraktionsflaggor | Klasshuvudet används; ingen fraktionspolicy krävs. |
| `--use-fraction-head` | Kräver `eligible`; annars stopp. |
| `--auto-fraction` | Kräver ett känt, giltigt beslut. `eligible` använder fraktioner; `unverified` och `ineligible` ger klassbaserad scoring med redovisat skäl. |

Båda fraktionsvalen kräver `--fraction-policy PATH` och
`--fraction-policy-sha256 HASH`. Okänd checkpoint, avvikande viktstorlek,
motsägande huvudflagga, saknad policy, fel hash eller tvetydiga/dubblerade
beslut ger fel. Policyn läses som stabila, vanliga filbytes; hash kontrolleras
före JSON-tolkning. Samma bytes tolkas, utan separat återöppning.

För den aktuella, negativa beslutstabellen är SHA-256:

```text
46366248c21fded308afd488d2c92640605019b1ca52cb048ebb609a71e9b299
```

Den förväntade hashen ska komma från den granskade körspecifikationen.
Beräkna inte automatiskt en ny förväntad hash från en ogranskad fil vid start.

Scoring sparar `_meta.fraction_scoring` med valt läge, beslut, skäl,
checkpoint och policy-/evidenshashar. Att automatiken avstår från fraktioner
ska föras vidare till jämförelsetabellen, så att frånvaro av L2a inte tolkas
som genomförd eller misslyckad fraktionsbehandling.

## Vad denna kontroll inte bevisar

Policyn är ett granskningsbeslut. Kontrollkoden verifierar identitet och
konsistens; den kan inte bevisa att en attestation är vetenskapligt riktig.
En framtida `eligible`-rad kräver aktivt huvud, registrerad icke-nollförlust
och hash till en granskad rapport som binder rätt fraktionsmål till träningen.
Att lägga till en hash är inte själva granskningen. Källrapporterna måste
arkiveras och det uppdaterade beslutet granskas före en ny körspecifikation.

Lågnivåfunktionen `make_cached_predict_fn(want_fracs=True)` får fortfarande
läsa råa fraktionsarrayer för tekniska paritetstester. Studiens scoring går
genom Stage B-grinden. Äldre direktvalidatorer har fortfarande egna
huvudbaserade val och ska inte användas för att kringgå kampanjens grind.

Kontrollen ger inte klartecken för observationsroller, oberoende från
träningsdata, val av trösklar eller användning av crop-holdouten. Dessa delar
av E01 återstår innan ett sluttest får köras.

## Verifiering 2026-09-18

- Samlad regression: 81 tester passerade, inklusive de 61 tidigare
  cache-/routing-/identitetstesterna.
- Efter komplettering av fyra behörighetsfall och rättning av den syntetiska
  LUCAS-kategorin passerade alla 24 fraktionstester igen, på 9,36 sekunder.
  Runtime-varningar behandlades som fel i denna slutkörning.
- Totalt är 85 unika testfall verifierade för oförändrad produktionskod.
  De positiva fallen använder enbart syntetiska policyer och data.
- Riktig NPZ-/parquet-I/O visar att NFI faktiskt byter från klassprediktion
  till fraktionskollaps när beslutet tillåter det, och att LUCAS läser
  fraktionskanalerna och ger förväntad dominerande-artöverensstämmelse.
- Saknat tillstånd stoppas före sanningsdata; hash-, checkpoint- och
  huvudkonflikter, dubbla beslut och otillräcklig supervision avvisas.
- Oberoende granskning: inga BLOCKER eller funktionella WARNING.
  Granskarens önskemål om kompletterande behörighetstester är genomfört.
  Granskaren kontrollerade även samtliga 28 beslut mot källrapporterna.

Kampanjens artifactkatalog innehåller `e01-fraction-policy-verification.json`,
`e01-fraction-policy-tests.xml`, `e01-fraction-policy-tests.final.xml` och
`e01-fraction-policy-review.md`. Första körningens varning är bevarad i
protokollet och härrörde från den syntetiska testkategorin; den försvann i
slutkörningen efter rättningen.

Inga nya jobb, checkpoints, fulla observationsfiler eller hållna testutfall
lästes i detta steg. Ingen commit eller deployment gjordes.
