# E01 — verifierad maskinversion i fraktionsförlusten

Status 2026-09-18: felet är reproducerat och en minimal lokal korrigering är
verifierad. Ingen träningskörning, image, checkpoint eller historisk poäng
har ändrats. Ingen fraktionscell har fått scoringklartecken.

## Fyndet

Datasetets `_load_frac_bundle` läser fraktionssidecarens `frac_unreliable`
och skapar en positiv mask: **1 betyder att pixeln ska superviseras**.
Tränaren skickar `batch['frac_mask']` oförändrad till `FractionLoss`.
Den gamla förlusten tolkade däremot samma värde som **1 = osäker, uteslut**.

Med `H = minst en positiv fraktion` och `U = sidecarens osäkerhetsmask` blir:

```text
Dataset:         M = H & ~U
Gammal förlust:  keep = ~M & H = U & H
Rättad förlust:  keep =  M & H = H & ~U
```

Den gamla vägen utesluter alltså pixlar som enligt masken är tillförlitliga
och använder osäkra pixlar med positiv signal. Detta gäller fraktionsförlustens
derivata mot pixlarnas fraktionslogits. Hårdklassförlusten och delade modellvikter
kan fortfarande uppdatera modellen; fyndet betyder inte att hela modellen står
otränad eller att alla dess prediktioner är ogiltiga.

## Reproduktion och versionsgräns

En syntetisk NPZ innehåller tre pixlar: tillförlitlig med signal, osäker med
signal och utan signal. Den ursprungliga implementationen ger fraktionsgradient
på bara den andra pixeln. Avsett utfall är gradient på bara den första.
Reproduktionen kör både L1 och smooth-L1.

| Källversion | Vad kontrollen visar |
|---|---|
| `69ccbb3d92a2b253f739658b249411987fbfb466` | Felet finns i introduktionen 2026-08-09. |
| `1fd08fad9ba9ab599415230938e6fade357cd5eb` | Felet finns i det deklarerade 4f-receptets källa. |
| `7d935b9a0210f9a88b641ef24e7d83206c2fdcb0` | Felet finns i den tidigare auditimagets källa. |
| `89875350d8b03152bdd837511fb5a5e23d6769fd` | Felet finns i denna worktrees HEAD före rättningen. |

De granskade dataset-, loss- och anropsdelarna har identiska AST-digester i
alla fyra versioner. Åtta fall kör dessa exakta källavsnitt; två ytterligare
fall kör de faktiskt importerade implementationerna före rättningen. Detta är
en reproduktion av kodkontraktet, **inte** fyra historiska träningskörningar i
deras ursprungliga miljöer.

Arkivet innehåller `reproduce_fraction_mask_contract.py` och
`e01-fraction-mask-reproduction.json`, med fulla filhashar, AST-digester,
källrader och observerade gradienter. Originalrapporten bevaras efter rättningen.

## Lokal korrigering och verifiering

Endast `FractionLoss` ändrar beteende: tredje argumentet heter nu `frac_mask`
och betyder positiv supervision. `keep = supervised & has_signal` följer
datasetets befintliga kontrakt. L1/Huber-beräkning, normalisering och
differentiabel nolla för osuperviserade batcher är bevarade. Dataset, tränarens
anrop och historiska källversioner är oförändrade.

- Före rättning: 9 misslyckade och 6 godkända nya regressioner.
- Efter rättning: **60 tester passerar på 22,10 sekunder**, inklusive samtliga
  15 nya fall, befintliga Lovász-tester, Tessera-integration och scoringpolicy.
- Tester läser verkliga syntetiska NPZ-filer, kör datasetets centercrop för
  Prithvi och Tessera, verifierar L1/smooth-L1 och gradienternas riktning samt
  kör ett SGD-steg som bara ändrar de tillåtna pixlarnas fraktionslogits.
- Osäkra pixlar, nollmål, saknade och korrupta sidecars ger ingen fraktionsgradient.
- Oberoende slutgranskning: **LGTM**, inga kvarvarande fynd.

Verifieringen ovan från 2026-09-18 omfattade ingen full träningsepok.
Den senare syntetiska körningen nedan verifierar hela tränaren med en riktig
modell. Före framtida klusterträning återstår preflight med riktiga mål,
korrekt pinnad körmiljö och separat resursgodkännande. Den förberedda
CPU-indexauditen påverkas inte: den använder inte tränaren eller
fraktionsförlusten.

## Komplettering 2026-09-24: en epok genom den riktiga tränaren

Två lokala CPU-fall passerade med syntetiska NPZ-tiles och fraktionssidecars.
Båda använde `UnifiedDataset`, den registrerade `tessera_v1`-modellen och
`LULCTrainer` för en hel epok med optimering, validering, loggning och
checkpointlagring. Varje fall hade två träningstiles och en separat
valideringstile. Miljön var lokal Torch 2.12.0 med två CPU-trådar; ingen
klusterresurs användes. Gradient-hooks observerade utfallet utan att ersätta
tränaren, modellen eller förlusten.

| Fall | Verifierat utfall |
|---|---|
| Blandad mask | Exakt 768 tillåtna pixlar i batchen fick fraktionsgradient; alla uteslutna pixlar fick noll. Fraktionsförlusten var positiv och fraktionshuvudets vikter ändrades. |
| Alla pixlar maskerade | Noll fraktionsgradient och noll fraktionsförlust. Fraktionshuvudets vikter var oförändrade. |
| Båda fallen | Sparad checkpoint lästes tillbaka med projektets begränsade laddare, laddades strikt i en ny modell och samtliga modelltensorer var exakt lika. |

Detta verifierar kopplingen dataset → modell → förlust → optimering →
checkpoint på syntetiska data. Det verifierar inte verkliga fraktionsmål,
historiska körningar, datamängdernas oberoende eller den pinnade
klustermiljön. De syntetiska kvalitetsmåtten används inte som studieresultat.
Den verkliga träningspreflighten och resursbesluten återstår; scoringpolicyn
är fortsatt 24 unverified, 4 ineligible och 0 eligible.

Reproduceraren är `campaign/fraction-trainer-smoke.py`. Evidensen finns i
`campaign/fraction-trainer-smoke-20260923/verification.json`, tillsammans
med indata, träningsloggar och de två checkpointfilerna. JSON-rapporten
binder resultatet till fullständiga käll- och checkpoint-hashar.

## Vad som kan sägas om de sju R4-cellerna

De redan hashbundna loggmetadata visar aktivt huvud, `lambda_frac=1` och
registrerad icke-noll fraktionsförlust för Clay, CROMA, Prithvi-300M 1f/4f,
TerraMind och Tessera. De saknar däremot `frac_dir` och `label_dir` i den
sparade konfigurationen. Dagens jobbmanifest pekar på `/cephfs/tradslag_fracs`,
vilket är en deklaration och inte bevis på de historiskt lästa målens bytes.
Prithvi-600M R4 har avstängt huvud och saknar registrerad fraktionsförlust.

En icke-noll fraktionsförlust bevisar därmed inte att avsedda tillförlitliga
pixlar användes: den felaktiga vägen kan också ge en sådan förlust. Samtidigt
saknas tillräcklig historisk koppling mellan körning och exekverad kod för
att säkert tillskriva var och en av de sex checkpointfilerna detta kodfel.

Scoringpolicyn lämnas oförändrad: **24 unverified, 4 ineligible, 0 eligible**.
Maskfyndet är ytterligare ett konkret krav i granskningen av målens ursprung.
Före ett positivt beslut behövs historiskt körd källa/runtime, målfilsidentitet,
pixelmaskernas faktiska användning och observationsroller. En korrigerad
träning blir ett nytt versionsbundet experiment; inga gamla resultat
omklassificeras eller körs om automatiskt.

Kvarstår även separat: frysta splitters punkt-/geografiöverlapp, historiska
foldtilldelningar och träningsmedverkan. En korrekt mask löser inte dessa frågor.

## Evidens

Kampanjarkivet innehåller även
`e01-fraction-mask-regression.before.xml`,
`e01-fraction-mask-regression.after.xml`,
`e01-fraction-mask-review.md` och `e01-fraction-mask-verification.json`.
De tidigare checkpoint- och loggunderlagen är oförändrade. Se även
[scoringgrinden](fm_e01_fraction_gate.md) och
[observationsrollerna](fm_e01_observation_roles.md).
