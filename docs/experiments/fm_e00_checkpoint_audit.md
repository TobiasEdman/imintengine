# E00 — säker checkpointkontroll

Status 2026-09-17: användargodkänd CPU-körning avslutad efter 277 sekunder.
28/28 checkpoints är hashverifierade, säkert lästa och har finita vikter.
Råstatus är 9 passed / 19 failed: verifieraren hade en för strikt huvudregel
för R1–R3, och Prithvi-600M R4 saknar aktiverad fraktionsbehandling.
Korrigeringen i lokal kod är inte en omkörning eller omklassificering.
Användaren beslutar separat om ytterligare kostnadsdrivande åtgärder.
Task 0010 äger arbetet. Gemensam crop-verifiering är en separat ändring.

## Vad kontrollen visar

Alla 28 inventerade checkpoints (sju modeller × R1–R4) binds till full
SHA-256 och storlek. Träningsloggen måste matcha sin inventerade SHA-256.
Befintlig säker laddare kopierar filen till en privat snapshot och använder
alltid `weights_only=True`. Ingen modell byggs, och inga testdata öppnas.

Kontrollen undersöker tensorformat, NaN/Inf i vikter, best epoch, sparad
byggkonfiguration och kända huvudformer. Modellarmarnas antal ramar,
klassantal och cropstorlek kontrolleras också. Huvudets existens följer
checkpointens explicita flagga; ett avstängt huvud i R4 är en separat
protokollavvikelse. Huvudets existens bevisar inte fraktionssupervision.
Saknade historiska uppgifter blir `incomplete`; motsägelser blir `failed`.
NaN i per-class-metrik för klasser utan stöd är tillåtet.

Ett `passed` visar endast dessa kontroller. Det bevisar inte strict model
load, forward-pass, preprocessing, träningskodens identitet, dataläckage
eller modellkvalitet. Dessa delar återstår i E00/E01.

## Förbered och verifiera

Från repo-roten med projektets Python:

```bash
python -m pytest -q -p no:cacheprovider tests/test_ladder_checkpoint_audit.py
python -m scripts.prepare_checkpoint_audit_job \
  --inventory /absolute/path/checkpoint-identities.jsonl \
  --output /absolute/path/checkpoint-audit-job.json
kubectl --context icekube -n prithvi-training-default \
  apply --dry-run=server -f /absolute/path/checkpoint-audit-job.json
```

Generatorn skapar en immutable ConfigMap med exakt skript och inventering,
samt ett Job med namn härlett från deras gemensamma hash. Filerna monteras
med read-only subPath eftersom den säkra läsaren avvisar symlänkar.
Generatorn skriver inte över befintliga filer och skickar inget jobb.

## Godkänd och avslutad resursbudget

- En befintlig image pinnad till digest `25919e5c…dc3599`, källa
  `7d935b9a0210f9a88b641ef24e7d83206c2fdcb0`; inget nytt imagebygge.
- 4 CPU, 16 GiB RAM, 8 GiB temporär disk, ingen GPU.
- Högst 1 800 sekunder för jobbet, inga automatiska omförsök.
- Som mest 2 CPU-timmar och 8 GiB-timmar RAM vid full tidsgräns;
  detta är resursbudget, inte ett pris i kronor.
- Inventerade viktfiler: cirka 34,4 GiB. Sekventiell läsning med ett nytt
  barn-Python per checkpoint begränsar samtidiga minnesreferenser.
- Endast checkpoints-katalogen monteras från PVC, read-only. Resultat
  skrivs till privat arbetsyta och som en JSON-rad per cell i Pod-loggen.
- Jobbet/Podden behålls sju dagar. Efter godkänd start måste bevakningen
  spara slutlogg, Job-/Pod-status och faktiskt imageID lokalt före TTL.
  En misslyckad/avbruten körning får aldrig tolkas som full täckning.

Varje resultat anger inventerings-, auditor- och loader-hash samt
checkpoint-/logg-hash, Pod UID, image, Python/Torch-version, tid och peak RSS.
För komplett kontroll krävs exakt 28 unika celler och granskning av samtliga
statusar. Resultatet är en integritetsrapport, inte ett experimentutfall.

## Lokal verifiering och kvarvarande runtime-kontroll

Tester gör riktiga torch.save/läsningar och en separat CPU-process med
JSONL på disk. De täcker korruption, ändrade filer, symlänkar,
felaktiga inventeringar, metadatafrånvaro, antal ramar och huvudformer.
En oberoende granskare har kontrollerat nycklarna mot modellimplementationerna.

Lokal Python/Torch skiljer sig från den pinnade containern. Serverdryrun
validerar API/antagning men kör inte containern. Den första CPU-körningen
verifierade den säkra läsningen i klustermiljön och blottlade huvudregelns
fel. Korrigerad kod har inte körts på de verkliga checkpointfilerna.

Efter korrigeringen passerar 31 av 32 lokala tester. Ett subprocess-test
nådde 240 sekunders tidsgräns; samma CLI-fixture passerade separat med
JSONL på disk. En fristående diagnostik mätte 244,11 sekunder för enbart
Torch-importen. Lokal uppstartsseghet är därmed reproducerad, men hela
väntan i det misslyckade testet är inte lokaliserad. Oberoende kodgranskning
gav LGTM. Korrigeringen är ännu inte fullständigt verifierad eller
committad. Testfelet ändrar inte den arkiverade klusterkontrollens evidens.

## Terminal evidens

Kampanjrapporten sparas under Codex-artifactens campaign-katalog som
checkpoint-audit-terminal-report.json och checkpoint-audit-terminal-report.md.
Exekverad källkod, råa JSONL-resultat och Job/Pod-identiteter är arkiverade
separat från den korrigerade arbetskopian. Inga vikter eller historiska
resultat har ändrats; ingen omträning eller auditomkörning startades.
