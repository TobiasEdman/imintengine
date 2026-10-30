# ImintEngine — Backlog

Lättviktig löpande lista över kända, ej-färdiga åtgärder. Kompletterar
`docs/conversation_log/` (kronologisk historik) med en *åtaganden-vy*
som ska gå att skanna på 30 sekunder.

**Inte ersättning för** stories/PR-discussions/agile-sprint-planering —
ImintEngine är ML-pipeline/forskning, inte produktutveckling. Det
finns ingen PO, ingen personas, inga sprintar. Den här filen är ett
*git-versionerat index* över småskaliga uppgifter som annars riskerar
glömmas mellan sessioner.

---

## Konventioner

**ID:** `IM-NNN` (ImintEngine, nollpaddad löpnummer). Välj nästa lediga
genom `grep -oE "IM-[0-9]+" docs/backlog.md | sort -u | tail -1`.

**Status:**
- `open` — inte påbörjad
- `in-progress` — aktiv branch eller pågående utredning
- `blocked` — väntar på extern händelse (annan PR, jobb-completion)
- `done` — landad i main; flyttas till "Done"-sektion nedan
- `wontfix` — beslutat att inte göra; flyttas till "Done" med motivering

**Typ:**
- `bug` — felaktigt beteende (kosmetiskt / funktionellt)
- `tech-debt` — kod fungerar men borde vara renare
- `infra` — k8s, PVC, scheduling, secrets
- `housekeeping` — committs, branch-merges, städning
- `investigation` — diagnostik utan färdig åtgärd
- `gate` — väntar på annan färdigställning

**Commit-referens:** prefixa commit-meddelande med ID, t.ex.
`IM-001: trainer writes tile_counts to training_log.json`. Då fungerar
`git log --grep="IM-001"` som leverans-historik per item.

**Status-uppdatering:** manuell. Den som mergar en PR (eller flyttar
en blocker) uppdaterar tabellen i samma PR där det är möjligt.

---

## Active

| ID | Titel | Typ | Status | Branch/PR | Anmärkning |
|---|---|---|---|---|---|
| IM-001 | Trainer skriver `train_tiles`/`val_tiles` till `training_log.json` | tech-debt | open | — | Förebyggande fix; just nu rekonstruerar dashboard-podden via MD5-split |
| IM-002 | Dashboard `<title>` + `<h1>` säger "19-Class" — uppdatera till "23-Class" | bug | open | — | Kosmetiskt; v5-schema är 23 klasser |
| IM-003 | Null-safe-fix för `train_tiles.toLocaleString()` till repo (just nu bara server-side) | tech-debt | in-progress | server-side i `k8s/training-dashboard-server.yaml` | Lyft till `dashboards/unified_training_dashboard.html` i nästa dashboard-PR |
| IM-004 | `nodeAffinity p02r08srv01` på dashboard-pod är skör — byt till `podAffinity` mot training-pod | infra | open | — | Bryts vid nästa H100-nodbyte. Bättre: colocate-by-label med aktiva training-pods |
| IM-005 | ~~Efter PR #13 merge: ta bort server-side patcher~~ | housekeeping | **done** 2026-05-21 | edit i `training-dashboard-server.yaml` | Patch-blocket borttaget; pod var crash-loop före (bakgrund-filter regex matchade 0). Bara sed null-safe (IM-003) kvar som server-side |
| IM-006 | Merga `feat/dataset-clay-croma-emission` | gate | blocked | blocked-on: Clay/CROMA training end-to-end validation | Branch innehåller `s2_clay`/`s2_croma`-keys i `UnifiedDataset`. Smoke PASS, men ej kört ett tränings-batch än |
| IM-007 | Råg → vete asymmetri 24.3% — diagnostisera och åtgärda | investigation | **findings-in** | — | 600M klar: råg-IoU 0.333 vs 300M:s 0.381 (REGRESSION). 600M bloat 1.71× vs 300M:s 1.25×. Slutsats: större modell hjälper INTE (allokerar params till dominerande klasser). Åtgärd: bumpa class weights / oversample → IM-017 |
| IM-008 | CLASS_NAMES single source of truth — just nu duplicerat i `imint/training/unified_schema.py`, `dashboards/unified_training_dashboard.html` | tech-debt | open | — | Föreslå: dashboard hämtar namn från `data.config.class_names` om trainern börjar exportera dem (kopplat till IM-001). Server-side dup borttagen sedan IM-005 |
| IM-009 | Otrackade k8s/-YAMLs (audit-rededge, probe-h100, train-prithvi-{300m,600m}, smoke-dataset, inspect-checkpoints, training-dashboard-server, inference-compare-raw-and-superpixel) | housekeeping | open | — | Alla körda + verifierade under sessionen 2026-05-20/21 |
| IM-010 | `trainer.py:1000-1009` sparar minimal config till checkpoint — saknar `backbone_name` | tech-debt | open | — | Tvingade inferens-scriptet att infera backbone från `pos_embed` embed_dim (1024=300M, 1280=600M). Fix: lägg till `backbone_name` + `backbone` + `patch_size` i checkpoint config-dict |
| IM-011 | `data/viz_tiles/` saknades i k8s git-clone (gitignore) — krävde `git add -f` på 18 MB fixtures | tech-debt | **partial-done** | commit `00b7cc7` på `fix/inference-comparison-patch-size` | Långsiktig fix: PVC-baserad fixture-distribution istället för repo-bloat |
| IM-012 | `/compare/` HTML: NIR-CIR / GT / NMD-kolumner visas som broken images | bug | open | — | viz_data5.json base64-strings rendererar inte korrekt. Modell-prediktioner OK |
| IM-013 | SLIC superpixel post-processing visuellt VÄRRE än raw | investigation | open | — | Förväntat: spektral-edge-snappning. Faktiskt: blockigare/grövre än raw. Möjliga orsaker: fel `--superpixel-segments`, fel input-format (6-band reflectance vs RGB), eller behöver `--cleanup` i kombination |
| IM-014 | Modell-meta i Compare-HTML: `mIoU ?` istället för faktiskt värde | bug | open | — | `_metric` field i prediction-JSON är `"?"` istället för 0.4716/0.4769. Fixa i `inference_comparison.py` eller `build_comparison_html` |
| IM-015 | Inference-fix-branch `fix/inference-comparison-patch-size` ej PR-öppnad | housekeeping | open | branch pushad, 3 commits (`c117ac3` + `00b7cc7` + `5e7860d`) | Öppna PR när compare verifierats end-to-end (görs nu) |
| IM-016 | Klassvikt-rebalansering: sänk över-viktade rare crops, höj lövskog | investigation | open | — | Direkt åtgärd för IM-007 findings. Föreslag: bete 2.10 → 1.5, potatis 3.02 → 2.0, majs 5.0 → 3.5, råg 4.11 → 3.5 för 600M. Switch till effective-number weighting |
| IM-018 | Repo-isolation för parallella workstreams: utvärdera monorepo + submodules vs separate repos | investigation | open | — | 2026-05-22 incident: bf18671 (aircraft_parallax GIF) läckte in i `fix/refetch-late-autumn-frames` via worktree-mess mellan parallella agent-sessioner. PR #16 visar nu en ovidkommande commit. Worktree-isolation (intended boundary) räcker inte. Alternativ: (1) status quo monorepo + striktare conventions, (2) separate repo per workstream, (3) monorepo + git submodules som hård gräns. Beslutas vid nästa retro |
| IM-021 | Showcase-sajten scrollar i sidled vid 375 px | bug | open | — | Headerns rad med sex `theme-tab` är 569 px bred, och `.sub-tab-nav` radbryter inte, så undersidornas knappar blir en rad per knapp. Fanns före radarundersidan (task 0104); uppmätt 2026-10-10 med headless Chromium. Fix i den delade `docs/css/styles.css` (wrap/scroll för nav), så alla flikar berörs |
| IM-022 | Ny källa för SKS-auxkanaler (markfukt, grunddata, trädhöjd) när nya tiles behöver dem | infra | **done** 2026-10-13 | #74 (task 0107, merge `2e34ff9`) + #78 (spegel stagad, mosaikfix, paritet) | Spegeln stagad på PVC 2026-10-13 01:04Z (Job `skg-open-stage` + `-r2`, VRT ombyggda från d6b60cc): 174 filer, 97,3 GiB rå + 79,2 GiB COG, 10 VRT, `MANIFEST.json`; paritet mätt nationellt mot proxy-hämtade cache-arrayer (`k8s/skg-open-parity-job.yaml`, siffror och tolkning i `docs/data/skg_open_mirror.md`): grunddata ≥ 99,75 % identiska pixlar (nfi2024-referenser ≥ 99,97 %), markfukt 35,8 % identiska / MAE 5 (trolig modellversionsskillnad, hypotes). Separata beslut (Tobias): trädhöjd (329 GiB rå, 1 m, reproducerar inte lagrad kanal; Jobbet stödjer `STAGE_PRODUCTS=tradhojd`) och keep/scrap av `raw/` efter berikning. Implementerad i #74 (task 0107, merge `2e34ff9`, 2026-10-12): markfukt-, grunddata- och trädhöjdsfetcharna läser en spegel av de öppna per-läns-zipparna via `imint/training/skg_open.py` (se `docs/data/skg_open_mirror.md`) och kastar `SkgMirrorMissingError` när spegeln saknas på disk. Ingen berikning av tiles är gjord än (census-sessionens pass). Historik (2026-10-12, före #74): kartor.skogsstyrelsen.se-proxyerna pensionerades 2026 (302 → skogenskarta / WAF 403); fetcharna kastar nu `SkgEndpointRetiredError` (task 0106). Befintliga tiles har kvar sina aux-kanaler. Öppen ersättare utan inloggning: per-läns-zip på geodpags.skogsstyrelsen.se/geodataport/data/ — `sksSLUMarkfuktighet<län>` (21 filer, 86,5 GB, 2 m uint16, nodata 65535, ej tilad), `sks_<Volym|Grundyta|Medeldiameter>_omdrev<1-3>_<län>` (~18 GB), trädhöjd `sksTradhojd<län>` (353,5 GB) eller `sks_Tradhojd_omdrev3` (9 län, 40,5 GB). Fjärrläsning ur zip fungerar inte (deflate + strips) → engångsspegling till PVC i 10 m + VRT, kodningsmappning verifierad mot befintliga tiles, grunddata-mosaik senaste omdrev. Görs först när en kampanj behöver nya tiles (beslut Tobias 2026-10-12) |
| IM-024 | Årskartor för de tio största kommunerna, alla TESSERA-år | infra | in-progress | task 0128 | Spår 1: få ut demon med befintlig modell, de tio kommunerna och samtliga publicerade TESSERA-år. Kiruna tätort kvar som extra demo. Demon prioriteras; båda spåren körs parallellt enligt Tobias 2026-10-29 |
| IM-025 | 2026 för demoområdena med Prithvi-600M och en 2016-komposit per kommun | infra | proposed | — | Plan: [`plans/lclu_2026_prithvi600.md`](plans/lclu_2026_prithvi600.md). Ingen befintlig 600M-checkpoint läser 2016-ramen och alla kräver HR-VPP för året (saknas för 2025 och 2026): kräver omträning utan VPP, armar A/B (+ kontroll C0). Beslut väntar hos Tobias; inget kört |
| IM-023 | Behåll barrblandskog separat från lövskog i nästa modell | bug | in-progress | task 0127 | Spår 2 parallellt med IM-024: utred mappningens ursprung och testa separerbarhet med befintliga vikter/resultat före schema- och träningsbeslut. Se planen nedan; Tobias 2026-10-29 |

---

## IM-024 — Tio kommuner och samtliga TESSERA-år

**Typ:** infra · **Status:** in-progress · **Planering:** spår 1, demo prioriterad.
**Källa:** Tobias, 2026-10-29: de tio kommunerna för alla TESSERA-år;
senaste mål: färdigställ demon för ImintEngines showcase. Spår 1 och 2 körs parallellt; demon behåller prioritet.

**Ordning:** leverera kommun- och årsutbyggnaden med den befintliga
verifierade modellen och dess nuvarande klasschema. Behåll förklaringarna
av sammanslagna klasser i legenden och proveniensen. IM-023:s historik-
och separerbarhetsanalys körs parallellt i en separat arbetskopia. Demon
väntar inte på klasschema-/modellbeslut eller nästa modellträning.

**Omfattning:** de tidigare utvalda tio största kommunerna efter folkmängd:
Stockholm (0180), Göteborg (1480), Malmö (1280), Uppsala (0380),
Linköping (0580), Västerås (1980), Örebro (1880), Helsingborg (1283),
Jönköping (0680) och Norrköping (0581). Urvalet finns sparat med SCB:s
befolkningsreferens 2025-12-31 i
[`config/lclu/stockholm.json`](../config/lclu/stockholm.json),
`boundary_source.municipalities`. Hela kommunernas områden ingår.
Kiruna tätort behålls som ett separat extra demoområde utöver dessa tio.

- [x] Inventera samtliga publicerade TESSERA-år mot ett pinnat
  producentregister. Nuvarande verifierade registerintervall är 2017–2025;
  bekräfta år och täckning per kommun, inklusive nödvändig randzon.
  Utbyggnadens mål är alla år, inte enbart den nu verifierade 2024-demon.
- [ ] Skapa en täckningsmatris kommun × år för TESSERA och HR-VPP per år. Övriga stöddata (DEM, trädhöjd,
  skogliga grunddata och markfukt) delas spatialt mellan alla år. Fyll tillgängliga källuckor och redovisa
  kvarstående hinder uttryckligen. Ett annat års data får inte ersätta
  det begärda året; enbart en post i TESSERA-registret kvalificerar inte
  en färdig årskarta.
  *2026-10-30:* matrisen finns i
  [`coverage-matrix.md`](verification/lclu-service/coverage-matrix.md). Före
  luckfyllnaden täckte TESSERA alla 99 celler men HR-VPP bara 30. Tobias
  godkände luckfyllnaden ([`vpp-gapfill-plan.json`](verification/lclu-service/vpp-gapfill-plan.json):
  133 MGRS × år-par via WEkEO, utan CDSE PU) och `k8s/lclu-vpp-gapfill-job.yaml`
  kördes samma dag. Därefter är **88 av 99 celler källklara: alla elva områden
  2017–2024**. 2025 saknar publicerad HR-VPP i hela kampanjen och står kvar som
  uttryckligt hinder; inget annat år ersätter det.
- [ ] Förbered förlustfria kompakta TESSERA-COG:ar och verifiera deras
  paritet med originalen. Återanvänd befintlig cache och deduplicera block
  som delas mellan kommuner. Läs aux-fönster vid behov från kvalificerade
  källor och cachelagra förberedda indata och resultat enligt demoupplägget.
- [ ] Dimensionera lagring, konvertering, samtidighet och drift för hela
  omfattningen. Uppdatera ICE-kostnadsunderlaget med engångsarbete och
  löpande kostnad innan den utökade driftsättningen.
- [ ] Utöka områdes- och årsväljaren samt API:ts katalog och routning.
  Aktivera varje kommun/år först när källor och modellkörning verifierats;
  visa kvarstående täckningsbrister tydligt.
- [ ] Verifiera hela kommun × år-matrisen, med skarpa provkörningar,
  årskorrekt proveniens och kontroller vid kommun-, block- och UTM-gränser.
  Redovisa temporal modellprestanda separat från teknisk datatillgång.
- [ ] Behåll NMD-anpassad legend, klassinformation vid klick, 100 %
  opacitet som standard, reglage och skarp pixelvisning vid zoom 15 och
  högre. Export och cache ska identifiera både modell- och schemaversion.

Underlag: [tjänst och källkrav](lclu-service.md),
[lagringsmätningar](tessera-geotiff-storage.md) och
[befintlig ICE-demo](../docker/lclu/README.md).

---

## IM-023 — Separera barrblandskog från lövskog

**Typ:** bug · **Status:** in-progress · **Planering:** spår 2 parallellt med IM-024.
**Källa:** Tobias, 2026-10-29: barrblandskog ska behållas separat och inte
slås ihop med lövskog. Befintliga vikter och resultat ska ge underlag för
vilka klasser som kan separeras eller lämpligen redovisas tillsammans.

**Känt:** Nuvarande `NMD2023_TO_UNIFIED` mappar 113 (barrblandskog på
fastmark) till samma klass som 115–117 (lövskogstyper). Kodkommentaren
beskriver grupperingen som ärvd från NMD2018 och bevarad för kompatibilitet,
inte för korrekthet. Kommentaren är ett spår att följa, inte bevis för hur
det ursprungliga beslutet fattades eller vilka etiketter R4 faktiskt använde.

**Öppet:** Klassens faktiska svårighetsgrad, sammanslagningens ursprung och
om en kommunikationsmiss bidrog. Ingen orsak eller ansvarsfördelning är
fastställd; användarens kontroll ska inte antas ha brustit.

**Verifierat 2026-10-29 (task 0127):** Historiken visar en felaktig tidig
basmappning och ett senare kompatibilitetsarv; inget belägg fastställer en
kommunikationsmiss eller ansvar. R4:s faktiska skogsetiketter förfinades
med en fältdatabaserad modell, så basmappningen beskriver inte ensam R4.
Modellen har fortfarande ingen separat barrblandskogsklass. Ett begränsat
interiörprov på 24 historiska träningstiles stöder att 113 och 115–117 kan
skiljas i representationen, men avgör inte separation mot tall/gran eller
prestanda på nya områden. Provet använder 2022-egenskaper mot fast NMD2023;
åldersskillnaden och urvalet begränsar slutsatsen. Nästa beslut kräver bredare
geografisk prövning, särskilt 113 mot 111/112/114, och orört sluttest.
Legendens benämningar korrigeras till Lövskog/Blandskog med en uttrycklig
modellklassnot; detta innebär ingen ny modell eller schemaändring.

**Spår A — Hur uppstod sammanslagningen?**

- [ ] Följ NMD2018-/NMD2023-mappningen genom git-historik, PR:er och
  tillgängliga beslutsanteckningar. Skilj ett uttryckligt klassbeslut från
  en implementeringsmiss eller ett kompatibilitetsarv. Redovisa belägg och
  luckor; avsaknad av dokumentation fastställer inte en orsak.
- [ ] Verifiera de faktiska träningsetiketterna och labelbyggets version
  för R4 och berörda tidigare checkpointar mot råkällorna. Dokumentera
  vilka modeller som påverkats; dra inte slutsatsen enbart från dagens kod.

**Spår B — Kan modellerna skilja klasserna åt?**

- [ ] Inventera och återanvänd befintliga vikter, förväxlingsmatriser,
  sannolikheter och modellegenskaper där de finns. Analysera förväxlingar
  per modell, klass och geografiskt område på samma jämförbara underlag.
- [ ] Börja med barrblandskog/lövskog: använd separata referensetiketter
  och testa en liten klassificerare på frysta modellegenskaper före
  beslut om full omträning. Den sammanslagna slutklassen kan inte ensam
  visa om den interna representationen fortfarande skiljer klasserna åt.
- [ ] Använd geografiskt separerade utvecklingsdata och redovisa antal
  exempel, klassvis IoU, precision/recall samt osäkerhet. Behåll sluttestet
  orört under schema- och modellval. NMD-etiketter mäter överensstämmelse
  med NMD; använd oberoende fältreferenser där de stöder klassdefinitionerna.
- [ ] Skilj bristande separerbarhet från etikettfel, få exempel och
  otillräckliga indata. Hög förväxling är inte ensam skäl för sammanslagning.
  Redovisa om underlaget stöder separata klasser, behov av bättre data eller
  en grövre redovisningsnivå. En högre poäng efter sammanslagning är inte
  i sig belägg för en bättre modell.

**Beslut och nästa modelliteration**

- [ ] Sammanställ spår A och B separat före schema- och träningsbeslut.
  Dokumentera varje ändring av klassernas betydelse, motiv, underlag och
  uttryckligt beslut med användaren. Bevara de ursprungliga klassernas
  identitet även om en grövre presentationsnivå används.
- [ ] Inför ett versionerat klasschema där barrblandskog har en separat
  målklass från lövskog och lövblandad barrskog (NMD 114). Uppdatera
  labelbygget och lägg regressionstest som hindrar att 113 och 115–117
  åter slås ihop. Dokumentera hanteringen av motsvarande våtmarksklasser.
- [ ] Återskapa berörda tränings- och utvärderingsetiketter från råkällorna
  och genomför den träningsinsats som separerbarhetstestet motiverar:
  nytt klassificeringshuvud, finjustering eller bredare omträning.
  Spara schemaversion,
  klasslista och etikettproveniens i checkpointen; redovisa hur äldre
  checkpointar kan användas vid initiering.
- [ ] Utvärdera barrblandskog och lövskog separat, med klassvis IoU,
  precision/recall och förväxlingsmatris, innan den nya modellen tas i bruk.
- [ ] Uppdatera legend, pixelinformation och exporter tillsammans med
  modellversionen. En namnändring skapar inte en separat barrblandskogsklass;
  befintliga R4-resultat behåller sina numeriska ID:n och sin modellbetydelse.

Bakgrund och verifierad klassmappning:
[NMD-legendkontroll](verification/lclu-service/nmd-legend.md).

---

## Done

| ID | Titel | Typ | Status | Branch/PR | Anmärkning |
|---|---|---|---|---|---|
| IM-019 | `save_tabbed_report`: odefinierad `grazing_prefix` | bug | done | [#76](https://github.com/TobiasEdman/imintengine/pull/76), `19b4c9b` | Fallback-grenen läser `grazing_dir/bands/bands_meta.json` när `grazing_meta.json` saknas. Verifierad rapportgenerering och regressionstester. |
| IM-020 | Bakgrundsväxlingen använder odefinierad JS-variabel `bg` | bug | done | [#76](https://github.com/TobiasEdman/imintengine/pull/76), `19b4c9b` | RGB-panelen använder `activeBg`; växling och opacity verifierade i webbläsare. |

---

## Källor

Items i tabellen ovan kommer från:
- Live session-arbete 2026-05-20 / 2026-05-21 (initial seed)
- `docs/conversation_log/*.md` — ärver "## Next Steps"-sektioner när någon
  hittar tid att skörda dem hit
- Savant-review / `/review` output — säkerhetsbuggar och tech-debt-fynd
- PR-discussions där en sidoupptäckt inte ryms i nuvarande scope
