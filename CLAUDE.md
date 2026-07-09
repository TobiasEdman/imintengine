# Claude Code Instructions — ImintEngine

## Repo-identitet

- **Namn:** imint-engine
- **Domän:** Geospatial satellitbildsanalys (Sentinel-2)
- **Nyckelgränssnitt:** `run_job(IMINTJob) → IMINTResult`, `BaseAnalyzer`, `ANALYZER_REGISTRY`
- **Beroenden:** Inga kodberoenden till andra repos. Syskonprojekt med space-ecosystem-v2.

## Schema — 23-klassers Unified Schema (v5)

Det enhetliga schemat (`imint/training/unified_schema.py`) slår samman NMD + LPIS-grödor + SKS-avverkning:

| Klass | Namn | Källa | SJV-koder |
|-------|------|-------|-----------|
| 0 | bakgrund | — | — |
| 1–5 | tallskog, granskog, lövskog, blandskog, sumpskog | NMD | — |
| 6 | tillfälligt ej skog | NMD | — |
| 7 | våtmark | NMD | — |
| 8 | öppen mark | NMD | — |
| 9 | bebyggelse | NMD | — |
| 10 | vatten | NMD | — |
| 11 | vete | LPIS | 4, 5, 307, 316 |
| 12 | korn | LPIS | 1, 2, 12, 13, 315 |
| 13 | havre | LPIS | 3, 10, 15 |
| 14 | oljeväxter | LPIS | 20-28, 38, 40-42, 85-88 |
| 15 | slåttervall | LPIS | 49, 50, 57-59, 62, 63, 302, 16, 80, 81 |
| 16 | bete | LPIS | 52-56, 61, 89, 90, 95 |
| 17 | potatis | LPIS | 45, 46, 70-72, 311 |
| 18 | sockerbetor | LPIS | 47, 48 |
| 19 | trindsäd | LPIS | 30-37, 39, 43 |
| 20 | råg | LPIS | 7, 8, 11, 14, 29, 317 |
| 21 | majs | LPIS | 9 |
| 22 | hygge | SKS | Avvdatum inom 5 år före tile-år |

**v5 ändringar:** Klass 21 ändrad från "övrig åker" (noise catch-all) till "majs" (spektralt distinkt C4-gröda). NMD cropland (raw 12) → bakgrund (0). Grönfoder (SJV 16,80,81) → slåttervall (15). Träda (SJV 60) → öppen mark (8). Skyddszon (SJV 66,77) → bakgrund. Omappade SJV-koder → bakgrund (0).

**SJV-koder är konsekventa 2018–2024.** Nya koder (7-9, 20-28, 45-47, 60) tillkom 2022 men gamla koder ändrades inte.
**LPIS rasteriseras med rå SJV-koder (uint16).** Mappning till unified sker i `merge_all()`.

## Multitemporal träning

Modellen tränas med 4 temporala ramar per tile:
- **Ram 0**: Höst (Sep–Okt) från *år-1* — stubble, höstsäd
- **Ramar 1–3**: VPP-styrda växtsäsongsramar (anpassade per tile-latitud)
- Backbone: Prithvi-EO-2.0 med `num_temporal_frames=4`
- Aux-kanaler: 11 st (träddata, DEM, VPP-fenologi, avverknings­sannolikhet)
- Input: `(4×6 + 11, H, W)` = 35 kanaler

## Dual-head arkitektur

Modellen har två output-huvuden från samma backbone:

1. **Head 1: LULC** — 20-klassers segmentering (softmax, focal loss)
2. **Head 2: Avverkningsmogen** — binär sannolikhetskarta (sigmoid, BCE loss)
   - Träningsdata: `n_mature_polygons > 0` från SKS avverkningsanmälan
   - Overlay på skogsklasser — visar vilka skogspixlar som är mogna för avverkning
   - Modellen lär sig spektral/aux-signatur för avverkningsmogen skog
   - Vid inferens: LULC-klass + avverkningsmogen-sannolikhet per pixel

## Nyckelmoduler — träning

| Modul | Syfte |
|-------|-------|
| `imint/training/unified_schema.py` | 20-klassers schemadefinition |
| `imint/training/unified_dataset.py` | Multitemporal dataset loader |
| `imint/training/tile_fetch.py` | Delad hämtningslogik (STAC → CDSE → DES fallback) |
| `scripts/fetch_unified_tiles.py` | Enhetlig 4-rams tile-hämtare (LULC + crop + urban) |
| `scripts/train_unified.py` | Träningsskript med `--enable-multitemporal` |
| `k8s/unified-training-job.yaml` | K8s-jobbspec för H100 |

## Dataregler — Tilehantering

- **Alla tiles i samma katalog.** LULC, crop och urban tiles blandas inte i underkatalog — allt ligger i en platt `unified_v2/`-katalog. Datasetet filtrerar/samplar internt.

## Datapipeline — 2 steg

### Steg 1: Fetch spektral (CDSE, CPU-pod, ~5h)
```bash
kubectl apply -f k8s/fetch-tiles-job.yaml   # fetch-lulc jobbet
```
- Hämtar 4-frame spektral från CDSE (höst + 3 VPP-ramar)
- Sparar rå data till `/data/unified_v2/` — INGEN label-remapping
- Raderar INTE source-tiles
- Skippar tiles som redan finns i unified_v2

### Steg 2: Build labels (CPU-pod, ~20min)
```bash
kubectl apply -f k8s/build-labels-job.yaml
```
- Bygger unified 20-class labels från scratch per tile
- Läser NMD-raster + LPIS-parquets + SKS-parquets (alla på PVC)
- Kör `merge_all()`: NMD-bas → LPIS-overlay → SKS-hygge
- Kör QC: nodata-filter (>5%) + frame-check (≥3/4) + class_stats.json
- Referensdata måste finnas på PVC: `/data/nmd/`, `/data/lpis/`, `/data/sks/`

### ALDRIG blanda fetch och label-logik
Fetch-scriptet (`fetch_unified_tiles.py`) hanterar BARA spektral.
Label-scriptet (`build_labels.py`) hanterar BARA NMD/LPIS/SKS → labels.
De är helt oberoende och körs i sekvens.

### VPP — ALLTID `VPP_SOURCE=wekeo` i fetch-yamls

`scripts/fetch_unified_tiles.py` kör VPP-prefetch via
`imint.training.cdse_vpp.fetch_vpp_tiles`. Default-routing (`_auto_fetch_vpp`)
är "cache-first med CDSE-fallback" — men på en cache-miss faller den
tillbaka till CDSE Sentinel Hub Process API, som **delar PU-pool med
spektralfetchen**. En större fetch mot ett delvis befolkat WEkEO-cache
dränerar därför CDSE-budgeten innan spektralfetchen ens börjat
(HTTP 403 "Insufficient processing units" → `credit_guard` markerar
hela CDSE som DEAD → spektralfetchen failar tyst → exit-0, 0 tiles).

Det här hände i `campaign-orphan-512` 2026-06-29 (commit `1927a1c`):
0/1147 tiles, silent exit-0. Diagnos i `feedback_vpp_source_wekeo_mandatory.md`.

**Regel — varje k8s-yaml som kör `fetch_unified_tiles.py` MÅSTE sätta:**

```yaml
env:
  - { name: VPP_SOURCE,    value: "wekeo" }      # cache-only, fail-loud on miss
  - { name: VPP_WEKEO_DIR, value: "/data/vpp_wekeo" }
envFrom:
  - secretRef: { name: wekeo-creds }              # WEKEO_USERNAME + WEKEO_PASSWORD
```

Med `VPP_SOURCE=wekeo` raises en cache-miss `RuntimeError` per tile
(synligt i loggen) istället för att tyst spendera PU. Cache-gaps fylls
med en separat `prefetch_vpp_wekeo.py`-körning (PU-fri, via WEkEO HDA)
INNAN spektralfetchen — aldrig genom att betala CDSE PU. Samma mönster
som `k8s/refetch-recoreg-vpp-job.yaml` (recoreg-kampanjen var 100%
PU-fri tack vare den här regeln).

Audit-status 2026-06-29: 12 av 13 `fetch_unified_tiles.py`-yamls
saknar dessa env vars (alla utom `campaign-orphan-512-job.yaml` efter
fix). De är inaktiva nu men måste retroaktivt patchas innan
återanvändning — annars upprepas drainen.

### DES `--workers` är **dynamiskt** — fråga teamet före bulk-fetch

Antalet samtidiga workers mot DES openEO (`openeo.digitalearth.se`)
styrs server-side av digitalearth.se-teamet och **varierar över tid**.
Att hårdkoda en siffra i fetch-yamls är fel mönster — för många
workers => andra worker:n köas tyst på serversidan (ingen
throughput-vinst, kanske 429), för få => suboptimal wall-time.

**Tidslinje:**
- **2026-06-15:** `--workers 2` var det säkra maxet (user-stated;
  bakad in i `campaign-phase1-des-recoreg-job.yaml` header som
  "HARD constraint").
- **2026-06-30:** teamet skruvade ner till **1 effektiv worker**
  (orphan-512 full-run observerade 14 tiles/h med `--workers 2`,
  dvs identiskt med single-worker → andra worker:n queued
  server-side). Datum-stämplad här.
- **2026-07-06:** user-bekräftat att allotment fortfarande är **1**
  ("det finns bara en worker på DES server sida") — recovery-körningen
  stannade på `--workers 1`. OBS: `--workers 2` mot allotment 1 var
  också en bidragande faktor till [408]-stormen 2026-07-05 (spektral +
  SCL-screen konkurrerade om samma enda server-slot).
- **2026-07-09:** user-meddelat: **"du kan öka till två workers på DES"**
  — allotment höjd till 2. Recovery-legget uppdaterat till `--workers 2`
  (OBS: SCL-screen kör också via DES sedan CDSE-402:an, så 2 workers =
  2 samtidiga strömmar totalt — matchar allotment 2 exakt; höj INTE
  workers utöver allotment igen, se [408]-stormen 07-05).
- **Framtid:** kan ramp:as upp till **~6** efter team-beslut.

**Regel:** innan en ny `fetch_unified_tiles.py`-körning **fråga
användaren / DES-teamet** vad aktuell allotment är. Sätt `--workers
N` till exakt det numret. Hardkoda inte `2` baserat på gamla
checkpoints (inkl. denna fil) eller äldre yaml-headers.

Verifiera live: per-tile loggraderna visar `DES: permits=N` —
matchar `_DES_SEMAPHORE.initial`. Reell concurrency = `min(--workers,
DES-server-allotment)`. Om wall-rate per worker är konstant oavsett
`--workers 1` vs `--workers 2` → server är bottleneck, sänk till
allotment.

### CDSE PU — **delad pool**, använd klokt

`imint.training.cdse_vpp._PU_POOL = "cdse"` är **delad** mellan tre
clients som alla räknas mot samma månatliga PU-budget:

1. **CDSE SH-Process API** (`--fetch-sources cdse` backend, plus
   per-tile aux/VPP fallthrough).
2. **CDSE openEO** (`--fetch-sources cdse-openeo` backend — separat
   protocol men samma PU-mätare server-side).
3. **VPP via SH-Process** (cdse_vpp auto-router cache-miss fallback;
   blockeras numera av `VPP_SOURCE=wekeo` per regeln ovan).

Drainen är **per session** — `credit_guard` markerar `cdse` DEAD vid
första HTTP 403, och då dör alla tre vägar för resten av poden.
Drainen är också **per månad** budgeten resetar månadsskiftet.

Princip: **DES tar bulk-spektral (fritt). CDSE PU reserveras för
saker DES inte kan göra.** Konkret:

- ✅ **Phase-2 sen2cor pre-2018 backfill** — `cdse` SH-Process eller
  `cdse-openeo`. M2-kapabilitet behövs inte (`--coreg-to-anchor`
  använder existing tile's ankare → 6-band/no-halo OK). Mätbar scope
  (orphan-Phase-2: ~546 pre-2018 tiles × 1-2 slots = ~600-1000 PU).
- ✅ **WEkEO-cache gap-fill** — engångs, mätbart per (mgrs, year).
- ✅ **Spot-fix** av enskilda DES-failade tiles (post-run cleanup).
- ⚠ **Bulk-spektral via cdse-openeo som PARALLELL BOOST** — acceptabelt
  när DES är server-side throttled så hårt att wall-timen blir en
  kampanjblockerare (t.ex. orphan-512 2026-06-30 med `DES: permits=1`
  → ~67h ETA för 1147 tiles). CDSE PU + DES parallellt på samma
  staging-dir (idempotent skip via `_valid_existing_tile`) halverar
  ungefär. Villkor: (a) PU-saldo verifierad, (b) scopet mätbart (t.ex.
  ~1000 tiles × 4 slots ≈ ~4000 openEO-jobs → jämför mot månadsbudget),
  (c) DES-jobbet lämnas kvar (kör vidare parallellt — inte swap).
  ALDRIG som primär när DES har kapacitet, och ALDRIG istället för
  Phase-2-sen2cor (använd GCS+l1c_sen2cor för sen2cor via
  `ghcr.io/tobiasedman/imint-sen2cor` — fritt).
- ❌ **VPP via SH-Process** — `VPP_SOURCE=wekeo` (cache-only) per
  regeln ovan; cache-miss raises hellre än drainar.

Datum-stämplad: **2026-06-29** orphan-512-runen dränerade hela poolen
i VPP-prefetch (yaml saknade `VPP_SOURCE=wekeo` — fix 64d9cab). Reset
2026-07-01 — nästa session som vill köra cdse-spektral måste först
verifiera PU-saldo (`fetch_vpp_tiles`-call returnerar HTTP 403 om
exhausted) och scopa körningen mot mätbar budget, aldrig bulk.

## Viktiga regler

- **Verifiera varje steg.** När du gör en transformation (flip, transpose, rotation), verifiera visuellt att resultatet är korrekt INNAN du applicerar på alla tiles. Gör INTE flera ändringar utan att kontrollera varje.
- **En ändring i taget.** Byt aldrig flera transformationer samtidigt — det gör det omöjligt att debugga.
- **Genomför instruktioner exakt.** Om användaren säger "applicera X" — gör exakt X, inte en approximation.

## Dataregler — Temporal matchning

- **Spektraldata och etiketter MÅSTE matcha per år.** En tile med LPIS-etiketter från 2022 ska ha Sentinel-2-spektraldata från 2022 (höstram från 2021). Blanda ALDRIG år mellan spektral och etiketter.
- **Ramstrategi:** 1 höstram (Sep–Okt, år-1) + 3 VPP-styrda växtsäsongsramar. Ingen fast månadsindelning — VPP-fenologi per tile styr ramfönstren.
- **SKS-årsmatchning:** SKS-avverkningsdata (2021–2026) måste överlappa med spektralets år. Tiles med 2018/2019-spektral kommer att sakna hygges-etiketter.
- **Refetch-mönster:** Vid omhämtning av spektral (`--mode refetch`), läs tile-året från befintligt `.npz` (`year`, `lpis_year` eller `dates`) och använd som primärt sökår.
- **Ingen årsfallback för grödor:** Tiles med LPIS-etiketter (crop) får ALDRIG falla tillbaka till andra år. Spektral måste matcha etikettåret exakt. Årsfallback är bara tillåtet för rena skog/vatten-tiles (NMD-klasser utan årsspecifika etiketter).

## Koregistrering — M1 → M2 → (M3)

Tre distinkta mekanismer, i denna ordning. **Relativ före absolut; helstacken, aldrig band-för-band.**

- **M1 — grid-snap (deterministisk, transform-baserad).** `imint.fetch._snap_to_target_grid`. Placerar varje frames *transform* på NMD:s 10 m-lattice. Per-scen, exakt, ej skattad.
- **M2 — relativ inter-frame koregistrering (ömsesidig information/MI på B04).** Tar bort den relativa orto-driften (~2 px) mellan ramar som M1 inte kan röra. MI (inte faskorrelation) eftersom ramarna spänner över säsonger — samma markpunkt ser radiometriskt olika ut, så intensitetskorrelation jagar fenologi, inte geometri. **Referens-ankare:** den klaraste ramen (`clearest_frame_idx`) är fast ankare; varje övrig ram registreras mot den med `estimate_mi_offset` och skiftas dit. Ankaret rörs aldrig — hela stacken landar på ankarets grid (ej centroid).
  - **Argordning + tecken är sign-bärande.** `estimate_mi_offset(moving, reference)` returnerar skiftet att **applicera på `moving`** (positivt tecken) för att registrera den mot `reference`. I `coregister_interframe` är `moving` den rörliga ramen och `reference` ankaret; i `coregister_to_reference` anropas `estimate_mi_offset(ref_band, cur_band)` och resultatet appliceras **positivt** på referensen (faskorrelations-vägen applicerade `−`). Fel argordning/tecken **förstärker** driften (buggen i `54b30a3`: känd +3,−2 px → +7,−4 px efter koreg).
  - **Testa koreg ALLTID med dot/center-of-mass** (entydig positionsmätning). Smooth-field-residual räcker INTE — den passerade på det inverterade tecknet. (Jfr `feedback_compensating_bugs`: reverse-fit är det skottsäkra testet — se `tests/test_coregistration.py::test_coregister_to_reference_removes_shift_dot_com`.)
  - Skatta-budget: inter-frame (`coregister_interframe`) = `CROP` (halo-bredden); `coregister_to_reference` (post-integer-align-residual) = `1.0` px. MI förkastar optimum ≥ `0.95·search_px`, så *applicerad* shift hålls halo-bunden.
- **M3 — absolut till NMD (klassgräns-edges).** ENDAST om mätning (dry-run + before/after-GIF + S2↦NMD-edge-korrelation) visar en kvarvarande per-tile-offset efter M1+M2. NMD är klasskoder → gradient-magnitud-*edges* är enda delade signalen mot S2-reflektans (ingen samregistrerad optisk referens finns). En helstacks-shift, efter M2 och före 520→512-croppen, confidence-guardad. NMD läses då **som geometrisk linjal, inte labels** — degraderar till no-op om rastret saknas (bevarar fetch/label-oberoendet).
- **Helstacken, aldrig band-för-band.** Skatta skiftet på *ett* ankarband (B04 — hög SNR, skarpa kanter) och applicera samma vektor på alla 12 band. Banden är redan inbördes samregistrerade av L2A; per-band-skift är brusigare på låg-SNR-band (B01/B09) och förstör inter-band-registreringen (kromatisk fransning). Geometrisk drift är en helbilds-egenskap.

## Arkitekturregler

- Nya analyzers ska subklassa `BaseAnalyzer` och registreras i `ANALYZER_REGISTRY`
- Executors bygger `IMINTJob` och anropar `run_job()` — engine är executor-agnostisk
- Modifiera ALDRIG andra repos direkt härifrån

### Återanvänd repo-egna pipelines — bygg inte om hjulet

Innan du skriver en ny lösning, sök i repot. Det här är en genomtänkt arkitektur, inte en samling lösa skript. Konkreta regler:

- **Sentinel-2-hämtning** → använd alltid `imint.training.optimal_fetch.optimal_fetch_dates(mode="era5_then_scl")` för att välja rena scener. Hardkodade datumlistor är förbjudna — Atmosfär-pipelinen är designad för att eliminera molniga scener före spektral-fetch.
- **Showcase-tabbar** → alltid `TAB_CONFIG[<key>]` i `docs/js/tab-data.js` + tom `<div class="tab-dynamic"></div>` i HTML. Mallen `renderTabDynamic` (`docs/js/app.js:88`) renderar paneler, opacity-sliders, bgToggle, legends och summary-cards konsekvent med övriga tabbar.
- **Showcasens visuella identitet (DES)** → ALLA sidor under `docs/` — tabbar OCH fristående `.html` (t.ex. `coregistration.html`) — ska följa den visuella identiteten i `docs/css/styles.css`. Länka `css/styles.css?v=2` + Space Grotesk; **vit bakgrund**, skogsgrön `#1A4338` (primär) / `#245045` (sekundär), mint `#cff8e4`-accent, `#171717` text, `#6b7280` grå, `#e5e7eb` linjer. Återanvänd klasserna: `.header` (vit topbar med `<span>IMINT</span>`-titel + `.theme-tab`-nav), `.summary-section`/`.summary-card`, `.testimonial` (callouts), `.section-byline`. Analysis-lager med kartöverlägg = TAB_CONFIG (Leaflet-paneler); metod-/info-sidor = fristående HTML, men ALDRIG eget tema/egen palett (off-brand `#27ae60`, mörk header etc.) — det bryter mot DES och mot resten av showcasen. (Lärdom 2026-06-09: koregistreringssidan byggdes först off-brand och fick göras om.)
- **Aux-channel-fetcher** → följ mönstret från `imint/training/skg_height.py`: `(west, south, east, north, *, size_px, cache_dir)`-signatur, EPSG:3006-bbox snappad till 10 m grid via `_to_nmd_grid_bounds`, `.npy`-cache med deterministisk nyckel.
- **Spektral-fetch** → `imint.fetch.fetch_seasonal_image` eller `fetch_des_data` — aldrig egna openEO-anrop.

När du undrar om en abstraktion finns: kör `Glob`/`Grep` först. Använd `Agent`-Explore om scope är osäkert. Duplicering kostar tid (för dig) och städning (för användaren).

## Docker- och processversionering — undvik repo-skew

Bakgrund: i commit `52d19ae` (`feat(water_quality): real ESA C2RCC + Pahlevan MDN`) committades 9 PNG-resultat utan att Dockerfile, SNAP graph-XML eller run-skript var versionerade. Sex månader senare gick det inte att replikera pipelinen för en ny AOI/period utan reverse-engineering. Se [governance-rapporten](docs/governance/avoiding_docker_repo_skew.md) för full analys.

### Per pipeline-image — tolv-punkts-checklista

För varje Docker-image som körs i en pipeline ska repot innehålla:

1. **`docker/<namn>/Dockerfile`** — bygger imagen från scratch utan host-beroenden.
2. **Pinnade FROM via digest, INTE bara tag.** `FROM mundialis/esa-snap@sha256:056f971...`, inte `:13.0.0` eller `:latest`. Tag-pinning räcker inte — tagen kan ompekas av registry-ägaren, eller existera bara i din fantasi (pause-incident 2026-05-07: `:13.0.0` fanns aldrig).
3. **Build-skript** (`build.sh` eller `Makefile`-target) som producerar samma tag-namn som körnings-skript förväntar sig.
4. **Run-skript** (`run.sh` eller Python-driver) — wrapper som tar input/output-paths och AOI-parametrar.
5. **Eventuella config-filer** (SNAP graph-XML, JSON-config, etc.) i samma katalog.
6. **README.md** i `docker/<namn>/` som beskriver bakgrund, build, kör-exempel, output-format.
7. **MANIFEST.json sidecar** i output-katalogen med `{image, image_digest, git_sha, run_args, input_data_hash, produced_at}` — så det går att binda PNG-filerna till exakt build + commit som producerade dem.
8. **Smoke-test som verifierar runtime, INTE bara att operatorer existerar.** Pause-incident 2026-05-07: `gpt -h | grep c2rcc.msi` lyckades på SNAP 9, men SNAP 9 saknade S2 product reader för 2025 SAFE-format. Smoke-tester ska minst verifiera: runtime-version (`cat /opt/snap/VERSION.txt`), att kärn-operatorerna i pipelinen finns (Read + c2rcc.msi), helst end-to-end mot en versionerad fixture-SAFE.
9. **k8s-yamls och `docker run`-anrop ska pinna till digest, INTE tag.** `image: ghcr.io/foo/bar@sha256:abc...`, inte `:latest` eller `:v1`. Mutable tags i pipeline = body-skifte.
10. **Lokal-image-vs-Dockerfile-reconciliation.** Om `docker images` visar en image som körs i pipeline (t.ex. `imint-snap13:latest`) men ingen Dockerfile i repot producerar den: STOPP. Bygg inte om från `docker history`-antaganden — det kan ge en helt annan image (mundialis SNAP 9 istället för ESA SNAP 13). Använd `docker save` + lager-inspektion eller bygg om från scratch + bevisa bit-ekvivalens.
11. **MANIFEST.json är obligatorisk för committade artefakter under `outputs/showcase/` och `docs/showcase/`.** Föreslagen lint: `tests/test_committed_outputs_have_manifest`.
12. **Governance-dokument ska ha motsvarande tester.** Varje regel i `docs/governance/*.md` ska ha en motsvarande `test_*` i `tests/test_repo_hygiene.py`. Otestbar regel = drift:ande regel (pause-incident 2026-05-07: `/usr/local/snap` och `:13.0.0` stod genomgående i docs men matchade aldrig den faktiska fungerande imagen).

Referens-implementation: [`docker/cloud-models/`](docker/cloud-models/) (cloud-detection-jämförelsen) + [`docker/c2rcc-snap/`](docker/c2rcc-snap/) (ESA SNAP C2RCC).

### Nolltolerans

- **Aldrig `:latest` i FROM eller `docker run`.** Använd explicita versioner.
- **`docker run X` i Python/shell kräver Dockerfile för X i repot.** Lint-test i `tests/test_repo_hygiene.py` blockerar PR där detta inte stämmer.
- **Output-artefakter får inte committas innan processen som producerade dem är committad.** Order: process → output, aldrig tvärtom.
- **Image-tag → git-SHA-mappning.** Ingår i MANIFEST.json eller commit-meddelandet som committar artefakter.
- **Lokalt-bara image får ALDRIG användas i en pipeline.** Om `docker images` har en image som inte har Dockerfile i repot — bygg om den från Dockerfile innan du använder den.

## Kodgranskningsstandard — obligatorisk vid alla kodändringar

**Agenten ska alltid följa detta arbetsflöde vid granskning och korrigering av kod:**

### 1. Identifiera
Granska koden och markera alla delar som är:
- Ineffektiva (onödiga loopar, dåliga datastrukturer, O(n²) när O(n) räcker)
- Redundanta (duplicerad logik, dead code, backward-compat-junk som aldrig används)
- Svårförståeliga (oklara variabelnamn, avsaknad av typannoteringar, magiska konstanter)
- Bryter mot god stil (PEP 8, explicit är bättre än implicit, YAGNI)

**Motivera varje problem:** vad är fel, varför är det problematiskt (prestanda / läsbarhet / underhållbarhet / säkerhet).

### 2. Ersätt
Skriv om till optimerad, robust, lättläst kod. Förklara varför lösningen är bättre.

```python
# Dåligt — indexering via range(len(...)):
for i in range(len(lst)):
    result.append(lst[i] * 2)

# Bra — direkt iteration, pythonisk, snabbare:
result = [x * 2 for x in lst]
```

### 3. Testa
Verifiera alltid förbättrad kod med explicita assertions eller pytest-tester. Visa att alla fall passerar.

**Testkrav:**
- Unit-tester räcker INTE ensamt. Testa alltid integrationen: att koden fungerar i sitt riktiga kontext (K8s, med riktig data, med riktiga API:er).
- Innan en ny feature deployas: verifiera att den fungerar end-to-end, inte bara att den parsar.
- Disk-I/O: testa alltid att data faktiskt sparas till disk och kan läsas tillbaka.
- Concurrency: testa med verklig belastning, inte bara mock-semaforer — verifiera att API:er inte throttlas.
- Cacha alltid mellanresultat till disk. Data som tar tid att hämta/beräkna får ALDRIG bara leva i minnet.

### Regler (nolltolerans)
- **Ingen slarv** — inga onödiga rader, ineffektiva algoritmer eller dålig kodstil accepteras
- **Inga backward-compat-shims** för schema-versioner som inte längre används (t.ex. 10-klassers NMD)
- **Explicit > implicit** — auto-detect-logik som ändrar beteende baserat på indatans max-värde är förbjuden
- **Motivera alltid** — varje ändring ska ha en tydlig motivering
- **All kod ska vara framtidssäkrad** — inget junk som "kan behövas senare"
- **Starta aldrig om jobb som förkastar redan klart arbete** — fråga alltid användaren först
- **Alla hämtade/beräknade data ska persisteras till disk** — aldrig bara i minnet

### Verifiera alltid ändringar (repo-specifik tolkning av global regel §6)

Varje icke-trivial ändring i detta repo måste avslutas med ett konkret verifieringssteg:

- Schema-ändringar: `pytest tests/test_schema.py -v` — 40+ regressionstester ska passera.
- Träningspipeline: kör en 1-epok smoke-test lokalt innan K8s-submit.
- Fetch-kod: kör mot en känd tile och jämför bit-likhet mot cachad referens.
- Dashboard/notebook: screenshot + rendera om — inga visuella regressioner.
- K8s-manifester: `kubectl apply --dry-run=server -f` före verklig submit.

Utan verifieringsartefakt: ingen commit. Se `~/.claude/CLAUDE.md` §6 + agentic_workflow `docs/lessons/external_patterns.md` §C2.
