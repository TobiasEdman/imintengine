# Claude Code Instructions — ImintEngine

## RAG-system: des-agent — OBLIGATORISK VID SESSIONSSTART

**KRITISKT: des-agent RAG MÅSTE vara igång innan arbete påbörjas i detta repo.**
Agenten MÅSTE starta dessa tjänster vid varje sessionsstart, utan att bli ombedd.

```bash
# Docker binary (macOS Docker Desktop):
export DOCKER=/Applications/Docker.app/Contents/Resources/bin/docker

# 1. Starta Qdrant (des-agents vektordatabas)
cd /Users/tobiasedman/Developer/des-agent
$DOCKER compose up -d
# Verifiera: curl -s http://localhost:6333/healthz

# 2. des-agent CLI
/Users/tobiasedman/Developer/des-agent/.venv/bin/des-agent query "test"
```

**Om vi även arbetar med space-ecosystem-v2 (KG), starta också Neo4j:**
```bash
cd /Users/tobiasedman/Developer/swedish-space-ecosystem-v2
$DOCKER compose up -d
# Verifiera: curl -s http://localhost:7474
```

### Använd vid kodändringar

Innan du implementerar en ändring, fråga des-agent om kontext:

```bash
des-agent query "hur fungerar [det du vill ändra]?"
```

För större ändringar, använd multi-agent planering:

```bash
des-agent plan "beskriv din ändring här"
```

### Efter commits — uppdatera index

```bash
des-agent ingest --repo imint-engine
```

Detta sker automatiskt via post-commit hook om den är installerad.

## Repo-identitet

- **Namn:** imint-engine
- **Domän:** Geospatial satellitbildsanalys (Sentinel-2)
- **Nyckelgränssnitt:** `run_job(IMINTJob) → IMINTResult`, `BaseAnalyzer`, `ANALYZER_REGISTRY`
- **Beroenden:** Inga kodberoenden till andra repos. Syskonprojekt med space-ecosystem-v2.

## Schema — 20-klassers Unified Schema (v3)

Det enhetliga schemat (`imint/training/unified_schema.py`) slår samman NMD + LPIS-grödor + SKS-avverkning:

| Klass | Namn | Källa |
|-------|------|-------|
| 0 | background | — |
| 1–5 | tallskog, granskog, lövskog, blandskog, sumpskog | NMD |
| 6 | tillfälligt ej skog | NMD (uppdelat från blandskog) |
| 7 | våtmark | NMD |
| 8 | öppen mark | NMD |
| 9 | bebyggelse | NMD |
| 10 | vatten | NMD |
| 11–18 | vete, korn, havre, oljeväxter, vall, potatis, trindsäd, övrig åker | LPIS |
| 19 | hygge | SKS |

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

## Arkitekturregler

- Nya analyzers ska subklassa `BaseAnalyzer` och registreras i `ANALYZER_REGISTRY`
- Executors bygger `IMINTJob` och anropar `run_job()` — engine är executor-agnostisk
- Modifiera ALDRIG andra repos direkt härifrån
