# Integrationsförslag — ERA5-prefilter i fetch-pipelinen

**Status:** Förslag, ej implementerat. Skapad som komplement till
[REPORT.md](REPORT.md).

Detta dokument skissar hur ett ERA5-baserat dagsfilter skulle slottas in i
befintlig fetch-pipeline (`imint/training/tile_fetch.py`,
`imint/training/cdse_s2.py`, `scripts/fetch_unified_tiles.py`) **utan att ändra
befintliga signaturer**. Beslut om att faktiskt commita ligger hos användaren.

## Var i pipelinen

Idag, sett uppifrån:

```
fetch_unified_tiles.py
  └── per tile, per säsongsram:
        VPP-fönster → (date_start, date_end)            ← redan finns
        _fetch_single_scene(bbox, coords, date_start, date_end, ...)
          └── _stac_available_dates(coords, date_start, date_end)
                → [(date, cloud_cover), ...] sorterad på cloud_cover
          └── topp-N kandidater → DES / CDSE / openEO (race)
```

Nytt steg, lagt **mellan** VPP-fönsterval och STAC-search:

```
fetch_unified_tiles.py
  └── per tile, per säsongsram:
        VPP-fönster → (date_start, date_end)
        ───────────────────────────────────────────────
        ⬇ NYTT: era5_prefilter(bbox, date_start, date_end)
                → [date_a, date_b, ...]                  ← gleslista
        ───────────────────────────────────────────────
        för varje sub-fönster av kandidat-datum:
          _fetch_single_scene(bbox, coords, sub_start, sub_end, ...)
```

## Två integrationsnivåer

### Nivå 1 — Prefilter före STAC-search *(rekommenderad)*

Kör ERA5-prefiltret en gång per (tile, säsongsram), använd resultatet för att
**dela upp** den ursprungliga date-rangen i ett fåtal smala sub-rangear. Varje
sub-range skickas separat till `_fetch_single_scene`.

**Vinst:** STAC-anropen blir färre (många smala fönster ger ofta tomma svar
snabbt) och cloud-ranking-listan innehåller bara dagar vi tror är användbara.
Denna nivå motsvarar siffrorna i REPORT.md (−65 % STAC-träffar, median
cloud cover 65 % → 44 %).

**Kostnad:** Ett ERA5-anrop per tile per säsongsram. Med
`era5_aux.py`-mönstret (Polytope point-batch + cache) är det försumbart jämfört
med en STAC-search.

### Nivå 2 — Prefilter inuti `_fetch_single_scene` (filter på `candidates`)

Behåll signaturen oförändrad. Filtrera `candidates`-listan
(`(date, cloud_cover)`-tuples efter `_stac_available_dates`) genom att slänga
datum som inte passerar väderregeln.

**Vinst:** STAC-search körs ändå men antal asset-nedladdningar (CDSE/openEO/DES
nedan i racet) minskar. Cloud-cover-medel på det som faktiskt processas
förbättras.

**Kostnad:** Vi sparar inte STAC-anropet — bara asset-pull. Mindre vinst, men
också mindre invasivt.

## Föreslagen modul

Ny fil: `imint/training/era5_prefilter.py` *(skapas inte i detta showcase)*.

```python
# Skiss — implementeras inte här. Återanvänder Polytope-pathen i era5_aux.py.

from datetime import date

class ERA5DayFilter:
    """Identifies Sentinel-2 candidate days from ERA5 daily aggregates.

    Caches per-bbox-per-year on disk (npz). Same caching pattern as
    imint/training/era5_aux.py.
    """
    def __init__(
        self,
        cache_dir: str,
        precip_today_max_mm: float = 0.5,
        precip_prev2d_max_mm: float = 3.0,
        t2m_mean_min_c: float = 10.0,
    ): ...

    def good_dates(
        self,
        bbox_wgs84: dict,            # {"west", "south", "east", "north"}
        date_start: str,             # ISO
        date_end: str,
        *,
        return_segments: bool = True,
    ) -> list[tuple[str, str]]:
        """Return list of (sub_start, sub_end) ISO date ranges.

        Each segment is a contiguous run of "good" days. If return_segments=False
        returns flat list of ISO dates instead.
        """
```

## Anrops-skiss i `tile_fetch.py`

Konceptuell diff — visar bara *var* anropet skulle ligga, inte färdig kod:

```python
# Pseudo-kod, INTE applicerad på tile_fetch.py

def fetch_seasonal_frame(bbox, date_start, date_end, ...):
    coords = bbox_3006_to_wgs84(bbox)

    # NYTT — endast om CLI-flaggan är på
    if config.use_era5_prefilter:
        segments = ERA5_FILTER.good_dates(coords, date_start, date_end)
        if not segments:
            # Inget väderfönster — fall tillbaka till hela rangen,
            # eller skippa ramen helt beroende på policy
            segments = [(date_start, date_end)]
    else:
        segments = [(date_start, date_end)]

    for sub_start, sub_end in segments:
        scene, used_date = _fetch_single_scene(
            bbox_3006=bbox,
            coords_wgs84=coords,
            date_start=sub_start,
            date_end=sub_end,
            ...
        )
        if scene is not None:
            return scene, used_date
    return None, ""
```

CLI:

```
python scripts/fetch_unified_tiles.py \
    --use-era5-prefilter \
    --era5-precip-today-max 0.5 \
    --era5-precip-prev2d-max 3.0 \
    --era5-t2m-mean-min 10.0
```

Defaults är ERA5-filter **av**. Existerande beteende oförändrat.

## Hänsyn till våra regler

| Regel (CLAUDE.md) | Hur förslaget hanterar den |
|---|---|
| **§3 Running code is read-only** | Förslaget är ny modul + ny opt-in flagga. Inga ändringar i existerande `tile_fetch.py`-kod-paths som ett pågående jobb skulle vara beroende av. |
| **§Cacha alltid mellanresultat** | ERA5-prefiltret cachar per bbox+år till disk (samma mönster som `era5_aux.py`). Aldrig in-memory only. |
| **§Verifiera varje steg** | Verifierbart med replicate_metafilter.py-typ av jämförelse: kör en tile med och utan flaggan, jämför STAC-träffar och cloud cover på resultatet. |
| **§ALDRIG blanda fetch och label-logik** | Prefiltret ligger i fetch-vägen, hanterar bara spektral-kandidatdatum. Rör inte `build_labels.py` eller NMD/LPIS. |

## Inte i scope

- **Tuning av tröskelvärden.** Ska göras separat genom att svepa över en uppsättning AOI/år och plotta recall (andel cloud-free dagar bevarade) mot precision (andel kvarvarande dagar som är cloud-free).
- **Ersätta cloud cover-ranking.** ERA5-prefiltret kompletterar — det rankar inte bland kvarvarande dagar. Den befintliga `cloud_cover`-rankningen i `_stac_available_dates` är fortfarande primär ordning inom varje sub-segment.
- **Bryta CDSE-fallback-kedjan.** STAC → CDSE → DES-fallbacks i `_fetch_single_scene` påverkas inte. Prefiltret bara begränsar vilka datum som kommer in i den kedjan.

## Nästa steg om vi vill commita

1. Bygg `imint/training/era5_prefilter.py` baserat på `era5_aux.py` Polytope-koden.
2. Lägg till opt-in CLI-flagga i `scripts/fetch_unified_tiles.py`.
3. Verifieringskörning: 5–10 tiles, 2 år (2022 torrt, 2023 blött), jämför mot kontrollkörning utan flaggan. Mät STAC-träffar, asset-pull, slutlig cloud cover.
4. Om verifieringen håller siffrorna från denna showcase: aktivera som default, dokumentera i CLAUDE.md.
