# PR-förslag mot `DigitalEarthSweden/space-datalab-metafilter`

## Status — 2026-06-16

Tre PR:er pushade mot `DigitalEarthSweden/space-datalab-metafilter`:

| # | Branch | Innehåll | Status |
|---|---|---|---|
| [#1](https://github.com/DigitalEarthSweden/space-datalab-metafilter/pull/1) | `feat/extended-filters-and-s1-support` | Schema + operatorer + härledda kolumner + S1 + CDS-cloud (5 commits, 45 tester) | Öppen (manuellt 2026-06-02 efter att routinen failade) |
| [#2](https://github.com/DigitalEarthSweden/space-datalab-metafilter/pull/2) | `fix/setup-friction` | Pinade requirements, README `-m`-invokering, lazy openEO-cred-validation (3 commits) | Öppen |
| (väntar) | `feat/open-meteo-backend` | Open-Meteo som alternativ ERA5-backend (5 commits, 65 tester) | Schemalagd 2026-06-16 09:00 CEST, villkorlig på #1-merge |

Plus en samlingsgren för externa konsumenter:

- **`TobiasEdman/space-datalab-metafilter:try/extended-with-pinned-deps`** —
  setup-fix + extended features i en enda branch som kan klonas direkt och
  fungera utan väntan på upstream-merge.

**Lokal workspace:** `~/Developer/space-datalab-metafilter/`.

**Follow-up för ImintEngine** *(öppna trådar att hantera när DES-PR:erna
mergar)*:

1. När #1 mergar — uppdatera `imint/training/optimal_fetch.py` att konsumera
   DES-paketets filterprofiler istället för Stage 1-prefiltret in-house.
2. När open-meteo-backend-PR mergar — `optimal_fetch.py` Stage 1 + den
   föreslagna `era5_prefilter.py` i [`integration_proposal.md`](integration_proposal.md)
   blir helt redundant. Refaktorera till en `backend: open-meteo`-profil-
   konsumtion istället.
3. Audit `imint/training/era5_aux.py` `_fetch_via_cdsapi`-fallback — samma
   `cftime`-gotcha som DES #2 fixade kan dyka upp där också.

Dokumentet nedan är designspecen som implementationen följer, bevarad för
framtida referens och som råmaterial till PR-beskrivningarna.

---

**Repo:** https://github.com/DigitalEarthSweden/space-datalab-metafilter
**Branch:** `feat/extended-filters-and-s1-support` (pushad till
TobiasEdman-forken)

---

## TL;DR

Utöka metafiltret från **2 ERA5-Land-variabler / 2 regler / S2-only** till:

1. **Fler S2-filter** — total/low cloud cover, daglig insolation, **flersteg
   nederbörds-lookback** (24h, 48h, 7d, 30d), GDD-ackumulering, dry-streak.
2. **S1-stöd** — ny `"sensor": "sentinel-1"` i `metafilter.json` som tolkar
   reglerna mot pass-tids-samplad markstatus (markfuktighet, frost, snö,
   våt kanopi) istället för dygnsmedel.
3. **Minimal schema-tillägg** — två nya operatorer (`between`, `abs_lt`),
   härledda kolumner i `calculate_daily_metrics`, ingen breaking change i
   befintliga filterfiler.

Befintligt default-beteende (`mean_temp_c > 15 AND total_precip_mm < 1`) är
oförändrat. Allt nytt är opt-in via `filters/sentinel2_extended.json` resp.
`filters/sentinel1_default.json`.

## Motivation

`metafilter.json` idag använder dygnsmedel av t2m + dygnssumma av precip som
proxy för "klar himmel". Fyra praktiska brister:

- **Precip är en svag molnproxy.** En torr morgon med stratus passerar filtret
  trots att S2-scenen blir oanvändbar. ERA5 har `total_cloud_cover` direkt —
  varför inte använda den?
- **Engångs-dygnsfönster** missar att blött markskikt och våt kanopi från
  *föregående* dag förstör både S2 (skuggor/dimma) och S1 (backscatter-bias).
  Vår parallella experimentpipeline (`demos/era5_metafilter/REPORT.md` i
  ImintEngine) visar +18 % "användbar scen"-yield när 2-dagars-lookback läggs
  ovanpå nuvarande regler.
- **Vecko- och månadskontext saknas.** Två S2-scener från samma kalenderdatum
  i olika år är *inte* samma observation om den ena ligger efter en torr
  månad och den andra efter en blöt. För både scenkvalitet och NDVI-tolkning
  behövs lookback på 7d/30d-skala:
  - *Scenkvalitet:* En vecka med tung nederbörd ger ofta morgonfukt och
    konvektiv eftermiddagsmolnighet i 3–5 dagar efter — längre svans än
    48h-filtret fångar.
  - *NDVI-tolkning:* En torr 30-dagarsperiod sänker NDVI via senescens, inte
    via strukturell vegetationsförändring. Ackumulerad GDD och precip-anomali
    behövs som covariates för att jämföra scener över tid.
  - *Jord-bakgrundsreflektans:* Mark-NIR/SWIR-bakgrund i glesa bestånd
    dominerar NDVI; våt vs torr jord skiftar NDVI med 0.05–0.10 utan att
    vegetationen ändrats. Månads-precip + swvl1 är prediktorn.
- **S1 är en annan värld.** Moln transparenta för C-band, men markfuktighet,
  frysning och snö flyttar backscatter med flera dB. Metafiltret är generellt
  nog att hantera båda sensorerna med samma rule-engine — bara `metric_column`
  och samplingsstrategi behöver byta.

Sammantaget gör tilläggen samma kod mer användbar för fler downstream-
projekt utan att utmana den befintliga design-principen ("regler i JSON,
data via CDS, plot via xarray").

## Scope

**I denna PR:**
- Schema-tillägg i `filters/metafilter.json` (`sensor`-fält, opt-in)
- Nya operatorer `between`, `abs_lt` i `process_era5.py`
- Härledda kolumner:
  - *Korta lookbacks:* `precip_prev24h_mm`, `precip_prev48h_mm`
  - *Långa lookbacks (NYA):* `precip_prev7d_mm`, `precip_prev30d_mm`,
    `ssrd_prev30d_mj_m2`, `gdd_prev30d_c`, `dry_streak_days`
  - *Pass-tid:* `tcc_mean_overpass`, `lcc_mean_overpass`, `skt_at_pass_c`
  - *Övrigt:* `ssrd_mj_m2`, `swvl1_mean`, `swvl1_delta_prev2d`,
    `snow_depth_mean_m`, `freeze_flag`, `min_temp_c`, `max_temp_c`
- Två nya filterprofiler: `filters/sentinel2_extended.json`,
  `filters/sentinel1_default.json`
- Andra CDS-retrieval-anrop för `reanalysis-era5-single-levels` (cloud cover —
  finns ej i ERA5-Land)
- **Buffer-månad i `download_era5_land`** — 30-dagars-lookback kräver att
  föregående månads data finns laddad. Wrappern hämtar nu (start_month - 1)
  automatiskt när långa lookback-regler är aktiva.
- Uppdaterad README.md-sektion

**Inte i denna PR (förslagna följd-PRs):**
- **Stratifierande regelläge** (label per dag istället för bool-gating) —
  större strukturell ändring, görs som PR #2.
- **Open-Meteo-backend** som alternativ till CDS API (en endpoint, ingen auth,
  redan inkluderar cloud cover) — PR #3.
- **Säsongsspecifika trösklar** (`valid_months: [6,7,8]` per regel) — PR #4.
- **Vikt-baserad scoring** istället för hård AND — PR #5.

**Brytande ändringar:** inga. Existerande `filters/metafilter.json` fortsätter
fungera identiskt.

---

## Schema-tillägg i `filters/metafilter.json`

Nytt toppnivåfält `sensor` (valfritt, default `"sentinel-2"`). Befintliga
filterfiler utan fältet behåller exakt nuvarande beteende.

```json
{
  "sensor": "sentinel-2",
  "overpass_time_utc": "10:30",
  "rules": {
    "temperature": { ... },
    "precipitation": { ... }
  }
}
```

`overpass_time_utc` är valfritt och styr vid vilken klocktid pass-känsliga
variabler samplas (relevant för cloud cover och S1-yt-tillstånd). Default per
sensor: S2 → `"10:30"`, S1 → `"05:30"` (descending) / `"17:00"` (ascending).

**Bakåtkompatibilitet i loadern:** `normalize_metafilter_rules()` accepterar
både den nya formen (`{sensor, rules: {...}}`) och den gamla platta formen
(`{temperature: {...}, precipitation: {...}}`).

```python
# scripts/process_era5.py
def load_metafilter_parameters(json_file):
    with open(json_file, "r") as file:
        payload = json.load(file)
    # Backward-compat: flat dict = legacy format = sentinel-2 default
    if "rules" not in payload:
        return {"sensor": "sentinel-2", "overpass_time_utc": "10:30", "rules": payload}
    payload.setdefault("sensor", "sentinel-2")
    payload.setdefault(
        "overpass_time_utc",
        "10:30" if payload["sensor"] == "sentinel-2" else "05:30",
    )
    return payload
```

## Nya operatorer

```python
# scripts/process_era5.py — utökning av COMPARISON_OPERATORS

COMPARISON_OPERATORS = {
    "gt": {"apply": lambda s, t: s > t, "symbol": ">"},
    "ge": {"apply": lambda s, t: s >= t, "symbol": ">="},
    "lt": {"apply": lambda s, t: s < t, "symbol": "<"},
    "le": {"apply": lambda s, t: s <= t, "symbol": "<="},
    # NYA:
    "between": {
        "apply": lambda s, t: (s >= t[0]) & (s <= t[1]),
        "symbol": "in",
    },
    "abs_lt": {
        "apply": lambda s, t: s.abs() < t,
        "symbol": "|·| <",
    },
}
```

Validatorn extends så `between`-regler kräver `threshold` som 2-elements lista.

```python
# I normalize_metafilter_rules():
if operator == "between":
    if not (isinstance(merged_rule["threshold"], list) and len(merged_rule["threshold"]) == 2):
        raise MetafilterConfigurationError(
            f"Rule '{rule_name}' uses 'between' but threshold is not [low, high]."
        )
```

## Härledda kolumner i `calculate_daily_metrics`

Funktionen byggs ut så den producerar ett superset av kolumner. Rule-engine
plockar bara de kolumner som faktiskt refereras av aktiva regler — så obrukade
kolumner är gratis (utöver minimal CPU-kostnad).

```python
def calculate_daily_metrics(file_path, area=AREA, *, sensor="sentinel-2",
                            overpass_time_utc="10:30",
                            cloud_file_path=None):
    """Compute daily aggregates for both ERA5-Land surface state and
    (optionally) ERA5 single-levels cloud cover.

    Args:
        file_path: path to ERA5-Land NetCDF (surface state).
        cloud_file_path: optional path to ERA5 single-levels NetCDF
            containing tcc / lcc. Required if any active rule references
            tcc_mean_overpass or lcc_mean_overpass.
        sensor: 'sentinel-2' or 'sentinel-1' — controls which pass-time
            sampling columns are produced.
        overpass_time_utc: HH:MM, used for pass-time sampling.

    Returns:
        pd.DataFrame indexed by date with all derived columns.
    """
    ds = xr.open_dataset(file_path)
    if "valid_time" in ds.coords or "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    ds = subset_dataset_to_area(ds, area)

    spatial = tuple(d for d in ("latitude", "longitude") if d in ds.dims)

    df = pd.DataFrame(index=pd.to_datetime(
        ds["time"].resample(time="1D").mean()["time"].values
    ).strftime("%Y-%m-%d").rename("date"))

    # ── Bibehållna kolumner (oförändrade) ───────────────────────────────
    if "t2m" in ds:
        t_c = ds["t2m"] - 273.15
        df["mean_temp_c"] = t_c.resample(time="1D").mean().mean(spatial).values
        df["min_temp_c"]  = t_c.resample(time="1D").min().mean(spatial).values
        df["max_temp_c"]  = t_c.resample(time="1D").max().mean(spatial).values
        df["freeze_flag"] = (df["min_temp_c"] < 0).astype(int)

    if "tp" in ds:
        precip = (ds["tp"] * 1000.0).resample(time="1D").sum().mean(spatial)
        df["total_precip_mm"] = precip.values
        # Korta lookbacks (samma fönster som tidigare iteration av PR:n)
        df["precip_prev24h_mm"] = pd.Series(df["total_precip_mm"]).shift(1).fillna(0).values
        df["precip_prev48h_mm"] = (
            pd.Series(df["total_precip_mm"]).rolling(2, min_periods=1)
            .sum().shift(1).fillna(0).values
        )
        # Långa lookbacks — fenologisk + atmosfärisk kontext
        df["precip_prev7d_mm"] = (
            pd.Series(df["total_precip_mm"]).rolling(7, min_periods=1)
            .sum().shift(1).fillna(np.nan).values
        )
        df["precip_prev30d_mm"] = (
            pd.Series(df["total_precip_mm"]).rolling(30, min_periods=1)
            .sum().shift(1).fillna(np.nan).values
        )
        # Dry-streak: antal sammanhängande dagar med precip < 0.5 mm fram t.o.m. *igår*
        wet = (pd.Series(df["total_precip_mm"]).shift(1).fillna(0) >= 0.5).astype(int)
        # cumulative-count-since-last-wet trick
        df["dry_streak_days"] = (
            wet.groupby(wet.cumsum()).cumcount().values
        )

    # ── Ackumulerad insolation (vegetations-energibudget) ──────────────
    if "ssrd" in ds:
        df["ssrd_mj_m2"] = (
            ds["ssrd"].resample(time="1D").sum().mean(spatial).values / 1e6
        )
        df["ssrd_prev30d_mj_m2"] = (
            pd.Series(df["ssrd_mj_m2"]).rolling(30, min_periods=1)
            .sum().shift(1).fillna(np.nan).values
        )

    # ── GDD-ackumulering (växtutvecklings-tid, base 5 °C) ──────────────
    if "mean_temp_c" in df.columns:
        gdd_daily = (df["mean_temp_c"].clip(lower=5) - 5).fillna(0)
        df["gdd_prev30d_c"] = (
            gdd_daily.rolling(30, min_periods=1).sum().shift(1).fillna(np.nan).values
        )

    # ── Yt-status (S1-relevant — men gratis att alltid producera) ──────
    if "skt" in ds:
        skt_c = ds["skt"] - 273.15
        df["skt_mean_c"] = skt_c.resample(time="1D").mean().mean(spatial).values
        df["skt_min_c"]  = skt_c.resample(time="1D").min().mean(spatial).values
        # Pass-tids-sampling — väljs av overpass_time_utc
        df["skt_at_pass_c"] = _sample_at_overpass(skt_c, overpass_time_utc, spatial)

    if "stl1" in ds:
        df["stl1_mean_c"] = (ds["stl1"] - 273.15).resample(time="1D").mean().mean(spatial).values

    if "swvl1" in ds:
        swvl1_daily = ds["swvl1"].resample(time="1D").mean().mean(spatial).values
        df["swvl1_mean"] = swvl1_daily
        df["swvl1_delta_prev2d"] = (
            pd.Series(swvl1_daily) - pd.Series(swvl1_daily).shift(2)
        ).fillna(0).values
        # 30-dagars-medel — proxy för "blöt vs torr säsong" / rotzon-historik
        df["swvl1_prev30d_mean"] = (
            pd.Series(swvl1_daily).rolling(30, min_periods=1)
            .mean().shift(1).fillna(np.nan).values
        )

    if "sd" in ds:  # snow_depth in metres
        df["snow_depth_mean_m"] = ds["sd"].resample(time="1D").mean().mean(spatial).values

    # ── Cloud cover (kräver separat ERA5 single-levels-fil) ────────────
    if cloud_file_path is not None:
        cds = xr.open_dataset(cloud_file_path)
        if "valid_time" in cds.coords:
            cds = cds.rename({"valid_time": "time"})
        cds = subset_dataset_to_area(cds, area)
        cspatial = tuple(d for d in ("latitude", "longitude") if d in cds.dims)
        if "tcc" in cds:
            df["tcc_mean_overpass"] = _sample_at_overpass(
                cds["tcc"], overpass_time_utc, cspatial
            )
        if "lcc" in cds:
            df["lcc_mean_overpass"] = _sample_at_overpass(
                cds["lcc"], overpass_time_utc, cspatial
            )

    return df.reset_index()


def _sample_at_overpass(var, overpass_time_utc, spatial_dims):
    """Sample an hourly variable at the closest hourly slot to overpass_time_utc.

    overpass_time_utc: 'HH:MM' string. Rounded to nearest hour.
    """
    hour = int(overpass_time_utc.split(":")[0])
    minute = int(overpass_time_utc.split(":")[1])
    if minute >= 30:
        hour = (hour + 1) % 24
    times = pd.to_datetime(var["time"].values)
    mask = (times.hour == hour)
    if not mask.any():
        # Fallback: daily mean if hourly data isn't available
        return var.resample(time="1D").mean().mean(spatial_dims).values
    var_at = var.isel(time=mask)
    daily = var_at.resample(time="1D").mean().mean(spatial_dims)
    return daily.values
```

## Ändringar i `scripts/download_era5.py`

Tre ändringar:

1. Utökad variabel-lista för ERA5-Land.
2. Andra `cdsapi.Client().retrieve(...)`-anrop för `reanalysis-era5-single-levels`
   när cloud cover-regler är aktiva.
3. **Auto-buffer av föregående månad** när långa lookback-regler är aktiva —
   `precip_prev30d_mm` på 1 augusti behöver 2 juli–31 juli som källdata.

```python
import json
from pathlib import Path

from utils.config import AREA, OUTPUT_DIR

# Defaults — inkluderar all data som behövs för S2 extended + S1 default.
# Anroparen kan trimma listan om de bara behöver delmängd.
ERA5_LAND_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "skin_temperature",
    "soil_temperature_level_1",
    "volumetric_soil_water_layer_1",
    "snow_depth",
    "snowfall",
    "2m_dewpoint_temperature",
]

ERA5_SINGLE_LEVELS_VARIABLES = [
    "total_cloud_cover",
    "low_cloud_cover",
]


def download_era5_land(year, month, variables=None, area=None):
    import cdsapi
    area = area or AREA
    variables = variables or ERA5_LAND_VARIABLES
    c = cdsapi.Client()
    out_path = f"{OUTPUT_DIR}/era5/era5_land_{year}_{month:02d}.nc"
    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": variables,
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [area["north"], area["west"], area["south"], area["east"]],
            "data_format": "netcdf",
        },
        out_path,
    )
    return out_path


def download_era5_cloud(year, month, area=None):
    """Cloud cover (saknas i ERA5-Land). Returnerar None om ej använt."""
    import cdsapi
    area = area or AREA
    c = cdsapi.Client()
    out_path = f"{OUTPUT_DIR}/era5/era5_clouds_{year}_{month:02d}.nc"
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ERA5_SINGLE_LEVELS_VARIABLES,
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [area["north"], area["west"], area["south"], area["east"]],
            "data_format": "netcdf",
        },
        out_path,
    )
    return out_path


def cloud_vars_needed(metafilter_path):
    """True if any active rule references tcc/lcc columns."""
    with open(metafilter_path) as f:
        cfg = json.load(f)
    rules = cfg.get("rules", cfg)
    return any(
        r.get("metric_column", "").startswith(("tcc_", "lcc_"))
        for r in rules.values()
    )


_LONG_LOOKBACK_PREFIXES = (
    "precip_prev7d_", "precip_prev30d_", "ssrd_prev30d_",
    "gdd_prev30d_", "swvl1_prev30d_", "dry_streak_",
)


def long_lookback_needed(metafilter_path):
    """True if any rule needs ≥ 7 days of trailing data."""
    with open(metafilter_path) as f:
        cfg = json.load(f)
    rules = cfg.get("rules", cfg)
    return any(
        any(r.get("metric_column", "").startswith(p) for p in _LONG_LOOKBACK_PREFIXES)
        for r in rules.values()
    )


def download_period(year, month, filter_path, area=None):
    """Hämta primärmånad + ev. buffer-månad + ev. cloud cover.

    Returnerar (land_paths, cloud_paths) — listor av lokala NetCDF-paths
    som ska konkateneras av calculate_daily_metrics innan aggregering.
    """
    months_to_fetch = [(year, month)]
    if long_lookback_needed(filter_path):
        prev_year, prev_month = (year, month - 1) if month > 1 else (year - 1, 12)
        months_to_fetch.insert(0, (prev_year, prev_month))

    land_paths = [download_era5_land(y, m, area=area) for y, m in months_to_fetch]
    cloud_paths = []
    if cloud_vars_needed(filter_path):
        cloud_paths = [download_era5_cloud(y, m, area=area) for y, m in months_to_fetch]
    return land_paths, cloud_paths


if __name__ == "__main__":
    # CLI exempel — full månadshämtning
    year, month = 2024, 8
    download_era5_land(year, month)
    # Endast om filterprofilen behöver cloud cover
    if cloud_vars_needed("filters/metafilter.json"):
        download_era5_cloud(year, month)
```

## Nya filterprofiler

### `filters/sentinel2_extended.json`

Tre delar:
1. **Scen-kvalitets-gating** (rad 1–6) — strikt boolean AND.
2. **Säsongs-kontext** (rad 7–9) — kräver långa lookbacks, filtrerar bort
   scener vars NDVI är fenologiskt avvikande från den jämförelsegrupp
   nedströms-analysen förväntar.
3. *Stratifierings-rules* (i scope för PR #2, ej här) — tagga snarare än
   filtrera.

```json
{
  "sensor": "sentinel-2",
  "overpass_time_utc": "10:30",
  "rules": {
    "temperature": {
      "name": "Daily mean temperature",
      "metric_column": "mean_temp_c",
      "operator": "gt",
      "threshold": 10.0,
      "unit": "Celsius",
      "description": "Avoid winter scenes where NDVI is undefined."
    },
    "precipitation_today": {
      "name": "Today's precipitation",
      "metric_column": "total_precip_mm",
      "operator": "lt",
      "threshold": 0.5,
      "unit": "mm",
      "description": "Dry surface → less likely cloud cover at overpass."
    },
    "precipitation_prev48h": {
      "name": "Precipitation, previous 48 hours",
      "metric_column": "precip_prev48h_mm",
      "operator": "lt",
      "threshold": 3.0,
      "unit": "mm",
      "description": "Atmosphere needs time to clear after rain events."
    },
    "precipitation_prev7d": {
      "name": "Precipitation, previous 7 days",
      "metric_column": "precip_prev7d_mm",
      "operator": "lt",
      "threshold": 25.0,
      "unit": "mm",
      "description": "Heavy week → residual humidity + convective afternoon cloud for 3-5 days after."
    },
    "cloud_total": {
      "name": "Total cloud cover at overpass",
      "metric_column": "tcc_mean_overpass",
      "operator": "lt",
      "threshold": 0.30,
      "unit": "fraction",
      "description": "Direct cloud signal at ~10:30 UTC overpass."
    },
    "cloud_low": {
      "name": "Low cloud cover at overpass",
      "metric_column": "lcc_mean_overpass",
      "operator": "lt",
      "threshold": 0.20,
      "unit": "fraction",
      "description": "Low clouds (stratus, fog) are the main S2 killer."
    },
    "solar_radiation": {
      "name": "Daily solar radiation",
      "metric_column": "ssrd_mj_m2",
      "operator": "gt",
      "threshold": 12.0,
      "unit": "MJ/m^2",
      "description": "High daily insolation correlates with clear-sky days."
    },
    "precipitation_prev30d": {
      "name": "Cumulative precipitation, previous 30 days",
      "metric_column": "precip_prev30d_mm",
      "operator": "between",
      "threshold": [30.0, 150.0],
      "unit": "mm",
      "description": "Stay in climatologically 'normal' band. Below → drought NDVI suppression; above → waterlogging + soil-NIR shift."
    },
    "gdd_accumulation": {
      "name": "Growing degree days, previous 30 days (base 5 C)",
      "metric_column": "gdd_prev30d_c",
      "operator": "gt",
      "threshold": 200.0,
      "unit": "degree-days",
      "description": "Ensures vegetation is in an active growth window — NDVI interpretation is otherwise unstable in cold spring/late autumn."
    },
    "soil_moisture_history": {
      "name": "Rootzone moisture, previous 30 days",
      "metric_column": "swvl1_prev30d_mean",
      "operator": "between",
      "threshold": [0.15, 0.40],
      "unit": "m^3/m^3",
      "description": "Avoid scenes where 30-day soil dryness or saturation dominates the spectral signal."
    }
  }
}
```

**Tröskelmotivering** (gäller mellansvenska AOI-er som Skåne/Östergötland):
- `precip_prev7d_mm < 25`: 25 mm/vecka ≈ medelnederbörd. Över det → våt vecka,
  förhöjd molnighet i svansen.
- `precip_prev30d_mm ∈ [30, 150]`: månadsklimatologi maj–aug är 50–80 mm.
  30 mm = torra månader (begränsad NDVI-tolkbarhet pga stress);
  150 mm = blöta månader (jord-NIR-bias + översvämningsrisk).
- `gdd_prev30d_c > 200`: motsvarar ~6–7 °C medeltemperatur över base 5 °C.
  Faller bort i april och oktober — exakt när NDVI inte är jämförbart med
  växtsäsongens mitt.
- `swvl1_prev30d_mean ∈ [0.15, 0.40]`: bredt fält-kapacitets-intervall för
  svensk åker (siltig lera–lerig mojord). Filtrerar bort uttalad torka och
  vattenmättnad.

Användare med andra AOI:er bör tuna trösklarna — vill man ha det generellt
hör det hemma i en följd-PR med per-AOI-config eller per-månads-trösklar
(PR #4 enligt scope).

### `filters/sentinel1_default.json`

```json
{
  "sensor": "sentinel-1",
  "overpass_time_utc": "05:30",
  "rules": {
    "frost_free": {
      "name": "Frost-free surface at overpass",
      "metric_column": "skt_at_pass_c",
      "operator": "gt",
      "threshold": 2.0,
      "unit": "Celsius",
      "description": "Avoid freeze-thaw days (large dielectric jump → backscatter artefact)."
    },
    "no_diurnal_freeze": {
      "name": "No diurnal freeze cycle",
      "metric_column": "skt_min_c",
      "operator": "gt",
      "threshold": 0.0,
      "unit": "Celsius",
      "description": "Daily minimum above 0 — soil dielectric stable across the day."
    },
    "snow_free": {
      "name": "Snow-free ground",
      "metric_column": "snow_depth_mean_m",
      "operator": "lt",
      "threshold": 0.02,
      "unit": "m",
      "description": "Snow cover invalidates vegetation backscatter interpretation."
    },
    "dry_canopy": {
      "name": "Dry vegetation canopy",
      "metric_column": "precip_prev24h_mm",
      "operator": "lt",
      "threshold": 1.0,
      "unit": "mm",
      "description": "Wet canopy adds 0.5-1 dB noise to crop backscatter time series."
    },
    "stable_soil_moisture": {
      "name": "Stable surface soil moisture",
      "metric_column": "swvl1_delta_prev2d",
      "operator": "abs_lt",
      "threshold": 0.05,
      "unit": "m^3/m^3",
      "description": "Sudden swvl1 change → unstable backscatter baseline."
    }
  }
}
```

## Bakåtkompatibilitet

| Befintlig artefakt | Påverkan |
|---|---|
| `filters/metafilter.json` (flat form) | Loader normaliserar till nytt schema med S2-default. **Identiskt beteende.** |
| `scripts/download_era5.py` (gamla anropare) | Variabel-lista är default-arg. Gamla `download_era5_land()` utan args → samma två variabler som idag. |
| `scripts/process_era5.py` outputs | `mean_temp_c` och `total_precip_mm` finns kvar oförändrade. Nya kolumner adderas, ingen tas bort. |
| `scripts/compare_ndvi.py`, `visualize.py` | Påverkas ej. Konsumerar bara den filtrerade datumlistan. |

Test som skickar nuvarande `filters/metafilter.json` ska ge bit-identisk
output mot pre-PR-versionen. Lägg som regressionstest.

## Test- och verifieringsplan

Måste passera innan PR:n mergas.

### Unit-tester
```
tests/test_operators.py            # gt/ge/lt/le/between/abs_lt — sant/falskt
tests/test_loader_legacy.py        # flat metafilter.json → normaliseras OK
tests/test_loader_new.py           # {sensor, rules} → normaliseras OK
tests/test_derived_columns.py      # precip_prev48h, swvl1_delta — kända input
                                   #   → kända output
tests/test_long_lookback.py        # NYA: precip_prev7d/30d, gdd_prev30d,
                                   #   dry_streak_days, swvl1_prev30d_mean —
                                   #   verifiera med syntetisk 60-dagars-input
                                   #   att rolling-fönster + shift är korrekt
                                   #   och att NaN-randen är förväntat lång.
tests/test_long_lookback_needed.py # NYA: detect-funktionen identifierar
                                   #   profiler som behöver buffer-månad
```

### Integrationstest
```
tests/test_end_to_end_s2_legacy.py
  # Kör pipeline med befintlig filters/metafilter.json mot fixture-NetCDF
  # för 2024-08. Assert: exakt samma kvarvarande datum som master-branchen.

tests/test_end_to_end_s2_extended.py
  # Kör med filters/sentinel2_extended.json. Assert:
  #   - färre kvarvarande datum (striktare)
  #   - alla kvarvarande datum har tcc_mean_overpass < 0.30 i fixture-datat
  #   - NYA: download_period() har hämtat 2024-07 + 2024-08 (buffer-månad)
  #     när long-lookback-regler är aktiva
  #   - NYA: första veckan i 2024-08 (där 30d-lookback inte kan ges utan
  #     buffer) faller bort om buffer saknas

tests/test_end_to_end_s1_default.py
  # Kör med filters/sentinel1_default.json. Assert:
  #   - vinterdagar (snow_depth > 0.02) filtreras bort
  #   - frost-dagar (skt_at_pass < 2) filtreras bort
```

### Empirisk benchmark (rapport, ej assert)

Kör de tre profilerna mot Skåne-AOI för 2022-04-01 — 2022-09-30 och plotta:

- Antal kvarvarande dagar per profil
- För S2-profilerna: faktiska eo:cloud_cover-värden för de S2-scener som
  matchar de kvarvarande dagarna (Element84 STAC, anonymt API)
- **NYA: NDVI-tidsserie-jämförelse.** För `sentinel2_extended.json` med
  vs utan långa-lookback-reglerna (`precip_prev7d_mm`, `precip_prev30d_mm`,
  `gdd_prev30d_c`, `swvl1_prev30d_mean`):
  - Plotta NDVI-fördelning per kvarvarande scen
  - Förvänta: striktare profil ger snävare NDVI-fördelning per fenologisk
    fas → tydligare säsongssignal, mindre brus från stress/torka/övermättnad
- För S1-profilen: faktiskt antal S1 IW-scener som finns för de kvarvarande
  dagarna, fördelat på asc/desc

Lägg resultatet som ny sektion i `README.md` under "Empirical results".

## Migrationsväg för befintliga användare

Bibehållen — ingen migration krävs. Inga ändringar i CLI-anrop, ingen flag
behöver sättas. En användare som vill ha mer än default-beteendet pekar bara
om `--filter` (eller motsvarande config-key) till en av de nya JSON-filerna.

## Öppna frågor till DES-maintainerna

1. **CDS-kvot:** Cloud-cover-anropet är ett separat `reanalysis-era5-single-levels`-
   retrieve — räknas som extra request mot CDS-kvoten. Är det OK eller vill ni
   hellre se Open-Meteo-vägen direkt (PR #3)?
2. **Filer eller fält?** Är `filters/sentinel2_extended.json` + `filters/sentinel1_default.json` rätt mönster, eller föredrar ni
   `filters/{sensor}/{profile}.json`-struktur (`filters/sentinel-2/default.json`,
   `filters/sentinel-2/extended.json`, `filters/sentinel-1/default.json`)?
3. **Tröskelkalibrering:** Värdena i de nya profilerna är baserade på vår
   Skåne-2022-validation. För nordlig latitud blir t.ex. `mean_temp_c > 10`
   för strikt. Vill ni att vi inkluderar `valid_months`-fält redan här, eller
   reserverar det för PR #4?
4. **S1-pass-tider:** `05:30` (desc) och `17:00` (asc) är ungefärliga banbanecentra.
   Vi väljer just nu en av dem via `overpass_time_utc`. Är det rimligt eller
   vill ni se båda samplade till olika kolumner (`skt_at_asc_c`, `skt_at_desc_c`)
   så samma config kan filtrera båda passen?
5. **Buffer-månad:** För långa lookbacks (`precip_prev30d_mm` etc.) hämtar
   wrappern föregående månad automatiskt. Är det rätt nivå att lösa det på,
   eller vill ni att `calculate_daily_metrics` istället tar en lista av
   NetCDF-paths och konkatenerar internt? Det första är mindre invasivt mot
   befintliga anrop, det andra mer explicit.

## Föreslagen commit-struktur

Om PR:n delas upp i atomiska commits:

```
1.  refactor(process_era5): normalize legacy + new filter config schema
2.  feat(operators): add 'between' and 'abs_lt' comparison operators
3.  feat(metrics): derive short-window precip lookbacks (24h, 48h) + swvl1 delta + freeze flag
4.  feat(metrics): derive long-window context (precip_prev7d, precip_prev30d,
                   ssrd_prev30d, gdd_prev30d, swvl1_prev30d_mean, dry_streak_days)
5.  feat(metrics): pass-time sampling helper + skt_at_pass column
6.  feat(download): ERA5 single-levels cloud cover retrieval
7.  feat(download): auto-buffer previous month when long-lookback rules are active
8.  feat(filters): add sentinel2_extended.json profile
9.  feat(filters): add sentinel1_default.json profile
10. test: regression + end-to-end coverage for new profiles
11. docs(README): document sensor field, new variables, new profiles
```

## Hur det relaterar till ImintEngine-experimenten

Detta förslag konsoliderar tre separata trådar från
`demos/era5_metafilter/`:

- `replicate_metafilter.py` — visade att 2-dagars precip-lookback ger
  mätbart bättre cloud-cover-distribution på kvarvarande dagar.
- `compute_cot.py` + `cot_metrics.json` — visade att direkt cloud-cover-
  proxy slår precip-proxy på precision.
- `integration_proposal.md` — internt förslag att slotta in ERA5-prefilter
  i `imint/training/tile_fetch.py`-pipelinen. Detta DES-PR-förslag
  *kompletterar* det interna förslaget: när DES-metafiltret stödjer S1 +
  utökade S2-regler kan vi senare ersätta vår interna `era5_prefilter.py`
  med en konsumtion av DES-paketet.
