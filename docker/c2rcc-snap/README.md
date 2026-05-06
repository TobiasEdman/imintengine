# C2RCC SNAP Docker

Kör ESA SNAP `c2rcc.msi` på Sentinel-2 L1C SAFE-arkiv för
vattenkvalitets-retrievals (chlorofyll-a, TSM, CDOM via IOPs).

## Filer

| Fil | Roll |
|---|---|
| `Dockerfile` | Bygger `imint-snap-c2rcc:latest` från `mundialis/esa-snap` + `s3tbx-c2rcc`-plugin |
| `c2rcc_msi_graph.xml` | SNAP graph: Read → Resample(B2) → Subset → c2rcc.msi(C2X-Nets) → Write(BEAM-DIMAP) |
| `run.sh` | Wrapper som tar SAFE-path, output-path, AOI bbox och kör graphen i container |

## Bakgrund

Den ursprungliga imagen `imint-snap-c2rcc:latest` (commit 52d19ae,
`feat(water_quality): real ESA C2RCC + Pahlevan MDN`) saknade Dockerfile
och kör-skript i versionkontroll. Detta är en rekonstruktion baserad på
`docker history`, befintlig output-struktur i `outputs/c2rcc_runs/` och
commit-beskrivningen.

Originalet kopierade `/usr/local/snap` från en host-installation; den här
versionen bygger från `mundialis/esa-snap`-base som har SNAP installerat
via ESA:s officiella installer. Det ger reproducerbarhet utan behov av
en host-installerad SNAP.

## Build

```bash
docker build -t imint-snap-c2rcc:latest -f docker/c2rcc-snap/Dockerfile docker/c2rcc-snap
```

Verifiera att `c2rcc.msi`-operatorn registrerades:

```bash
docker run --rm imint-snap-c2rcc:latest /usr/local/snap/bin/gpt -h | grep c2rcc.msi
```

## Användning

```bash
./docker/c2rcc-snap/run.sh \
    <SAFE_PATH> <OUTPUT.dim> \
    --west <W> --south <S> --east <E> --north <N>
```

Exempel — Lilla Karlsö 2025-05-12:

```bash
./docker/c2rcc-snap/run.sh \
    demos/lilla_karlso_birds/cache_l1c/S2A_MSIL1C_20250512T100031_*.SAFE \
    outputs/c2rcc_runs_lilla_karlso/2025-05-12.dim \
    --west 17.91 --south 57.21 --east 18.21 --north 57.41
```

## Output

BEAM-DIMAP-format: `<output>.dim` (XML-header) + `<output>.data/` (ENVI-band).

Band per scen:

| Band | Enhet | Beräkning av slutprodukt |
|---|---|---|
| `iop_apig` | m⁻¹ | chl_a = (apig^1.04) × 21.0 mg/m³ |
| `iop_agelb` | m⁻¹ | CDOM rapporteras direkt |
| `iop_bpart` | m⁻¹ | TSM = 1.72·bpart + 3.1·bwit g/m³ |
| `iop_bwit` | m⁻¹ | (samma) |
| `iop_adet` | m⁻¹ | detrital absorption (sällan använd) |
| `kd489` | m⁻¹ | diffuse attenuation @ 489 nm |
| `kdmin` | m⁻¹ | minimum kd |
| `rhow_B*` | – | water-leaving reflectance per band |
| `c2rcc_flags` | bit-flag | Rtosa_OOS, Cloud_risk, m.fl. |

Filtrera bort NN-floor-rester nedströms:

```python
chl[iop_apig < 0.001] = np.nan
```

## C2X-Nets vs default C2RCC-Nets

Imagen kör default `netSet=C2X-Nets` (Case-2 eXtreme) eftersom
default-NN klampar pixlar utanför sin tränings-range till en konstant
NN-floor. Per commit 52d19ae mätningar på Bohuslän-data:

| Net | Land | NN-floor | Real retrievals |
|---|---|---|---|
| Default C2RCC-Nets | 65 % | 19 % | 16 % |
| C2X-Nets | 59 % | 1.8 % | **38.8 %** |

För Östersjö-applikationer (hög CDOM från avrinning) är C2X-Nets nästan
alltid rätt val.
