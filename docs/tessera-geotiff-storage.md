# Lagringsbehov för TESSERA GeoTIFF över Sverige

Beräknat 2026-10-29 för TESSERA **v1/vultr**, alla **128 embeddingband**,
cirka **10 meters upplösning**, och åren **2018–2024**.

Vanliga float32-COG:ar med ZSTD, utan översiktsnivåer, uppskattas kräva
**2,17 TB per år** eller **15,18 TB för sju år**. En rimlig planeringsbudget
med marginal är **3 TB för ett år** eller **20 TB för 2018–2024**.

## Uppmätt komprimering och beräknad volym

| Lagring | Ett år, representerat av 2024 | 2018–2024 |
|---|---:|---:|
| Float32, okomprimerad datapayload | 2,57 TB | 17,98 TB |
| GeoTIFF med LZW | 2,30 TB | 16,10 TB |
| COG med ZSTD, utan översikter | 2,17 TB | 15,18 TB |
| COG med ZSTD och översikter | 2,88 TB | 20,11 TB |
| Kompakt: int8-COG + float32-skalfaktorer, utan översikter | 0,54 TB | 3,76 TB |

TB avser decimal lagring: 1 TB = 10¹² byte. Inventeringen omfattar
8 335–8 349 kompletta block per år, totalt 58 364 block. Hela block som
skär Sveriges grova landpolygon ingår, även deras kust- och gränspixlar.
Små öar kan saknas i polygonen. En registerpost för 2024 saknade uppgift om
skalfaktorfilens storlek och har uteslutits; detta är ingen fullständig
verifiering av geografisk täckning.

Komprimeringen mättes på ett verkligt Uppsalablock från 2024, med
1 135 × 605 pixlar och 128 band. Dess 90,64 MB originaldata blev 314,89 MB
med LZW, 296,83 MB som ZSTD-COG utan översikter och 393,23 MB med översikter
i faktorerna 2, 4 och 8. Rikssiffrorna extrapolerar detta enda prov;
komprimeringen kan variera mellan områden och år.

Det kompakta alternativet behåller TESSERAs ursprungliga int8-värden och
skalfaktorer i två GeoTIFF-filer, totalt 73,53 MB för provblocket. Läsaren
måste återskapa värdena som `float32(int8-värde) × skalfaktor`.
Alla 128 band återskapades bitexakt i provet, utan ytterligare kvantisering.
Även övriga format kontrollerades bitexakt vid full upplösning, med samma
koordinatsystem, geotransform och dimensioner. Översiktsnivåernas medelvärden
är inte kvalificerade som ersättning för modellens fullupplösta indata.

## Planeringsbudget

- **Float32-COG utan översikter:** 3 TB för ett år eller 20 TB för sju år.
- **Float32-COG med översikter:** 25–30 TB för sju år.
- **Int8-COG och skalfaktorer utan översikter:** cirka 5 TB för sju år,
  förutsatt att läsaren stöder tvåfilsformatet.

Budgetarna gäller en lagrad kopia med marginal. Backup eller replikering,
stöddata, förberedda inferenscacher och arbetsutrymme vid konvertering
tillkommer. Om originalfilerna i NPY-format också behålls kräver de ytterligare
cirka **0,66 TB per år**, totalt **4,63 TB för 2018–2024**.

## Sparat underlag

- [Mätningar, inventering, filhashar och beräkningar](verification/lclu-service/tessera-storage.json).
- [Verifieringsrapport](verification/lclu-service/verification.md#tessera-geotiff-storage-estimate--sweden).
- Lokala provfiler, kvitton och beräkningsskript:
  `outputs/tessera-storage-20261029/` (ignoreras av Git).
- Formatdokumentation: [GeoTessera](https://geotessera.readthedocs.io/en/latest/architecture.html)
  och [GDAL COG](https://gdal.org/en/stable/drivers/raster/cog.html).

Ingen rikstäckande rasterhämtning eller WCS-driftsättning ingår i mätningen.
