# Skogsstyrelsen open-data mirror (IM-022)

Source of the forestry aux channels (`height`, `volume`, `basal_area`,
`diameter`) and SLU `markfukt` since 2026-10. Replaces the kartportal
ImageServer proxies (`kartor.skogsstyrelsen.se/kartor/*ArcGIS93Rest`),
which Skogsstyrelsen retired in 2026 (task 0106, PR #73).

| | |
|---|---|
| Source | `https://geodpags.skogsstyrelsen.se/geodataport/data/` (IIS directory, HTTP range supported, no login) |
| Mirror | `/data/skg_open` on `training-data-cephfs` (`SKG_OPEN_DIR`) |
| Staging | `scripts/stage_skg_open.py` via `k8s/skg-open-stage-job.yaml` |
| Reader | `imint/training/skg_open.py`, used by `skg_grunddata.py` and `slu_markfukt.py` |
| Manifest | `/data/skg_open/MANIFEST.json`: per zip URL, bytes, sha256, index date, download date; per derived raster method, sha256, grid; git sha |
| Licence | Skogsstyrelsen open data (CC0 per product descriptions) |

## Why a mirror and not windowed HTTP reads

The server supports range requests, but every zip holds one strip-TIFF
(one row per strip, LZW) inside deflate or Deflate64. Reading a window
means inflating the zip stream up to that row, so random access over
HTTP is not possible. Each raster is therefore downloaded once and
re-encoded as a tiled COG (512 px blocks, DEFLATE, no overviews) that the
reader windows locally.

## Products

| Channel | Published file | Native | Mirror | Reader policy |
|---|---|---|---|---|
| volume, basal_area, diameter | `sks_{Volym,Grundyta,Medeldiameter}_omdrev{1,2,3}_<län>.zip` | omdrev 2/3: 10 m int16, nodata −1; omdrev 1: 12.5 m | lossless COG per county + one VRT per variable and omdrev | newest omdrev per pixel: 3 → 2 → 1 (omdrev 1 resampled bilinearly) |
| markfukt | `sksSLUMarkfuktighet<län>.zip` | 2 m uint16 index 0–100, 101 water, 255 fill, nodata 65535 | lossless COG per county + VRT | nearest 2 m sample at each 10 m cell centre (65535 → 0) |
| height | `sksTradhojd<län>.zip` | 1 m int16 dm, nodata −1 | 10 m derivative per county (`tradhojd_10m_<län>.tif`) + VRT | 10 m value as is, dm (fetcher converts to m) |

Omdrev = laser-scanning round. Omdrev 1 is the 2009–2015 scanning
(12.5 m product), omdrev 2 the 2018– national re-scan ("complete"
2025-12-09 per the product description), omdrev 3 the third round
published 2026-04 for nine counties (07, 10, 12, 13, 21–25).

## Parity against proxy-fetched arrays (2026-10-12, Blekinge, county 10)

References: `.npy` cache arrays written by the proxy fetchers, 512×512
at 10 m, from `/data/unified_v2_512/cache` (2026-06-25),
`/data/unified_v2_512_orphans_staging/cache` (2026-07-04) and
`/data/unified_v2_512_nfi2024_staging/cache` (2026-10-02, the last run
before the proxies went away). Script: session scratchpad
`parity_grunddata.py` / `parity_markfukt.py`; summary below.

**Grunddata** (`volume`, `basal_area`, `diameter`), newest-omdrev policy:

| Reference cache | Tiles | Exact match | Note |
|---|---|---|---|
| nfi2024_staging (2026-10-02) | 6 | 100 % where the county file has data (5 tiles 1.000, 1 tile 0.855) | the 0.855 tile straddles the Skåne border; its remaining pixels live in county 12's file |
| orphans_staging (2026-07-04) | 4 | 1.000 | omdrev 3 covers 92–100 % of these tiles; the rest is omdrev 2 |
| unified_v2_512 (2026-06-25) | 6 | 0.992 mean, 0.951 min | the mismatching pixels of the one 0.951 tile are all outside every Blekinge file (county-12 area) |

Reading omdrev 2 alone matched 100 % on the 2026-10-02 references and
only 70 % on the 2026-07-04 ones, which is how the "newest omdrev per
pixel" policy was identified. The national VRTs remove the county-border
cases.

**Markfukt**: the published county files are not the model version the
proxy served. Nodata mask (0.178 vs 0.181) and water mask (0.194 vs
0.194) agree; the index values do not (23–39 % exact, mean absolute
difference 4–6 on the 0–100 scale, no reduction method does better).
Counties 20, 22, 24 and 25 were re-published 2026-04 and may be the
newer version; to be measured on the cluster once staged. Treated as a
documented per-channel difference: new tiles get the published version,
stored tiles keep the proxy version.

**Height**: see the parity note below once measured against
`sksTradhojd10` (1 m).

## Operations

Dry run of the plan (no writes):

```bash
python scripts/stage_skg_open.py --dest /tmp/x --products grunddata markfukt --dry-run
```

Stage on the cluster (resumable; re-apply under a new Job name after a
failure, finished files are skipped):

```bash
kubectl apply -f k8s/skg-open-stage-job.yaml
```

## Staged (2026-10-13 01:04Z, Job `skg-open-stage` + resumed `skg-open-stage-r2`)

| | |
|---|---|
| Files | 174 (grunddata omdrev 1–3 × 3 variables = 153, markfukt 21), all sha256-recorded |
| Raw zips `raw/` | 97.3 GiB |
| Derived COGs | 79.2 GiB (grunddata 16 GiB, markfukt 64 GiB), lossless, pixel-verified on five windows each |
| VRTs | 10 (volym/grundyta/medeldiameter × omdrev 1, 2, 3 + markfukt) |
| Manifest git sha | 798f18b (branch `agent/te/claude/im022-skg-geotiff-source`) |
| PVC free | 626 GiB before, 333 GiB after (the mirror is ~176 GiB; other jobs wrote the rest meanwhile) |

The first run (8 GiB memory) was OOM-killed while converting the 10.6 GiB
markfukt county 23 after 171 files: the page cache of a sequential
10 GiB write counts against the container's cgroup. The re-run with
24 GiB and `GDAL_CACHEMAX=512` skipped the 171 finished files and
converted counties 23–25 (peak 14 GiB resident). The manifest therefore
requires 24 GiB for any future county of that size.

Trädhöjd (21 files, 329.2 GiB raw at 1 m) is not staged: it does not
reproduce the stored height channel (closest statistic p90, MAE ≈ 1 m)
and would take half the remaining PVC. Run `STAGE_PRODUCTS=tradhojd`
once that cost is approved; the 10 m derivative is ~3 GiB and the raw
zips can be dropped afterwards with `--drop-raw`.

### Raw files after enrichment

The COGs are lossless re-encodings verified pixel-for-pixel on sampled
windows, every raw zip's sha256 is in `MANIFEST.json`, and the source
stays published, so deleting `raw/` (97.3 GiB) after the enrichment pass
loses no reproducibility: a re-download can be checked against the
recorded hashes. Keeping it only buys re-derivation without network.
