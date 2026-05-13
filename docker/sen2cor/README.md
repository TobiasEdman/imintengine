# Sen2Cor v2.12.04 — Sentinel-2 L1C → L2A atmospheric correction

ESA-official standalone Linux64 installer, packaged as a Docker image
for use by the ImintEngine pipeline to back-fill `frame_2016` /
`frame_2015` tiles where DES openEO has no indexed L2A.

## Why this exists

DES STAC has zero L2A scenes indexed for 2016 and 2017 (verified
2026-04-29 via `probe-2017-stac-2gv4m` and 2026-05-12 via
`probe-l1c-2016-lktlw`). However:

- CDSE STAC has **27 L1C scenes** for summer 2016 over our southern
  Sweden test bbox
- DES STAC has **15 L1C scenes** for summer 2016 with assets b01–b08

We can fetch L1C from either CDSE or Google Cloud's public Sentinel-2
mirror (`gs://gcp-public-data-sentinel-2`, accessible via the existing
`imint.fetch.fetch_l1c_safe_from_gcp()` helper), then run sen2cor
locally to produce L2A reflectance — same format as the rest of our
`frame_2016` data.

## Build

```bash
cd docker/sen2cor
bash build.sh                                       # local build
IMAGE_TAG=ghcr.io/myorg/sen2cor:2.12.04 bash build.sh
PUSH=1 IMAGE_TAG=ghcr.io/myorg/sen2cor:2.12.04 bash build.sh
```

The build automatically smoke-tests `L2A_Process --help`. If sen2cor's
bundled Python doesn't load (broken installer, missing libstdc++,
etc.) the build fails before the layer is committed.

After the first push, capture the registry digest:

```
docker inspect --format='{{index .RepoDigests 0}}' imintengine/sen2cor:2.12.04
```

And update `MANIFEST.json` + every k8s manifest that pulls this image
to use `image@sha256:...` rather than `image:tag` (per CLAUDE.md
docker policy).

## Run

The container's `ENTRYPOINT` is `L2A_Process`, so the standard sen2cor
CLI works directly:

```bash
docker run --rm \
    -v /path/to/SAFE_in:/work/input \
    -v /path/to/SAFE_out:/work/output \
    imintengine/sen2cor:2.12.04 \
    /work/input/S2A_MSIL1C_20160821T..._T33VUD_..._SAFE \
    --output_dir /work/output
```

The output is a sibling `*_MSIL2A_*` SAFE archive in `--output_dir`.

## Smoke test

Built into the Dockerfile — fails the build if the bundled Python
import path is broken. To re-run manually:

```bash
docker run --rm imintengine/sen2cor:2.12.04 --help
```

## Inputs supported

Sen2cor 02.12.04 accepts L1C SAFE archives from baseline PSD 02.03 and
later (2016 onwards). Older 2015-era SAFE archives use the legacy
format and may require a per-scene check before bulk processing.

## Pipeline integration

See `k8s/sen2cor-back-fill-frame2016-*.yaml` (TBD) for the consumer
job manifests that:

1. STAC-search via `optimal_fetch_dates(mode="atmosphere")` for ERA5-
   clean candidate dates
2. Greedy set-cover to pick scenes that maximise tile coverage
3. Per scene: `fetch_l1c_safe_from_gcp()` → mount into this image →
   run `L2A_Process` → crop to per-tile bbox → write `frame_2016`

## License

Sen2Cor is distributed by ESA under their own terms. Refer to
[the official Sen2Cor page](https://step.esa.int/main/snap-supported-plugins/sen2cor/)
for redistribution rules. The Dockerfile in this directory only
encodes the installation procedure — the installer binary is fetched
from the official ESA STEP server at build time.
