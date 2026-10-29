# Private Stockholm LCLU demo

Temporary, single-worker TESSERA R4 inference for public clicks inside Stockholm
municipality. The map is 504×504 pixels at 10 m; each input has 512×512 pixels.
The municipal polygon restricts the snapped centre. Its 3,700 m source buffer
covers windows at the boundary, including windows crossing into neighbouring
municipalities. The buffer is not an extension of click admission.

## Sources and runtime

- `config/lclu/stockholm.json` records the SCB boundary, original public registry
  SHA256 and ordered regional block inventory. Both 2024 and 2025 have 28 blocks
  covering the buffered rectangle, 2,589,836,344 bytes of NPY vectors/scales per
  year. Conversion preserves int8 vectors and float32 scales in separate COGs;
  all base pixels are reopened and compared before publication.
- ICE preflight on 2026-10-29 verified 20 local HR-VPP source files for Stockholm
  2024, covering all five metrics across the buffered area. **No 2025 VPP files
  were present.** The initial demo therefore explicitly requests 2024. There is
  no automatic fallback from 2025 to 2024.
- Existing `/data/vpp_wekeo` and `/data/skg_open` are read-only. Only public
  TESSERA originals/COGs and GLO-30 DEM are downloaded to the demo volume.
  No DES/CDSE calls, WEkEO login, training or checkpoint export occurs.
- The R4 checkpoint remains on ICE and must match SHA256
  `1270bb2af88074840020f3b481ed5e98a72f0b7ceb51befac2b8fa4675385668`.
- Requests create and cache aux/embedding windows on demand. The server
  verifies source identities, exact years, model identity and implementation
  identity. Only qualified region/years are advertised.

`requirements.lock` contains exact distribution hashes and the Dockerfile pins
the official Python base digest. `build.py` includes only listed source/UI files;
no checkpoint, imagery, credentials or output enters the context. An exact file
manifest records uncommitted prototype code as such, without claiming that HEAD
alone identifies it.

## Build and smoke check

```sh
python docker/lclu/build.py --tag imint-lclu-demo:stockholm-20261029 \
  --output outputs/lclu-stockholm-20261029/build
```

The Dockerfile runs `smoke.py`: imports the clean API/model/data runtime,
constructs the TESSERA head, exports and reopens a known categorical GeoTIFF,
checks its geometry/nodata, and verifies the PNG. No random-weight map is served.
After a built image has been published to an authorized registry, render a Job
with its observed `repository@sha256:…` digest. Never use a mutable tag in K8s.

For this session the local Docker build installed its locked dependencies but
failed committing the layer with a Docker storage I/O error; the host had less
than 0.5 GiB free. The alternative below reproduces the locked environment on
the isolated demo PVC. It is explicitly recorded as `source-recipe` execution,
not as a successfully built/published application image.

```sh
python docker/lclu/build.py --tag imint-lclu-demo:stockholm-20261029 \
  --output outputs/lclu-stockholm-20261029/recipe --bundle-only
python docker/lclu/render_job.py \
  --image python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f \
  --recipe outputs/lclu-stockholm-20261029/recipe/source.tar.gz \
  --year 2024 --output outputs/lclu-stockholm-20261029/demo-job.json
kubectl --context icekube -n prithvi-training-default create --dry-run=server \
  -f outputs/lclu-stockholm-20261029/demo-job.json
```

Recipe mode mounts an immutable ConfigMap. Its name and environment pin the
archive SHA256; the executed bootstrap is extracted from that same archive and
checked against its source manifest. Bootstrap verifies every included file,
installs the hash-locked requirements in a dedicated virtual environment, runs
`pip check` and the export smoke check, then stages sources and serves the API.
`system_runtime.py` supplies Debian's libexpat1 2.5.0-1+deb12u5, pinned by
official package size and SHA256. It extracts into an isolated library directory;
the same installer is used by the Dockerfile and recipe bootstrap. The first ICE
attempt caught the missing XML library in the import smoke test before staging.
Recovery reuses the installed virtual environment and existing cache volume.
Use `create`, not client-side `apply`: the latter's last-applied annotation is
too large for this bundled ConfigMap.

For a recovery, render a new immutable archive and use `--reuse-volume`,
`--job-name lclu-stockholm-0124-r1` and `--deadline-seconds` equal to the remaining
approved duration. For this run the original expiry is 2026-10-30 12:46:25 UTC;
a retry must not reset the 24-hour budget. Keep the previous archive and Job
manifest as evidence. The active UI revision uses
`outputs/lclu-stockholm-20261029/demo-job-r3.json`. Revision r2 also preserves
the venv executable symlink when launching preparation subprocesses; resolving
that link incorrectly selected the package-free system interpreter. Revision r3
changes only the two UI assets: DES forest-green legend text and selection that
preserves the user's current detail zoom (15 verified; maximum 18).

## Start, access and stop

Actual creation requires the operator's explicit approval for the demo scope.
The completed short metadata preflight was a separate, smaller approved test.
For a new initial deployment, the rendered default Job and volume must not
already exist:

```sh
kubectl --context icekube -n prithvi-training-default create \
  -f outputs/lclu-stockholm-20261029/demo-job.json
kubectl --context icekube -n prithvi-training-default logs -f job/lclu-stockholm-0124
kubectl --context icekube -n prithvi-training-default port-forward \
  --address 127.0.0.1 job/lclu-stockholm-0124 8788:8000
```

For the already-created 2026-10-29 recovery, connect to its actual Job instead;
do not recreate the initial resources:

```sh
kubectl --context icekube -n prithvi-training-default port-forward \
  --address 127.0.0.1 job/lclu-stockholm-0124-r3 8788:8000
```

Open `http://127.0.0.1:8788/lclu`. The recipe creates no public Service, Ingress or
IP. It mounts no credentials and disables the service-account token. Resource
requests equal limits: 4 CPU, 8 GiB RAM. The Job stops after 24 hours including
setup and is cleaned up one day later. The separate 20 GiB CephFS PVC retains
verified originals, COGs, prepared inputs, results and receipts across sessions.
Restarting a completed/expired demo must reuse or deliberately replace that
cache; never discard it implicitly. Stop the current Job before changing code
or source configuration used by it. Stop the active 2026-10-29 recovery with:

```sh
kubectl --context icekube -n prithvi-training-default delete job lclu-stockholm-0124-r3
```

The cache remains billable until separately removed. Remove the PVC only after
exporting wanted results and obtaining authorization to discard its contents.
At ICE's public rates checked 2026-10-29, 4 cores + 8 GiB + 20 GiB CephFS are
approximately SEK 24.26 for 24 hours, excluding VAT. Retaining only that PVC is
about SEK 40.06 per 730-hour month. Existing reference storage is already
allocated. Rates: https://ice.ri.se/ .

## Evidence

Local evidence is in `outputs/lclu-stockholm-20261029/`: completed preflight Job,
source coverage check, COG parity, build log, source manifest and rendered Job.
The preflight confirms source inventory, not the scientific accuracy of R4.
Real Stockholm inference, canonical pixel parity, cache timing and browser
verification passed on 2026-10-29; see `docs/verification/lclu-service/verification.md`.
New maps took 18–19 s, repeated HTTP retrieval 0.15 s, verified disk-cache reuse
1.04 s. The active Job stops by 2026-10-30 13:45:02 Stockholm time. `MANIFEST.json` on the demo volume
records base/application image, execution mode, exact source and input hashes.
