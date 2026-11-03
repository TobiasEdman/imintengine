# LCLU demo runtime and showcase release

The current municipality campaign covers the ten largest municipalities and
Kiruna tätort for inventoried TESSERA years 2017–2025. It reuses an existing
694 GiB PVC. The initial two-area deployment examples and costs below are
historical; use the release preparation steps below for the current campaign.
Preparation, model qualification and public publication are separate steps.

## Initial two-area demo

Temporary, single-worker TESSERA R4 inference for public clicks inside Stockholm
municipality and Kiruna tätort (SCB 2023 boundary). The map is 504×504 pixels at 10 m; each input has 512×512 pixels.
Each area polygon restricts the snapped centre. Its 3,700 m source buffer
covers windows at the boundary, including windows crossing into neighbouring
areas. The buffer is not an extension of click admission.

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

On 2026-10-30 Tobias extended the approval to one week, so `render_job.py`
accepts at most 604,800 s. Revision r7 serves the corrected forest legend
(Lövskog/Blandskog with the field-refinement note) on the same areas, budget
and cache, with `--deadline-seconds 604800`: about SEK 170 for the week at
the rates below, excluding VAT.

For a recovery, render a new immutable archive and use `--reuse-volume`,
`--job-name lclu-stockholm-0124-r1` and `--deadline-seconds` equal to the remaining
approved duration. For this run the original expiry is 2026-10-30 12:46:25 UTC;
a retry must not reset the approved budget. Keep the previous archive and Job
manifest as evidence. The active UI revision uses
`outputs/lclu-stockholm-20261029/demo-job-r6.json`. Revision r2 also preserves
the venv executable symlink when launching preparation subprocesses; resolving
that link incorrectly selected the package-free system interpreter. Revision r3
changes only the two UI assets: DES forest-green legend text and selection that
preserves the user's current detail zoom (15 verified; maximum 18). Revision r4
sets default result opacity to 100%, retains the slider, and renders categorical
pixels without smoothing or animated zoom/fade transitions. Revision r5 adds
Kiruna tätort and the area selector, retaining the same resource budget.
Revision r6 adds native-pixel class inspection and the NMD presentation
palette; the numerical classifications remain identical. Its actual expiry
is 2026-10-30 12:40:48 UTC, within the original approval.

Kiruna uses `config/lclu/kiruna_tatort.json`: the exact SCB tätort polygon,
not the full municipality. It has a separate source inventory and persistent
cache in `/work/kiruna_tatort`; Stockholm remains in `/work/stockholm`. The
launcher qualifies both before publishing `/work/demo/serving.json`. A single
resident R4 model routes each request to its exact region and year.

## Municipality campaign (IM-024)

`render_job.py --campaign-years 2017 … 2025 --max-original-bytes B
--storage-gib N` renders the ten municipalities plus Kiruna tätort from
`config/lclu/municipalities.json`. `B` is the campaign preflight's
`source_ready_original_bytes`; only `source_ready` area-years are staged and
each is qualified with one real R4 prediction before serving, so a year
without HR-VPP is absent rather than substituted. The volume must hold the
retained originals plus COGs (the renderer requires `N` GiB ≥ 2.3·B + 10 GiB).
The existing 694 GiB campaign volume meets that bound; no expansion is needed
for this release.

Serving inputs, results and retained diagnostics share an 8 GiB budget
(`IMINT_LCLU_STORAGE_MAX_BYTES`) and require 2 GiB of estimated filesystem
free space (`IMINT_LCLU_STORAGE_MIN_FREE_BYTES`). Writers reserve their
allowance before starting; all regions and warm-up/map workers share the
guard. Native inputs are bounded before compression; preparation metadata
and child diagnostics are each capped at 1 MiB. One failure log is retained
per input key. Admission uses a startup byte count updated by each writer,
without recursively scanning the cache on every request.

When space is needed, the service reclaims the least recently used prepared
inputs and failure logs. A lease protects each input through its last model
read. Finished map artifacts are retained; their result sidecars persist a
request-to-result lookup so maps still open after input eviction and process
restart. Hash and source-configuration checks still apply. After acquiring
its exclusive output-root lock, startup removes abandoned `.preparing-*`
and `.pending-*` directories before reconciling the budget. No source COGs,
reference data or qualification inputs are evicted.

If retained results, active leases or the filesystem reserve leave too little
space even after reclamation, new warm-ups return 503 and new map jobs fail
with a Swedish storage message. Existing prepared keys and verified map files
remain readable. This bounds the staged-COG campaign's serving cache, separately
from source staging; remote-prefetch logs are outside this contract and that
path is disabled by the staged COG inventory. This release serves only LCLU:
the public ingress excludes `/v1/analyze`; do not run a mixed general-analysis
workload in its output root. The byte counter covers the declared LCLU writers,
while the live filesystem free-space check also observes other PVC consumers.
CephFS can report writes and deletions late. Each admission samples free space
once, credits completed evictions and retains completed write debits across
requests. Lower observations made while no writer is active reduce that
estimate; later higher observations do not restore the allowance until restart.
This can conservatively refuse work after other consumers free space or a stale
post-deletion observation. During active writes, lower observations affect only
that admission so their bytes are not debited again on completion. This is a
single-process guard, not a filesystem quota for other PVC writers. Eviction
renames an input to `.preparing-*` before deletion, so interrupted cleanup cannot
leave a partial directory that looks prepared.
If retained maps themselves
fill the budget, preserve/export the results and prepare a separately approved
release with sufficient capacity; do not delete inputs or shared source trees
by hand to repair a live service.

Staging takes hours, so the campaign runs beside the two-area demo on the same
volume (Tobias 2026-10-30: no downtime). It keeps its own manifest, process
`MANIFEST.json` and API output directory under `/work/campaign`, because the
API holds a single-worker flock in its output directory; the running demo's
`/work/demo` and `/work/results` stay untouched. Move the port-forward and stop
the old Job once the campaign Job is Ready. Its deadline is the remainder of
the approved week, not a new week.

`docker/lclu/run_full_demo.sh` performs the whole sequence unattended from the
operator's machine: wait for the HR-VPP gap-fill Job, commit its preflight,
expand the PVC, create r8, wait until it serves, move the 8788 port-forward
and stop r7. Each step is skipped when already done, so it is re-run after an
interruption; it stops before changing anything if an upstream Job failed.

## Prepare a public showcase release

Start with a fresh, hash-bound Job proposal from `render_job.py` using
`--reuse-volume --isolate-campaign`, a versioned Job name, the existing
campaign byte budget and a deadline within the remaining approval. An
isolated release writes to `/work/releases/<job-name>` and reuses verified
COGs and DEM from `/work/campaign/shared`; it has its own VPP metadata
snapshots, prepared inputs, manifests, results and API lock. Keep
`/work/campaign/shared` after retiring r8: the release's verified COG records
still reference those files by absolute path. The running
campaign's source and configuration remain immutable. Recipe bootstrap also
installs its system libraries under that release root, so it cannot overwrite
libraries loaded by the running service. The hash-addressed Python environment
is reused once its READY marker exists.

The r9 proposal reuses the completed r8 inventory. A later release that adds
sources needs an explicit predecessor-cache plan; this launcher does not scan
arbitrary older release directories.

Keep concurrent compute within its existing approval. When replacing r8,
wait until it is ready and retire the older two-area r7 service before
starting r9, so the new release does not add a third reserved worker. Keep
r8 serving until the new release passes acceptance and traffic is switched.

```sh
python docker/lclu/render_showcase.py \
  --job-manifest <job-proposal.json> \
  --api-host <confirmed-api-hostname> \
  --showcase-origin https://tobiasedman.github.io \
  --showcase-origin https://digitalearth.se \
  --output <release-directory>
kubectl --context icekube -n prithvi-training-default apply \
  --server-side --dry-run=server \
  -f <release-directory>/job-with-origins.json \
  -f <release-directory>/routing.json
```

The renderer adds exact CORS origins, a Service selecting only the proposed
Job and an nginx TLS Ingress using the namespace's `letsencrypt-prod` Issuer.
It exposes only annual-map routes and assets. Source hashes, compute, storage
and expiry are preserved. The commands above create no resources; the
generated `release.json` records file hashes and outstanding release gates.

After all usable area/year pairs are qualified, run the acceptance audit in
the serving environment and retain its output:

```sh
python scripts/verify_lclu_campaign.py \
  --campaign config/lclu/municipalities.json \
  --work /work/releases/<job-name> \
  --api http://127.0.0.1:8000 > campaign-acceptance.json
```

The audit must pass, including explicit source gaps, before switching traffic.
Verify real Stockholm and Kiruna results in the browser: class clicks, NMD
legend, full opacity and slider, sharp zoom to at least 15, GeoTIFF download,
mobile layout and showcase navigation. Verify DNS, TLS and CORS from the
actual showcase origin. Publish the generated `lclu-config.js` as
`docs/js/lclu-config.js` only after these checks. A successful dry-run alone
does not prove public reachability or a working map.

## Initial deployment: start, access and stop

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
  --address 127.0.0.1 job/lclu-stockholm-0124-r6 8788:8000
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
kubectl --context icekube -n prithvi-training-default delete job lclu-stockholm-0124-r6
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
1.04 s. The active Job stops by 2026-10-30 13:42:51 Stockholm time. `MANIFEST.json` on the demo volume
records base/application image, execution mode, exact source and input hashes.
