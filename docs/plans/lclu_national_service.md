# National annual land-cover service: own copy for our areas, Zarr for the rest of Sweden

Status: proposal, 2026-11-01. Direction from Tobias the same day: analysis for
all of Sweden through the TESSERA Zarr store, and for our municipalities from
our own copy. Each cluster step needs Tobias' approval and
`kubectl create --dry-run=server` first. Decisions that are Tobias' to make
are marked **Decision**.

## Summary

The service already chooses a source per region (`RegionalSources`), so a
national service is two kinds of region:
- **Our areas** — the ten municipalities and Kiruna tätort — keep the
  staged copy, which holds the exact values R4 was trained on.
- **The rest of Sweden** reads TESSERA on demand from the published Zarr
  store, with a disk cache.

Every map has exactly one embedding source, named in its provenance and in
the viewer. TESSERA is not the only national input. HR-VPP, DEM and the
forestry layers must also cover the whole country, and HR-VPP does not yet.

## Where we are

| Fact | Where |
|---|---|
| One served model, TESSERA R4 (28 classes, 11 aux), years 2017–2025, only in operator-qualified regions | `imint/inference/lclu.py`, `imint/inference/lclu_sources.py` |
| A new Stockholm map took 18.1 s: 12.9 s preparation, 4.6 s model, 0.6 s export; a cached one 1.0 s | `docs/verification/lclu-service/verification.md` |
| The viewer now warms a place's inputs while the year is chosen, and a map request reuses them (one preparation per key) | `POST /v1/lclu/prepare`, `af1f209a`, `ce8cc8f5` |
| Zarr reader with its own identity `ZARR_EMBEDDING`; bit-identical to the tile path for a tile on the zone lattice | `imint/inference/lclu_zarr.py`, `tests/lclu/test_zarr.py` |
| The store rounds tile origins to the 10 m lattice (2–9 m off for sampled Swedish tiles, so up to a 5 m shift) and resolves overlaps last-written-wins | `imint/inference/lclu_zarr.py` (docstring); geotessera 0.10.2 `zarr.py` `_tile_pixel_offset` |
| A cluster job compares Zarr windows with the cohort's stored embeddings; not run yet | `k8s/tessera-zarr-compare-job.yaml`, `scripts/compare_tessera_zarr.py` |
| HR-VPP (WEkEO cache) covers only the demo areas' MGRS tiles; 2025 is unpublished everywhere | `docs/verification/lclu-service/coverage-matrix.md` |
| DEM (GLO-30) is staged per area; forestry and markfukt are per-county mosaics on the PVC; county masks limit windows to Sweden | `scripts/stage_lclu_demo.py`, `imint/training/skg_open.py` |
| From this cloud session `data.source.coop` is blocked; from ICE it is unverified | session network policy |

## Target

| Region | TESSERA source | Identity | First new map (estimate) | Storage |
|---|---|---|---|---|
| Our 11 areas | staged COGs on the PVC (S3 only if read from outside ICE) | `EMBEDDING` | about 5 s after a warm-up (model + export) | about 300 GB, exists |
| Rest of Sweden | Zarr store over HTTPS, aligned block cache | `ZARR_EMBEDDING` | a few seconds more for a cold Zarr read, hidden by the warm-up when the user picks a year | cache only, bounded |

- **Window rule (proposed).** A 512 px window uses the own copy only when it
  lies entirely inside one of our areas; otherwise it uses Zarr. Sources are
  never mixed inside one map.
- **Labelling.** Provenance and the viewer state the source, e.g. "TESSERA,
  egen kopia" or "TESSERA via Zarr, rikstäckning". A national map also
  states that its embeddings can sit up to 5 m from those the model was
  trained on, once the measurement gives the real figure.
- **Licence.** Check the TESSERA terms and the required attribution before a
  public national service. Show the attribution in the viewer and in
  provenance.

## Work packages

1. **Click warm-up — done.** `POST /v1/lclu/prepare`; the newest request
   wins; map requests reuse the input. Measure the live effect after the
   next deployment: preparation should leave the new-map time.
2. **Zarr measurement → Decision D1.**
   - Run `k8s/tessera-zarr-compare-job.yaml`: identical fraction, validity
     agreement, worst cosine percentile.
   - Extend it with what the map sees: R4 predictions on both embeddings
     for the same tiles, with per-class agreement.
   - Write the acceptance rule down before the result is read.
3. **National auxiliary coverage audit.** Per year and per source, list
   what the country has:
   - HR-VPP: WEkEO index against the MGRS tiles covering Sweden;
   - forestry and markfukt: counties present in the mosaics;
   - DEM: GLO-30 tiles;
   - county masks.

   The output is a national counterpart of `coverage-matrix.md`.
4. **HR-VPP national gap-fill → Decision D4.** Use the WEkEO path of the
   demo gap-fill (no CDSE units). The demo's 133 MGRS-year pairs were about
   74 GB, so all of Sweden for 2017–2024 is likely several hundred GB;
   size it from the audit. 2025 stays unavailable until it is published.
5. **DEM on demand.** Read GLO-30 windows from the public COGs per request,
   with a disk cache, instead of staging per area.
6. **Serving contract for two identities.**
   - `lclu.py`: the identity comes from the region's source, not one
     constant; it is in the service identity, the cache key and
     provenance; catalogues are homogeneous.
   - `lclu_sources.py`: a Zarr-backed source class and the window rule in
     `RegionalSources`.
   - `prepare_lclu.py`: a `zarr_store` branch that uses `lclu_zarr` plus
     the same local aux path.
   - API and viewer labels, and tests that pin each identity.
7. **Aligned block cache and prefetch levels 2–3.**
   - The Zarr cache moves from exact windows to blocks fixed on the zone
     lattice (e.g. 256 px, a multiple of the 32 px chunks), so any click
     reuses what was fetched.
   - View prefetch on pan/zoom, and municipality prefetch on area choice
     from the view outward.
   - Both have byte budgets, and click work always comes first.
   - Evict by recency, as the enrichment's `_LRUCacheGuard` does.
8. **Qualification of "Sverige via Zarr".**
   - Take a deterministic national sample of windows per year (stratified
     by county).
   - Check canonical pixel parity against `scripts/inference_comparison.py`,
     pixel inspection and validity.
   - Put the receipts in `docs/verification/lclu-service/`.
   - A year is offered nationally only with receipts.
9. **Deployment (separate approval).**
   - `zarr` and `fsspec[http]` in the demo image, pinned.
   - Egress from ICE to `data.source.coop`, verified.
   - Cache volume sizing.
   - A new serving manifest.

## Decisions

- **D1** Whether the measured Zarr difference is acceptable outside our
  areas (after package 2).
- **D2** The window rule at region borders (proposed: own copy only when
  entirely inside).
- **D3** Own copy on the PVC (fast inside ICE) or in S3 (only needed for
  access from outside).
- **D4** National HR-VPP storage (size from package 3).
- **D5** Cache and prefetch budgets per area and in total.
- **D6** How national maps are labelled in the viewer and provenance.

## Order

1. Packages 2 and 3 in parallel (measurement and audit).
2. Then D1, D4.
3. Packages 4–6, then 7 and 8.
4. Package 9 last.

Each package ships with its own tests and receipts; nothing national is
offered before package 8. These are estimates, not measurements, until
packages 1 and 2 report.

## Risks

- **External host.** National maps depend on Source Cooperative's
  availability and policy; the block cache softens outages but cannot
  replace the store.
- **Model mismatch.** R4 was trained on unshifted tile values; package 2
  measures the cost.
- **HR-VPP gaps.** A year without national HR-VPP cannot be offered
  nationally (no year substitution).
- **Abuse and cost.** Prefetch can be driven by anyone with the viewer.
  Newest-wins, byte budgets and the job queue bound it.
