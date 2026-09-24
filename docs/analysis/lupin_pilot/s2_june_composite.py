#!/usr/bin/env python3
"""Cloud-masked June median composite from CDSE openEO.

Why a composite instead of picking a date: single-date fetching needs a
clear-sky date chosen in advance, and on 2026-09-09 every pre-screen was
unavailable (DES HTTP 503; the CDSE SCL stack returned zero dates). ERA5
alone was measured picking dates that turned out 17-71 % cloudy — 7
fetch attempts for 2 usable groups, a third of groups lost entirely.

A masked median over the whole month removes the date choice from the
problem: every scene in June contributes, clouds are masked per pixel,
and the median of what remains is cloud-free wherever at least a couple
of scenes saw the ground. It also removes the acquisition-date confound
— with single dates the acquisition day varies systematically between
groups, exactly the kind of hidden structure that produced the year
confound in the TESSERA run.

Reuses the repo's own primitives (``_connect_cdse``, ``_to_nmd_grid``,
the 10/20/60 m band split and resample-to-10 m pattern from
``_load_s2_cube``). The one genuinely new piece is SCL masking before
temporal reduction, which no existing fetcher does.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# SCL classes to drop before compositing: 0 nodata, 1 saturated,
# 3 cloud shadow, 8 cloud medium, 9 cloud high, 10 cirrus, 11 snow.
# 2 (dark), 4 (veg), 5 (bare), 6 (water), 7 (unclassified) are kept —
# dropping 7 would punch holes in perfectly good vegetation.
SCL_DROP = [0, 1, 3, 8, 9, 10, 11]

# Visible-brightness ceiling used as a second cloud screen on top of SCL.
# Healthy vegetation sits near 0.05-0.09 mean visible reflectance; bare
# soil and sand reach ~0.20. 0.25 therefore rejects cloud without eating
# legitimate bright ground.
BRIGHT_MAX = 0.25

BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]
BANDS_60M = ["B09"]
ALL_BANDS = BANDS_10M + BANDS_20M + BANDS_60M

# Band sets. CDSE bills by input pixels, so dropping the resampled 20/60 m
# bands cuts cost roughly in proportion: measured 5.2 credits/group for
# the full 11-band set. "10m" keeps the four native-resolution bands,
# which carry the visible colour the bloom indices are built from plus
# NIR for vegetation vigour; SCL is still fetched for masking either way
# (it is 20 m natively and is not part of the output).
BAND_SETS = {"all": ALL_BANDS, "10m": BANDS_10M}
INDEX_NAMES = ["blueness", "magenta", "brightness"]


def feature_names(band_set: str = "all") -> list[str]:
    return BAND_SETS[band_set] + INDEX_NAMES


def bloom_indices(cube):
    """Per-pixel June maxima of three flower-colour indices.

    L. polyphyllus is colour-polymorphic — blue-violet, pink and white
    inflorescences occur in the same population — so one colour channel
    would miss part of the stand. Each index is reduced with ``max`` over
    June, i.e. per pixel we keep the peak-bloom value rather than a
    monthly average that dilutes a two-week flowering window.

    Returns a single cube of three bands: blueness, magenta, brightness.
    """
    # SCL alone leaves thin cloud and cloud edges, and a max-reducer seeks
    # out exactly those: measured brightness max 1.43 and blueness max
    # 0.37, both impossible for vegetation. Clouds are bright and grey, so
    # drop any timestep whose visible brightness exceeds BRIGHT_MAX before
    # taking the maximum. Without this, "bluest pixel in June" reliably
    # finds the cloudiest day rather than peak bloom.
    vis_mean = (cube.band("B02") + cube.band("B03") + cube.band("B04")) / 3.0
    cube = cube.mask(vis_mean.apply(lambda v: v > BRIGHT_MAX))

    b, g, r = cube.band("B02"), cube.band("B03"), cube.band("B04")
    # Differences, NOT ratios. The first version normalised by
    # (B02+B03+B04), which blows up wherever that sum approaches zero —
    # dark water and shadow after the -1000 offset — and measured
    # blueness up to 18 and magenta up to 313 on real data. Differences
    # are bounded by reflectance itself. Illumination normalisation is
    # not lost: the 11 median bands travel in the same feature vector,
    # so a linear probe can form any ratio it needs.
    blueness = b - g                         # blue-violet morphs
    magenta = b + r - 2 * g                  # pink/magenta morphs
    brightness = (b + g + r) / 3.0           # white morphs, bright canopy
    out = None
    for idx, name in ((blueness, "blueness"), (magenta, "magenta"),
                      (brightness, "brightness")):
        peak = idx.reduce_dimension(dimension="t", reducer="max") \
                  .add_dimension("bands", name, "bands")
        out = peak if out is None else out.merge_cubes(peak)
    return out


def fetch_june_composite(coords: dict, year: int, *, conn=None,
                         reducer: str | None = "median"):
    """Masked June median composite for *coords*.

    Args:
        coords: WGS84 bbox dict (west, south, east, north).
        year: Calendar year; the window is that year's June.
        conn: Optional pre-authenticated openEO connection, so a batch
            run pays the auth handshake once rather than per group.

    Returns:
        ``(img, crs, transform)`` with ``img`` of shape
        ``(len(ALL_BANDS), H, W)`` float32 reflectance, band order
        ``ALL_BANDS``. NOTE ``crs`` is whatever CDSE returns — measured
        EPSG:32632 (UTM 32N), NOT SWEREF99 — so callers must reproject
        their sample points into it rather than assuming EPSG:3006.
    """
    import rasterio
    from imint.fetch import (CDSE_COLLECTION, _connect_cdse, _to_nmd_grid,
                             _unpack_openeo_gtiff_bytes)

    conn = conn or _connect_cdse()
    projected = _to_nmd_grid(coords)
    temporal = [f"{year}-06-01", f"{year}-07-01"]   # end is exclusive

    def _load(bands):
        return conn.load_collection(
            collection_id=CDSE_COLLECTION, spatial_extent=projected,
            temporal_extent=temporal, bands=bands)

    cube = _load(BANDS_10M)
    ref = cube
    for bands, method in ((BANDS_20M, "bilinear"), (BANDS_60M, "bilinear")):
        cube = cube.merge_cubes(
            _load(bands).resample_cube_spatial(target=ref, method=method))

    # Cloud mask from SCL, nearest-neighbour so class codes stay intact.
    scl = _load(["SCL"]).resample_cube_spatial(target=ref, method="near")
    mask = scl.band("SCL").apply(
        lambda v: sum((v == c) for c in SCL_DROP)).apply(lambda v: v > 0)
    cube = cube.mask(mask)

    if reducer:
        # Median over the masked stack: cloud-free wherever any scene saw
        # ground, and robust to a single bad acquisition.
        cube = cube.reduce_dimension(dimension="t", reducer=reducer)

    data = _unpack_openeo_gtiff_bytes(cube.download(format="gtiff"))
    if not data:
        raise RuntimeError(f"empty composite for {coords} {year}")
    with rasterio.open(io.BytesIO(data)) as src:
        raw = src.read().astype(np.float32)
        crs, transform, nodata = src.crs, src.transform, src.nodata

    # Masked (cloudy) pixels come back as the file's nodata, which GTiff
    # writes as 0. Left alone they become (0-1000)/10000 = -0.1 and are
    # then averaged in as if they were real dark ground — measured as a
    # NEGATIVE median visible reflectance, which is unphysical. Convert
    # them to NaN so downstream code must handle them explicitly.
    raw[raw == (nodata if nodata is not None else 0)] = np.nan

    # DN -> reflectance is a plain /10000 here: NO -1000 offset.
    # fetch_copernicus_data subtracts 1000 because its path receives raw
    # DN, but load_collection returns values with the L2A baseline offset
    # ALREADY applied. Subtracting it again shifted every band by -0.1 and
    # was measured as blue -0.071 / red -0.066 / NIR +0.143 over
    # vegetation; adding the 0.1 back gives 0.029 / 0.034 / 0.243, i.e.
    # textbook vegetation. Negative visible reflectance was the tell.
    img = raw / 10000.0
    return img, crs, transform


FEATURE_NAMES = ALL_BANDS + ["blueness", "magenta", "brightness"]


def build_feature_cube(conn, coords: dict, year: int,
                       band_set: str = "all"):
    """One 14-band cube: June median reflectance + peak-bloom indices.

    Both halves reduce the temporal dimension away, so they share a grid
    and merge into a single cube — and therefore a single openEO job.
    That matters for batch execution: one job per group instead of two
    halves the queue and removes any chance of the two downloads landing
    on different grids.

    Band order is ``FEATURE_NAMES``. Reflectance is DN/10000 with NO
    -1000 offset (see fetch_june_composite for why).
    """
    from imint.fetch import CDSE_COLLECTION, _to_nmd_grid

    projected = _to_nmd_grid(coords)
    temporal = [f"{year}-06-01", f"{year}-07-01"]

    def _load(bands):
        return conn.load_collection(
            collection_id=CDSE_COLLECTION, spatial_extent=projected,
            temporal_extent=temporal, bands=bands)

    cube = _load(BANDS_10M)
    ref = cube
    if band_set == "all":
        for bands in (BANDS_20M, BANDS_60M):
            cube = cube.merge_cubes(
                _load(bands).resample_cube_spatial(target=ref,
                                                   method="bilinear"))

    scl = _load(["SCL"]).resample_cube_spatial(target=ref, method="near")
    mask = scl.band("SCL").apply(
        lambda v: sum((v == c) for c in SCL_DROP)).apply(lambda v: v > 0)
    refl = cube.mask(mask) / 10000.0

    median = refl.reduce_dimension(dimension="t", reducer="median")
    return median.merge_cubes(bloom_indices(refl))


def fetch_june_features(coords: dict, year: int, *, conn=None):
    """June median reflectance PLUS peak-bloom colour indices.

    Two downloads per group: the 11-band masked median, and the 3-band
    peak-bloom stack. Keeping them separate is deliberate — the median
    describes the habitat, the maxima describe the flowering event, and
    a probe can tell us which of the two actually carries the signal.

    Returns ``(img, crs, transform)`` with ``img`` of shape
    ``(14, H, W)``, band order ``FEATURE_NAMES``.
    """
    import io as _io
    import rasterio
    from imint.fetch import (CDSE_COLLECTION, _connect_cdse, _to_nmd_grid,
                             _unpack_openeo_gtiff_bytes)

    conn = conn or _connect_cdse()
    med, crs, transform = fetch_june_composite(coords, year, conn=conn)

    projected = _to_nmd_grid(coords)
    temporal = [f"{year}-06-01", f"{year}-07-01"]

    def _load(bands):
        return conn.load_collection(
            collection_id=CDSE_COLLECTION, spatial_extent=projected,
            temporal_extent=temporal, bands=bands)

    vis_cube = _load(["B02", "B03", "B04"])
    scl = _load(["SCL"]).resample_cube_spatial(target=vis_cube, method="near")
    mask = scl.band("SCL").apply(
        lambda v: sum((v == c) for c in SCL_DROP)).apply(lambda v: v > 0)
    # Same /10000, no -1000 (see fetch_june_composite). The differences
    # would survive a constant offset, but ``brightness`` would not.
    vis_refl = vis_cube.mask(mask) / 10000.0

    data = _unpack_openeo_gtiff_bytes(
        bloom_indices(vis_refl).download(format="gtiff"))
    with rasterio.open(_io.BytesIO(data)) as src:
        idx = src.read().astype(np.float32)
        if src.nodata is not None:
            idx[idx == src.nodata] = np.nan

    if idx.shape[1:] != med.shape[1:]:
        raise RuntimeError(
            f"bloom-index grid {idx.shape[1:]} != median grid "
            f"{med.shape[1:]} — refusing to stack misaligned rasters")
    return np.concatenate([med, idx], axis=0), crs, transform


if __name__ == "__main__":   # tiny manual check on one bbox
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--west", type=float, required=True)
    ap.add_argument("--south", type=float, required=True)
    ap.add_argument("--east", type=float, required=True)
    ap.add_argument("--north", type=float, required=True)
    ap.add_argument("--year", type=int, required=True)
    a = ap.parse_args()
    img, crs, tr = fetch_june_features(
        {"west": a.west, "south": a.south, "east": a.east, "north": a.north},
        a.year)
    finite = np.isfinite(img).all(axis=0)
    print(f"img={img.shape} crs={crs} bands={len(FEATURE_NAMES)}")
    print(f"finite pixels: {finite.mean():.1%}")
    print(f"reflectance range: {np.nanmin(img[:11]):.3f} .. "
          f"{np.nanmax(img[:11]):.3f}")
    print("  per-band median (vegetation: B08 high ~0.2-0.35, "
          "water: B08 < 0.05):")
    for i, name in enumerate(ALL_BANDS):
        print(f"    {name}: {np.nanmedian(img[i]):+.4f}")
    for i, name in enumerate(FEATURE_NAMES[11:], start=11):
        v = img[i][np.isfinite(img[i])]
        print(f"  {name}: {v.min():.4f} .. {v.max():.4f} (median {np.median(v):.4f})")
