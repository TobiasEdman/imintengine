#!/usr/bin/env python3
"""S2 June reflectance at the lupine pilot points (all bands, CDSE).

Follow-up to the TESSERA probe (docs/analysis/lupin_tessera_finetune_
exploration.md). That probe reached AUC 0.745 against controls, but the
annual 128-D embedding compresses a whole year while lupine's signature
is a magenta bloom lasting a few weeks. This fetches raw Sentinel-2
reflectance inside the bloom window for the SAME points, so the two can
be compared on identical labels, folds and controls: if raw spectra
carry more, the bottleneck is temporal compression, not the task.

Design notes:
  - Dates come from ``optimal_fetch_dates(mode="era5_then_scl")`` — the
    repo's own cloud screening. Hard-coded date lists are forbidden.
  - Fetching is ``fetch_sentinel2_data``, not bespoke openEO calls.
    ``fetch_seasonal_image`` would be the closer fit but returns only
    ``(image, date)``; we need ``FetchResult.geo`` to map SWEREF99 point
    coordinates to pixels, so we call one level down and stack the bands
    ourselves in the same order.
  - One fetch per (block, year) over the bounding box of THAT group's
    points padded by PAD_M, not the whole 0.1-degree block: the points
    cluster tightly, so this is far smaller than a full block.
  - Output schema is byte-compatible with the TESSERA run's features.npz
    (X, y, lat, lon, year, month, block, dist_lupin, dist_road), so
    ``probe_train.py`` runs unchanged and the numbers are comparable.

Cost: defaults to CDSE because DES returned HTTP 503 all through
2026-09-09. CDSE is PU-billed from a pool shared with Phase-2 backfill
and VPP, so the run is scoped, not open-ended: 490 (block, year) groups,
median bbox 5.4 km², ~1.8 PU each => **~890 PU** for the full set at 11
bands, plus the SCL screen when ``--scl-backend cdse``. Always prove the
path with a small ``--max-groups`` first; a 403 means the pool is
exhausted and every CDSE path dies for the rest of the session.
With ``--source des`` the run is free but competes with the crop
campaign for capacity — set ``--workers`` to the current allotment then
(ask the team; 6 is a ceiling, not a default).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

PAD_M = 200.0          # padding around a group's point cloud, metres
BLOOM = ("06-01", "06-30")   # June: peak Lupinus polyphyllus bloom


def group_bbox(pts: list[dict], to4326) -> dict:
    """WGS84 bbox around a group's points, padded by PAD_M in SWEREF99."""
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    w, s = min(xs) - PAD_M, min(ys) - PAD_M
    e, n = max(xs) + PAD_M, max(ys) + PAD_M
    lon0, lat0 = to4326.transform(w, s)
    lon1, lat1 = to4326.transform(e, n)
    return {"west": lon0, "south": lat0, "east": lon1, "north": lat1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=str(HERE / "chosen_points.json"))
    ap.add_argument("--road-dist", default=str(HERE / "road_dist.json"))
    ap.add_argument("--dist-lupin", default=str(HERE / "dist_lupin.json"),
                    help="reuse the TESSERA run's values so the veto sweep "
                         "compares like with like")
    ap.add_argument("--out", default=str(HERE / "features_s2.npz"))
    ap.add_argument("--cache-dir", default=str(HERE / "s2_cache"))
    ap.add_argument("--max-groups", type=int, default=0,
                    help="0 = all; set small first to prove the path")
    ap.add_argument("--workers", type=int, default=1,
                    help="for DES, MUST match the current allotment (ask "
                         "first); irrelevant for CDSE, which is PU-billed")
    ap.add_argument("--source", default="copernicus",
                    choices=["copernicus", "des"],
                    help="copernicus = CDSE (PU-billed, ~1.8 PU/group); "
                         "des = free but capacity-shared with the crop "
                         "campaign, and down with HTTP 503 on 2026-09-09")
    ap.add_argument("--scl-backend", default="cdse", choices=["cdse", "des"],
                    help="which endpoint runs the SCL screen; cdse keeps the "
                         "whole path off DES but also costs PU")
    ap.add_argument("--screen", default="composite",
                    choices=["composite", "plan", "era5_fetch"],
                    help="composite (default) = cloud-masked June median "
                         "plus peak-bloom colour indices, no date choice at "
                         "all; see s2_june_composite.py. The two single-date "
                         "modes below are kept for comparison but measured "
                         "worse: 7 fetches for 2 usable groups.")
    ap.add_argument("--legacy-screen", dest="_unused", default=None,
                    choices=["plan", "era5_fetch"],
                    help="plan = pre-screen with optimal_fetch_dates "
                         "(preferred, no PU wasted on cloudy scenes). "
                         "era5_fetch = ERA5 candidates only, then let the "
                         "fetch itself reject cloudy scenes via SCL. Use "
                         "when BOTH screening backends are unavailable: on "
                         "2026-09-09 DES was HTTP 503 and the CDSE SCL "
                         "stack returned zero dates, leaving ERA5 as the "
                         "only working stage. Costs PU per rejected scene, "
                         "so --max-tries bounds the damage.")
    ap.add_argument("--max-tries", type=int, default=3,
                    help="era5_fetch only: PU ceiling per group")
    ap.add_argument("--cloud-max", type=float, default=0.15,
                    help="era5_fetch only: max SCL cloud fraction accepted")
    args = ap.parse_args()

    from pyproj import Transformer
    from imint.training.optimal_fetch import (optimal_fetch_dates,
                                          era5_prefilter_dates)
    from imint.fetch import fetch_sentinel2_data, S2L2A_SPECTRAL_BANDS

    # "All bands" in this codebase: B02-B12 incl. B8A and B09 (11 bands).
    # B01/B10 are not in the repo's L2A spectral set.
    bands = list(S2L2A_SPECTRAL_BANDS)

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    to4326 = Transformer.from_crs(3006, 4326, always_xy=True)

    pts = json.loads(Path(args.points).read_text())
    road = json.loads(Path(args.road_dist).read_text())
    dlup = json.loads(Path(args.dist_lupin).read_text())
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for p in pts:
        groups[(p["block"], p["year"])].append(p)
    keys = sorted(groups)
    if args.max_groups:
        keys = keys[: args.max_groups]
    print(f"{len(pts)} points -> {len(groups)} (block, year) groups; "
          f"fetching {len(keys)}", flush=True)

    # Shared across groups: one openEO connection (auth handshake is
    # slow) and one Transformer per raster CRS.
    _shared: dict = {}
    if args.screen == "composite":
        from imint.fetch import _connect_cdse
        _shared["conn"] = _connect_cdse()

    cols: dict[str, list] = defaultdict(list)
    n_fail = 0
    for i, key in enumerate(keys, 1):
        block, year = key
        gpts = groups[key]
        tag = f"{block}_{year}"
        # Cache key includes the mode: composite emits 14 features, the
        # single-date modes 11, and a shared key silently mixes the two
        # into one feature matrix. Caught as an inhomogeneous-shape crash
        # on the first run that reused a single-date cache entry.
        npz = cache / f"{tag}__{args.screen}.npz"

        if npz.exists():
            cached = np.load(npz)
            img, transform_arr, date = (cached["img"], cached["transform"],
                                        str(cached["date"]))
            img_crs = str(cached["crs"]) if "crs" in cached else "EPSG:3006"
        else:
            bbox = group_bbox(gpts, to4326)
            try:
                if args.screen == "composite":
                    from s2_june_composite import (FEATURE_NAMES,
                                                   fetch_june_features)
                    img, img_crs, tr_obj = fetch_june_features(
                        bbox, year, conn=_shared.get("conn"))
                    _shared.setdefault("names", FEATURE_NAMES)
                    date = f"{year}-06-composite"
                    transform_arr = np.asarray(
                        [float(v) for v in tuple(tr_obj)[:6]],
                        dtype=np.float64)
                    np.savez_compressed(npz, img=img, transform=transform_arr,
                                        date=np.array(date),
                                        crs=str(img_crs))
                elif args.screen == "plan":
                    plan = optimal_fetch_dates(
                        bbox, f"{year}-{BLOOM[0]}", f"{year}-{BLOOM[1]}",
                        mode="era5_then_scl", scl_backend=args.scl_backend)
                    if not plan.dates:
                        print(f"[{i}/{len(keys)}] {tag}: no clean bloom date "
                              f"({plan.notes})", flush=True)
                        n_fail += 1
                        continue
                    cands, screen_at_fetch = plan.dates[:1], False
                else:
                    # ERA5 is the only screening stage still reachable.
                    cands = era5_prefilter_dates(
                        bbox, f"{year}-{BLOOM[0]}", f"{year}-{BLOOM[1]}"
                    )[: args.max_tries]
                    if not cands:
                        print(f"[{i}/{len(keys)}] {tag}: no ERA5 clear-sky "
                              f"candidate in June", flush=True)
                        n_fail += 1
                        continue
                    screen_at_fetch = True

                res = None
                for cand in [] if args.screen == "composite" else cands:
                    try:
                        res = fetch_sentinel2_data(
                            source=args.source, date=cand, coords=bbox,
                            # screen_at_fetch: the fetch computes cloud
                            # fraction from SCL and rejects above this.
                            cloud_threshold=(args.cloud_max
                                             if screen_at_fetch else 1.0),
                            include_scl=screen_at_fetch)
                        date = cand
                        break
                    except Exception as inner:
                        print(f"    {tag}: {cand} rejected "
                              f"({type(inner).__name__}: "
                              f"{str(inner)[:60]})", flush=True)
                if args.screen != "composite":
                    if res is None:
                        raise RuntimeError(
                            f"no usable scene in {len(cands)} candidate(s)")
                    missing = [b for b in bands if b not in res.bands]
                    if missing:
                        raise RuntimeError(f"missing bands {missing}")
                    if res.geo is None:
                        raise RuntimeError("no GeoContext — cannot map points")
                    img_crs = res.geo.crs
                    img = np.stack([res.bands[b] for b in bands],
                                   axis=0).astype(np.float32)
                    transform_arr = np.asarray(
                        [float(v) for v in tuple(res.geo.transform)[:6]],
                        dtype=np.float64)
                    np.savez_compressed(npz, img=img,
                                        transform=transform_arr,
                                        date=np.array(date),
                                        crs=str(img_crs))
            except Exception as e:
                print(f"[{i}/{len(keys)}] {tag}: FAIL {type(e).__name__}: "
                      f"{str(e)[:90]}", flush=True)
                n_fail += 1
                continue

        # Sample each point; img is (bands, H, W) on the fetched grid.
        # The raster CRS is NOT assumed: CDSE returns UTM (measured
        # EPSG:32632 and :32633 for Swedish bboxes), so points are
        # projected from lon/lat into whatever CRS the image carries.
        # Reusing the SWEREF99 x/y here would sample the wrong pixels
        # and still return plausible-looking reflectance.
        import rasterio.transform as rt
        tr = rt.Affine(*[float(v) for v in np.asarray(transform_arr).ravel()[:6]])
        to_img = _shared.get(f"proj_{img_crs}")
        if to_img is None:
            to_img = Transformer.from_crs(4326, img_crs, always_xy=True)
            _shared[f"proj_{img_crs}"] = to_img
        H, W = img.shape[1], img.shape[2]
        kept = 0
        for p in gpts:
            px, py = to_img.transform(p["lon"], p["lat"])
            row, col = rt.rowcol(tr, px, py)
            if not (0 <= row < H and 0 <= col < W):
                continue
            vec = img[:, row, col].astype(np.float32)
            if not np.isfinite(vec).all() or np.all(vec == 0):
                continue
            if cols["X"] and len(vec) != len(cols["X"][0]):
                raise RuntimeError(
                    f"{tag}: feature width {len(vec)} != "
                    f"{len(cols['X'][0])} — refusing to mix schemas")
            cols["X"].append(vec)
            cols["y"].append(p["label"])
            cols["lat"].append(p["lat"]); cols["lon"].append(p["lon"])
            cols["year"].append(year); cols["month"].append(p["month"])
            cols["block"].append(block)
            k = f"{p['lon']:.6f},{p['lat']:.6f}"
            cols["dist_lupin"].append(dlup.get(k, np.nan))
            cols["dist_road"].append(road.get(k, np.nan))
            kept += 1
        print(f"[{i}/{len(keys)}] {tag}: {kept}/{len(gpts)} pts "
              f"(total {len(cols['y'])})", flush=True)
        np.savez_compressed(Path(args.out).with_suffix(".partial.npz"),
                            **{k: np.asarray(v) for k, v in cols.items()})

    if not cols.get("X"):
        # Every group failed — say so and exit nonzero rather than crashing
        # on a missing key or, worse, writing an empty features file that
        # later looks like a real but disappointing result.
        print(f"NO DATA: all {len(keys)} groups failed "
              f"({n_fail} failures). Nothing written.", file=sys.stderr)
        raise SystemExit(2)

    X = np.asarray(cols.pop("X"), dtype=np.float32)
    arrs = {k: np.asarray(v) for k, v in cols.items()}
    print(f"total: X={X.shape}, pos={int(arrs['y'].sum())}, "
          f"neg={int((arrs['y'] == 0).sum())}, failed_groups={n_fail}")
    if n_fail:
        print(f"WARNING: {n_fail}/{len(keys)} groups missing — coverage is "
              f"partial, do not read the result as complete", file=sys.stderr)
    np.savez_compressed(args.out, X=X, **arrs)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
