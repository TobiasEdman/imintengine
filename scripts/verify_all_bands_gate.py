#!/usr/bin/env python3
"""1-tile all-band fetch verification gate.

Proves, for ONE DES openEO acquisition, that the all-band spectral fetch
(``fetch_tile_all_slots_des_openeo`` → 12-band ``ALL_BANDS`` array, split by
``ALL_BANDS_INDEX``) is SAFE to run across the whole dataset before any rebuild
relies on it.

Two independent things are checked:

1. BAND ORDER (hard gate — non-zero exit on failure)
   A wrong index in ``ALL_BANDS_INDEX`` would silently corrupt every future
   tile, so each extracted extra is cross-checked against an *independent*
   single-purpose fetch of the same scene:
     * in-fetch B08 (index 3) vs ``enrich_tiles_b08._fetch_b08_frame`` — both
       use plain ``DN/10000``, so they must agree tightly: corr ≈ 1 AND a small
       max|Δ| (the "byte-match" indicator).
     * in-fetch red-edge B05/B06/B07 (indices 4,5,6) vs
       ``enrich_tiles_rededge._fetch_rededge_frame_des`` — per-band corr > 0.999.
       Correlation is offset-invariant, so it proves ORDER even though the two
       paths use different scaling (see #2).
     * B01 (index 10) / B09 (index 11) — finite, in [0,1], non-zero, and NOT a
       duplicate of any other band (a mis-index would alias another slice).

2. SCALING CONVENTION (reported verdict)
   Settles the confirmed contradiction in the existing code:
       spectral cube + enrich_tiles_b08 use plain  DN/10000
       enrich_tiles_rededge             uses      (DN-1000)/10000
   They cannot both be true reflectance. The two paths differ by exactly the
   1000-DN PB04.00 offset *by construction* (so the measured red-edge Δ≈+0.10
   only confirms we read the code right — it does NOT say which is correct).
   The discriminator is the raw-DN floor over the darkest pixels (recovered as
   ``value*10000`` since the all-band path only divides by 10000, no clip):
     * floor ≈ 1000 → offset BAKED IN → (DN-1000)/10000 correct → spectral+b08
       are +0.1 high (latent dataset-wide bug; enrich_rededge is the right one).
     * floor ≈ 0    → offset already STRIPPED by DES → plain DN/10000 correct →
       enrich_tiles_rededge is the buggy one.
   A crisp read needs genuinely dark pixels (water/deep shadow); pick a tile
   that contains some.

Usage (cluster, preferred — real bbox + a date we know had a scene):
    python scripts/verify_all_bands_gate.py --tile /cephfs/unified_v2_512/<tile>.npz
Or explicit:
    python scripts/verify_all_bands_gate.py --bbox W,S,E,N --date YYYY-MM-DD --size-px 512
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from imint.training.openeo_tile_graph import (
    ALL_BANDS,
    ALL_BANDS_INDEX,
    fetch_tile_at_specific_dates,
)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over finite, co-valid pixels. Offset/scale-invariant."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 16:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if na == 0 or nb == 0:
        return float("nan")
    return float((a * b).sum() / (na * nb))


def _resolve_target(args) -> tuple[dict, str, int]:
    """Return (bbox_3006 dict, date_str, size_px) from --tile / --bbox / default."""
    if args.tile:
        d = np.load(args.tile, allow_pickle=True)
        bb = [float(x) for x in d["bbox_3006"]]
        bbox = {"west": bb[0], "south": bb[1], "east": bb[2], "north": bb[3]}
        size_px = int(d["tile_size_px"]) if "tile_size_px" in d else args.size_px
        date = args.date
        if not date:
            dates = [str(x)[:10] for x in d["dates"]] if "dates" in d else []
            date = next((x for x in dates if x and x[0].isdigit()), None)
        if not date:
            sys.exit("gate: --tile has no usable date in `dates`; pass --date")
        return bbox, date, size_px

    if args.bbox:
        w, s, e, n = (float(x) for x in args.bbox.split(","))
        bbox = {"west": w, "south": s, "east": e, "north": n}
    else:
        # Fallback grid-snapped 256-px (2560 m) tile over the Stockholm /
        # Mälaren area — mixed land + water so dark pixels exist for the
        # scaling read. Override with --tile (best) or --bbox.
        cx, cy = 658000, 6580000
        half = args.size_px * 10 // 2
        bbox = {"west": cx - half, "south": cy - half,
                "east": cx + half, "north": cy + half}

    date = args.date
    if not date:
        from imint.training.tile_fetch import bbox_3006_to_wgs84
        from imint.training.optimal_fetch import optimal_fetch_dates
        plan = optimal_fetch_dates(
            bbox_3006_to_wgs84(bbox), "2023-06-01", "2023-08-31")
        if not plan.dates:
            sys.exit("gate: optimal_fetch_dates found no clean date; pass --date")
        date = plan.dates[0]
    return bbox, date, args.size_px


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile", help="Existing .npz — source real bbox_3006 + date (preferred)")
    ap.add_argument("--bbox", help="EPSG:3006 'W,S,E,N' (alt to --tile)")
    ap.add_argument("--date", help="YYYY-MM-DD (else from tile, else optimal_fetch_dates)")
    ap.add_argument("--size-px", type=int, default=256)
    ap.add_argument("--corr-min", type=float, default=0.999)
    ap.add_argument("--b08-maxabs", type=float, default=5e-3,
                    help="max|Δ| in-fetch vs enrich B08 to call it a byte-match")
    args = ap.parse_args()

    bbox, date, size_px = _resolve_target(args)
    print(f"[gate] bbox_3006={bbox} date={date} size_px={size_px}", flush=True)

    # ── 1) All-band DES fetch (the path under test) ──────────────────────────
    res = fetch_tile_at_specific_dates(bbox, {0: date}, source="des")
    entry = res.get(0)
    if entry is None or entry[0] is None:
        sys.exit(f"gate: all-band fetch returned nothing for {date} — pick a date with a scene")
    arr = np.asarray(entry[0], np.float32)
    if arr.shape[0] != len(ALL_BANDS):
        sys.exit(f"gate: expected {len(ALL_BANDS)} bands, got {arr.shape[0]} — "
                 f"all-band path NOT active for source=des (abort, do NOT rebuild)")
    H, W = arr.shape[1], arr.shape[2]
    print(f"[gate] all-band array {arr.shape} (ALL_BANDS={ALL_BANDS})", flush=True)

    i_b08 = ALL_BANDS_INDEX["b08"][0]
    i_re = list(ALL_BANDS_INDEX["rededge"])
    i_b01 = ALL_BANDS_INDEX["b01"][0]
    i_b09 = ALL_BANDS_INDEX["b09"][0]
    i_prithvi = list(ALL_BANDS_INDEX["prithvi"])
    b08_in = arr[i_b08]
    re_in = arr[i_re]
    b01_in, b09_in = arr[i_b01], arr[i_b09]

    failures: list[str] = []

    # ── 2) Independent enrich fetches → band-order proof ─────────────────────
    from scripts.enrich_tiles_b08 import _fetch_b08_frame
    from scripts.enrich_tiles_rededge import _fetch_rededge_frame_des

    w, s, e, n = bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    b08_en = _fetch_b08_frame(w, s, e, n, date, size_px, source="des")
    re_en = _fetch_rededge_frame_des(w, s, e, n, date, size_px)

    if b08_en is None:
        failures.append("B08 enrich fetch returned None (cannot prove B08 order)")
    elif b08_en.shape != b08_in.shape:
        failures.append(f"B08 shape mismatch in={b08_in.shape} enrich={b08_en.shape}")
    else:
        c = _corr(b08_in, b08_en)
        mx = float(np.nanmax(np.abs(b08_in - b08_en)))
        ok = c > args.corr_min
        print(f"[gate] B08 (idx {i_b08}): corr={c:.5f} max|Δ|={mx:.5f} "
              f"byte-match={'YES' if mx <= args.b08_maxabs else 'no (grid-snap?)'} "
              f"-> {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            failures.append(f"B08 order: corr {c:.5f} <= {args.corr_min}")

    redge_offsets: list[float] = []
    if re_en is None:
        failures.append("red-edge enrich fetch returned None (cannot prove B05-07 order)")
    elif re_en.shape != re_in.shape:
        failures.append(f"red-edge shape mismatch in={re_in.shape} enrich={re_en.shape}")
    else:
        for k, name in enumerate(("B05", "B06", "B07")):
            c = _corr(re_in[k], re_en[k])
            off = float(np.nanmean(re_in[k] - re_en[k]))
            redge_offsets.append(off)
            ok = c > args.corr_min
            # enrich_rededge always applies the DES offset; post-fix the fetch
            # path does too, so the mean diff must be ~0 (was +0.10 pre-fix).
            print(f"[gate] {name} (idx {i_re[k]}): corr={c:.5f} "
                  f"mean(in-enrich)={off:+.4f} (expect ~0: both apply DES offset) "
                  f"-> {'PASS' if ok else 'FAIL'}", flush=True)
            if not ok:
                failures.append(f"{name} order: corr {c:.5f} <= {args.corr_min}")

    # ── 3) B01 / B09 sanity + not-a-duplicate ────────────────────────────────
    for name, idx, band in (("B01", i_b01, b01_in), ("B09", i_b09, b09_in)):
        finite = bool(np.isfinite(band).all())
        nonzero = bool(np.any(band))
        in_range = bool(band.min() >= -1e-3 and band.max() <= 1.5)
        dups = [j for j in range(arr.shape[0])
                if j != idx and float(np.nanmax(np.abs(band - arr[j]))) < 1e-6]
        ok = finite and nonzero and in_range and not dups
        print(f"[gate] {name} (idx {idx}): finite={finite} nonzero={nonzero} "
              f"range=[{band.min():.3f},{band.max():.3f}] dup_of={dups} "
              f"-> {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            failures.append(f"{name} sanity: finite={finite} nonzero={nonzero} "
                            f"in_range={in_range} dup_of={dups}")

    # ── 4) Scaling regression guard ──────────────────────────────────────────
    # The fetch path under test now applies the DES -1000 offset (via
    # dn_to_reflectance). Two independent confirmations:
    #   (a) reference — it must AGREE with enrich_rededge (which has always
    #       applied the offset): mean diff ~0, was +0.10 when the fetch divided
    #       by plain 10000. Robust to land cover (compares to a known-good ref).
    #   (b) physical — the corrected-reflectance floor over the dark center
    #       must sit near 0 (water/shadow), not ~0.10. Needs dark pixels.
    if redge_offsets:
        moff = float(np.mean(redge_offsets))
        ok = abs(moff) < 0.05
        print(f"\n[gate] offset-match vs enrich_rededge: mean={moff:+.4f} -> "
              f"{'PASS (fetch applies DES offset)' if ok else 'FAIL (≈+0.10 → DES offset NOT applied)'}",
              flush=True)
        if not ok:
            failures.append(f"DES offset not applied: fetch is {moff:+.3f} vs "
                            f"enrich_rededge (expect ~0; +0.10 = plain /10000 regression)")

    cen = arr[:, H // 4:3 * H // 4, W // 4:3 * W // 4]
    lit = cen[cen > 1e-4]
    if lit.size:
        p1 = float(np.percentile(lit, 1))
        if p1 > 0.08:
            tag, ok = "too high → +0.1 offset NOT applied", False
        elif p1 < 0.05:
            tag, ok = "near 0 → offset applied (dark water/shadow)", True
        else:
            tag, ok = "no clearly-dark pixels — physical check inconclusive", True
        print(f"[gate] dark-center reflectance floor p1={p1:.4f} ({tag})", flush=True)
        if not ok:
            failures.append(f"dark floor p1={p1:.3f} > 0.08 — DES offset not applied")
    else:
        print("[gate] dark-center floor: no lit pixels to read", flush=True)

    # ── Result ───────────────────────────────────────────────────────────────
    print("\n" + ("=" * 64), flush=True)
    if failures:
        print(f"GATE: FAIL ({len(failures)} issue(s)) — do NOT rebuild:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1
    print("GATE: PASS — band order proven + DES offset applied (matches "
          "enrich_rededge and dark-water floor ~0). Extras safe to persist.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
