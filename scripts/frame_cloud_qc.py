#!/usr/bin/env python3
"""Per-frame cloud QC + replace for the re-coreg campaign (SPEC §3).

Two phases over every tile in ``unified_v2_512_recoreg`` and, for each tile,
every temporal frame (spectral slots 0-3 + ``frame_2016``):

PHASE A — measure (cheap, in-place, no network)
    cloud verdict = ERA5-criteria check on the frame's stored date
                    (``optimal_fetch.era5_prefilter_dates``)
                    AND an actual-pixel cloud fraction.

    Pixel-cloud metric — **B02 brightness proxy on the stored reflectance**.
    The stored npz carries ONLY the 6-band Prithvi cube
    (``B02,B03,B04,B8A,B11,B12``); there is no SCL band and no 12-band TOA
    stack on disk, so the ``cot_l1c`` analyzer (needs 12 bands incl. B10
    cirrus) and an SCL cloud-fraction both require re-downloading the SAFE —
    which defeats the whole point of a *cheap* measure pass. The repo already
    gates haze on B02 mean reflectance (``imint.training.cdse_s2`` "B02 haze
    gate"); we reuse that signal as a per-pixel fraction: clouds are bright in
    the blue band, vegetation/soil/water are dark, so
    ``mean(B02_valid > 0.2)`` is a zero-network cloud proxy computed directly
    on the stored cube. ERA5 gates the *date* (a predictor; misses real
    clouds — the frame-0 ~30%-cloud case passed ERA5 yet was visibly cloudy);
    the B02 fraction gates the *actual* pixels. A frame FAILS when its B02
    cloud fraction exceeds ``--threshold`` (default 0.20).

    Output: a JSON report — per (tile, slot) ERA5 verdict + B02 cloud fraction
    + pass/fail, plus per-slot fail-rate aggregates to size Phase B.

PHASE B — replace (heavy; des / sen2cor)
    For each frame that FAILS the pixel test, re-select a cleaner date in the
    SAME year + SAME slot window via
    ``optimal_fetch.optimal_fetch_dates(mode="era5_then_scl")``, then re-fetch
    that single slot via ``fetch_spectral.fetch_tile_spectral`` (M1 grid-snap +
    M2 inter-frame MI coreg to the tile anchor — see CLAUDE.md "Koregistrering
    M1→M2"; the M2 sign convention lives inside ``coregister_interframe`` and is
    NOT touched here). des for >=2018 slots; the sen2cor path for pre-2018
    (``frame_2016`` + slot 0 of 2018-tiles).

    TWO HARD GUARDS:
      1. NEVER-WORSE — only overwrite the stored frame if the candidate's B02
         cloud fraction is *strictly lower*; otherwise keep the original.
      2. TEMPORAL-MATCHING — the replacement date MUST stay in the same
         (year, slot-window). The candidate window is derived from the slot's
         own year (slot 0 = tile_year-1 autumn; slots 1-3 = tile_year growing
         season; frame_2016 = 2016) via the canonical
         ``select_scenes._window_for_target``. A candidate outside that window
         is impossible by construction (the selector is only ever asked for
         dates inside it), and we additionally assert the chosen date's year
         matches the slot year before fetching. This keeps crop frames on the
         label-year signal (CLAUDE.md "Dataregler — Temporal matchning").

This DEVIATES from the campaign's "reuse stored dates" rule by re-selecting for
failed frames — intentional and signed off (SPEC §3); the same-year+window
guard is non-negotiable.

Cross-cutting (SPEC §Sequencing): atomic writes (tmp + ``os.replace``);
idempotent / resumable; ``--dry-run`` plans without touching disk;
``--workers`` for the des-billed replace pass.

Usage
-----
    # Phase A — measure (cheap python:3.11-slim pod)
    python scripts/frame_cloud_qc.py --phase measure \
        --data-dir /data/unified_v2_512_recoreg \
        --threshold 0.2 \
        --report /data/debug/frame_cloud_qc_report.json \
        --workers 8

    # Phase B — replace (needs des creds + the sen2cor image for pre-2018)
    python scripts/frame_cloud_qc.py --phase replace \
        --data-dir /data/unified_v2_512_recoreg \
        --report /data/debug/frame_cloud_qc_report.json \
        --threshold 0.2 \
        --workers 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Canonical band order of the stored 6-band Prithvi cube. B02 (blue) is slot
# index 0 within every 6-band frame — the cloud-brightness band. Imported as a
# fact, not a literal, so it can never drift from the producer.
from imint.training.tile_fetch import PRITHVI_BANDS  # noqa: E402

# Frame layout in the stored npz.
N_BANDS = 6
N_SLOTS = 4
B02_OFFSET = PRITHVI_BANDS.index("B02")  # 0 — within-frame index of the blue band

# Logical frame keys: the four temporal slots plus the 2016 background frame.
SLOT_KEYS = [f"slot:{i}" for i in range(N_SLOTS)]
FRAME_2016_KEY = "frame_2016"
ALL_FRAME_KEYS = SLOT_KEYS + [FRAME_2016_KEY]

# B02 brightness threshold (reflectance) above which a pixel is counted "cloud".
# Matches the repo's B02 haze gate (imint.training.cdse_s2). Not a tunable knob
# of this script — the *tile-level* pass/fail threshold is --threshold; this is
# the per-pixel brightness cut that defines the fraction.
B02_BRIGHT_REFLECTANCE = 0.20

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ── Pixel-cloud metric ───────────────────────────────────────────────────────


def b02_cloud_fraction(
    frame: np.ndarray,
    *,
    bright: float = B02_BRIGHT_REFLECTANCE,
) -> float:
    """Cloud fraction of one ``(6, H, W)`` frame via the B02 brightness proxy.

    Fraction of *valid* pixels whose blue-band (B02) reflectance exceeds
    ``bright``. "Valid" excludes the exact-zero nodata fill that empty / masked
    pixels carry (zeros would otherwise dilute the fraction and make a cloudy
    frame look clean). An all-zero / empty frame has no valid pixels → returns
    ``1.0`` (treated as maximally cloudy / unusable) so it never passes QC.

    Args:
        frame: ``(6, H, W)`` float32 reflectance in ``PRITHVI_BANDS`` order.
        bright: per-pixel B02 reflectance cut (default 0.20).

    Returns:
        Cloud fraction in [0, 1].
    """
    if frame is None or frame.ndim != 3 or frame.shape[0] < N_BANDS:
        return 1.0
    b02 = np.asarray(frame[B02_OFFSET], dtype=np.float32)
    valid = b02 > 0.0  # exact-zero = nodata / unfilled
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 1.0
    return float(np.count_nonzero((b02 > bright) & valid) / n_valid)


# ── npz frame access ─────────────────────────────────────────────────────────


def _slot_frame(data: dict, slot_idx: int) -> np.ndarray | None:
    """The ``(6, H, W)`` frame for temporal slot ``slot_idx``, or ``None``.

    ``None`` when the spectral cube is absent / too short, the temporal mask
    marks the slot empty, or the slice is all-zero (unfilled).
    """
    spec = data.get("spectral")
    if spec is None:
        return None
    spec = np.asarray(spec, dtype=np.float32)
    if spec.shape[0] < (slot_idx + 1) * N_BANDS:
        return None
    tmask = data.get("temporal_mask")
    if tmask is not None and slot_idx < len(tmask) and int(tmask[slot_idx]) == 0:
        return None
    frame = spec[slot_idx * N_BANDS:(slot_idx + 1) * N_BANDS]
    if not np.any(frame):
        return None
    return frame


def _frame_2016(data: dict) -> np.ndarray | None:
    """The ``(6, H, W)`` ``frame_2016`` background frame, or ``None`` if absent."""
    if int(data.get("has_frame_2016", 0)) != 1:
        return None
    f = data.get("frame_2016")
    if f is None:
        return None
    f = np.asarray(f, dtype=np.float32)
    if f.ndim != 3 or f.shape[0] < N_BANDS or not np.any(f):
        return None
    return f


def _frame_date(data: dict, key: str) -> str:
    """Stored ISO date (YYYY-MM-DD) for a frame key, or ``""`` if absent."""
    if key == FRAME_2016_KEY:
        raw = data.get("frame_2016_date")
        return str(raw)[:10] if raw is not None else ""
    slot_idx = int(key.split(":", 1)[1])
    dates = data.get("dates")
    if dates is None or slot_idx >= len(dates):
        return ""
    return str(dates[slot_idx])[:10]


def _get_frame(data: dict, key: str) -> np.ndarray | None:
    """Dispatch a frame key to its ``(6, H, W)`` array, or ``None``."""
    if key == FRAME_2016_KEY:
        return _frame_2016(data)
    return _slot_frame(data, int(key.split(":", 1)[1]))


# ── Temporal-matching: slot → (year, window) ─────────────────────────────────


def slot_year(tile_year: int, key: str) -> int:
    """The calendar year a frame key belongs to, given the tile's label year.

    slot 0  → autumn of (tile_year - 1)        [CLAUDE.md temporal-matching]
    slot 1-3→ growing season of tile_year
    frame_2016 → 2016 (fixed background)
    """
    if key == FRAME_2016_KEY:
        return 2016
    slot_idx = int(key.split(":", 1)[1])
    return tile_year - 1 if slot_idx == 0 else tile_year


def frame_window(tile_year: int, key: str) -> tuple[str, str]:
    """``(date_start, date_end)`` for a frame key — the SAME-WINDOW guard.

    Reuses the canonical per-target month windows from the sen2cor selector
    (``select_scenes._window_for_target``) so this script and the sen2cor
    Phase-2 pass agree on slot windows by construction. ``frame_2016`` maps to
    its 2016 summer window; ``slot:N`` to the slot's window in ``slot_year``.
    """
    from scripts.sen2cor_pipeline.select_scenes import _window_for_target
    return _window_for_target(key, slot_year(tile_year, key))


def _tile_year(data: dict) -> int | None:
    """Canonical tile (label) year — ``tessera_year``, or ``None`` if absent."""
    ty = data.get("tessera_year")
    if ty is None:
        ty = data.get("lpis_year")
    if ty is None:
        ty = data.get("label_year")
    try:
        return int(ty) if ty is not None else None
    except (TypeError, ValueError):
        return None


# ── Phase A: measure ─────────────────────────────────────────────────────────


def _era5_pass(bbox_wgs84: dict, date_str: str) -> bool:
    """True if ``date_str`` is in the ERA5-clear set for a 1-day window.

    Reuses ``optimal_fetch.era5_prefilter_dates`` (the SPEC-mandated date
    predictor). A 1-day window asks ERA5 exactly about the stored date.
    """
    from imint.training.optimal_fetch import era5_prefilter_dates
    if not date_str:
        return False
    try:
        clear = set(era5_prefilter_dates(bbox_wgs84, date_str, date_str))
    except Exception as e:  # ERA5 is best-effort; don't crash the measure pass.
        _log(f"    [measure] ERA5 lookup failed for {date_str}: "
             f"{type(e).__name__}: {str(e)[:120]}")
        return False
    return date_str in clear


def measure_tile(
    npz_path: str,
    *,
    threshold: float,
    bright: float,
    with_era5: bool,
) -> dict:
    """Per-frame cloud verdict for one tile npz.

    Returns a dict ``{tile, tile_year, frames: {key: {...}}}`` where each frame
    entry carries the stored date, the B02 cloud fraction, the ERA5 verdict, and
    the pass/fail decision (fail ⇔ ``cloud_frac > threshold``). Frames that are
    absent / empty are reported with ``present=False`` and excluded from
    fail-rate aggregates.
    """
    name = Path(npz_path).stem
    try:
        with np.load(npz_path, allow_pickle=True) as d:
            data = {k: d[k] for k in d.files}
    except Exception as e:
        return {"tile": name, "error": f"{type(e).__name__}: {str(e)[:160]}",
                "frames": {}}

    tile_year = _tile_year(data)
    bbox_wgs84 = None
    if with_era5:
        bbox = data.get("bbox_3006")
        if bbox is not None and len(bbox) == 4:
            from imint.training.tile_fetch import bbox_3006_to_wgs84
            w, s, e, n = (int(x) for x in bbox)
            bbox_wgs84 = bbox_3006_to_wgs84(
                {"west": w, "south": s, "east": e, "north": n})

    frames: dict[str, dict] = {}
    for key in ALL_FRAME_KEYS:
        frame = _get_frame(data, key)
        if frame is None:
            frames[key] = {"present": False}
            continue
        date_str = _frame_date(data, key)
        cloud_frac = b02_cloud_fraction(frame, bright=bright)
        era5_ok: bool | None = None
        if with_era5 and bbox_wgs84 is not None:
            era5_ok = _era5_pass(bbox_wgs84, date_str)
        # The pixel measure gates the actual frame; ERA5 is recorded alongside
        # as the date-predictor diagnostic. Fail ⇔ pixels exceed threshold.
        pixel_fail = cloud_frac > threshold
        frames[key] = {
            "present": True,
            "date": date_str,
            "cloud_frac": round(cloud_frac, 4),
            "era5_clear": era5_ok,
            "pass": not pixel_fail,
        }
    return {"tile": name, "tile_year": tile_year, "frames": frames}


def run_measure(args: argparse.Namespace) -> int:
    """Phase A driver — measure every frame, write the JSON report atomically."""
    npz_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not npz_paths:
        _log(f"[measure] no .npz under {args.data_dir}")
        return 1
    _log(f"[measure] {len(npz_paths)} tiles; threshold={args.threshold} "
         f"B02>{args.bright}; ERA5={'on' if not args.no_era5 else 'off'}; "
         f"workers={args.workers}")

    results: list[dict] = []

    def _one(p: str) -> dict:
        return measure_tile(
            p, threshold=args.threshold, bright=args.bright,
            with_era5=not args.no_era5)

    if args.workers <= 1:
        for p in npz_paths:
            results.append(_one(p))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_one, p): p for p in npz_paths}
            for fut in as_completed(futs):
                results.append(fut.result())

    results.sort(key=lambda r: r["tile"])
    report = _build_report(results, args)
    if args.dry_run:
        _log(f"[measure] DRY-RUN — would write report for {len(results)} tiles "
             f"to {args.report}")
        _log(json.dumps(report["summary"], indent=2))
        return 0
    _atomic_write_json(args.report, report)
    _log(f"[measure] wrote {args.report}")
    _log(json.dumps(report["summary"], indent=2))
    return 0


def _build_report(results: list[dict], args: argparse.Namespace) -> dict:
    """Assemble the per-tile results + per-slot fail-rate summary."""
    per_slot_present: dict[str, int] = {k: 0 for k in ALL_FRAME_KEYS}
    per_slot_fail: dict[str, int] = {k: 0 for k in ALL_FRAME_KEYS}
    replace_list: list[dict] = []
    for r in results:
        for key, fr in r.get("frames", {}).items():
            if not fr.get("present"):
                continue
            per_slot_present[key] += 1
            if not fr.get("pass", True):
                per_slot_fail[key] += 1
                replace_list.append(
                    {"tile": r["tile"], "slot": key,
                     "date": fr.get("date", ""),
                     "cloud_frac": fr.get("cloud_frac")})
    summary = {
        "n_tiles": len(results),
        "threshold": args.threshold,
        "b02_bright": args.bright,
        "metric": "b02_brightness_fraction",
        "per_slot": {
            k: {"present": per_slot_present[k], "fail": per_slot_fail[k],
                "fail_rate": round(per_slot_fail[k] / per_slot_present[k], 4)
                if per_slot_present[k] else 0.0}
            for k in ALL_FRAME_KEYS},
        "n_frames_to_replace": len(replace_list),
    }
    return {"summary": summary, "tiles": results, "replace_list": replace_list}


# ── Phase B: replace ─────────────────────────────────────────────────────────


def _select_cleaner_date(
    bbox_wgs84: dict,
    tile_year: int,
    key: str,
    *,
    max_aoi_cloud: float,
) -> str | None:
    """Re-select the cleanest date in the SAME (year, slot-window).

    Calls ``optimal_fetch_dates(mode="era5_then_scl")`` bounded to the slot's
    own window (``frame_window``). Returns the first (sorted ascending) clean
    date, or ``None`` if the selector found none. The returned date is, by
    construction, inside the slot window; we additionally assert its year equals
    ``slot_year(tile_year, key)`` (the temporal-matching guard) before returning.
    """
    from imint.training.optimal_fetch import optimal_fetch_dates

    date_start, date_end = frame_window(tile_year, key)
    want_year = slot_year(tile_year, key)
    try:
        plan = optimal_fetch_dates(
            bbox_wgs84, date_start, date_end,
            mode="era5_then_scl", max_aoi_cloud=max_aoi_cloud)
    except Exception as e:
        _log(f"    [replace] date-select failed ({date_start}..{date_end}): "
             f"{type(e).__name__}: {str(e)[:140]}")
        return None
    for d in plan.dates:
        # TEMPORAL-MATCHING GUARD: same year as the slot, full stop.
        if int(d[:4]) == want_year:
            return d
    return None


def _fetch_replacement_frame(
    data: dict,
    key: str,
    new_date: str,
    *,
    halo_px: int,
) -> np.ndarray | None:
    """Re-fetch ONE slot on ``new_date`` via ``fetch_tile_spectral`` (M1+M2).

    Routes >=2018 dates through des and pre-2018 dates through the sen2cor
    fallthrough (``fetch_tile_spectral`` picks the path internally: pre-2018
    slots skip the des tile-graph and take the ``l1c_sen2cor`` fallthrough,
    which no-ops off the sen2cor image). The fetch is co-centred and co-gridded
    with the stored tile (same EPSG:3006 centre + ``tile_size_px``), and M2
    coregisters the new frame onto the tile anchor's grid. Returns the
    ``(6, H, W)`` frame, or ``None`` if the fetch yielded nothing for the slot.
    """
    from imint.training.fetch_spectral import fetch_tile_spectral
    from imint.training.tile_config import TileConfig

    east = data.get("easting")
    north = data.get("northing")
    size_px = data.get("tile_size_px")
    if east is None or north is None or size_px is None:
        _log(f"    [replace] tile missing geometry (easting/northing/size)")
        return None
    tile = TileConfig(size_px=int(size_px), gsd_m=10)
    # backend is ALWAYS "des": fetch_tile_spectral only accepts an M2-capable
    # backend ("des"/"cdse-openeo"; sen2cor is 6-band/no-halo so it cannot be
    # passed here). It then routes pre-2018 dates INTERNALLY to the l1c_sen2cor
    # fallthrough on the SAME halo grid (the >=2018/pre-2018 split is governed by
    # fetch_spectral.DES_L2A_FLOOR there) — so one "des" entry handles both eras.
    backend = "des"
    # We fetch a SINGLE slot (1-frame request) → M1 grid-snap onto the tile's
    # halo grid (M2 is a no-op on one frame). The caller then coregisters this
    # frame onto the tile's STORED anchor (_coreg_to_stored_anchor) to remove the
    # ~2 px relative ortho drift M1 can't touch — without that the replacement is
    # misaligned with the rest of the stack + the label.
    try:
        res = fetch_tile_spectral(
            (int(east), int(north)),
            tile=tile,
            dates={0: new_date},
            n_frames=1,
            backend=backend,
            halo_px=halo_px,
            coregister=True,
        )
    except Exception as e:
        _log(f"    [replace] fetch_tile_spectral failed {new_date}: "
             f"{type(e).__name__}: {str(e)[:160]}")
        return None
    if res is None:
        return None
    spec = res.get("spectral")
    if spec is None:
        return None
    spec = np.asarray(spec, dtype=np.float32)
    if spec.shape[0] < N_BANDS:
        return None
    return spec[:N_BANDS]


def _coreg_to_stored_anchor(
    data: dict, frame: np.ndarray, key: str,
) -> tuple[np.ndarray | None, str]:
    """Coregister a fresh ``(6, H, W)`` replacement onto the tile's stored Phase-1
    anchor (``coreg_ref_frame``).

    M1 grid-snap alone leaves the ~2 px relative ortho drift between acquisition
    dates that M2/coreg-to-anchor exists to remove — so a standalone-fetched
    replacement lands misaligned with the rest of the stack AND the label. Mirrors
    ``run_sen2cor_per_scene._coreg_frame_to_anchor`` (anchor = fixed ``target``,
    fresh = ``reference``; take the 2nd return; ``reference_band=2`` = B04;
    ``*_transform=None`` ⇒ sub-pixel only; the MI estimator auto-rejects ≳1 px so a
    bad match is a no-op). Returns ``(aligned, "ok")`` or ``(None, reason)`` — never
    write a mis-aligned frame. Refuses to replace the anchor slot itself (that would
    require re-coregistering the whole stack onto a new reference)."""
    from imint.coregistration import coregister_to_reference

    ref_idx = int(data["coreg_ref_frame"]) if "coreg_ref_frame" in data else -1
    spec = data.get("spectral")
    if ref_idx < 0 or spec is None:
        return None, "no-anchor"
    if key != FRAME_2016_KEY and int(key.split(":", 1)[1]) == ref_idx:
        return None, "is-anchor-frame"
    spec = np.asarray(spec, dtype=np.float32)
    if spec.shape[0] < (ref_idx + 1) * N_BANDS:
        return None, "anchor-out-of-range"
    anchor = spec[ref_idx * N_BANDS:(ref_idx + 1) * N_BANDS]
    if not bool(np.any(anchor)):
        return None, "anchor-empty"
    anchor_hwc = np.transpose(anchor, (1, 2, 0))
    fresh_hwc = np.transpose(np.asarray(frame, np.float32), (1, 2, 0)).copy()
    _a, aligned, _m = coregister_to_reference(
        target=anchor_hwc, reference=fresh_hwc,
        target_transform=None, reference_transform=None,
        subpixel=True, reference_band=2)
    return np.ascontiguousarray(np.transpose(aligned, (2, 0, 1)), np.float32), "ok"


def _write_replacement(
    npz_path: str,
    key: str,
    frame: np.ndarray,
    new_date: str,
    new_cloud_frac: float,
) -> None:
    """Atomically write a replacement frame into the tile npz (tmp + os.replace).

    Updates the frame array, its stored date, the temporal mask (for slots), and
    a per-frame provenance/QC record so a later audit can tell a QC-replaced
    frame from an original one. Everything else in the npz is preserved verbatim.
    """
    with np.load(npz_path, allow_pickle=True) as d:
        out = {k: d[k] for k in d.files}

    frame = np.asarray(frame, dtype=np.float32)
    if key == FRAME_2016_KEY:
        out["frame_2016"] = frame
        out["has_frame_2016"] = np.int32(1)
        out["frame_2016_date"] = np.str_(new_date)
        out["frame_2016_year"] = np.int32(int(new_date[:4]) if new_date else 0)
    else:
        slot_idx = int(key.split(":", 1)[1])
        spec = (np.asarray(out["spectral"], dtype=np.float32).copy()
                if "spectral" in out else
                np.zeros((N_SLOTS * N_BANDS, *frame.shape[1:]), dtype=np.float32))
        spec[slot_idx * N_BANDS:(slot_idx + 1) * N_BANDS] = frame
        out["spectral"] = spec
        # dates / temporal_mask kept length-N_SLOTS and consistent.
        dates = ([str(x) for x in out["dates"]] if "dates" in out
                 else [""] * N_SLOTS)
        while len(dates) < N_SLOTS:
            dates.append("")
        dates[slot_idx] = new_date
        out["dates"] = np.array(dates)
        tmask = (list(out["temporal_mask"]) if "temporal_mask" in out
                 else [0] * N_SLOTS)
        while len(tmask) < N_SLOTS:
            tmask.append(0)
        tmask[slot_idx] = 1
        out["temporal_mask"] = np.array(tmask, dtype=np.uint8)

    # Provenance — survives across idempotent re-runs, records the QC verdict.
    safe_key = key.replace(":", "_")
    out[f"qc_replaced_{safe_key}"] = np.int32(1)
    out[f"qc_replaced_{safe_key}_date"] = np.str_(new_date)
    out[f"qc_replaced_{safe_key}_cloud_frac"] = np.float32(new_cloud_frac)

    tmp_base = str(npz_path) + ".tmp"
    np.savez_compressed(tmp_base, **out)
    os.replace(tmp_base + ".npz", npz_path)


def replace_tile_frame(
    npz_path: str,
    key: str,
    orig_cloud_frac: float,
    *,
    threshold: float,
    bright: float,
    halo_px: int,
    max_aoi_cloud: float,
    dry_run: bool,
) -> dict:
    """Try to replace ONE failing frame in ONE tile, guarded.

    Steps: re-select a same-(year,window) cleaner date → fetch it (M1+M2) →
    measure its B02 cloud fraction → apply the never-worse guard → write
    atomically. Returns a verdict dict describing the action taken.
    """
    name = Path(npz_path).stem
    try:
        with np.load(npz_path, allow_pickle=True) as d:
            data = {k: d[k] for k in d.files}
    except Exception as e:
        return {"tile": name, "slot": key, "action": "error",
                "reason": f"{type(e).__name__}: {str(e)[:140]}"}

    # Idempotency: a frame already QC-replaced (and now passing) is skipped.
    safe_key = key.replace(":", "_")
    if int(data.get(f"qc_replaced_{safe_key}", 0)) == 1:
        cur = _get_frame(data, key)
        cur_frac = b02_cloud_fraction(cur, bright=bright) if cur is not None else 1.0
        if cur_frac <= threshold:
            return {"tile": name, "slot": key, "action": "skip-already-replaced",
                    "cloud_frac": round(cur_frac, 4)}

    tile_year = _tile_year(data)
    if tile_year is None:
        return {"tile": name, "slot": key, "action": "skip-no-tile-year"}

    bbox = data.get("bbox_3006")
    if bbox is None or len(bbox) != 4:
        return {"tile": name, "slot": key, "action": "skip-no-bbox"}
    from imint.training.tile_fetch import bbox_3006_to_wgs84
    w, s, e, n = (int(x) for x in bbox)
    bbox_wgs84 = bbox_3006_to_wgs84({"west": w, "south": s, "east": e, "north": n})

    new_date = _select_cleaner_date(
        bbox_wgs84, tile_year, key, max_aoi_cloud=max_aoi_cloud)
    if new_date is None:
        return {"tile": name, "slot": key, "action": "no-candidate",
                "window": list(frame_window(tile_year, key))}

    # Belt-and-braces year assertion (the guard is already enforced inside
    # _select_cleaner_date; assert here so a future refactor can't silently
    # break temporal matching).
    want_year = slot_year(tile_year, key)
    assert int(new_date[:4]) == want_year, (
        f"temporal-matching violated: {new_date} not in year {want_year}")

    if dry_run:
        return {"tile": name, "slot": key, "action": "would-replace",
                "orig_date": _frame_date(data, key),
                "orig_cloud_frac": round(orig_cloud_frac, 4),
                "new_date": new_date, "new_year": want_year,
                "window": list(frame_window(tile_year, key))}

    new_frame = _fetch_replacement_frame(data, key, new_date, halo_px=halo_px)
    if new_frame is None:
        return {"tile": name, "slot": key, "action": "fetch-failed",
                "new_date": new_date}

    # Align the fresh frame onto the tile's stored Phase-1 anchor (coreg_ref_frame)
    # so it lands on the same grid as the rest of the stack + the label — M1 alone
    # leaves ~2 px relative drift. Skip (keep original) if there's no valid anchor
    # or this IS the anchor slot.
    aligned, careason = _coreg_to_stored_anchor(data, new_frame, key)
    if aligned is None:
        return {"tile": name, "slot": key, "action": f"skip-{careason}",
                "new_date": new_date}
    new_frame = aligned

    new_frac = b02_cloud_fraction(new_frame, bright=bright)
    # NEVER-WORSE GUARD — overwrite only if strictly cleaner.
    if new_frac >= orig_cloud_frac:
        return {"tile": name, "slot": key, "action": "keep-original-never-worse",
                "orig_cloud_frac": round(orig_cloud_frac, 4),
                "candidate_cloud_frac": round(new_frac, 4),
                "new_date": new_date}

    _write_replacement(npz_path, key, new_frame, new_date, new_frac)
    return {"tile": name, "slot": key, "action": "replaced",
            "orig_cloud_frac": round(orig_cloud_frac, 4),
            "new_cloud_frac": round(new_frac, 4),
            "new_date": new_date, "new_year": want_year}


def run_replace(args: argparse.Namespace) -> int:
    """Phase B driver — read the report's replace-list, replace each frame.

    Requires the Phase-A report (``--report``); it sizes the work and is the
    list of (tile, slot) failures to act on. Idempotent: re-reading the same
    report after a partial run skips frames already replaced-and-passing.
    """
    report = _load_json(args.report)
    if report is None:
        _log(f"[replace] report not found / unreadable: {args.report}. "
             f"Run --phase measure first.")
        return 1
    replace_list = report.get("replace_list", [])
    if not replace_list:
        _log("[replace] nothing to replace (empty replace_list).")
        return 0
    _log(f"[replace] {len(replace_list)} failing frames; threshold="
         f"{args.threshold}; workers={args.workers}; "
         f"{'DRY-RUN' if args.dry_run else 'LIVE'}")

    def _one(item: dict) -> dict:
        npz_path = os.path.join(args.data_dir, f"{item['tile']}.npz")
        if not os.path.exists(npz_path):
            return {"tile": item["tile"], "slot": item["slot"],
                    "action": "skip-missing-tile"}
        return replace_tile_frame(
            npz_path, item["slot"], float(item.get("cloud_frac") or 1.0),
            threshold=args.threshold, bright=args.bright,
            halo_px=args.halo_px, max_aoi_cloud=args.max_aoi_cloud,
            dry_run=args.dry_run)

    verdicts: list[dict] = []
    if args.workers <= 1:
        for item in replace_list:
            verdicts.append(_one(item))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_one, it): it for it in replace_list}
            for fut in as_completed(futs):
                verdicts.append(fut.result())

    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v["action"]] = counts.get(v["action"], 0) + 1
    _log(f"[replace] verdicts: {json.dumps(counts, indent=2)}")

    if not args.dry_run and args.replace_log:
        _atomic_write_json(args.replace_log,
                           {"counts": counts, "verdicts": verdicts})
        _log(f"[replace] wrote verdict log {args.replace_log}")
    return 0


# ── IO helpers ───────────────────────────────────────────────────────────────


def _atomic_write_json(path: str, payload: dict) -> None:
    """Write JSON atomically (tmp + os.replace) — safe under concurrent runs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _load_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-frame cloud QC + replace (SPEC §3).")
    p.add_argument("--phase", required=True, choices=["measure", "replace"],
                   help="measure = B02+ERA5 verdict per frame → report; "
                        "replace = re-fetch failing frames (same year+window).")
    p.add_argument("--data-dir", required=True,
                   help="unified_v2_512_recoreg dir (the tiles being QC'd).")
    p.add_argument("--report", default="/data/debug/frame_cloud_qc_report.json",
                   help="measure: output path; replace: input replace-list.")
    p.add_argument("--replace-log", default=None,
                   help="replace: optional path for the per-frame verdict log.")
    p.add_argument("--threshold", type=float, default=0.2,
                   help="tile-level cloud-fraction fail threshold (default 0.2).")
    p.add_argument("--bright", type=float, default=B02_BRIGHT_REFLECTANCE,
                   help="per-pixel B02 reflectance 'cloud' cut (default 0.20).")
    p.add_argument("--max-aoi-cloud", type=float, default=0.10,
                   help="SCL post-filter ceiling for date re-selection "
                        "(optimal_fetch_dates max_aoi_cloud; default 0.10).")
    p.add_argument("--halo-px", type=int, default=8,
                   help="M2 coreg halo (2*crop) for the replacement fetch.")
    p.add_argument("--workers", type=int, default=4,
                   help="thread workers (measure: I/O+ERA5; replace: des — "
                        "keep at 2 for the DES rate-limit).")
    p.add_argument("--no-era5", action="store_true",
                   help="measure: skip the ERA5 date check (pixel metric only).")
    p.add_argument("--dry-run", action="store_true",
                   help="plan only; write nothing to disk.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "measure":
        return run_measure(args)
    return run_replace(args)


if __name__ == "__main__":
    raise SystemExit(main())
