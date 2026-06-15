"""Tile-assembly + halo-crop helpers shared by the spectral-fetch pipeline.

Pure (no fetch, no IO) so they are unit-testable in isolation. Promoted out of
``scripts/regrid_national_512.py`` so the canonical fetch entry
(``imint/training/fetch_spectral.py::fetch_tile_spectral``) and the regrid
orchestrator share ONE assembly path instead of duplicating it.

Geometry is passed explicitly (``crop``/``canon``) rather than hard-coded to the
520→512 case — the helpers are size-agnostic and the caller owns the halo/canon
contract.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from imint.training.openeo_tile_graph import ALL_BANDS_INDEX


def date_to_doy(date_str: str) -> int:
    """Day-of-year for an ISO date, or 0 for an empty/invalid string."""
    if not date_str:
        return 0
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").timetuple().tm_yday
    except ValueError:
        return 0

# Band-group indices into an ALL_BANDS (12-band) frame.
_I_PRITHVI = list(ALL_BANDS_INDEX["prithvi"])   # (0,1,2,7,8,9) B02,B03,B04,B8A,B11,B12
_I_B08 = ALL_BANDS_INDEX["b08"][0]              # 3
_I_RE = list(ALL_BANDS_INDEX["rededge"])        # (4,5,6) B05,B06,B07
_I_B01 = ALL_BANDS_INDEX["b01"][0]              # 10
_I_B09 = ALL_BANDS_INDEX["b09"][0]              # 11


def _has_signal(frames: list[np.ndarray]) -> np.int32:
    """1 if any frame carries non-zero data, else 0 — the ``has_*`` extra flag."""
    return np.int32(1 if any(bool(np.any(f)) for f in frames) else 0)


def crop_halo(arr: np.ndarray, *, crop: int, canon: int) -> np.ndarray:
    """Centre-crop a ``(..., canon+2*crop, canon+2*crop)`` array to ``(..., canon, canon)``.

    The halo (``crop`` px per side) absorbs the sinc wrap-around that the M2
    sub-pixel shift leaves at the frame edges, so the inner ``canon`` square is
    clean.
    """
    return arr[..., crop:crop + canon, crop:crop + canon]


def assemble_fresh(
    frames: dict[int, np.ndarray],
    dates: list[str],
    n_frames: int,
    *,
    canon: int,
) -> tuple[np.ndarray, dict]:
    """Assemble cropped fresh all-band frames into the spectral cube + extras.

    Every frame is fresh (re-gridded), so there is no keep-if-clean branch: a
    fetched+cropped frame's 6 Prithvi bands go straight into the cube and its
    B08/red-edge/B01/B09 into the extras. A frame absent from ``frames`` failed to
    fetch and is zero-filled with an empty date (downstream QC drops <3/4-frame
    tiles). Output keys/shapes match the enrich-script .npz contract (b08/b01/b09
    stacked on a new axis; rededge concatenated band×frame → frame-major
    B05,B06,B07 per frame; each with ``*_dates`` + ``has_*``).

    Args:
        frames:   ``{slot: (12, canon, canon)}`` — already halo-cropped.
        dates:    ISO date per slot (``""`` for a missing slot).
        n_frames: number of temporal slots in the output.
        canon:    output edge in pixels (zero-fill shape for missing slots).
    """
    h = w = canon
    spec, b08_f, re_f, b01_f, b09_f = [], [], [], [], []
    b08_d, re_d, b01_d, b09_d = [], [], [], []
    for fi in range(n_frames):
        f = frames.get(fi)
        d = dates[fi] if fi < len(dates) else ""
        if f is not None:
            spec.append(f[_I_PRITHVI])
            b08_f.append(f[_I_B08]); re_f.append(f[_I_RE])
            b01_f.append(f[_I_B01]); b09_f.append(f[_I_B09])
            b08_d.append(d); re_d.append(d); b01_d.append(d); b09_d.append(d)
        else:
            spec.append(np.zeros((6, h, w), np.float32))
            b08_f.append(np.zeros((h, w), np.float32))
            re_f.append(np.zeros((3, h, w), np.float32))
            b01_f.append(np.zeros((h, w), np.float32))
            b09_f.append(np.zeros((h, w), np.float32))
            b08_d.append(""); re_d.append(""); b01_d.append(""); b09_d.append("")
    spectral = np.concatenate(spec, axis=0).astype(np.float32)   # (T*6, H, W)
    extras = {
        "b08": np.stack(b08_f, 0), "b08_dates": np.array(b08_d), "has_b08": _has_signal(b08_f),
        "rededge": np.concatenate(re_f, 0), "rededge_dates": np.array(re_d),
        "has_rededge": _has_signal(re_f),
        "b01": np.stack(b01_f, 0), "b01_dates": np.array(b01_d), "has_b01": _has_signal(b01_f),
        "b09": np.stack(b09_f, 0), "b09_dates": np.array(b09_d), "has_b09": _has_signal(b09_f),
    }
    return spectral, extras
