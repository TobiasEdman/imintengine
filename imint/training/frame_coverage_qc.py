"""Per-frame spectral coverage QC — does each present temporal frame actually
carry data, or is it a swath-edge / empty-post-crop no-data wedge?

A frame can be marked present (``temporal_mask=1``) yet be largely no-data: a
Sentinel-2 swath edge leaves a black wedge, and a frame the M2 shift + inner
crop landed on the edge can end up ~100% black *post-crop* — which the raw-halo
all-zero guard never sees. Such a frame is no-data the model treats as real.

``coreg_anchor_valid_frac`` only ever stored the *anchor* frame's scalar; this
recomputes every present frame's valid (non-zero) fraction directly from the
stored ``spectral`` cube — no re-fetch. The same :func:`frame_valid_fraction`
primitive is used at fetch time (``fetch_spectral``) to drop sub-threshold
frames from ``temporal_mask``, so the gate and the fetch can never disagree on
"what counts as covered". Companion to :mod:`imint.training.aux_alignment_qc`.
"""
from __future__ import annotations

import numpy as np

# No-data is the exact-zero fill, so a pixel is valid if ANY band is non-zero.
_NODATA_EPS = 1e-6


def frame_valid_fraction(frame: np.ndarray) -> float:
    """Valid (non-no-data) fraction of one ``(bands, H, W)`` frame.

    The single source of truth for "how much of this frame is real data",
    shared by the QC gate and the fetch path.
    """
    fr = np.asarray(frame)
    if fr.ndim != 3 or fr.shape[0] == 0:
        return 0.0
    return float((np.abs(fr) > _NODATA_EPS).any(axis=0).mean())


def _keys(npz) -> list[str]:
    return list(npz.files) if hasattr(npz, "files") else list(npz)


def frame_valid_fractions(npz) -> list[tuple[int, float]]:
    """``[(slot, valid_frac)]`` for each PRESENT frame (``temporal_mask=1``).

    Absent frames (mask 0) are omitted — a missing frame is a different concern
    from a partial one. Returns ``[]`` when the tile has no ``spectral`` cube.
    """
    keys = _keys(npz)
    if "spectral" not in keys:
        return []
    spec = np.asarray(npz["spectral"])
    nb = int(npz["num_bands"]) if "num_bands" in keys else 6
    nf = spec.shape[0] // nb if nb else 0
    mask = (np.asarray(npz["temporal_mask"]) if "temporal_mask" in keys
            else np.ones(nf, np.uint8))
    out: list[tuple[int, float]] = []
    for f in range(nf):
        if f < len(mask) and int(mask[f]) == 0:
            continue
        out.append((f, frame_valid_fraction(spec[f * nb:(f + 1) * nb])))
    return out


def check_frame_coverage(npz, *, min_valid_frac: float = 0.90) -> dict:
    """Verdict on a tile's per-frame spectral coverage.

    ``status``:
      * ``skipped`` — no ``spectral`` cube / no present frames.
      * ``pass`` — every present frame has ``valid_frac >= min_valid_frac``.
      * ``fail`` — a present frame is a partial / empty no-data wedge (the
        swath edge, or an empty-post-crop frame still marked valid).
    """
    fracs = frame_valid_fractions(npz)
    if not fracs:
        return {"status": "skipped", "reason": "no_present_frames"}
    bad = [(f, round(v, 3)) for f, v in fracs if v < min_valid_frac]
    return {
        "status": "fail" if bad else "pass",
        "valid_frac": {f: round(v, 3) for f, v in fracs},
        "min_valid_frac": round(min(v for _, v in fracs), 3),
        "bad_frames": bad,
    }
