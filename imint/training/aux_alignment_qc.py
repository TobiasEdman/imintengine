"""Cross-raster alignment QC — does a stored tile's aux actually sit on the
NMD label grid?

The 256/512 aux-misalignment (volume/basal_area/diameter/dem/markfukt/vpp
rendered on a 2560 m extent at 512 px = 5 m/px, the central quarter stretched
2x) was silent: every array kept the right *shape*, only the *ground* it
covered was wrong. Nothing re-validated a stored tile's cross-raster alignment,
so it survived undetected until a human noticed it on the dashboard.

This module is the standing guard. A correctly georeferenced **forestry** aux
layer (height/volume/basal_area/diameter — physically defined on forest land)
has its valid (>0) pixels concentrated on the NMD forest mask; a wrong-grid
layer has a near-random footprint vs forest. The phi coefficient (Matthews
correlation) between ``aux>0`` and the forest mask separates the two cleanly:
aligned ≈ 0.3-0.6, misaligned ≈ 0.

Forestry aux is the canary for the WHOLE aux stack: every aux channel is fetched
on the same grid via :func:`imint.training.tile_bbox.resolve_fetch_bbox`, so if
the forestry channels align with the label, dem/markfukt/vpp (fetched at the
same bbox) do too. Reference is the unified ``label`` (classes 1-6 = forest),
read from the NMD raster on the NMD lattice — the canonical grid.
"""
from __future__ import annotations

import numpy as np

# Unified forest classes — imint.training.unified_schema._NMD_FOREST.
_FOREST_CLASSES = (1, 2, 3, 4, 5, 6)
# Aux channels physically defined on forest land (the alignment canaries).
FORESTRY_AUX = ("height", "volume", "basal_area", "diameter")


def _keys(npz) -> list[str]:
    """Key list of an NpzFile or a plain dict."""
    return list(npz.files) if hasattr(npz, "files") else list(npz)


def forest_mask(npz) -> np.ndarray | None:
    """Boolean forest reference mask from the tile's unified ``label`` (1-6).

    Returns ``None`` when the tile carries no ``label`` (e.g. a mid-campaign
    ``_recoreg`` tile before label-restore) — the caller treats that as
    "not evaluable", never as a pass.
    """
    if "label" not in _keys(npz):
        return None
    return np.isin(np.asarray(npz["label"]), _FOREST_CLASSES)


def phi_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Matthews correlation (phi) between two boolean masks.

    0 ⇒ the masks are independent (a wrong-grid footprint vs forest);
    >0 ⇒ they co-occur (a correctly placed forestry layer on forest).
    Base-rate-normalised, so it is not fooled by differing coverage fractions
    the way a raw overlap count would be.
    """
    a = np.asarray(a, bool).ravel()
    b = np.asarray(b, bool).ravel()
    tp = float(np.count_nonzero(a & b))
    tn = float(np.count_nonzero(~a & ~b))
    fp = float(np.count_nonzero(a & ~b))
    fn = float(np.count_nonzero(~a & b))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


def check_tile_alignment(
    npz,
    *,
    min_forest_frac: float = 0.05,
    max_forest_frac: float = 0.95,
    min_phi: float = 0.15,
) -> dict:
    """Verdict on one tile's forestry-aux ↔ forest alignment.

    ``status``:
      * ``skipped`` — no ``label`` reference, no forestry aux, or a forest
        fraction outside ``[min_forest_frac, max_forest_frac]`` (all-water /
        all-urban / all-forest tiles carry no discriminating signal).
      * ``pass`` — every present forestry aux has ``phi(aux>0, forest) >=
        min_phi``.
      * ``fail`` — at least one forestry aux is near-random (or shape-mismatched)
        vs the forest mask → it is on the wrong grid.
    """
    keys = _keys(npz)
    forest = forest_mask(npz)
    if forest is None:
        return {"status": "skipped", "reason": "no_label_reference"}

    ffrac = float(forest.mean())
    if not (min_forest_frac <= ffrac <= max_forest_frac):
        return {"status": "skipped",
                "reason": f"forest_frac={ffrac:.3f}_uninformative"}

    present = [k for k in FORESTRY_AUX if k in keys]
    if not present:
        return {"status": "skipped", "reason": "no_forestry_aux"}

    phi: dict[str, float | None] = {}
    failed: list[str] = []
    for k in present:
        aux = np.asarray(npz[k], np.float64)
        if aux.shape != forest.shape:        # different size ⇒ misaligned grid
            phi[k] = None
            failed.append(k)
            continue
        valid = np.isfinite(aux) & (aux > 0)
        p = phi_coefficient(valid, forest)
        phi[k] = round(p, 3)
        if p < min_phi:
            failed.append(k)

    return {
        "status": "fail" if failed else "pass",
        "forest_frac": round(ffrac, 3),
        "phi": phi,
        "failed_aux": failed,
    }
