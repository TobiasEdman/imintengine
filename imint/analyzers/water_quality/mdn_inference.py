"""Pahlevan Mixture Density Network (MDN) inference wrapper.

The MDN by Pahlevan et al. is a pre-trained neural retrieval for
chlorophyll-a from Sentinel-2 surface reflectance. The network's
mixture-of-Gaussians head returns a point estimate for chl-a; the
v1 wrapper exposes only chl-a (no TSS/aCDOM — those are separate model
variants under upstream's ``benchmarks/tss/SOLID`` and require their own
weights and inference paths).

References:
    Pahlevan et al. (2020) — Seamless retrievals of chlorophyll-a from
        Sentinel-2 (MSI) and Sentinel-3 (OLCI) in inland and coastal
        waters. Remote Sensing of Environment 240: 111604.

Upstream repository:
    https://github.com/BrandonSmithJ/MDN — clone to a local directory and
    point ``mdn_repo_path`` (or env var ``IMINT_MDN_PATH``) at it. The
    pre-trained weight bundle ships inside the repo at ``Weights/MSI/``
    and is loaded by the upstream ``image_estimates`` entry point.

Design contract:
    Heavy deps (``tensorflow``, ``tensorflow_probability``, the upstream
    MDN package) are imported lazily inside :func:`run_mdn`. Importing
    this module never fails. If the upstream package or TF stack is
    missing, ``run_mdn`` raises :class:`MDNUnavailable` and the
    orchestrator skips MDN outputs (skip-and-warn pattern, per SPEC.md).

Sentinel-2 band layout:
    MDN expects MSI bands at wavelengths
    ``[443, 490, 560, 665, 705, 740, 783] nm`` =
    ``[B01, B02, B03, B04, B05, B06, B07]``.
    The caller must supply *all seven* — including B01 (coastal aerosol)
    which the canonical ImintEngine fetch only includes when configured
    with ``BANDS_60M_ALL``.

Reflectance convention:
    Inputs are remote-sensing reflectance ``Rrs`` (units 1/sr). When the
    ImintEngine pipeline supplies surface reflectance ``ρ`` (unitless,
    [0, 1] from L2A), :func:`run_mdn` converts via ``Rrs = ρ / π``
    before invoking the upstream model.
"""
from __future__ import annotations

import logging
import os
import warnings
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel-2 bands required by the MDN MSI model, in canonical order.
MDN_BANDS: tuple[str, ...] = ("B01", "B02", "B03", "B04", "B05", "B06", "B07")


class MDNUnavailable(RuntimeError):
    """Raised when MDN cannot run.

    Reasons include: upstream MDN package not on ``PYTHONPATH`` /
    ``mdn_repo_path``, TensorFlow stack missing or broken, required
    bands absent from the input dict, or no water pixels available
    after masking.
    """


def _import_mdn(mdn_repo_path: str | None) -> "callable":
    """Lazy-import the upstream :func:`MDN.image_estimates` entry point.

    Args:
        mdn_repo_path: Filesystem path to a clone of
            ``BrandonSmithJ/MDN``. If ``None``, the env var
            ``IMINT_MDN_PATH`` is consulted, then a default clone
            location at ``~/code/MDN``.

    Raises:
        MDNUnavailable: If the package can't be imported (missing
            dependency, wrong path, broken TF stack).
    """
    import sys

    candidates: list[str] = []
    if mdn_repo_path:
        candidates.append(mdn_repo_path)
    env_path = os.environ.get("IMINT_MDN_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.expanduser("~/code"))  # parent of MDN/

    for cand in candidates:
        if cand not in sys.path:
            sys.path.insert(0, cand)

    # Quiet the noisy upstream stack
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    try:
        from MDN import image_estimates  # type: ignore
    except ImportError as e:
        raise MDNUnavailable(
            f"upstream MDN package not importable from any of {candidates}: {e}"
        ) from e
    except Exception as e:
        raise MDNUnavailable(f"upstream MDN import failed: {e}") from e
    return image_estimates


def _stack_rrs(
    bands: dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Build an (H, W, 7) Rrs cube from the bands dict, NaN-filling invalid pixels.

    Args:
        bands: dict containing at least the keys in :data:`MDN_BANDS`,
            each a (H, W) float array of L2A surface reflectance ρ.
        valid_mask: (H, W) boolean — True for water pixels eligible
            for inference.

    Raises:
        MDNUnavailable: If any required band is missing.

    Returns:
        Rrs cube of shape (H, W, 7), float32. Invalid pixels are NaN
        (the upstream model handles NaN by skipping those pixels and
        returning NaN in the output).
    """
    missing = [b for b in MDN_BANDS if bands.get(b) is None]
    if missing:
        raise MDNUnavailable(f"MDN requires bands {missing} which are absent")

    h, w = valid_mask.shape
    cube = np.empty((h, w, len(MDN_BANDS)), dtype=np.float32)
    for i, name in enumerate(MDN_BANDS):
        cube[..., i] = bands[name]
    # ρ → Rrs (Pahlevan et al. assume input in 1/sr units)
    cube /= np.float32(np.pi)
    cube[~valid_mask] = np.nan
    return cube


def run_mdn(
    bands: dict[str, np.ndarray],
    water_mask: np.ndarray,
    mdn_repo_path: str | None = None,
) -> dict[str, np.ndarray]:
    """Run the upstream Pahlevan MDN on water pixels of an S2 scene.

    Args:
        bands: Sentinel-2 surface reflectance dict. Must contain all
            keys in :data:`MDN_BANDS` (note B01 — coastal aerosol).
        water_mask: (H, W) boolean — True for valid water pixels.
        mdn_repo_path: Optional path to the upstream MDN clone.

    Returns:
        Dict with one entry::

            "chlorophyll_a" — chl-a, mg/m³, shape (H, W), NaN over land

    Raises:
        MDNUnavailable: If the upstream package or required bands are
            absent, or no water pixels are available.
    """
    if not water_mask.any():
        raise MDNUnavailable("no water pixels in tile — MDN inference would be empty")

    image_estimates = _import_mdn(mdn_repo_path)
    rrs = _stack_rrs(bands, water_mask)

    try:
        chla, _idxs = image_estimates(rrs, sensor="MSI")
    except Exception as e:
        raise MDNUnavailable(f"MDN forward pass failed: {e}") from e

    if chla.ndim == 3 and chla.shape[-1] == 1:
        chla = chla[..., 0]
    chla = chla.astype(np.float32)
    chla[~water_mask] = np.nan
    return {"chlorophyll_a": chla}
