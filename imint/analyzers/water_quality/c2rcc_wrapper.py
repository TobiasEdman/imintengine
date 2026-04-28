"""ACOLITE C2RCC inference wrapper.

C2RCC (Case-2 Regional Coast Colour) is ESA's neural-net retrieval for
chlorophyll-a, total suspended matter (TSM) and CDOM in optically
complex (Case-2) water. The original implementation is Java/SNAP; this
wrapper uses the ACOLITE Python port to avoid a JVM dependency.

References:
    Brockmann, Doerffer, Peters, Stelzer, Embacher, Ruescas (2016) —
        Evolution of the C2RCC Neural Network for Sentinel 2 and 3 for
        the Retrieval of Ocean Colour Products in Normal and Extreme
        Optically Complex Waters. ESA Living Planet Symposium.
    Vanhellemont, Q. (ACOLITE) — open-source atmospheric correction and
        product generation code, https://github.com/acolite/acolite

Design contract identical to ``mdn_inference``:
    Soft-import. ``import acolite`` is deferred to call time so this
    module never fails on import. Missing acolite → ``C2RCCUnavailable``.

Note on input reflectance:
    SPEC.md tradeoff: this wrapper passes Sen2Cor L2A surface
    reflectance directly to C2RCC, bypassing C2RCC's own atmospheric
    correction. Absolute Chl-a is therefore biased in optically complex
    water; relative spatial patterns remain valid for v1 visual
    verification.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Bands C2RCC expects on Sentinel-2 MSI per Brockmann et al. (2016).
C2RCC_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A"]


class C2RCCUnavailable(RuntimeError):
    """Raised when the C2RCC backend cannot run.

    The orchestrator catches this and skips C2RCC outputs while still
    emitting NDCI / MCI / MDN.
    """


def _import_acolite() -> Any:
    """Lazy import. Raises C2RCCUnavailable if absent or broken."""
    try:
        import acolite  # type: ignore
    except ImportError as e:
        raise C2RCCUnavailable(f"acolite not installed: {e}") from e
    except Exception as e:
        raise C2RCCUnavailable(f"acolite import failed: {e}") from e
    return acolite


def _check_bands(bands: dict[str, np.ndarray]) -> None:
    missing = [b for b in C2RCC_BANDS if bands.get(b) is None]
    if missing:
        raise C2RCCUnavailable(f"C2RCC requires bands {missing} which are absent")


def run_c2rcc(
    bands: dict[str, np.ndarray],
    water_mask: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Run ACOLITE C2RCC on water pixels of a Sentinel-2 scene.

    Args:
        bands: dict with at least the keys in :data:`C2RCC_BANDS`.
        water_mask: (H, W) boolean — True for valid water pixels.
        config: pass-through to ACOLITE (e.g. ``{"l2w_parameters":
            ["chl_re_mishra", "tur_nechad2016"]}``). Defaults to None,
            in which case ACOLITE's S2 Case-2 defaults apply.

    Returns:
        Dict with keys::

            "chlorophyll_a" — mg/m³, shape (H, W), NaN over land
            "tsm"           — g/m³, shape (H, W)
            "cdom"          — m⁻¹, shape (H, W)

        Negative retrievals (occasional in shallow/turbid pixels) are
        clipped to 0 and the count is logged.

    Raises:
        C2RCCUnavailable: acolite missing, required bands missing, no
            water pixels, or ACOLITE forward call failed.
    """
    acolite = _import_acolite()
    _check_bands(bands)

    if not water_mask.any():
        raise C2RCCUnavailable("no water pixels in tile — C2RCC inference would be empty")

    config = dict(config or {})
    H, W = water_mask.shape

    # ACOLITE's Python API surfaces vary by version; the wrapper trusts
    # an attribute named ``run_c2rcc_array`` taking a band stack and
    # returning a dict of named retrievals. If the installed version
    # exposes a different entry point, the call below raises and we
    # convert it to C2RCCUnavailable so the orchestrator can skip.
    rho_stack = np.stack([bands[b] for b in C2RCC_BANDS], axis=0).astype(np.float32)

    try:
        retrievals = acolite.run_c2rcc_array(  # type: ignore[attr-defined]
            rho=rho_stack,
            band_names=C2RCC_BANDS,
            water_mask=water_mask,
            **config,
        )
    except AttributeError as e:
        raise C2RCCUnavailable(
            f"acolite.run_c2rcc_array entry point missing — installed acolite version "
            f"does not expose the expected API: {e}"
        ) from e
    except Exception as e:
        raise C2RCCUnavailable(f"C2RCC forward pass failed: {e}") from e

    if not isinstance(retrievals, dict):
        raise C2RCCUnavailable(
            f"unexpected acolite return type {type(retrievals)}; expected dict"
        )

    out: dict[str, np.ndarray] = {}
    name_map = {"chlorophyll_a": "chl", "tsm": "tsm", "cdom": "cdom"}
    for canonical, source_key in name_map.items():
        arr = retrievals.get(source_key)
        if arr is None:
            raise C2RCCUnavailable(
                f"acolite return missing key '{source_key}'; got {list(retrievals)}"
            )
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape != (H, W):
            raise C2RCCUnavailable(
                f"acolite '{source_key}' shape {arr.shape} != expected {(H, W)}"
            )
        # Clip negatives, log count
        neg = int((arr < 0).sum())
        if neg:
            logger.warning("C2RCC '%s': clipping %d negative pixels to 0", canonical, neg)
            arr = np.where(arr < 0, 0.0, arr)
        # Mask land/cloud
        arr[~water_mask] = np.nan
        out[canonical] = arr

    return out
