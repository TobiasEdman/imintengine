"""Version contract for the per-tile Sentinel-1 season-composite enrichment.

Single source of truth for the ``s1_enrich_v`` tile marker. The producer
(``scripts/enrich_tiles_s1.py``) stamps this version into every enriched
tile; the consumer (``UnifiedDataset``) refuses tiles stamped with anything
else. Both import the constant from here — a version bump that only touches
one side cannot happen again. (It happened once: the v3→v4 bump on
2026-08-24 updated the enricher only, so the dataset's ``== 3`` gate
silently dropped every SAR tile and train-croma-v3 died with "No tiles
found" — twice, the first time with all evidence reaped.)

Version history — why older stamps are rejected outright:

- v1: ±3-day per-frame stack, shape ``(T*2, H, W)`` — wrong shape for the
  season-composite readers.
- v2: CDSE dB σ⁰ composite — dB units double-log under normalizers that
  apply ``10*log10`` internally.
- v3: linear γ⁰ RTC composite, but the composite year was derived from
  ``dates[0]`` (the autumn frame of year-1), so every tile without
  ``lpis_year`` got a wrong-season composite.
- v4: composited over the label year (year-0). Current. Accepted only
  together with ``s1_source == S1_SOURCE`` (task 0113).
"""
from __future__ import annotations

S1_ENRICH_VERSION = 4

# Provenance stamp (``s1_source``) of the one backend whose tiles the SAR
# models may consume: Planetary Computer RTC γ⁰ (``imint.training.pc_s1_rtc``).
# The producer writes it, the consumer requires it next to the version. The
# CDSE GRD σ⁰ backend ("cdse-grd-sigma0") is retired — its tie points sat at
# ellipsoid height (task 0113) — and a tile with any other or no source is
# refused. A PVC count on 2026-10-22 found every v4 tile stamped with this
# value, so requiring it drops no current tile.
S1_SOURCE = "pc-rtc-gamma0"


def s1_source_of(value) -> str | None:
    """Decode a tile's ``s1_source`` (``np.bytes_`` / 0-d array / str) to str."""
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    return str(value)


def s1_unusable_reason(data) -> str | None:
    """Why a tile's S1 composite may not be used, or ``None`` if it may.

    The one rule every SAR gate applies (dataset, pinned plot set, plot
    feature extraction, k8s preflights): ``s1_enrich_v == S1_ENRICH_VERSION``
    AND ``s1_source == S1_SOURCE``. ``data`` is any mapping with ``.get``
    over the tile's keys — a dict, or an open ``np.load`` NpzFile (only the
    two scalar stamps are read). A missing stamp counts as unusable.
    """
    version = data.get("s1_enrich_v")
    if getattr(version, "size", 1) != 1:
        return "s1_enrich_v=<not a scalar>"
    if version is not None and hasattr(version, "item"):
        version = version.item()
    version = 0 if version is None else int(version)
    if version != S1_ENRICH_VERSION:
        return f"s1_enrich_v={version}"
    source = s1_source_of(data.get("s1_source"))
    if source != S1_SOURCE:
        return f"s1_source={source!r}"
    return None


def s1_usable(data) -> bool:
    """True iff the tile's S1 passes :func:`s1_unusable_reason`."""
    return s1_unusable_reason(data) is None
