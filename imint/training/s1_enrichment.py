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
- v4: composited over the label year (year-0). Current.
"""
from __future__ import annotations

S1_ENRICH_VERSION = 4
