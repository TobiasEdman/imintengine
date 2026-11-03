"""Tile bounds shared by serving and training, on the same national lattice."""
from __future__ import annotations


def bounds_from_center(east: int | float, north: int | float,
                       size_px: int, gsd_m: float = 10.0) -> dict[str, int]:
    """Snap the centre and return an even-sized tile's integer metre bounds."""
    if size_px <= 0:
        raise ValueError(f'size_px must be positive, got {size_px}')
    if gsd_m <= 0:
        raise ValueError(f'gsd_m must be positive, got {gsd_m}')
    gsd = int(gsd_m)
    half = int(size_px * gsd_m) // 2
    if half % gsd:
        raise ValueError(
            f'TileConfig(size_px={size_px}, gsd_m={gsd_m}): '
            f'half_m={half} is not a multiple of gsd={gsd} — size_px must be '
            'even for national-lattice alignment.'
        )
    cx = round(east / gsd) * gsd
    cy = round(north / gsd) * gsd
    return {'west': cx - half, 'east': cx + half,
            'south': cy - half, 'north': cy + half}
