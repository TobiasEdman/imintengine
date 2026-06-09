"""
imint/training/tile_config.py — Runtime tile geometry configuration.

Single source of truth for "how big is a tile?". Created once at CLI / job
entry, threaded explicitly through every function that needs it. No module-
level constants, no monkey-patching, no env-var reads deep in the stack.

The previous design used module-level ``TILE_SIZE_M`` / ``TILE_SIZE_PX``
constants that callers mutated via ``_tf.TILE_SIZE_M = ...``. That broke
because ``from X import Y`` creates a local binding that does NOT track
subsequent mutations of ``X.Y`` — resulting in tiles fetched with the
wrong bbox extent at the wrong GSD. TileConfig replaces all of it.

Usage:
    tile = TileConfig(size_px=512)       # 512 × 10m = 5120m square
    bbox = tile.bbox_from_center(east=281280, north=6471280)
    tile.assert_bbox_matches(bbox)       # raises if a caller passed a
                                         # stale 2560m bbox by mistake
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileConfig:
    """Tile geometry parameters.

    Attributes:
        size_px: Side length in pixels (e.g. 256, 512, 1024).
        gsd_m: Ground sample distance in meters. Defaults to 10.0 for
            Sentinel-2 native. Pass a different value for other sensors.
    """

    size_px: int
    gsd_m: float = 10.0

    def __post_init__(self) -> None:
        if self.size_px <= 0:
            raise ValueError(f"size_px must be positive, got {self.size_px}")
        if self.gsd_m <= 0:
            raise ValueError(f"gsd_m must be positive, got {self.gsd_m}")

    @property
    def size_m(self) -> int:
        """Tile side length in meters (size_px * gsd_m)."""
        return int(self.size_px * self.gsd_m)

    @property
    def half_m(self) -> int:
        """Half the tile side length in meters (for center → bbox)."""
        return self.size_m // 2

    def bbox_from_center(self, east: int | float, north: int | float) -> dict[str, int]:
        """Build a dict bbox centered at the given EPSG:3006 coordinates,
        snapped to the national ``gsd_m`` lattice.

        The centre is rounded to the nearest ``gsd_m`` multiple so every bbox
        edge lands on the Swedish national grid (NMD's 10 m lattice at the
        default GSD; its origin is itself on exact 10 m multiples). This makes
        ``rasterio.from_bounds`` yield integer window offsets, so label/aux
        reads are pixel-exact with no resampling displacement and all temporal
        frames of a tile share one grid.

        Returns:
            dict with integer ``west, south, east, north`` keys, guaranteed
            ``(east - west) == (north - south) == size_m`` and every edge an
            exact ``gsd_m`` multiple.

        Raises:
            ValueError: if ``size_px`` is odd, which makes ``half_m`` half a
                pixel off the ``gsd_m`` lattice.
        """
        gsd = int(self.gsd_m)
        h = self.half_m
        if h % gsd:
            raise ValueError(
                f"TileConfig(size_px={self.size_px}, gsd_m={self.gsd_m}): "
                f"half_m={h} is not a multiple of gsd={gsd} — size_px must be "
                f"even for national-lattice alignment."
            )
        cx = round(east / gsd) * gsd
        cy = round(north / gsd) * gsd
        return {
            "west":  cx - h,
            "east":  cx + h,
            "south": cy - h,
            "north": cy + h,
        }

    def assert_bbox_matches(self, bbox: dict) -> None:
        """Raise ``ValueError`` when bbox extent disagrees with tile size.

        This is the defense-in-depth check that catches the "wrong GSD"
        bug at the API boundary — if Sentinel Hub is about to compute
        GSD as ``(east - west) / size_px``, the bbox MUST match or the
        returned raster will be silently up/downsampled.

        Raises:
            ValueError: if ``(east - west) ≠ size_m`` or
                ``(north - south) ≠ size_m`` by more than 1 metre.
        """
        ew = bbox["east"] - bbox["west"]
        ns = bbox["north"] - bbox["south"]
        if abs(ew - self.size_m) > 1 or abs(ns - self.size_m) > 1:
            raise ValueError(
                f"TileConfig(size_px={self.size_px}, gsd_m={self.gsd_m}) "
                f"expects {self.size_m}m bbox; got ew={ew}m ns={ns}m. "
                f"Caller passed a stale or mismatched bbox — rebuild via "
                f"TileConfig.bbox_from_center()."
            )

    def __repr__(self) -> str:
        return f"TileConfig(size_px={self.size_px}, gsd_m={self.gsd_m})"
