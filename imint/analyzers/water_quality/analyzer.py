"""WaterQualityAnalyzer — orchestrator.

Combines four retrieval backends — Pahlevan MDN, ACOLITE C2RCC, NDCI,
MCI — on Sentinel-2 L2A surface reflectance for Bohuslän coastal Case-2
water. Outputs are kept separate (no fusion); a spread map highlights
inter-method disagreement.

Per-method failures are logged and skipped — the analyzer returns
``success=True`` whenever at least one method produces output. See
SPEC.md for the full contract.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..base import AnalysisResult, BaseAnalyzer
from . import classical_indices, water_mask
from .c2rcc_wrapper import C2RCCUnavailable, run_c2rcc
from .mdn_inference import MDNUnavailable, run_mdn

logger = logging.getLogger(__name__)

DEFAULT_AOI_PATH = Path(__file__).parent / "aoi" / "stigfjorden_skagerrak.geojson"

# Bands required by at least one method (NDCI / MCI need B04, B05, B06)
_INDEX_BANDS = {"B04", "B05", "B06"}


class WaterQualityAnalyzer(BaseAnalyzer):
    """Sentinel-2 water quality analyzer (MDN + C2RCC + NDCI + MCI).

    Configurable via ``analyzers.yaml`` ``water_quality`` section.
    """

    name = "water_quality"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        methods_cfg = (self.config.get("methods") or {})
        self._mdn_enabled = (methods_cfg.get("mdn") or {}).get("enabled", True)
        self._c2rcc_enabled = (methods_cfg.get("c2rcc") or {}).get("enabled", True)
        self._ndci_enabled = (methods_cfg.get("ndci") or {}).get("enabled", True)
        self._mci_enabled = (methods_cfg.get("mci") or {}).get("enabled", True)
        self._mdn_repo_path = (methods_cfg.get("mdn") or {}).get("repo_path")
        self._aoi_path = self.config.get("aoi_geojson") or str(DEFAULT_AOI_PATH)
        self._bloom_threshold = float(self.config.get("bloom_threshold_chl", 5.0))

    # ------------------------------------------------------------------ #
    #  AOI handling
    # ------------------------------------------------------------------ #

    def _load_aoi_mask(self, geo: Any | None, raster_shape: tuple[int, int]) -> np.ndarray | None:
        """Build an in-AOI boolean mask aligned to the raster.

        Returns None when the AOI file is missing OR ``geo`` is absent —
        callers fall back to "use the whole tile".
        """
        aoi_file = Path(self._aoi_path)
        if not aoi_file.exists():
            logger.warning("AOI GeoJSON not found at %s — using full tile", aoi_file)
            return None
        if geo is None:
            logger.warning("GeoContext absent — cannot project AOI; using full tile")
            return None

        try:
            import rasterio
            from rasterio.features import geometry_mask
            from rasterio.warp import transform_geom
        except ImportError as e:
            logger.warning("rasterio missing (%s) — using full tile", e)
            return None

        with open(aoi_file) as f:
            gj = json.load(f)

        try:
            from shapely.geometry import shape, box
            from shapely.ops import unary_union

            features = gj["features"]
            geoms_4326 = [feat["geometry"] for feat in features]
            # Project polygons to data CRS, then use their AXIS-ALIGNED ENVELOPE
            # in that CRS as the AOI mask. This matches the data-fetch contract:
            # `_to_nmd_grid` asks DES openEO for the EPSG:3006-axis-aligned bbox
            # of the WGS84 polygon's projection. If the analyzer here used the
            # raw projected polygon (a parallelogram when the source was a
            # WGS84 rectangle), it would NaN-clip the corner wedges of data
            # that the fetcher already legitimately delivered. Envelope keeps
            # both ends consistent.
            projected = unary_union([
                shape(transform_geom("EPSG:4326", geo.crs, g))
                for g in geoms_4326
            ])
            envelope = box(*projected.bounds)
            out_of_aoi = geometry_mask(
                [envelope.__geo_interface__],
                out_shape=raster_shape,
                transform=geo.transform,
                invert=False,
            )
            in_aoi = ~out_of_aoi
        except Exception as e:
            logger.warning("AOI projection failed (%s) — using full tile", e)
            return None

        if not in_aoi.any():
            logger.warning("AOI polygon does not intersect tile — empty mask")
        return in_aoi

    # ------------------------------------------------------------------ #
    #  Saving
    # ------------------------------------------------------------------ #

    @staticmethod
    def _save_geotiff(
        path: Path,
        array: np.ndarray,
        geo: Any | None,
        nodata: float | None = None,
        dtype: str | None = None,
    ) -> None:
        """Write a single-band COG-style GeoTIFF.

        Falls back to a plain GeoTIFF if rasterio is missing or geo is None;
        in that case writes a .npy alongside so the showcase script can still
        render PNGs.
        """
        target_dtype = dtype or array.dtype.name
        try:
            import rasterio
        except ImportError:
            np.save(path.with_suffix(".npy"), array)
            return

        if geo is None:
            np.save(path.with_suffix(".npy"), array)
            return

        H, W = array.shape
        profile = {
            "driver": "GTiff",
            "height": H,
            "width": W,
            "count": 1,
            "dtype": target_dtype,
            "crs": geo.crs,
            "transform": geo.transform,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        if nodata is not None:
            profile["nodata"] = nodata

        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array.astype(target_dtype), 1)

    # ------------------------------------------------------------------ #
    #  Per-method dispatch
    # ------------------------------------------------------------------ #

    def _run_classical_indices(
        self,
        bands: dict[str, np.ndarray],
        valid: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute NDCI + MCI on water pixels; NaN elsewhere."""
        outputs: dict[str, np.ndarray] = {}
        if self._ndci_enabled and "B04" in bands and "B05" in bands:
            ndci = classical_indices.compute_ndci(bands["B04"], bands["B05"])
            ndci[~valid] = np.nan
            outputs["ndci"] = ndci

        if self._mci_enabled and {"B04", "B05", "B06"}.issubset(bands):
            mci = classical_indices.compute_mci(bands["B04"], bands["B05"], bands["B06"])
            mci[~valid] = np.nan
            outputs["mci"] = mci
        return outputs

    def _run_mdn(
        self,
        bands: dict[str, np.ndarray],
        valid: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if not self._mdn_enabled:
            return {}
        try:
            return run_mdn(bands, valid, mdn_repo_path=self._mdn_repo_path)
        except MDNUnavailable as e:
            logger.warning("MDN skipped: %s", e)
            return {}

    def _run_c2rcc(
        self,
        bands: dict[str, np.ndarray],
        valid: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if not self._c2rcc_enabled:
            return {}
        try:
            return run_c2rcc(bands, valid)
        except C2RCCUnavailable as e:
            logger.warning("C2RCC skipped: %s", e)
            return {}

    @staticmethod
    def _compute_chl_spread(
        mdn_chl: np.ndarray | None,
        c2rcc_chl: np.ndarray | None,
    ) -> np.ndarray | None:
        """Std across MDN + C2RCC Chl-a (only over pixels valid in both)."""
        if mdn_chl is None or c2rcc_chl is None:
            return None
        stack = np.stack([mdn_chl, c2rcc_chl], axis=0)
        with np.errstate(invalid="ignore"):
            spread = np.nanstd(stack, axis=0)
        return spread.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Statistics
    # ------------------------------------------------------------------ #

    @staticmethod
    def _stats(array: np.ndarray) -> dict[str, float]:
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return {"n_pixels": 0}
        return {
            "n_pixels": int(finite.size),
            "mean": float(finite.mean()),
            "p50": float(np.median(finite)),
            "p95": float(np.percentile(finite, 95)),
            "max": float(finite.max()),
        }

    # ------------------------------------------------------------------ #
    #  BaseAnalyzer interface
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        rgb: np.ndarray,
        bands: dict[str, np.ndarray] | None = None,
        date: str | None = None,
        coords: dict | None = None,
        output_dir: str = "outputs",
        scl: np.ndarray | None = None,
        geo: Any | None = None,
    ) -> AnalysisResult:
        if bands is None:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error="WaterQualityAnalyzer requires multi-band input",
            )

        # 1. Water mask (SCL primary, MNDWI fallback)
        try:
            wm, mask_method = water_mask.build_water_mask(bands, scl=scl)
        except ValueError as e:
            return AnalysisResult(
                analyzer=self.name, success=False, error=str(e),
            )

        # 2. AOI clip
        aoi_mask = self._load_aoi_mask(geo, wm.shape)
        if aoi_mask is not None:
            valid = wm & aoi_mask
            if not valid.any():
                return AnalysisResult(
                    analyzer=self.name, success=True,
                    outputs={}, metadata={
                        "out_of_aoi": True,
                        "mask_method": mask_method,
                    },
                )
        else:
            valid = wm

        # 3. Run methods
        rasters: dict[str, np.ndarray] = {}
        rasters.update(self._run_classical_indices(bands, valid))
        mdn_out = self._run_mdn(bands, valid)
        c2rcc_out = self._run_c2rcc(bands, valid)

        # Namespace MDN outputs: chlorophyll_a → chlorophyll_mdn, etc.
        mdn_renamed = {f"{k}_mdn": v for k, v in mdn_out.items()}
        c2rcc_renamed = {f"{k}_c2rcc": v for k, v in c2rcc_out.items()}
        rasters.update(mdn_renamed)
        rasters.update(c2rcc_renamed)

        # 4. Inter-method spread (MDN + C2RCC chl only)
        spread = self._compute_chl_spread(
            mdn_out.get("chlorophyll_a"),
            c2rcc_out.get("chlorophyll_a"),
        )
        if spread is not None:
            rasters["chlorophyll_spread"] = spread

        # 5. Always save the water mask
        rasters["water_mask"] = valid.astype(np.uint8) * 255

        if not rasters or set(rasters) == {"water_mask"}:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error="all retrieval methods failed; only water_mask produced",
                metadata={"mask_method": mask_method},
            )

        # 6. Save outputs
        date_safe = date or "undated"
        out_root = Path(output_dir) / "water_quality" / date_safe
        out_root.mkdir(parents=True, exist_ok=True)

        # dtype map: physical units → float32, indices → float32, mask → uint8
        for key, arr in rasters.items():
            if key == "water_mask":
                self._save_geotiff(out_root / f"{key}.tif", arr, geo, nodata=0, dtype="uint8")
            else:
                # Replace inf with NaN before saving
                arr_safe = np.where(np.isfinite(arr), arr, np.nan).astype(np.float32)
                self._save_geotiff(
                    out_root / f"{key}.tif", arr_safe, geo, nodata=float("nan"), dtype="float32",
                )

        # 7. Summary
        summary: dict[str, Any] = {
            "date": date,
            "mask_method": mask_method,
            "aoi_applied": aoi_mask is not None,
            "n_water_pixels": int(valid.sum()),
            "methods_succeeded": {
                "ndci": "ndci" in rasters,
                "mci": "mci" in rasters,
                "mdn": bool(mdn_out),
                "c2rcc": bool(c2rcc_out),
            },
            "stats": {key: self._stats(arr) for key, arr in rasters.items() if key != "water_mask"},
            "bloom_above_threshold": {},
        }
        # Above-threshold area (per-method)
        for key in ("chlorophyll_a_mdn", "chlorophyll_a_c2rcc"):
            arr = rasters.get(key)
            if arr is not None:
                above = int(np.sum(arr > self._bloom_threshold))
                summary["bloom_above_threshold"][key] = {
                    "threshold_mg_m3": self._bloom_threshold,
                    "n_pixels_above": above,
                    "fraction_of_water": above / max(1, int(valid.sum())),
                }

        with open(out_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs=rasters,
            metadata={
                "date": date,
                "mask_method": mask_method,
                "output_dir": str(out_root),
                "methods_succeeded": summary["methods_succeeded"],
                "reference": (
                    "Pahlevan et al. 2020/2022 (MDN); Brockmann et al. 2016 (C2RCC); "
                    "Mishra & Mishra 2012 (NDCI); Gower et al. 2005 (MCI)"
                ),
            },
        )
