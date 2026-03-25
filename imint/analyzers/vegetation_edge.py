"""
imint/analyzers/vegetation_edge.py — Vegetation edge detection via NDVI

Detects the boundary between vegetated and non-vegetated zones using
NDVI (Normalized Difference Vegetation Index) thresholding — inspired
by the VedgeSat toolkit (Muir et al., 2024) and its tropical
validation by Nugraha et al. (2026).

Spectral index:
    NDVI = (NIR − Red) / (NIR + Red)    → B08, B04

The analyzer produces:
  - 3-class segmentation map: water / non-vegetated / vegetated
  - Binary vegetation edge mask (1-pixel-wide centerline)
  - Vegetation edge contours (vectorised)
  - Per-class statistics

Thresholding methods:
  - Otsu: automatic threshold selection (default)
  - Weighted Peaks: VedgeSat's dual-Gaussian approach
    Io = w_veg * z_veg(I) + w_nonveg * z_nonveg(I)
    Default weights: w_veg=0.2, w_nonveg=0.8 (Muir et al., 2024)
  - Fixed: manual threshold (0.1–0.3 typical for Swedish coasts)

References:
  Muir et al. (2024), "VedgeSat: An automated, open-source toolkit for
  coastal change monitoring using satellite-derived vegetation edges",
  Earth Surf. Process. Landf., 49, 2405–2423.

  Nugraha et al. (2026), "Extending a scalable satellite-based
  vegetation edge detection framework to diverse tropical coasts",
  Front. Mar. Sci., 13, 1757991.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .base import AnalysisResult, BaseAnalyzer

CLASS_NAMES = ["water", "non_vegetated", "vegetated"]
CLASS_LABELS_SV = ["Vatten", "Ej vegetation", "Vegetation"]


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalized difference ratio with epsilon to avoid division by zero."""
    return (a - b) / (a + b + 1e-10)


class VegetationEdgeAnalyzer(BaseAnalyzer):
    """NDVI-based vegetation edge detector (VedgeSat approach).

    Works on multi-band Sentinel-2 data.  Requires bands B04 (Red)
    and B08 (NIR).  Optionally uses B03 (Green) and B11 (SWIR) for
    water masking via NDWI/MNDWI.
    """

    name = "vegetation_edge"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._ndvi_threshold: float | None = None

    # ------------------------------------------------------------------ #
    #  Spectral indices
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """NDVI = (NIR − Red) / (NIR + Red).  Vegetation > ~0.2."""
        return _safe_ratio(nir, red)

    @staticmethod
    def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """NDWI = (Green − NIR) / (Green + NIR).  Water > 0."""
        return _safe_ratio(green, nir)

    # ------------------------------------------------------------------ #
    #  Threshold estimation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _otsu(index: np.ndarray, mask: np.ndarray | None = None) -> float:
        """Compute Otsu threshold on a normalised index image.

        Args:
            index: (H, W) float array (e.g. NDVI).
            mask: optional (H, W) bool — True for valid pixels.

        Returns:
            Optimal threshold in the original index scale.
        """
        if mask is not None:
            valid = index[mask & np.isfinite(index)]
        else:
            valid = index[np.isfinite(index)]

        if len(valid) == 0:
            return 0.3  # safe default for vegetation

        lo = float(np.nanpercentile(valid, 1))
        hi = float(np.nanpercentile(valid, 99))
        scaled = np.clip(
            (index - lo) / (hi - lo + 1e-10) * 255, 0, 255
        ).astype(np.uint8)
        thresh_val, _ = cv2.threshold(
            scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return lo + (thresh_val / 255.0) * (hi - lo)

    @staticmethod
    def _weighted_peaks(
        ndvi: np.ndarray,
        mask: np.ndarray | None = None,
        w_veg: float = 0.2,
        w_nonveg: float = 0.8,
        n_bins: int = 256,
    ) -> float:
        """VedgeSat Weighted Peaks threshold (Muir et al., 2024).

        Estimates the NDVI threshold as the weighted mean of the
        peaks of the vegetated and non-vegetated NDVI distributions:
            Io = w_veg * peak_veg + w_nonveg * peak_nonveg

        Args:
            ndvi: (H, W) float NDVI array.
            mask: optional validity mask.
            w_veg: weight for vegetation peak (default 0.2).
            w_nonveg: weight for non-vegetation peak (default 0.8).

        Returns:
            Estimated NDVI threshold.
        """
        if mask is not None:
            values = ndvi[mask & np.isfinite(ndvi)]
        else:
            values = ndvi[np.isfinite(ndvi)]

        if len(values) < 100:
            return 0.3  # safe fallback

        # Build histogram
        lo = float(np.percentile(values, 1))
        hi = float(np.percentile(values, 99))
        hist, bin_edges = np.histogram(values, bins=n_bins, range=(lo, hi))
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Smooth histogram for peak finding
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(hist.astype(float), sigma=3)

        # Find the two highest peaks
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(smoothed, distance=n_bins // 8)

        if len(peaks) < 2:
            # Fall back to Otsu if we can't find two peaks
            return VegetationEdgeAnalyzer._otsu(ndvi, mask)

        # Sort by peak height, take top two
        sorted_peaks = sorted(peaks, key=lambda p: smoothed[p], reverse=True)
        peak_positions = sorted([bin_centres[sorted_peaks[0]],
                                  bin_centres[sorted_peaks[1]]])

        # Lower peak = non-vegetation, higher peak = vegetation
        peak_nonveg = peak_positions[0]
        peak_veg = peak_positions[1]

        threshold = w_veg * peak_veg + w_nonveg * peak_nonveg
        return float(threshold)

    # ------------------------------------------------------------------ #
    #  Classification
    # ------------------------------------------------------------------ #

    def classify(
        self,
        bands: dict[str, np.ndarray],
        rgb: np.ndarray | None = None,
        scl: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Produce a 3-class segmentation from NDVI thresholding.

        Classes:
            0 = water          (NDWI > 0 or very low NDVI)
            1 = non-vegetated  (NDVI below threshold)
            2 = vegetated      (NDVI above threshold)

        Returns:
            seg_map: (H, W) uint8, values 0–2.
            info: dict with thresholds and fractions.
        """
        red = bands.get("B04")
        nir = bands.get("B08")

        if red is None or nir is None:
            raise ValueError("Bands B04 (Red) and B08 (NIR) required")

        H, W = red.shape
        ndvi = self.compute_ndvi(red, nir)

        # Cloud mask from SCL if available
        valid_mask = np.ones((H, W), dtype=bool)
        if scl is not None:
            # SCL classes to mask: 3=cloud_shadow, 8=cloud_medium,
            # 9=cloud_high, 10=cirrus
            cloud_classes = {3, 8, 9, 10}
            for cls in cloud_classes:
                valid_mask &= (scl != cls)

        # Water detection using NDWI if Green band available
        green = bands.get("B03")
        if green is not None:
            ndwi = self.compute_ndwi(green, nir)
            water_mask = ndwi > 0
        else:
            # Fallback: very low NDVI = water
            water_mask = ndvi < -0.1

        # Determine NDVI threshold
        cfg = self.config
        fixed_threshold = cfg.get("ndvi_threshold")
        use_weighted_peaks = cfg.get("weighted_peaks", False)

        if fixed_threshold is not None:
            ndvi_thresh = float(fixed_threshold)
        elif use_weighted_peaks:
            w_veg = cfg.get("w_veg", 0.2)
            w_nonveg = cfg.get("w_nonveg", 0.8)
            # Only estimate threshold on non-water, valid pixels
            land_mask = valid_mask & (~water_mask)
            ndvi_thresh = self._weighted_peaks(
                ndvi, mask=land_mask, w_veg=w_veg, w_nonveg=w_nonveg
            )
        else:
            # Otsu on non-water, valid pixels
            land_mask = valid_mask & (~water_mask)
            ndvi_thresh = self._otsu(ndvi, mask=land_mask)

        self._ndvi_threshold = ndvi_thresh

        # Build segmentation map
        seg_map = np.full((H, W), 1, dtype=np.uint8)  # default: non-vegetated
        seg_map[water_mask] = 0                         # water
        seg_map[ndvi > ndvi_thresh] = 2                 # vegetated

        # Morphological cleanup on vegetation class
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        veg_mask = (seg_map == 2).astype(np.uint8)
        veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, kernel)
        veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, kernel)
        # Restore cleaned vegetation
        seg_map[(seg_map != 0) & (veg_mask == 1)] = 2
        seg_map[(seg_map == 2) & (veg_mask == 0)] = 1

        # Apply cloud mask — mark invalid pixels as non-vegetated
        if scl is not None:
            seg_map[~valid_mask] = 1

        total = H * W
        info = {
            "ndvi_threshold": round(float(ndvi_thresh), 4),
            "threshold_method": (
                "fixed" if fixed_threshold is not None
                else "weighted_peaks" if use_weighted_peaks
                else "otsu"
            ),
            "water_fraction": float((seg_map == 0).sum()) / total,
            "non_vegetated_fraction": float((seg_map == 1).sum()) / total,
            "vegetation_fraction": float((seg_map == 2).sum()) / total,
        }

        return seg_map, info

    # ------------------------------------------------------------------ #
    #  Vegetation edge extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_vegetation_edge(seg_map: np.ndarray) -> np.ndarray:
        """Extract vegetation edge as 1-pixel-wide centerline.

        The vegetation edge is the boundary between vegetated (class 2)
        and non-vegetated zones (classes 0 + 1).

        Uses morphological gradient to find the edge band, then
        skeletonizes to a single-pixel centerline.

        Args:
            seg_map: (H, W) uint8 — class indices (2 = vegetated).

        Returns:
            (H, W) uint8 — binary edge mask (255 = edge pixel).
        """
        from skimage.morphology import skeletonize

        vegetation = (seg_map == 2).astype(np.uint8)

        # Morphological cleanup for smoother edges
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        vegetation = cv2.morphologyEx(vegetation, cv2.MORPH_CLOSE, kernel_clean)
        vegetation = cv2.morphologyEx(vegetation, cv2.MORPH_OPEN, kernel_clean)

        # Edge via morphological gradient (produces ~3px wide band)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(vegetation, cv2.MORPH_GRADIENT, kernel_edge)

        # Skeletonize to 1-pixel-wide centerline
        skeleton = skeletonize(edges > 0).astype(np.uint8)

        # Remove border pixels to avoid edge artifacts
        margin = 3
        skeleton[:margin, :] = 0
        skeleton[-margin:, :] = 0
        skeleton[:, :margin] = 0
        skeleton[:, -margin:] = 0

        return (skeleton * 255).astype(np.uint8)

    @staticmethod
    def extract_contours(
        edge_mask: np.ndarray, min_length: int = 15
    ) -> list[np.ndarray]:
        """Trace skeleton into ordered polylines.

        Walks the 1-pixel skeleton directly instead of using
        cv2.findContours, which would trace both edges of even a thin
        skeleton and produce doubled parallel lines.

        Returns list of (N, 2) arrays in pixel coordinates [x, y].
        """
        from scipy import ndimage

        binary = (edge_mask > 0).astype(np.uint8)
        labeled, num_components = ndimage.label(
            binary, structure=np.ones((3, 3))
        )

        results = []
        for comp_id in range(1, num_components + 1):
            ys, xs = np.where(labeled == comp_id)
            if len(xs) < min_length:
                continue

            # Build adjacency map (8-connected)
            coords_set = set(zip(xs.tolist(), ys.tolist()))
            adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
            for x, y in coords_set:
                nbrs = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        n = (x + dx, y + dy)
                        if n in coords_set:
                            nbrs.append(n)
                adjacency[(x, y)] = nbrs

            # Start from an endpoint (degree ≤ 1) for a clean walk
            start = None
            for pt, nbrs in adjacency.items():
                if len(nbrs) <= 1:
                    start = pt
                    break
            if start is None:
                start = next(iter(adjacency))

            # Greedy DFS walk
            path = [start]
            visited = {start}
            current = start
            while True:
                moved = False
                for nbr in adjacency[current]:
                    if nbr not in visited:
                        path.append(nbr)
                        visited.add(nbr)
                        current = nbr
                        moved = True
                        break
                if not moved:
                    break

            if len(path) >= min_length:
                results.append(np.array(path))

        return results

    # ------------------------------------------------------------------ #
    #  Multi-year batch processing
    # ------------------------------------------------------------------ #

    def predict_multi_year(
        self,
        yearly_bands: dict[int, dict[str, np.ndarray]],
        yearly_rgbs: dict[int, np.ndarray] | None = None,
        yearly_scl: dict[int, np.ndarray] | None = None,
    ) -> dict[int, dict]:
        """Run classification for multiple years.

        Returns:
            {year: {"seg_map": ..., "edge": ..., "contours": [...],
                     "info": {...}}}
        """
        results = {}
        for year in sorted(yearly_bands.keys()):
            print(f"    {year}...", end=" ", flush=True)
            rgb = yearly_rgbs[year] if yearly_rgbs else None
            scl = yearly_scl[year] if yearly_scl else None
            seg_map, info = self.classify(yearly_bands[year], rgb, scl)
            edge = self.extract_vegetation_edge(seg_map)
            contours = self.extract_contours(edge)
            results[year] = {
                "seg_map": seg_map,
                "edge": edge,
                "contours": contours,
                "info": info,
            }
            print(
                f"veg={info['vegetation_fraction']:.1%}, "
                f"{len(contours)} contours, "
                f"NDVI thresh={info['ndvi_threshold']:.3f}"
            )
        return results

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
    ) -> AnalysisResult:
        """Run vegetation edge detection as part of the IMINT pipeline."""
        if bands is None:
            return AnalysisResult(
                analyzer=self.name,
                success=False,
                error="Multi-band input required (B04, B08)",
            )

        seg_map, info = self.classify(bands, rgb, scl)
        edge_mask = self.extract_vegetation_edge(seg_map)
        contours = self.extract_contours(edge_mask)

        class_fractions = {}
        total = seg_map.shape[0] * seg_map.shape[1]
        for i, name in enumerate(CLASS_NAMES):
            class_fractions[name] = float((seg_map == i).sum()) / total

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "segmentation_map": seg_map,
                "vegetation_edge_mask": edge_mask,
                "vegetation_fraction": info["vegetation_fraction"],
                "class_fractions": class_fractions,
                "ndvi": self.compute_ndvi(bands["B04"], bands["B08"]),
                "n_contours": len(contours),
            },
            metadata={
                "date": date,
                "method": f"NDVI + {info['threshold_method']} "
                          f"(VedgeSat approach)",
                "ndvi_threshold": info["ndvi_threshold"],
                "threshold_method": info["threshold_method"],
                "reference": "Muir et al. (2024) VedgeSat; "
                             "Nugraha et al. (2026)",
            },
        )
