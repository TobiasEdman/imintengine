"""
imint/analyzers/nmd.py — NMD (Nationellt Marktäckedata) LULC analyzer

Cross-references other analyzer results with land cover classes from NMD.
Instead of just reporting "40% change", we can say "change occurred in
60% forest, 30% cropland, 10% developed land".

NMD 2018 Basskikt class codes:
    111-118  Forest outside wetland (pine, spruce, deciduous, mixed, temp. non-forest)
    121-128  Forest on wetland
    2        Open wetland
    3        Cropland
    41-42    Other open land (without/with vegetation)
    51-53    Developed land (buildings, infrastructure, roads)
    61-62    Water (lakes, sea)
    255      Unclassified / cloud

This analyzer runs last and consumes previous_results from change_detection,
spectral, and object_detection analyzers.
"""
from __future__ import annotations

import numpy as np

from .base import BaseAnalyzer, AnalysisResult
from ..fetch import fetch_nmd_data, FetchError


# ── NMD Class Mapping ────────────────────────────────────────────────────────

# Level 1: broad categories
NMD_LEVEL1 = {
    "forest": set(range(111, 119)) | set(range(121, 129)),
    "wetland": {2},
    "cropland": {3},
    "open_land": {41, 42},
    "developed": {51, 52, 53},
    "water": {61, 62},
}

# Level 2: more specific categories
NMD_LEVEL2 = {
    "forest_pine": {111},
    "forest_spruce": {112},
    "forest_deciduous": {113},
    "forest_mixed": {114},
    "forest_temp_non_forest": {115, 116, 117, 118},
    "forest_wetland_pine": {121},
    "forest_wetland_spruce": {122},
    "forest_wetland_deciduous": {123},
    "forest_wetland_mixed": {124},
    "forest_wetland_temp": {125, 126, 127, 128},
    "open_wetland": {2},
    "cropland": {3},
    "open_land_bare": {41},
    "open_land_vegetated": {42},
    "developed_buildings": {51},
    "developed_infrastructure": {52},
    "developed_roads": {53},
    "water_lakes": {61},
    "water_sea": {62},
}

# Color palette for Level 1 overlay (RGB tuples)
NMD_COLORS_L1 = {
    "forest": (34, 139, 34),
    "wetland": (139, 90, 43),
    "cropland": (255, 215, 0),
    "open_land": (210, 180, 140),
    "developed": (255, 0, 0),
    "water": (0, 0, 255),
    "unclassified": (128, 128, 128),
}


def nmd_code_to_l1(code: int) -> str:
    """Map an NMD class code to its Level 1 category name.

    Args:
        code: NMD class code (e.g. 111, 3, 61).

    Returns:
        Level 1 category name (e.g. "forest", "cropland", "water").
    """
    for name, codes in NMD_LEVEL1.items():
        if code in codes:
            return name
    return "unclassified"


def nmd_code_to_l2(code: int) -> str:
    """Map an NMD class code to its Level 2 category name.

    Args:
        code: NMD class code (e.g. 111, 3, 61).

    Returns:
        Level 2 category name (e.g. "forest_pine", "cropland").
    """
    for name, codes in NMD_LEVEL2.items():
        if code in codes:
            return name
    return "unclassified"


def nmd_raster_to_l1(nmd_raster: np.ndarray) -> np.ndarray:
    """Convert NMD class codes to Level 1 integer codes.

    Mapping:
        0 = unclassified, 1 = forest, 2 = wetland, 3 = cropland,
        4 = open_land, 5 = developed, 6 = water

    Args:
        nmd_raster: 2D uint8 array with NMD class codes.

    Returns:
        2D uint8 array with Level 1 codes (0-6).
    """
    l1 = np.zeros_like(nmd_raster, dtype=np.uint8)
    l1_names = ["unclassified", "forest", "wetland", "cropland",
                "open_land", "developed", "water"]
    for idx, name in enumerate(l1_names):
        if name == "unclassified":
            continue
        for code in NMD_LEVEL1[name]:
            l1[nmd_raster == code] = idx
    return l1


# ── Analyzer ─────────────────────────────────────────────────────────────────

class NMDAnalyzer(BaseAnalyzer):
    """NMD Land Cover analyzer.

    Fetches NMD data from DES, computes per-class statistics, and
    cross-references with previous analyzer results (change detection,
    spectral indices, object detection).
    """

    name = "nmd"

    def analyze(
        self,
        rgb: np.ndarray,
        bands=None,
        date=None,
        coords=None,
        output_dir="outputs",
        previous_results: list[AnalysisResult] | None = None,
    ) -> AnalysisResult:
        if coords is None:
            return AnalysisResult(
                analyzer=self.name,
                success=True,
                outputs={"nmd_available": False},
                metadata={"reason": "no_coords"},
            )

        # Fetch NMD raster
        h, w = rgb.shape[:2]
        try:
            nmd_result = fetch_nmd_data(
                coords=coords,
                target_shape=(h, w),
            )
            nmd_raster = nmd_result.nmd_raster
        except (FetchError, ImportError) as e:
            return AnalysisResult(
                analyzer=self.name,
                success=True,
                outputs={"nmd_available": False},
                metadata={"reason": f"fetch_failed: {e}"},
            )

        # Compute class statistics
        class_stats = _compute_class_stats(nmd_raster)

        # Build Level 1 raster for exports
        l1_raster = nmd_raster_to_l1(nmd_raster)

        # Cross-reference with previous results
        cross_ref = {}
        if previous_results:
            cross_ref = _cross_reference(nmd_raster, previous_results)

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "nmd_available": True,
                "nmd_raster": nmd_raster,
                "l1_raster": l1_raster,
                "class_stats": class_stats,
                "cross_reference": cross_ref,
            },
            metadata={
                "from_cache": nmd_result.from_cache,
                "image_shape": (h, w),
            },
        )


# ── Statistics ────────────────────────────────────────────────────────────────

def _compute_class_stats(nmd_raster: np.ndarray) -> dict:
    """Compute per-class pixel counts and fractions.

    Returns:
        Dict with "level1" and "level2" sub-dicts, each mapping
        category name to {"pixel_count": int, "fraction": float}.
    """
    total = max(nmd_raster.size, 1)

    # Level 1
    l1_stats = {}
    for name, codes in NMD_LEVEL1.items():
        mask = np.isin(nmd_raster, list(codes))
        count = int(mask.sum())
        if count > 0:
            l1_stats[name] = {
                "pixel_count": count,
                "fraction": round(count / total, 4),
            }

    # Unclassified (anything not in known classes)
    all_known = set()
    for codes in NMD_LEVEL1.values():
        all_known |= codes
    unclassified_mask = ~np.isin(nmd_raster, list(all_known))
    unc_count = int(unclassified_mask.sum())
    if unc_count > 0:
        l1_stats["unclassified"] = {
            "pixel_count": unc_count,
            "fraction": round(unc_count / total, 4),
        }

    # Level 2
    l2_stats = {}
    for name, codes in NMD_LEVEL2.items():
        mask = np.isin(nmd_raster, list(codes))
        count = int(mask.sum())
        if count > 0:
            l2_stats[name] = {
                "pixel_count": count,
                "fraction": round(count / total, 4),
            }

    return {"level1": l1_stats, "level2": l2_stats}


# ── Cross-reference ──────────────────────────────────────────────────────────

def _cross_reference(
    nmd_raster: np.ndarray,
    previous_results: list[AnalysisResult],
) -> dict:
    """Cross-reference NMD classes with other analyzer outputs.

    Finds results from change_detection, spectral, and object_detection
    among previous_results and computes per-LULC-class breakdowns.

    Returns:
        Dict with keys "spectral", "change_detection", "object_detection"
        (only present if the corresponding result was found).
    """
    cross_ref = {}

    # Find results by analyzer name
    results_by_name = {r.analyzer: r for r in previous_results if r.success}

    # Spectral cross-reference
    spectral = results_by_name.get("spectral")
    if spectral:
        cross_ref["spectral"] = _spectral_cross_ref(nmd_raster, spectral)

    # Change detection cross-reference
    change = results_by_name.get("change_detection")
    if change:
        cross_ref["change_detection"] = _change_cross_ref(nmd_raster, change)

    # Object detection cross-reference
    objdet = results_by_name.get("object_detection")
    if objdet:
        cross_ref["object_detection"] = _anomaly_cross_ref(nmd_raster, objdet)

    return cross_ref


def _spectral_cross_ref(nmd_raster: np.ndarray, spectral: AnalysisResult) -> dict:
    """Compute mean spectral indices and land cover fractions per NMD class.

    For each Level 1 LULC class, reports:
        - mean_ndvi, mean_ndwi: mean spectral index values
        - vegetation_fraction, water_fraction, built_up_fraction: from spectral land cover

    Args:
        nmd_raster: NMD class code raster.
        spectral: AnalysisResult from SpectralAnalyzer.

    Returns:
        Dict mapping L1 class name to spectral statistics.
    """
    indices = spectral.outputs.get("indices", {})
    ndvi = indices.get("NDVI")
    ndwi = indices.get("NDWI")
    land_cover = spectral.outputs.get("land_cover")

    result = {}
    for name, codes in NMD_LEVEL1.items():
        mask = np.isin(nmd_raster, list(codes))
        n_pixels = int(mask.sum())
        if n_pixels == 0:
            continue

        entry = {}
        if ndvi is not None and ndvi.shape == nmd_raster.shape:
            entry["mean_ndvi"] = round(float(np.mean(ndvi[mask])), 4)
        if ndwi is not None and ndwi.shape == nmd_raster.shape:
            entry["mean_ndwi"] = round(float(np.mean(ndwi[mask])), 4)

        if land_cover is not None and land_cover.shape == nmd_raster.shape:
            lc_in_class = land_cover[mask]
            entry["vegetation_fraction"] = round(float((lc_in_class == 2).sum()) / n_pixels, 4)
            entry["water_fraction"] = round(float((lc_in_class == 1).sum()) / n_pixels, 4)
            entry["built_up_fraction"] = round(float((lc_in_class == 3).sum()) / n_pixels, 4)

        if entry:
            result[name] = entry

    return result


def _change_cross_ref(nmd_raster: np.ndarray, change: AnalysisResult) -> dict:
    """Compute change fraction per NMD class.

    For each Level 1 LULC class, reports the fraction of pixels
    that were flagged as changed.

    Args:
        nmd_raster: NMD class code raster.
        change: AnalysisResult from ChangeDetectionAnalyzer.

    Returns:
        Dict mapping L1 class name to {"change_fraction": float, "changed_pixels": int}.
    """
    change_mask = change.outputs.get("change_mask")
    if change_mask is None:
        return {}

    if change_mask.shape != nmd_raster.shape:
        return {}

    result = {}
    for name, codes in NMD_LEVEL1.items():
        lulc_mask = np.isin(nmd_raster, list(codes))
        n_pixels = int(lulc_mask.sum())
        if n_pixels == 0:
            continue

        changed_in_class = int((change_mask & lulc_mask).sum())
        result[name] = {
            "change_fraction": round(changed_in_class / n_pixels, 4),
            "changed_pixels": changed_in_class,
            "total_pixels": n_pixels,
        }

    return result


def _anomaly_cross_ref(nmd_raster: np.ndarray, objdet: AnalysisResult) -> dict:
    """Count detected anomalies/objects per NMD class.

    Uses the center point of each detection's bounding box to determine
    which LULC class it falls in.

    Args:
        nmd_raster: NMD class code raster.
        objdet: AnalysisResult from ObjectDetectionAnalyzer.

    Returns:
        Dict mapping L1 class name to {"count": int, "detections": list}.
    """
    regions = objdet.outputs.get("regions", [])
    if not regions:
        return {}

    h, w = nmd_raster.shape
    result = {}

    for region in regions:
        bbox = region.get("bbox", {})
        cy = (bbox.get("y_min", 0) + bbox.get("y_max", 0)) // 2
        cx = (bbox.get("x_min", 0) + bbox.get("x_max", 0)) // 2

        # Clamp to raster bounds
        cy = min(max(cy, 0), h - 1)
        cx = min(max(cx, 0), w - 1)

        code = int(nmd_raster[cy, cx])
        l1_name = nmd_code_to_l1(code)

        if l1_name not in result:
            result[l1_name] = {"count": 0, "detections": []}

        result[l1_name]["count"] += 1
        result[l1_name]["detections"].append({
            "center": (cy, cx),
            "score": region.get("score", 0),
            "label": region.get("label", "unknown"),
        })

    return result
