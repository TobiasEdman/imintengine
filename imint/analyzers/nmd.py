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


# Level 2 class order (matches NMD_LEVEL2 dict order)
L2_CLASSES = list(NMD_LEVEL2.keys())


def nmd_raster_to_l2(nmd_raster: np.ndarray) -> np.ndarray:
    """Convert NMD class codes to Level 2 integer codes.

    Mapping:
        0 = unclassified, 1 = forest_pine, 2 = forest_spruce, ..., 19 = water_sea

    Args:
        nmd_raster: 2D uint8 array with NMD class codes.

    Returns:
        2D uint8 array with Level 2 codes (0-19).
    """
    l2 = np.zeros_like(nmd_raster, dtype=np.uint8)
    for idx, name in enumerate(L2_CLASSES):
        for code in NMD_LEVEL2[name]:
            l2[nmd_raster == code] = idx + 1  # 0 reserved for unclassified
    return l2


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

        # Build Level 1 and Level 2 rasters for exports
        l1_raster = nmd_raster_to_l1(nmd_raster)
        l2_raster = nmd_raster_to_l2(nmd_raster)

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
                "l2_raster": l2_raster,
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

    # Prithvi segmentation cross-reference (burn scars / flood)
    prithvi = results_by_name.get("prithvi")
    if prithvi and prithvi.outputs.get("seg_mask") is not None:
        cross_ref["prithvi"] = _prithvi_cross_ref(nmd_raster, prithvi)

    return cross_ref


def _spectral_cross_ref(nmd_raster: np.ndarray, spectral: AnalysisResult) -> dict:
    """Compute mean spectral indices per NMD Level 2 class.

    For each Level 2 LULC class, reports:
        - mean_ndvi, mean_ndwi: mean spectral index values

    Args:
        nmd_raster: NMD class code raster.
        spectral: AnalysisResult from SpectralAnalyzer.

    Returns:
        Dict mapping L2 class name to spectral statistics.
    """
    indices = spectral.outputs.get("indices", {})
    ndvi = indices.get("NDVI")
    ndwi = indices.get("NDWI")

    result = {}
    for name, codes in NMD_LEVEL2.items():
        mask = np.isin(nmd_raster, list(codes))
        n_pixels = int(mask.sum())
        if n_pixels == 0:
            continue

        entry = {}
        if ndvi is not None and ndvi.shape == nmd_raster.shape:
            entry["mean_ndvi"] = round(float(np.mean(ndvi[mask])), 4)
        if ndwi is not None and ndwi.shape == nmd_raster.shape:
            entry["mean_ndwi"] = round(float(np.mean(ndwi[mask])), 4)

        if entry:
            result[name] = entry

    return result


def _change_cross_ref(nmd_raster: np.ndarray, change: AnalysisResult) -> dict:
    """Compute change fraction per NMD Level 2 class.

    For each Level 2 LULC class, reports the fraction of pixels
    that were flagged as changed.

    Args:
        nmd_raster: NMD class code raster.
        change: AnalysisResult from ChangeDetectionAnalyzer.

    Returns:
        Dict mapping L2 class name to {"change_fraction": float, "changed_pixels": int}.
    """
    change_mask = change.outputs.get("change_mask")
    if change_mask is None:
        return {}

    if change_mask.shape != nmd_raster.shape:
        return {}

    result = {}
    for name, codes in NMD_LEVEL2.items():
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
    """Count detected anomalies/objects per NMD Level 2 class.

    Uses the center point of each detection's bounding box to determine
    which LULC class it falls in.

    Args:
        nmd_raster: NMD class code raster.
        objdet: AnalysisResult from ObjectDetectionAnalyzer.

    Returns:
        Dict mapping L2 class name to {"count": int, "detections": list}.
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
        l2_name = nmd_code_to_l2(code)

        if l2_name not in result:
            result[l2_name] = {"count": 0, "detections": []}

        result[l2_name]["count"] += 1
        result[l2_name]["detections"].append({
            "center": (cy, cx),
            "score": region.get("score", 0),
            "label": region.get("label", "unknown"),
        })

    return result


def _prithvi_cross_ref(nmd_raster: np.ndarray, prithvi: AnalysisResult) -> dict:
    """Compute Prithvi segmentation fraction per NMD Level 2 class.

    For each Level 2 LULC class, reports the fraction of pixels
    classified as each Prithvi class (e.g., burned vs. no_burn).

    Args:
        nmd_raster: NMD class code raster.
        prithvi: AnalysisResult from PrithviAnalyzer (segmentation mode).

    Returns:
        Dict mapping L2 class name to per-class fractions and pixel counts.
    """
    seg_mask = prithvi.outputs.get("seg_mask")
    if seg_mask is None:
        return {}
    if seg_mask.shape != nmd_raster.shape:
        return {}

    class_stats = prithvi.outputs.get("class_stats", {})
    # Build name lookup: {0: "no_burn", 1: "burned"}
    class_names = {
        int(k): v.get("name", f"class_{k}")
        for k, v in class_stats.items()
    }

    task_head = prithvi.metadata.get("task_head", "unknown")

    result = {}
    for name, codes in NMD_LEVEL2.items():
        lulc_mask = np.isin(nmd_raster, list(codes))
        n_pixels = int(lulc_mask.sum())
        if n_pixels == 0:
            continue

        seg_in_class = seg_mask[lulc_mask]
        entry = {"total_pixels": n_pixels, "task_head": task_head}
        unique, counts = np.unique(seg_in_class, return_counts=True)
        for cls_id, cnt in zip(unique, counts):
            cls_int = int(cls_id)
            cls_name = class_names.get(cls_int, f"class_{cls_int}")
            entry[f"{cls_name}_fraction"] = round(int(cnt) / n_pixels, 4)
            entry[f"{cls_name}_pixels"] = int(cnt)
        result[name] = entry

    return result
