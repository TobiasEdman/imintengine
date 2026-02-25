"""
imint/engine.py — Analysis engine

run_job() is the single entry point for all executors.
It is completely independent of how the job was scheduled or submitted.
"""
from __future__ import annotations

import os
import json
import yaml
import numpy as np
from pathlib import Path

from .job import IMINTJob, IMINTResult
from .analyzers.base import AnalysisResult
from .analyzers.change_detection import ChangeDetectionAnalyzer
from .analyzers.spectral import SpectralAnalyzer
from .analyzers.object_detection import ObjectDetectionAnalyzer
from .analyzers.prithvi import PrithviAnalyzer
from .analyzers.nmd import NMDAnalyzer
from .analyzers.cot import COTAnalyzer
from .analyzers.marine_vessels import MarineVesselAnalyzer
from .exporters.export import (
    save_rgb_png, save_change_overlay, save_ndvi_colormap,
    save_regions_geojson, save_geotiff, save_summary_report,
    save_nmd_overlay, save_nmd_stats, save_nmd_visualization,
    save_prithvi_overlay, save_prithvi_embedding_viz,
    save_ndvi_clean_png, save_spectral_index_clean_png,
    save_prithvi_seg_clean_png,
    save_cot_clean_png, save_cloud_class_clean_png,
    save_dnbr_clean_png, save_change_gradient_png,
    save_vessel_overlay,
)
from .exporters.html_report import save_html_report


ANALYZER_REGISTRY = {
    "change_detection": ChangeDetectionAnalyzer,
    "spectral": SpectralAnalyzer,
    "object_detection": ObjectDetectionAnalyzer,
    "prithvi": PrithviAnalyzer,
    "nmd": NMDAnalyzer,
    "cot": COTAnalyzer,
    "marine_vessels": MarineVesselAnalyzer,
}


def _load_config(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {name: {"enabled": True} for name in ANALYZER_REGISTRY}


def run_job(job: IMINTJob) -> IMINTResult:
    """
    Run all enabled analyzers on a cloud-free image.

    This function knows nothing about ColonyOS, cron, Airflow, etc.
    All execution context is already resolved in the IMINTJob.

    Args:
        job: A fully populated IMINTJob (rgb and coords must be set)

    Returns:
        IMINTResult with per-analyzer results and output paths
    """
    if job.rgb is None:
        return IMINTResult(
            job_id=job.job_id, date=job.date, success=False,
            error="IMINTJob.rgb is None — image data must be set before calling run_job()"
        )

    os.makedirs(job.output_dir, exist_ok=True)
    config = _load_config(job.config_path)
    results: list[AnalysisResult] = []

    print(f"\n{'='*60}")
    print(f"  IMINT Engine  |  {job.date}  |  job_id={job.job_id}")
    print(f"  Image: {job.rgb.shape}  |  Output: {job.output_dir}")
    if job.geo:
        print(f"  CRS: {job.geo.crs}  |  Grid: {job.geo.bounds_projected}")
    print(f"{'='*60}")

    prefix = f"{job.date}_" if job.date else ""
    save_rgb_png(job.rgb, os.path.join(job.output_dir, f"{prefix}rgb.png"))
    _save_bands_cache(job)

    for name, cls in ANALYZER_REGISTRY.items():
        cfg = config.get(name, {})
        if not cfg.get("enabled", True):
            print(f"  [{name}] skipped")
            continue

        print(f"  [{name}] running...")
        analyzer = cls(config=cfg)
        result = analyzer.run(
            job.rgb, bands=job.bands or None,
            date=job.date, coords=job.coords,
            output_dir=job.output_dir,
            previous_results=results,
            scl=job.scl,
        )
        results.append(result)
        print(f"  {result.summary()}")
        _export(result, job)

    summary_path = save_summary_report(results, job.date, job.output_dir)
    print(f"\n  Summary → {summary_path}")

    # Generate interactive HTML report
    _generate_html_report(job, prefix)
    print(f"{'='*60}\n")

    return IMINTResult(
        job_id=job.job_id,
        date=job.date,
        success=all(r.success for r in results),
        analyzer_results=results,
        summary_path=summary_path,
    )


def _save_bands_cache(job: IMINTJob) -> None:
    """Save band arrays as .npy for future reuse (e.g., re-running segmentation).

    Creates a ``bands/`` subdirectory under ``job.output_dir`` with one .npy
    file per band plus a ``bands_meta.json`` for coordinates and GeoContext.
    Skips silently if no bands are present.
    """
    if not job.bands:
        return
    prefix = f"{job.date}_" if job.date else ""
    bands_dir = os.path.join(job.output_dir, "bands")
    os.makedirs(bands_dir, exist_ok=True)

    for name, arr in job.bands.items():
        np.save(os.path.join(bands_dir, f"{prefix}{name}.npy"), arr)

    meta = {
        "date": job.date,
        "coords": job.coords,
        "band_names": list(job.bands.keys()),
        "shape": list(job.bands[list(job.bands.keys())[0]].shape),
    }
    if job.geo:
        meta["geo"] = {
            "crs": str(job.geo.crs),
            "bounds_projected": job.geo.bounds_projected,
            "bounds_wgs84": job.geo.bounds_wgs84,
            "shape": list(job.geo.shape),
            "transform": list(job.geo.transform)[:6],
        }

    meta_path = os.path.join(bands_dir, f"{prefix}bands_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    saved: {bands_dir}/ ({len(job.bands)} bands cached)")


def load_bands_cache(bands_dir: str, prefix: str = "") -> dict:
    """Load cached band arrays and metadata from a previous pipeline run.

    Args:
        bands_dir: Path to the ``bands/`` directory.
        prefix: Date prefix, e.g. ``"2023-07-15_"``.

    Returns:
        Dict with keys ``"bands"``, ``"coords"``, ``"geo_meta"``, ``"date"``.

    Raises:
        FileNotFoundError: If metadata file is missing.
    """
    meta_path = os.path.join(bands_dir, f"{prefix}bands_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    bands = {}
    for band_name in meta["band_names"]:
        path = os.path.join(bands_dir, f"{prefix}{band_name}.npy")
        bands[band_name] = np.load(path)

    return {
        "bands": bands,
        "coords": meta["coords"],
        "geo_meta": meta.get("geo"),
        "date": meta.get("date"),
    }


def _export(result: AnalysisResult, job: IMINTJob) -> None:
    if not result.success:
        return
    prefix = f"{job.date}_" if job.date else ""
    out = job.output_dir

    if result.analyzer == "change_detection":
        mask = result.outputs.get("change_mask")
        if mask is not None and mask.any():
            save_change_overlay(job.rgb, mask, os.path.join(out, f"{prefix}change_overlay.png"))
            regions = result.outputs.get("regions", [])
            if regions:
                save_regions_geojson(regions, os.path.join(out, f"{prefix}change_regions.geojson"),
                                     geo=job.geo, coords=job.coords, image_shape=job.rgb.shape)

        # Gradient change magnitude
        change_diff = result.outputs.get("change_diff")
        if change_diff is not None:
            save_change_gradient_png(change_diff, os.path.join(out, f"{prefix}change_gradient.png"))

        # dNBR (burn severity)
        dnbr = result.outputs.get("dnbr")
        if dnbr is not None:
            save_dnbr_clean_png(dnbr, os.path.join(out, f"{prefix}dnbr_clean.png"))

    elif result.analyzer == "spectral":
        indices = result.outputs.get("indices", {})
        ndvi = indices.get("NDVI")
        if ndvi is not None:
            save_ndvi_colormap(ndvi, os.path.join(out, f"{prefix}ndvi.png"))
            save_ndvi_clean_png(ndvi, os.path.join(out, f"{prefix}ndvi_clean.png"))

        # Save clean PNGs for all spectral indices (for HTML report)
        index_cmaps = {
            "NDWI": ("RdBu", -1, 1),
            "NDBI": ("RdYlBu_r", -1, 1),
            "EVI": ("RdYlGn", -1, 1),
            "NBR": ("RdYlGn", -1, 1),
        }
        for idx_name, (cmap, vmin, vmax) in index_cmaps.items():
            idx_arr = indices.get(idx_name)
            if idx_arr is not None:
                save_spectral_index_clean_png(
                    idx_arr,
                    os.path.join(out, f"{prefix}{idx_name.lower()}_clean.png"),
                    cmap_name=cmap, vmin=vmin, vmax=vmax,
                )

    elif result.analyzer == "object_detection":
        regions = result.outputs.get("regions", [])
        if regions:
            save_regions_geojson(regions, os.path.join(out, f"{prefix}detections.geojson"),
                                 geo=job.geo, coords=job.coords, image_shape=job.rgb.shape)

    elif result.analyzer == "marine_vessels":
        regions = result.outputs.get("regions", [])
        if regions:
            save_regions_geojson(regions, os.path.join(out, f"{prefix}vessels.geojson"),
                                 geo=job.geo, coords=job.coords, image_shape=job.rgb.shape)
        save_vessel_overlay(
            job.rgb, regions,
            os.path.join(out, f"{prefix}vessels_clean.png"),
        )

    elif result.analyzer == "prithvi":
        mode = result.metadata.get("mode", "embeddings")
        if mode == "segmentation":
            seg_mask = result.outputs.get("seg_mask")
            if seg_mask is not None:
                # Extract class names from class_stats for legend display
                class_stats = result.outputs.get("class_stats", {})
                class_names = {
                    int(k): v["name"]
                    for k, v in class_stats.items()
                    if "name" in v
                } or None
                save_prithvi_overlay(
                    seg_mask,
                    os.path.join(out, f"{prefix}prithvi_seg.png"),
                    rgb=job.rgb,
                    class_names=class_names,
                )
                save_prithvi_seg_clean_png(
                    seg_mask,
                    os.path.join(out, f"{prefix}prithvi_seg_clean.png"),
                )
                save_geotiff(seg_mask, os.path.join(out, f"{prefix}prithvi_seg.tif"),
                             geo=job.geo, coords=job.coords)
        elif mode == "embeddings":
            embedding = result.outputs.get("embedding")
            if embedding is not None:
                np.save(os.path.join(out, f"{prefix}prithvi_embedding.npy"), embedding)
                print(f"    saved: {os.path.join(out, f'{prefix}prithvi_embedding.npy')}")
                save_prithvi_embedding_viz(
                    embedding, job.rgb,
                    os.path.join(out, f"{prefix}prithvi_embedding.png"),
                )

    elif result.analyzer == "cot":
        cot_map = result.outputs.get("cot_map")
        cloud_class = result.outputs.get("cloud_class")
        if cot_map is not None:
            save_geotiff(cot_map, os.path.join(out, f"{prefix}cot.tif"),
                         geo=job.geo, coords=job.coords)
            save_cot_clean_png(cot_map, os.path.join(out, f"{prefix}cot_clean.png"))
            if cloud_class is not None:
                save_cloud_class_clean_png(cloud_class, os.path.join(out, f"{prefix}cloud_class_clean.png"))
            _save_cot_visualization(cot_map, cloud_class, job.rgb,
                                    os.path.join(out, f"{prefix}cot.png"))

    elif result.analyzer == "nmd":
        if result.outputs.get("nmd_available"):
            l2_raster = result.outputs.get("l2_raster")
            class_stats = result.outputs.get("class_stats")
            cross_ref = result.outputs.get("cross_reference")
            if l2_raster is not None:
                # Simple color-coded overlay (kept for quick reference)
                save_nmd_overlay(l2_raster, os.path.join(out, f"{prefix}nmd_overlay.png"))
                # Rich multi-panel visualization with charts and cross-reference
                if class_stats:
                    save_nmd_visualization(
                        l2_raster, job.rgb, class_stats, cross_ref,
                        os.path.join(out, f"{prefix}nmd_analysis.png"),
                    )
            if class_stats:
                save_nmd_stats(class_stats, cross_ref, os.path.join(out, f"{prefix}nmd_stats.json"))


def _generate_html_report(job: IMINTJob, prefix: str) -> None:
    """Generate interactive HTML report after all analyzers complete."""
    out = job.output_dir
    h, w = job.rgb.shape[:2]

    # Collect image paths (only include those that exist)
    path_candidates = {
        "rgb": f"{prefix}rgb.png",
        "nmd": f"{prefix}nmd_overlay.png",
        "change": f"{prefix}change_overlay.png",
        "ndvi": f"{prefix}ndvi_clean.png",
        "ndwi": f"{prefix}ndwi_clean.png",
        "ndbi": f"{prefix}ndbi_clean.png",
        "evi": f"{prefix}evi_clean.png",
        "nbr": f"{prefix}nbr_clean.png",
        "dnbr": f"{prefix}dnbr_clean.png",
        "change_gradient": f"{prefix}change_gradient.png",
        "prithvi_seg": f"{prefix}prithvi_seg_clean.png",
        "cot": f"{prefix}cot_clean.png",
        "vessels": f"{prefix}vessels_clean.png",
    }
    image_paths = {}
    for key, filename in path_candidates.items():
        full_path = os.path.join(out, filename)
        if os.path.exists(full_path):
            image_paths[key] = full_path

    if not image_paths:
        print("  [html_report] skipped — no images available")
        return

    # Load nmd_stats if available
    nmd_stats_path = os.path.join(out, f"{prefix}nmd_stats.json")
    nmd_stats = {}
    if os.path.exists(nmd_stats_path):
        with open(nmd_stats_path) as f:
            nmd_stats = json.load(f)

    # Load imint_summary
    summary_path = os.path.join(out, f"{prefix}imint_summary.json")
    imint_summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            imint_summary = json.load(f)

    html_path = os.path.join(out, f"{prefix}imint_report.html")
    save_html_report(
        image_paths=image_paths,
        nmd_stats=nmd_stats,
        imint_summary=imint_summary,
        image_shape=(h, w),
        date=job.date or "unknown",
        output_path=html_path,
    )
    print(f"  HTML Report → {html_path}")


def _save_cot_visualization(cot_map, cloud_class, rgb, path):
    """Save COT visualization: RGB and histogram-stretched COT heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .exporters.export import _cot_stretch_range

    vmin, vmax = _cot_stretch_range(cot_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: RGB
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB")
    axes[0].axis("off")

    # Panel 2: Continuous COT heatmap (histogram stretched)
    im = axes[1].imshow(cot_map, cmap="hot_r", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Cloud Optical Thickness (stretch {vmin:.4f}\u2013{vmax:.4f})")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="COT")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {path}")
