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
from .exporters.export import (
    save_rgb_png, save_change_overlay, save_ndvi_colormap,
    save_regions_geojson, save_geotiff, save_summary_report,
    save_nmd_overlay, save_nmd_stats, save_prithvi_overlay,
    save_prithvi_embedding_viz,
)


ANALYZER_REGISTRY = {
    "change_detection": ChangeDetectionAnalyzer,
    "spectral": SpectralAnalyzer,
    "object_detection": ObjectDetectionAnalyzer,
    "prithvi": PrithviAnalyzer,
    "nmd": NMDAnalyzer,
    "cot": COTAnalyzer,
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
        )
        results.append(result)
        print(f"  {result.summary()}")
        _export(result, job)

    summary_path = save_summary_report(results, job.date, job.output_dir)
    print(f"\n  Summary → {summary_path}")
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

    elif result.analyzer == "spectral":
        ndvi = result.outputs.get("indices", {}).get("NDVI")
        if ndvi is not None:
            save_ndvi_colormap(ndvi, os.path.join(out, f"{prefix}ndvi.png"))
        lc = result.outputs.get("land_cover")
        if lc is not None:
            save_geotiff(lc, os.path.join(out, f"{prefix}land_cover.tif"), geo=job.geo, coords=job.coords)

    elif result.analyzer == "object_detection":
        regions = result.outputs.get("regions", [])
        if regions:
            save_regions_geojson(regions, os.path.join(out, f"{prefix}detections.geojson"),
                                 geo=job.geo, coords=job.coords, image_shape=job.rgb.shape)

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
            _save_cot_visualization(cot_map, cloud_class, job.rgb,
                                    os.path.join(out, f"{prefix}cot.png"))

    elif result.analyzer == "nmd":
        if result.outputs.get("nmd_available"):
            l1_raster = result.outputs.get("l1_raster")
            if l1_raster is not None:
                save_nmd_overlay(l1_raster, os.path.join(out, f"{prefix}nmd_overlay.png"))
            class_stats = result.outputs.get("class_stats")
            cross_ref = result.outputs.get("cross_reference")
            if class_stats:
                save_nmd_stats(class_stats, cross_ref, os.path.join(out, f"{prefix}nmd_stats.json"))


def _save_cot_visualization(cot_map, cloud_class, rgb, path):
    """Save COT visualization: RGB, COT heatmap, and cloud classification."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: RGB
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    # Panel 2: COT heatmap
    im = axes[1].imshow(cot_map, cmap="hot_r", vmin=0, vmax=0.3)
    axes[1].set_title("Cloud Optical Thickness")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="COT")

    # Panel 3: Cloud classification (clear / thin / thick)
    cmap = ListedColormap(["#2196F3", "#FFC107", "#F44336"])
    axes[2].imshow(cloud_class, cmap=cmap, vmin=0, vmax=2)
    axes[2].set_title("Cloud Classification")
    axes[2].axis("off")
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#2196F3", label="Clear"),
        Patch(facecolor="#FFC107", label="Thin cloud"),
        Patch(facecolor="#F44336", label="Thick cloud"),
    ]
    axes[2].legend(handles=legend, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {path}")
