"""
imint/engine.py — Analysis engine

run_job() is the single entry point for all executors.
It is completely independent of how the job was scheduled or submitted.
"""

import os
import yaml
import numpy as np
from pathlib import Path

from .job import IMINTJob, IMINTResult
from .analyzers.base import AnalysisResult
from .analyzers.change_detection import ChangeDetectionAnalyzer
from .analyzers.spectral import SpectralAnalyzer
from .analyzers.object_detection import ObjectDetectionAnalyzer
from .exporters.export import save_rgb_png, save_change_overlay, save_ndvi_colormap, save_regions_geojson, save_geotiff, save_summary_report


ANALYZER_REGISTRY = {
    "change_detection": ChangeDetectionAnalyzer,
    "spectral": SpectralAnalyzer,
    "object_detection": ObjectDetectionAnalyzer,
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
    print(f"{'='*60}")

    prefix = f"{job.date}_" if job.date else ""
    save_rgb_png(job.rgb, os.path.join(job.output_dir, f"{prefix}rgb.png"))

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
                                     coords=job.coords, image_shape=job.rgb.shape)

    elif result.analyzer == "spectral":
        ndvi = result.outputs.get("indices", {}).get("NDVI")
        if ndvi is not None:
            save_ndvi_colormap(ndvi, os.path.join(out, f"{prefix}ndvi.png"))
        lc = result.outputs.get("land_cover")
        if lc is not None:
            save_geotiff(lc, os.path.join(out, f"{prefix}land_cover.tif"), coords=job.coords)

    elif result.analyzer == "object_detection":
        regions = result.outputs.get("regions", [])
        if regions:
            save_regions_geojson(regions, os.path.join(out, f"{prefix}detections.geojson"),
                                 coords=job.coords, image_shape=job.rgb.shape)
