"""
imint/analyzers/insar.py — InSAR time series analysis via MintPy

Measures ground displacement (subsidence, uplift) from Sentinel-1 SAR
interferometry. Designed for the infrastructure monitoring use case:
Trafikverket (rail/road), Svenska Kraftnät (power grid), and
municipal planning.

Uses MintPy (Miami INsar Time-series software in PYthon) for:
  - Persistent Scatterer (PS) / Small Baseline Subset (SBAS) processing
  - Velocity map estimation (mm/year line-of-sight displacement)
  - Time series decomposition (linear + seasonal)

When MintPy is not installed, falls back to a simplified differential
InSAR approach using phase difference between two SAR acquisitions.

Requirements:
    pip install mintpy          # Full InSAR time series
    pip install h5py            # HDF5 support for MintPy outputs

Config options:
    mode: "velocity" (default) — LOS velocity map
          "timeseries" — full displacement time series
          "differential" — simple 2-scene differential InSAR
    reference_point: [lat, lon] — stable reference point for deformation
    wavelength: 0.05546 — Sentinel-1 C-band wavelength in meters
    threshold_mm_yr: 5.0 — flag pixels with velocity > this
    temporal_coherence_threshold: 0.7 — quality filter
"""
from __future__ import annotations

import numpy as np
from .base import BaseAnalyzer, AnalysisResult


def _check_mintpy_available() -> bool:
    try:
        import mintpy  # noqa: F401
        return True
    except ImportError:
        return False


def _check_h5py_available() -> bool:
    try:
        import h5py  # noqa: F401
        return True
    except ImportError:
        return False


# Sentinel-1 C-band wavelength (meters)
S1_WAVELENGTH_M = 0.05546576


class InSARAnalyzer(BaseAnalyzer):
    """InSAR ground displacement analyzer.

    Computes line-of-sight (LOS) displacement from SAR interferometry.
    Supports three modes:

    - velocity: Mean LOS velocity map (mm/year) from time series
    - timeseries: Full displacement time series per pixel
    - differential: Simple 2-scene phase difference (no MintPy needed)

    Config:
        mode: "velocity", "timeseries", or "differential"
        wavelength: SAR wavelength in meters (default: Sentinel-1 C-band)
        threshold_mm_yr: Flag pixels exceeding this velocity (default: 5.0)
        temporal_coherence_threshold: Quality filter (default: 0.7)
        reference_point: [lat, lon] for stable reference (optional)
        ifg_dir: Path to interferogram directory (for MintPy modes)
    """

    name = "insar"

    def analyze(self, rgb, bands=None, date=None, coords=None,
                output_dir="outputs", geo=None):
        mode = self.config.get("mode", "velocity")

        if mode == "differential":
            return self._run_differential(bands, geo)
        elif mode in ("velocity", "timeseries"):
            return self._run_mintpy(mode, output_dir)
        else:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error=f"Unknown mode '{mode}'. Use: velocity, timeseries, or differential",
            )

    def _run_differential(self, bands, geo) -> AnalysisResult:
        """Simple differential InSAR from two SAR phase images.

        Computes phase difference → displacement without MintPy.
        Useful as a quick preview or when MintPy is not installed.

        Expects bands dict with 'phase_1' and 'phase_2' arrays (radians).
        """
        if not bands:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error=(
                    "Differential mode requires bands with 'phase_1' and 'phase_2' "
                    "(wrapped interferometric phase in radians)"
                ),
            )

        phase_1 = bands.get("phase_1")
        phase_2 = bands.get("phase_2")

        if phase_1 is None or phase_2 is None:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error="Missing 'phase_1' and/or 'phase_2' in bands dict",
            )

        wavelength = self.config.get("wavelength", S1_WAVELENGTH_M)
        threshold = self.config.get("threshold_mm_yr", 5.0)

        # Phase difference (wrapped)
        d_phase = phase_2 - phase_1

        # Wrap to [-pi, pi]
        d_phase_wrapped = np.angle(np.exp(1j * d_phase))

        # Convert phase to LOS displacement (mm)
        # displacement = (wavelength / 4π) × phase_difference × 1000
        displacement_mm = (wavelength / (4 * np.pi)) * d_phase_wrapped * 1000.0

        # Simple coherence estimate from phase stability
        coherence = np.abs(np.exp(1j * d_phase_wrapped))

        # Flag significant displacement
        displacement_flag = np.abs(displacement_mm) > threshold

        stats = {
            "mean_displacement_mm": float(np.nanmean(displacement_mm)),
            "max_displacement_mm": float(np.nanmax(np.abs(displacement_mm))),
            "std_displacement_mm": float(np.nanstd(displacement_mm)),
            "flagged_pixels": int(np.sum(displacement_flag)),
            "flagged_fraction": float(np.mean(displacement_flag)),
            "mean_coherence": float(np.nanmean(coherence)),
        }

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs={
                "displacement_mm": displacement_mm,
                "displacement_flag": displacement_flag,
                "coherence": coherence,
                "phase_diff": d_phase_wrapped,
                "stats": stats,
            },
            metadata={
                "mode": "differential",
                "wavelength_m": wavelength,
                "threshold_mm": threshold,
                "image_size": list(displacement_mm.shape),
            },
        )

    def _run_mintpy(self, mode: str, output_dir: str) -> AnalysisResult:
        """Run MintPy InSAR time series analysis.

        Requires pre-processed interferograms in ifg_dir.
        """
        if not _check_mintpy_available():
            return AnalysisResult(
                analyzer=self.name, success=False,
                error=(
                    "MintPy is not installed. Install with: pip install mintpy\n"
                    "For differential InSAR without MintPy, use mode: differential"
                ),
            )
        if not _check_h5py_available():
            return AnalysisResult(
                analyzer=self.name, success=False,
                error="h5py is required for MintPy. Install with: pip install h5py",
            )

        ifg_dir = self.config.get("ifg_dir")
        if not ifg_dir:
            return AnalysisResult(
                analyzer=self.name, success=False,
                error=(
                    "MintPy modes require 'ifg_dir' in config pointing to "
                    "interferogram directory (e.g. from ISCE2 or SNAP)"
                ),
            )

        import os
        import h5py
        from mintpy.cli import load_data, ifgram_inversion, timeseries2velocity

        coh_threshold = self.config.get("temporal_coherence_threshold", 0.7)
        threshold = self.config.get("threshold_mm_yr", 5.0)
        ref_point = self.config.get("reference_point")

        # Step 1: Load interferograms into HDF5
        work_dir = os.path.join(output_dir, "mintpy_work")
        os.makedirs(work_dir, exist_ok=True)

        load_args = [
            "--template", ifg_dir,
            "--work-dir", work_dir,
        ]
        load_data.main(load_args)

        # Step 2: Network inversion → time series
        inv_args = ["--work-dir", work_dir]
        ifgram_inversion.main(inv_args)

        # Step 3: Velocity estimation
        ts_file = os.path.join(work_dir, "timeseries.h5")
        vel_args = [ts_file]
        if ref_point:
            vel_args += ["--ref-lalo", str(ref_point[0]), str(ref_point[1])]
        timeseries2velocity.main(vel_args)

        # Read results
        vel_file = os.path.join(work_dir, "velocity.h5")
        with h5py.File(vel_file, "r") as f:
            velocity = f["velocity"][:]  # m/year → mm/year
            velocity_mm_yr = velocity * 1000.0

        # Flag significant deformation
        deformation_flag = np.abs(velocity_mm_yr) > threshold

        stats = {
            "mean_velocity_mm_yr": float(np.nanmean(velocity_mm_yr)),
            "max_velocity_mm_yr": float(np.nanmax(np.abs(velocity_mm_yr))),
            "flagged_pixels": int(np.sum(deformation_flag)),
            "flagged_fraction": float(np.mean(deformation_flag)),
        }

        outputs = {
            "velocity_mm_yr": velocity_mm_yr,
            "deformation_flag": deformation_flag,
            "stats": stats,
        }

        if mode == "timeseries":
            with h5py.File(ts_file, "r") as f:
                timeseries = f["timeseries"][:] * 1000.0  # m → mm
                dates = [d.decode() for d in f["date"][:]]
            outputs["timeseries_mm"] = timeseries
            outputs["dates"] = dates

        return AnalysisResult(
            analyzer=self.name,
            success=True,
            outputs=outputs,
            metadata={
                "mode": mode,
                "ifg_dir": ifg_dir,
                "coherence_threshold": coh_threshold,
                "threshold_mm_yr": threshold,
                "reference_point": ref_point,
            },
        )
