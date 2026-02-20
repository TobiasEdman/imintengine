"""
executors/local.py — Local executor

Runs the IMINT engine directly on your laptop.
No ColonyOS, no Docker, no network required (beyond DES for data fetch).

Usage:
    python executors/local.py --date 2022-06-15 --west 14.5 --south 56.0 --east 15.5 --north 57.0

Or from Python / a notebook:
    from executors.local import LocalExecutor
    executor = LocalExecutor()
    result = executor.execute(date="2022-06-15", coords={...})
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from executors.base import BaseExecutor
from imint.job import IMINTJob, IMINTResult


class LocalExecutor(BaseExecutor):
    """
    Runs a single IMINT job locally.

    Data fetching and cloud detection are optional — if you pass rgb directly
    (e.g. from a numpy file or a notebook) it skips straight to analysis.
    This makes local development fast: no DES account needed to test analyzers.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        config_path: str = "config/analyzers.yaml",
    ):
        self.output_dir = output_dir
        self.config_path = config_path

    def build_job(
        self,
        date: str,
        coords: dict,
        rgb: np.ndarray | None = None,
        bands: dict | None = None,
        job_id: str | None = None,
    ) -> IMINTJob:
        """
        Build an IMINTJob.

        If rgb is not provided, tries to fetch from DES via openEO + run
        cloud detection. Falls back to a synthetic image with a warning
        (useful for analyzer development without DES access).
        """
        if rgb is not None:
            print(f"[LocalExecutor] Using provided RGB array {rgb.shape}")
        else:
            rgb, bands = self._fetch_and_check(date, coords)

        return IMINTJob(
            date=date,
            coords=coords,
            rgb=rgb,
            bands=bands or {},
            output_dir=os.path.join(self.output_dir, date),
            config_path=self.config_path,
            job_id=job_id or f"local_{date}",
        )

    def _fetch_and_check(self, date: str, coords: dict):
        """
        Attempt to fetch real Sentinel-2 data from DES and run cloud detection.
        Falls back to synthetic data if DES is not configured.
        """
        try:
            # Import the original ai-pipelines-poc modules if available
            # These live in my_cloud_filtering/ in the original repo
            from my_cloud_filtering.get_data import get_data
            from my_cloud_filtering.main import pred_cloudy

            print(f"[LocalExecutor] Fetching Sentinel-2 data for {date}...")
            bands_raw = get_data(date, coords)
            rgb = self._bands_to_rgb(bands_raw)

            if pred_cloudy(rgb):
                print(f"[LocalExecutor] Image is cloudy — skipping {date}")
                return None, None

            return rgb, bands_raw

        except ImportError:
            print("[LocalExecutor] WARNING: my_cloud_filtering not found.")
            print("  Generating synthetic image for analyzer development.")
            print("  To use real data, add my_cloud_filtering/ to this repo or PYTHONPATH.")
            return self._synthetic_image()

    def _bands_to_rgb(self, bands: dict) -> np.ndarray:
        """Convert band dict to normalized RGB (B04=R, B03=G, B02=B)."""
        r = bands.get("B04", np.zeros((256, 256)))
        g = bands.get("B03", np.zeros((256, 256)))
        b = bands.get("B02", np.zeros((256, 256)))
        rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
        # Normalize to [0, 1] using the 2nd/98th percentile
        p2, p98 = np.percentile(rgb, [2, 98])
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
        return rgb

    def _synthetic_image(self):
        """Synthetic image for local dev — no DES needed."""
        h, w = 256, 256
        rgb = np.random.rand(h, w, 3).astype(np.float32)
        bands = {
            "B02": np.random.rand(h, w).astype(np.float32),
            "B03": np.random.rand(h, w).astype(np.float32),
            "B04": np.random.rand(h, w).astype(np.float32),
            "B08": (np.random.rand(h, w) + 0.3).astype(np.float32),
            "B11": np.random.rand(h, w).astype(np.float32),
        }
        return rgb, bands

    def handle_result(self, result: IMINTResult) -> None:
        if result.success:
            print(f"[LocalExecutor] ✓ Job {result.job_id} complete → {result.summary_path}")
        else:
            print(f"[LocalExecutor] ✗ Job {result.job_id} failed: {result.error}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run IMINT engine locally")
    parser.add_argument("--date", required=True, help="ISO date, e.g. 2022-06-15")
    parser.add_argument("--west",  type=float, required=True)
    parser.add_argument("--south", type=float, required=True)
    parser.add_argument("--east",  type=float, required=True)
    parser.add_argument("--north", type=float, required=True)
    parser.add_argument("--rgb",   help="Path to .npy RGB array (skip data fetch)")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--config", default="config/analyzers.yaml")
    args = parser.parse_args()

    coords = {"west": args.west, "south": args.south, "east": args.east, "north": args.north}
    rgb = np.load(args.rgb) if args.rgb else None

    executor = LocalExecutor(output_dir=args.output_dir, config_path=args.config)
    executor.execute(date=args.date, coords=coords, rgb=rgb)


if __name__ == "__main__":
    main()
