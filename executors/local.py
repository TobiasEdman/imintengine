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
        cloud_threshold: float = 0.3,
    ):
        self.output_dir = output_dir
        self.config_path = config_path
        self.cloud_threshold = cloud_threshold

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
        cloud detection using SCL. Falls back to a synthetic image with a
        warning (useful for analyzer development without DES access).
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
        Attempt to fetch real Sentinel-2 data from DES and run SCL cloud detection.
        Falls back to synthetic data if openeo is not installed or DES is not configured.
        """
        try:
            from imint.fetch import fetch_des_data, FetchError

            print(f"[LocalExecutor] Fetching Sentinel-2 data for {date}...")
            result = fetch_des_data(
                date=date,
                coords=coords,
                cloud_threshold=self.cloud_threshold,
            )

            if result.cloud_fraction > self.cloud_threshold:
                print(
                    f"[LocalExecutor] Cloud fraction {result.cloud_fraction:.1%} "
                    f"> threshold {self.cloud_threshold:.0%} — skipping {date}"
                )
                return None, None

            print(
                f"[LocalExecutor] Fetched {len(result.bands)} bands, "
                f"cloud fraction {result.cloud_fraction:.1%}"
            )
            return result.rgb, result.bands

        except ImportError:
            print("[LocalExecutor] WARNING: openeo not installed.")
            print("  Generating synthetic image for analyzer development.")
            print("  To use real data: pip install openeo")
            return self._synthetic_image()

        except Exception as e:
            print(f"[LocalExecutor] WARNING: DES fetch failed: {e}")
            print("  Generating synthetic image for analyzer development.")
            return self._synthetic_image()

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
    parser.add_argument("--cloud-threshold", type=float, default=0.3,
                        help="Max cloud fraction (0.0-1.0, default: 0.3)")
    args = parser.parse_args()

    coords = {"west": args.west, "south": args.south, "east": args.east, "north": args.north}
    rgb = np.load(args.rgb) if args.rgb else None

    executor = LocalExecutor(
        output_dir=args.output_dir,
        config_path=args.config,
        cloud_threshold=args.cloud_threshold,
    )
    executor.execute(date=args.date, coords=coords, rgb=rgb)


if __name__ == "__main__":
    main()
