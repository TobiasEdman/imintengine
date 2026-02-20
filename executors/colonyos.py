"""
executors/colonyos.py — ColonyOS executor

Wraps the IMINT engine for use inside a ColonyOS container job.

The ColonyOS job spec (config/colonyos_job.json) calls:
    python3 executors/colonyos.py

Environment variables are set by ColonyOS from the job spec:
    DATE, WEST, SOUTH, EAST, NORTH, DES_TOKEN, CLOUD_THRESHOLD, OUTPUTS_DIR
"""
from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from executors.base import BaseExecutor
from imint.job import IMINTJob, IMINTResult
from imint.fetch import fetch_des_data, FetchError


class ColonyOSExecutor(BaseExecutor):
    """
    Executor for ColonyOS container jobs.

    Reads job parameters from environment variables (set by ColonyOS)
    and writes results back to the ColonyOS filesystem (/cfs/outputs/).

    Authentication: The DES_TOKEN env var must be set in the ColonyOS job spec.
    """

    def build_job(self, **kwargs) -> IMINTJob:
        # ColonyOS sets these via the job spec env block
        date   = os.environ["DATE"]
        west   = float(os.environ["WEST"])
        south  = float(os.environ["SOUTH"])
        east   = float(os.environ["EAST"])
        north  = float(os.environ["NORTH"])
        output_dir = os.environ.get("OUTPUTS_DIR", "/cfs/outputs")
        job_id = os.environ.get("COLONY_PROCESS_ID", None)
        cloud_threshold = float(os.environ.get("CLOUD_THRESHOLD", "0.3"))

        coords = {"west": west, "south": south, "east": east, "north": north}

        rgb, bands, is_cloudy = self._fetch_and_check(date, coords, cloud_threshold)

        if is_cloudy or rgb is None:
            print(f"[ColonyOSExecutor] Cloudy or no data for {date} — skipping.")
            return IMINTJob(
                date=date, coords=coords, rgb=None,
                output_dir=output_dir, job_id=job_id,
            )

        return IMINTJob(
            date=date, coords=coords, rgb=rgb, bands=bands,
            output_dir=os.path.join(output_dir, date),
            config_path=os.environ.get("CONFIG_PATH", "config/analyzers.yaml"),
            job_id=job_id,
        )

    def _fetch_and_check(self, date: str, coords: dict, cloud_threshold: float = 0.3):
        """Fetch Sentinel-2 data from DES and check for clouds using SCL band."""
        print(f"[ColonyOSExecutor] Fetching data for {date}...")

        try:
            result = fetch_des_data(
                date=date,
                coords=coords,
                cloud_threshold=cloud_threshold,
            )
        except FetchError as e:
            print(f"[ColonyOSExecutor] Fetch failed: {e}")
            return None, None, False

        is_cloudy = result.cloud_fraction > cloud_threshold
        if is_cloudy:
            print(
                f"[ColonyOSExecutor] Cloud fraction {result.cloud_fraction:.1%} "
                f"> threshold {cloud_threshold:.0%} — marking as cloudy"
            )

        return result.rgb, result.bands, is_cloudy

    def handle_result(self, result: IMINTResult) -> None:
        if result.success:
            print(f"[ColonyOSExecutor] ✓ Complete — outputs at {result.summary_path}")
        else:
            print(f"[ColonyOSExecutor] ✗ Failed: {result.error}")
            raise RuntimeError(result.error)


if __name__ == "__main__":
    executor = ColonyOSExecutor()
    executor.execute()
