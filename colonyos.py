"""
executors/colonyos.py — ColonyOS executor

Wraps the IMINT engine for use inside a ColonyOS container job.
This replaces the original my_cloud_filtering/main.py logic.

The ColonyOS job spec (get_cloud_free.json) calls:
    python3 executors/colonyos.py

Environment variables are set by ColonyOS from the job spec:
    DATE, WEST, SOUTH, EAST, NORTH, OUTPUTS_DIR
"""

import os
import numpy as np

from executors.base import BaseExecutor
from imint.job import IMINTJob, IMINTResult


class ColonyOSExecutor(BaseExecutor):
    """
    Executor for ColonyOS container jobs.

    Reads job parameters from environment variables (set by ColonyOS)
    and writes results back to the ColonyOS filesystem (/cfs/outputs/).
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

        coords = {"west": west, "south": south, "east": east, "north": north}

        rgb, bands, is_cloudy = self._fetch_and_check(date, coords)

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

    def _fetch_and_check(self, date: str, coords: dict):
        """Fetch Sentinel-2 data and check for clouds using the original modules."""
        from my_cloud_filtering.get_data import get_data
        from my_cloud_filtering.main import pred_cloudy
        from my_cloud_filtering.utils import to_rgb

        print(f"[ColonyOSExecutor] Fetching data for {date}...")
        bands = get_data(date, coords)
        rgb = to_rgb(bands)

        cloudy, _ = pred_cloudy(rgb)
        return rgb, bands, cloudy

    def handle_result(self, result: IMINTResult) -> None:
        if result.success:
            print(f"[ColonyOSExecutor] ✓ Complete — outputs at {result.summary_path}")
        else:
            print(f"[ColonyOSExecutor] ✗ Failed: {result.error}")
            raise RuntimeError(result.error)


if __name__ == "__main__":
    executor = ColonyOSExecutor()
    executor.execute()
