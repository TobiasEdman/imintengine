"""
executors/base.py — Executor interface

Any job runner (ColonyOS, local, Airflow, cron, CLI) implements this.
The engine never imports from here — only executors do.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from imint.job import IMINTJob, IMINTResult


class BaseExecutor(ABC):
    """
    Contract for all executors.

    An executor is responsible for:
      1. Receiving or generating job specs (dates, coords, config)
      2. Fetching satellite data and running cloud detection
      3. Constructing an IMINTJob with rgb + bands populated
      4. Calling imint.run_job(job) and handling the result

    The engine (run_job) knows nothing about executors.
    """

    @abstractmethod
    def build_job(self, **kwargs) -> IMINTJob:
        """
        Construct an IMINTJob from executor-specific inputs.
        Must set job.rgb before returning (after cloud check).
        """
        ...

    @abstractmethod
    def handle_result(self, result: IMINTResult) -> None:
        """
        Do something with the result — push to ColonyOS FS, log, notify, etc.
        """
        ...

    def execute(self, **kwargs) -> IMINTResult:
        """Full execution cycle: build → run → handle."""
        from imint import run_job
        job = self.build_job(**kwargs)
        result = run_job(job)
        self.handle_result(result)
        return result
