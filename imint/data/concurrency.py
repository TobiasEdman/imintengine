"""Shared provider concurrency limits used by all fetch paths."""
from __future__ import annotations

import threading

class AdaptiveSemaphore:
    """Semaphore that adjusts concurrency based on success/failure rates.

    Starts at ``initial`` permits, increases by 1 (up to ``max_permits``)
    after ``ramp_up_after`` consecutive successes, decreases by 1 (down to
    ``min_permits``) on any failure or timeout.
    """

    def __init__(
        self,
        initial: int = 3,
        min_permits: int = 1,
        max_permits: int = 8,
        ramp_up_after: int = 10,
        name: str = "",
    ):
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(initial)
        self._permits = initial
        self._min = min_permits
        self._max = max_permits
        self._ramp_up_after = ramp_up_after
        self._consecutive_ok = 0
        self._name = name
        self._total_success = 0
        self._total_failure = 0

    @property
    def permits(self) -> int:
        return self._permits

    @property
    def stats(self) -> str:
        return f"{self._name}: ok={self._total_success} fail={self._total_failure} permits={self._permits}"

    def acquire(self, timeout: float | None = None) -> bool:
        return self._sem.acquire(timeout=timeout)

    def release(self) -> None:
        self._sem.release()

    def report_success(self) -> None:
        with self._lock:
            self._total_success += 1
            self._consecutive_ok += 1
            if self._consecutive_ok >= self._ramp_up_after and self._permits < self._max:
                self._permits += 1
                self._consecutive_ok = 0
                self._sem.release()  # add a permit
                print(f"    [{self._name}] ↑ concurrency → {self._permits}")

    def report_failure(self) -> None:
        with self._lock:
            self._total_failure += 1
            self._consecutive_ok = 0
            if self._permits > self._min:
                self._permits -= 1
                # consume a permit (don't release — effectively reduces slots)
                self._sem.acquire(timeout=0)
                print(f"    [{self._name}] ↓ concurrency → {self._permits}")


# DES openEO: raised 2026-05-26 to 6 concurrent slots after CDSE openEO
# became the primary source (single-flight) and DES needed to absorb the
# parallel-worker load. Race-bug fix (commit bbea8af) means a DES hang
# no longer blocks tile completion — workers time out at 180 s and
# threads are abandoned via shutdown(wait=False, cancel_futures=True).
_DES_SEMAPHORE = AdaptiveSemaphore(
    initial=6, min_permits=2, max_permits=6,
    ramp_up_after=10, name="DES",
)
# CDSE SH Process API allows 300 req/min but each 512px request takes
# ~20s. 10 concurrent = ~30 req/min, well within quota.
_CDSE_SEMAPHORE = AdaptiveSemaphore(
    initial=10, min_permits=3, max_permits=20,
    ramp_up_after=20, name="CDSE",
)
# CDSE openEO enforces a HARD per-account ceiling of 1 concurrent
# connection (verified 2026-05-26: synchronous fetches over that limit
# return `[429] max connections reached: 1` at preflight, before any
# process graph runs). Adaptive ramp-up would just bounce us repeatedly
# into 429-spam, so we lock the semaphore at single-flight. Throughput
# tradeoff: ~60-120 frames/h via this source alone — acceptable because
# (a) the SH PU pool is exhausted and (b) DES openEO can race in
# parallel as opportunistic secondary.
_CDSE_OPENEO_SEMAPHORE = AdaptiveSemaphore(
    initial=1, min_permits=1, max_permits=1,
    ramp_up_after=10, name="CDSE-OPENEO",
)
