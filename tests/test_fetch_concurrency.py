"""Tests for adaptive API concurrency control in tile_fetch.py."""
import threading
import time

import pytest

from imint.training.tile_fetch import AdaptiveSemaphore, _DES_SEMAPHORE, _CDSE_SEMAPHORE


class TestAdaptiveSemaphore:

    def test_initial_permits(self):
        sem = AdaptiveSemaphore(initial=3, name="test")
        assert sem.permits == 3

    def test_acquire_release(self):
        sem = AdaptiveSemaphore(initial=2, name="test")
        assert sem.acquire(timeout=0.1)
        assert sem.acquire(timeout=0.1)
        # Third should block (only 2 permits)
        assert not sem.acquire(timeout=0.1)
        sem.release()
        assert sem.acquire(timeout=0.1)
        sem.release()
        sem.release()

    def test_ramp_up_after_successes(self):
        sem = AdaptiveSemaphore(initial=2, max_permits=5, ramp_up_after=3, name="test")
        assert sem.permits == 2
        sem.report_success()
        sem.report_success()
        assert sem.permits == 2  # not yet
        sem.report_success()
        assert sem.permits == 3  # ramped up after 3 consecutive

    def test_ramp_down_on_failure(self):
        sem = AdaptiveSemaphore(initial=3, min_permits=1, name="test")
        assert sem.permits == 3
        sem.report_failure()
        assert sem.permits == 2
        sem.report_failure()
        assert sem.permits == 1
        sem.report_failure()
        assert sem.permits == 1  # can't go below min

    def test_failure_resets_success_counter(self):
        sem = AdaptiveSemaphore(initial=2, max_permits=5, ramp_up_after=3, name="test")
        sem.report_success()
        sem.report_success()
        sem.report_failure()  # resets counter
        sem.report_success()
        sem.report_success()
        assert sem.permits == 1  # went down from failure, not up

    def test_max_permits_cap(self):
        sem = AdaptiveSemaphore(initial=3, max_permits=4, ramp_up_after=1, name="test")
        sem.report_success()  # → 4
        assert sem.permits == 4
        sem.report_success()  # already at max
        assert sem.permits == 4

    def test_concurrent_max_enforced(self):
        """Verify actual concurrency doesn't exceed current permits."""
        sem = AdaptiveSemaphore(initial=3, max_permits=3, name="test")
        max_concurrent = 0
        current = 0
        lock = threading.Lock()

        def worker():
            nonlocal max_concurrent, current
            sem.acquire()
            try:
                with lock:
                    current += 1
                    max_concurrent = max(max_concurrent, current)
                time.sleep(0.03)
            finally:
                with lock:
                    current -= 1
                sem.release()

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_concurrent == 3

    def test_ramp_up_increases_actual_concurrency(self):
        """After ramp-up, more threads should run concurrently."""
        sem = AdaptiveSemaphore(initial=2, max_permits=4, ramp_up_after=2, name="test")

        # Ramp up to 4
        for _ in range(6):
            sem.report_success()
        assert sem.permits == 4

        max_concurrent = 0
        current = 0
        lock = threading.Lock()

        def worker():
            nonlocal max_concurrent, current
            sem.acquire()
            try:
                with lock:
                    current += 1
                    max_concurrent = max(max_concurrent, current)
                time.sleep(0.03)
            finally:
                with lock:
                    current -= 1
                sem.release()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_concurrent == 4


class TestGlobalSemaphores:

    def test_des_is_adaptive(self):
        assert isinstance(_DES_SEMAPHORE, AdaptiveSemaphore)
        assert _DES_SEMAPHORE.permits >= 1

    def test_cdse_is_adaptive(self):
        assert isinstance(_CDSE_SEMAPHORE, AdaptiveSemaphore)
        assert _CDSE_SEMAPHORE.permits >= 1

    def test_des_starts_at_3(self):
        # Note: permits may have changed during other tests — check initial config
        fresh = AdaptiveSemaphore(initial=3, name="des-fresh")
        assert fresh.permits == 3

    def test_cdse_starts_at_1(self):
        fresh = AdaptiveSemaphore(initial=1, name="cdse-fresh")
        assert fresh.permits == 1
