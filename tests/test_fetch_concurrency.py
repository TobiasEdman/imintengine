"""Tests for global API concurrency control in tile_fetch.py."""
import threading
import time

import numpy as np
import pytest

from imint.training.tile_fetch import _DES_SEMAPHORE, _CDSE_SEMAPHORE


class TestGlobalSemaphores:

    def test_des_semaphore_exists(self):
        assert isinstance(_DES_SEMAPHORE, threading.Semaphore)

    def test_cdse_semaphore_exists(self):
        assert isinstance(_CDSE_SEMAPHORE, threading.Semaphore)

    def test_des_max_3_concurrent(self):
        """At most 3 threads can hold the DES semaphore simultaneously."""
        max_concurrent = 0
        current = 0
        lock = threading.Lock()

        def worker():
            nonlocal max_concurrent, current
            _DES_SEMAPHORE.acquire()
            try:
                with lock:
                    current += 1
                    max_concurrent = max(max_concurrent, current)
                time.sleep(0.05)
            finally:
                with lock:
                    current -= 1
                _DES_SEMAPHORE.release()

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_concurrent == 3, f"Expected max 3 concurrent, got {max_concurrent}"

    def test_cdse_max_1_concurrent(self):
        """At most 1 thread can hold the CDSE semaphore simultaneously."""
        max_concurrent = 0
        current = 0
        lock = threading.Lock()

        def worker():
            nonlocal max_concurrent, current
            _CDSE_SEMAPHORE.acquire()
            try:
                with lock:
                    current += 1
                    max_concurrent = max(max_concurrent, current)
                time.sleep(0.05)
            finally:
                with lock:
                    current -= 1
                _CDSE_SEMAPHORE.release()

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_concurrent == 1, f"Expected max 1 concurrent, got {max_concurrent}"

    def test_des_and_cdse_independent(self):
        """DES and CDSE semaphores don't block each other."""
        des_held = threading.Event()
        cdse_held = threading.Event()
        both_held = threading.Event()

        def hold_des():
            _DES_SEMAPHORE.acquire()
            try:
                des_held.set()
                both_held.wait(timeout=2)
            finally:
                _DES_SEMAPHORE.release()

        def hold_cdse():
            _CDSE_SEMAPHORE.acquire()
            try:
                cdse_held.set()
                both_held.wait(timeout=2)
            finally:
                _CDSE_SEMAPHORE.release()

        t1 = threading.Thread(target=hold_des)
        t2 = threading.Thread(target=hold_cdse)
        t1.start()
        t2.start()

        # Both should be acquired within 1 second (they're independent)
        assert des_held.wait(timeout=1), "DES semaphore should be acquirable"
        assert cdse_held.wait(timeout=1), "CDSE should not be blocked by DES"
        both_held.set()
        t1.join()
        t2.join()

    def test_semaphores_release_on_exception(self):
        """Semaphores must be released even if the worker raises."""
        # Acquire all 3 DES slots
        for _ in range(3):
            _DES_SEMAPHORE.acquire()

        # Release them (simulating normal cleanup)
        for _ in range(3):
            _DES_SEMAPHORE.release()

        # Verify we can acquire again (not leaked)
        acquired = _DES_SEMAPHORE.acquire(timeout=0.5)
        assert acquired, "Semaphore should be available after release"
        _DES_SEMAPHORE.release()

    def test_total_concurrent_4(self):
        """Total concurrent API calls should be max 4 (3 DES + 1 CDSE)."""
        max_total = 0
        current = 0
        lock = threading.Lock()

        def des_worker():
            nonlocal max_total, current
            _DES_SEMAPHORE.acquire()
            try:
                with lock:
                    current += 1
                    max_total = max(max_total, current)
                time.sleep(0.05)
            finally:
                with lock:
                    current -= 1
                _DES_SEMAPHORE.release()

        def cdse_worker():
            nonlocal max_total, current
            _CDSE_SEMAPHORE.acquire()
            try:
                with lock:
                    current += 1
                    max_total = max(max_total, current)
                time.sleep(0.05)
            finally:
                with lock:
                    current -= 1
                _CDSE_SEMAPHORE.release()

        # Simulate 3 tiles each submitting 3 DES + 1 CDSE
        threads = []
        for _ in range(3):
            for _ in range(3):
                threads.append(threading.Thread(target=des_worker))
            threads.append(threading.Thread(target=cdse_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_total <= 4, f"Expected max 4 total concurrent, got {max_total}"
