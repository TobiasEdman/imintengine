"""Run the complete cache parity gate in one clean producer process.

Other test modules legitimately import producer dependencies during collection.
A production cache must refuse those unobserved preloads. Keep the assertion
suite unchanged in a private, non-discovered module rather than weakening that
contract or installing process-wide producer observers in the pytest parent.
"""
import builtins
import importlib
import os
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_cached_validation_parity_in_clean_process(tmp_path):
    # Regression: the parent may already contain this module from unrelated
    # tests, exactly as in full-suite CI collection. Never unload/reload it.
    from imint.training import tile_fetch
    hooks = (builtins.__import__, importlib._bootstrap._gcd_import)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    # Parent selection flags must not silently deselect private parity cases.
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider",
         str(ROOT / "tests/_cache_validation_cases.py"),
         "--basetemp", str(tmp_path / "parity-child")],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    # A zero exit also covers skipped or deselected cases; require that every
    # collected case ran and passed.
    collected = re.search(r"collected (\d+) items", result.stdout)
    assert collected and int(collected.group(1)) > 0, result.stdout
    summary = result.stdout.strip().splitlines()[-1]
    outcomes = {kind: int(n) for n, kind in re.findall(r"(\d+) (\w+)", summary)
                if not kind.startswith("warning")}
    assert outcomes == {"passed": int(collected.group(1))}, summary
    assert sys.modules["imint.training.tile_fetch"] is tile_fetch
    assert hooks == (builtins.__import__, importlib._bootstrap._gcd_import)
