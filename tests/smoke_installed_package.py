"""Run with python -I after installing a built wheel in a clean environment.

With --serving, also import the LCLU service modules; install the wheel with
its [fetch] extra first. Neither mode may load torch or imint.training, and
--serving fails if any module named scripts loads.
"""
import argparse
import hashlib
import importlib.abc
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile


class NoTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith(("torch.", "imint.training")):
            raise AssertionError("Lightweight import loaded " + fullname)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--serving", action="store_true",
                    help="also import the LCLU serving modules (needs the [fetch] extra)")
args = parser.parse_args()

sys.meta_path.insert(0, NoTraining())
import imint
import numpy as np
from imint.fetch import GeoContext
from imint.data.cdse_s2 import fetch_s2_scene
from imint.data.optimal_fetch import optimal_fetch_dates
from imint.schema.unified_schema import NUM_UNIFIED_CLASSES
from imint.metrics import compute_miou
from imint.analyzers.cot_l1c import load_ensemble_l1c

repo = Path(__file__).resolve().parents[1]
package = Path(imint.__file__).resolve().parent
assert repo not in package.parents, package
assert "imint.engine" not in sys.modules
assert compute_miou(np.array([1, 1, 2]), np.array([1, 2, 2]), 3)["miou"] == 0.5
assert NUM_UNIFIED_CLASSES > 0
assert len(load_ensemble_l1c()) == 10
assert len(list((package / "fm/cot_models").glob("*.pt"))) == 3
assert (package / "config/analyzers.yaml").read_bytes() == (repo / "config/analyzers.yaml").read_bytes()
files = importlib.metadata.files("imint-engine")
assert any(str(p).endswith("licenses/THIRD_PARTY_LICENSES.md") for p in files)
report = {"installed_package": str(package), "lightweight_imports": "pass",
          "packaged_models_config_and_notices": "pass"}

if args.serving:
    # The LCLU service imports these; scripts/ is not part of the wheel.
    from imint.inference import lclu, lclu_cog, lclu_sources, lclu_zarr  # noqa: F401
    from imint.sealed_io import atomic_write, file_identity

    with tempfile.TemporaryDirectory() as folder:
        probe = Path(folder) / "probe.bin"
        atomic_write(probe, lambda stream: stream.write(b"imint"))
        assert file_identity(probe)["sha256"] == hashlib.sha256(b"imint").hexdigest()
    assert lclu.vpp_year_for(2025) == 2024 and lclu.legend()
    loaded = sorted(m for m in sys.modules if m.split(".")[0] == "scripts")
    assert not loaded, loaded
    report["serving_imports"] = "pass"

print(json.dumps(report))
