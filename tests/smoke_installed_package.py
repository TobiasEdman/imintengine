"""Run with python -I after installing a built wheel in a clean environment."""
import importlib.abc
import importlib.metadata
import json
from pathlib import Path
import sys


class NoTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith(("torch.", "imint.training")):
            raise AssertionError("Lightweight import loaded " + fullname)


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
print(json.dumps({"installed_package": str(package), "lightweight_imports": "pass",
                  "packaged_models_config_and_notices": "pass"}))
