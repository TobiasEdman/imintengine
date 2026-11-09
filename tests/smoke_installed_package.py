"""Run with python -I after installing a built wheel in a clean environment.

With --eval, also run the field-truth evaluation modules' runtime checks;
install pandas first. Neither mode may load torch or imint.training.
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
parser.add_argument("--eval", action="store_true",
                    help="also run the evaluation modules' runtime checks (needs pandas)")
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

if args.eval:
    # The IM-023 consumers authenticate their sealed runtime through these;
    # scripts/ is not part of the wheel. Importing fieldtruth resolves its
    # runtime-check imports; sealed_runtime below runs nfi_amendment's.
    from imint import crop_distill_runtime as runtime
    from imint.eval import fieldtruth, nfi_amendment  # noqa: F401

    with tempfile.TemporaryDirectory() as folder:
        tree = Path(folder).resolve()  # no symlinked components (macOS /var)
        (tree / "probe.txt").write_bytes(b"imint")
        entries = runtime.snapshot_tree(tree)
        assert entries == [{"path": "probe.txt", "type": "file", "size_bytes": 5,
                            "sha256": hashlib.sha256(b"imint").hexdigest()}]
        # The encoding every sealed tree manifest is hashed with.
        assert runtime.tree_payload_sha256(entries) == (
            "f3c79c39f63e3b8f69c357fd6371e57d6702941476cb17bf6d8288af9219a67e")
        try:
            nfi_amendment.sealed_runtime(tree, tree / "runtime.json", source_git_sha="0" * 40,
                                         image_ref="probe@sha256:" + "0" * 64, environment="model")
        except runtime.ProvenanceError as error:
            assert "cannot securely read runtime manifest" in str(error), error
        else:
            raise AssertionError("sealed_runtime accepted a missing runtime manifest")
    loaded = sorted(m for m in sys.modules if m.split(".")[0] == "scripts")
    assert not loaded, loaded
    report["evaluation_runtime_checks"] = "pass"

print(json.dumps(report))
