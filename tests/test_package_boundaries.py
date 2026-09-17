"""Regression checks for installed-library and training boundaries."""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_shared_capabilities_do_not_import_training():
    forbidden = []
    for path in (ROOT / "imint").rglob("*.py"):
        if "training" in path.relative_to(ROOT / "imint").parts:
            continue
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                if node.level:
                    from importlib.util import resolve_name
                    prefix = resolve_name("." * node.level + prefix, package)
                names = [prefix] + [prefix + "." + item.name for item in node.names]
            if any(name == "imint.training" or name.startswith("imint.training.") for name in names):
                forbidden.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert forbidden == [], forbidden


def test_data_and_metrics_import_without_model_or_training_stack():
    code = r"""
import importlib.abc
import sys
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith(('torch.', 'imint.training')):
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, BlockTraining())
import imint
assert 'imint.engine' not in sys.modules
from imint.fetch import GeoContext
from imint.data.optimal_fetch import optimal_fetch_dates
from imint.data.cdse_s2 import fetch_s2_scene
from imint.schema.unified_schema import NUM_UNIFIED_CLASSES
from imint.metrics import compute_miou
import numpy as np
r = compute_miou(np.array([1, 1, 2]), np.array([1, 2, 2]), 3)
assert r['miou'] == 0.5
assert r['overall_accuracy'] == 0.6667
assert NUM_UNIFIED_CLASSES > 0
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_training_and_evaluation_use_identical_metric_function():
    from imint.metrics import compute_miou
    from imint.training.evaluate import compute_miou as training_miou
    from imint.eval.metrics import per_class_iou
    import numpy as np
    assert training_miou is compute_miou
    prediction = np.array([0, 1, 2, 2])
    target = np.array([0, 1, 1, 2])
    a = compute_miou(prediction, target, 3)
    b = per_class_iou(prediction, target, 3)
    assert a['miou'] == b['miou']
    np.testing.assert_array_equal(a['confusion_matrix'], b['confusion_matrix'])


def test_training_manifests_request_training_dependencies():
    for path in (ROOT / "k8s").rglob("*.yaml"):
        for line in path.read_text().splitlines():
            if "pip install" in line and "-e ." in line:
                assert "--no-deps" in line, f"{path}: base-only installation omits training dependencies"
