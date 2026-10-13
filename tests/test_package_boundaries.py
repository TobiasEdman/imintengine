"""Regression checks for installed-library and training boundaries."""
import ast
import shlex
import re

import yaml
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


def _shell_commands(script):
    """Split static shell commands without treating quoted operators as separators."""
    command = []
    quote = None
    i = 0
    while i < len(script):
        char = script[i]
        if char == "\\" and quote != "'" and i + 1 < len(script):
            following = script[i + 1]
            if following != "\n":
                command.extend((char, following))
            i += 2
            continue
        if quote:
            command.append(char)
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            command.append(char)
        elif char == "#" and (not command or command[-1].isspace()):
            # A comment ends at the physical newline, even after a backslash.
            i = script.find("\n", i)
            if i == -1:
                break
            continue
        elif char in ";&|()\n":
            if command:
                yield shlex.split("".join(command), comments=False)
                command = []
        else:
            command.append(char)
        i += 1
    if command:
        yield shlex.split("".join(command), comments=False)


def _manifest_containers(document):
    """Visit PodSpecs, including workload templates and Kubernetes Lists."""
    if not isinstance(document, dict):
        return
    for item in document.get("items", []):
        yield from _manifest_containers(item)
    spec = document.get("spec", {})
    for field in ("containers", "initContainers"):
        yield from spec.get(field, [])
    for field in ("template", "jobTemplate"):
        yield from _manifest_containers(spec.get(field))


def _container_commands(container):
    # Without command, the image ENTRYPOINT is unknown; args alone are not shell.
    argv = container.get("command", []) + container.get("args", [])
    if not container.get("command"):
        return
    if Path(argv[0]).name not in ("sh", "bash", "dash", "ksh", "zsh"):
        yield argv
        return
    i = 1
    while i < len(argv) and argv[i].startswith("-") and argv[i] != "--":
        option = argv[i]
        if option in ("-o", "+o", "-O"):
            i += 2
            continue
        if not option.startswith("--") and "c" in option[1:]:
            assert i + 1 < len(argv), "shell -c requires a script"
            for args in _shell_commands(argv[i + 1]):
                # Static shell control words and assignment prefixes do not
                # change which executable receives the remaining arguments.
                while args and (
                    args[0] in ("if", "then", "elif", "else", "do", "!")
                    or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=.*", args[0], re.DOTALL)
                ):
                    args = args[1:]
                yield args
            return  # Later argv values are shell positional parameters.
        i += 1


def test_training_manifests_request_training_dependencies():
    for path in (ROOT / "k8s").rglob("*.yaml"):
        for document in yaml.safe_load_all(path.read_text()):
            for container in _manifest_containers(document):
                for args in _container_commands(container):
                    if args[:3] in (["python", "-m", "pip"], ["python3", "-m", "pip"]):
                        args = args[2:]
                    if args[:2] not in (["pip", "install"], ["pip3", "install"]):
                        continue
                    for i, arg in enumerate(args):
                        if arg in ("-e", "--editable"):
                            requirement = args[i + 1]
                        elif arg.startswith("--editable="):
                            requirement = arg.split("=", 1)[1]
                        else:
                            continue
                        if requirement == ".":
                            extras = []
                        elif requirement.startswith(".[") and requirement.endswith("]"):
                            extras = [extra.strip() for extra in requirement[2:-1].split(",")]
                        else:
                            continue
                        assert "training" in extras or "--no-deps" in args, (
                            f"{path}: editable install {requirement!r} omits training dependencies"
                        )
