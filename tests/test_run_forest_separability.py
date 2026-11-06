"""run_forest_separability.sh refuses bad runs before anything reaches kubectl.

The script submits real Jobs, so it only ever runs here behind a fake
``kubectl`` first on PATH that records every call and fails loudly, with
KUBECONFIG=/dev/null and a temporary HOME. Every case asserts zero calls.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_forest_separability.sh"
MAIN = ["--model-name", "main", "--checkpoint",
        "/data/checkpoints/im023/tessera_b28/best_model.pt", "--num-classes", "29"]
FAKE_KUBECTL = """#!/bin/bash
echo "$*" >> "$FAKE_KUBECTL_LOG"
echo "FAKE KUBECTL CALLED (a test reached kubectl): $*" >&2
exit 97
"""


@pytest.fixture
def sandbox(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(FAKE_KUBECTL)
    kubectl.chmod(0o755)
    log = tmp_path / "kubectl.log"
    env = {"PATH": os.pathsep.join([str(bin_dir), str(Path(sys.executable).parent),
                                    "/usr/bin", "/bin"]),
           "HOME": str(tmp_path / "home"), "KUBECONFIG": os.devnull,
           "FAKE_KUBECTL_LOG": str(log)}
    found = subprocess.run(["/bin/bash", "-c", "command -v kubectl"], env=env,
                           capture_output=True, text=True, check=True).stdout.strip()
    assert found == str(kubectl)
    return env, log, tmp_path / "home"


def _run(env, *args):
    renders = ROOT / "outputs" / "forest-separability"
    before = set(renders.iterdir()) if renders.is_dir() else set()
    result = subprocess.run(["/bin/bash", str(SCRIPT), *args], env=env, capture_output=True,
                            text=True, timeout=120, stdin=subprocess.DEVNULL)
    after = set(renders.iterdir()) if renders.is_dir() else set()
    assert after == before, "a refused run rendered a Job"
    return result


@pytest.mark.parametrize("args", [
    ["forest5", "2022", "--model-name", "main"],            # trailing flag taken for a year
    ["forest5", "2022", "2024"],                            # mixed valid and invalid years
    ["forest5", "2022", "2022"],
    ["forest4", "2022"],
    ["forest5"],
    ["--bogus", "forest5", "2022"],
    ["--model-name", "main", "forest5", "2022"],            # partial model options
    ["--model-name", "r4", "--checkpoint", "/data/checkpoints/im023/tessera_b28/best_model.pt",
     "--num-classes", "28", "forest5", "2022"],             # R4's name on other weights
    [*MAIN[:4], "--num-classes", "30", "forest5", "2022"],
])
def test_bad_arguments_are_refused_before_kubectl(sandbox, args):
    env, log, _ = sandbox
    result = _run(env, *args)
    assert result.returncode == 2, result.stderr
    assert "FAKE KUBECTL" not in result.stderr
    assert not log.exists()


@pytest.mark.parametrize("args,tag", [(["forest5", "2022"], "forest5-2022"),
                                      ([*MAIN, "forest5", "2023", "2022"], "main-forest5-2022")])
def test_existing_evidence_is_never_overwritten(sandbox, args, tag):
    env, log, home = sandbox
    evidence = home / ".agents/continuity/evidence/ImintEngine/class-separability-0127" / tag
    evidence.mkdir(parents=True)
    (evidence / "probe.json").write_text("{}")
    result = _run(env, *args)
    assert result.returncode == 1 and "finns redan" in result.stderr, result.stderr
    assert not log.exists()
    assert (evidence / "probe.json").read_text() == "{}"
