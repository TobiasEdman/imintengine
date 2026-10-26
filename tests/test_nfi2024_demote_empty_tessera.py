"""The campaign Tessera demotion Job, run on a temporary campaign root."""
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

JOB = Path(__file__).resolve().parents[1] / "k8s/nfi2024-demote-empty-tessera-job.yaml"
SCRIPT = textwrap.dedent(re.search(r"python - <<'PY'\n(.*?)\n\s*PY\n", JOB.read_text(), re.S).group(1))


def _tile(path, *, flag, tessera, source="geotessera-0.10.2"):
    np.savez_compressed(path, has_tessera=np.int32(flag), tessera=tessera, tessera_source=source,
                        spectral=np.arange(24 * 4, dtype=np.float32).reshape(24, 2, 2))


@pytest.fixture
def campaign(tmp_path):
    root, manifests = tmp_path / "campaign", tmp_path / "manifests"
    root.mkdir(); manifests.mkdir()
    _tile(root / "good.npz", flag=1, tessera=np.ones((128, 2, 2), np.float32))
    _tile(root / "bad.npz", flag=1, tessera=np.zeros((128, 2, 2), np.float32))
    _tile(root / "gap.npz", flag=0, tessera=np.zeros((128, 2, 2), np.float32))
    return root, manifests


def _run(root, manifests, expected=1):
    env = dict(os.environ, CAMPAIGN_DIR=str(root), MANIFEST_DIR=str(manifests), EXPECTED=str(expected))
    return subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True)


def test_demotes_exactly_the_flagged_empty_tile_and_keeps_every_other_array(campaign):
    root, manifests = campaign
    good_before = (root / "good.npz").read_bytes()
    with np.load(root / "bad.npz") as d:
        original = {k: d[k] for k in d.files}
    result = _run(root, manifests)
    assert result.returncode == 0, result.stderr
    with np.load(root / "bad.npz", allow_pickle=False) as d:
        assert int(d["has_tessera"]) == 0 and d["has_tessera"].dtype == original["has_tessera"].dtype
        assert str(d["tessera_gap_reason"]).startswith("tessera_flag_on_empty")
        for k in ("tessera", "tessera_source", "spectral"):
            assert np.array_equal(d[k], original[k])
    assert (root / "good.npz").read_bytes() == good_before
    record = json.loads((manifests / "nfi2024_empty_tessera_demoted_20261026.json").read_text())
    assert [r["tile"] for r in record["demoted"]] == ["bad"] and record["remaining"] == 0
    assert not list(root.glob("*.tmp.npz"))


@pytest.mark.parametrize("change, message", [
    ({"expected": 2}, "count differs"),
    ({"source": "dl2.geotessera.org"}, "not geotessera-0.10.2"),
    ({"record": True}, "already demoted"),
])
def test_refuses_and_changes_nothing(campaign, change, message):
    root, manifests = campaign
    if "source" in change:
        _tile(root / "bad.npz", flag=1, tessera=np.zeros((128, 2, 2), np.float32), source=change["source"])
    if change.get("record"):
        (manifests / "nfi2024_empty_tessera_demoted_20261026.json").write_text("{}")
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    result = _run(root, manifests, expected=change.get("expected", 1))
    assert result.returncode != 0 and message in result.stderr + result.stdout
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
