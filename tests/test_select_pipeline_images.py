"""The image workflow must report one stable check on every PR."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / ".github/scripts/select_pipeline_images.py"
WORKFLOW = REPO / ".github/workflows/build-pipeline-images.yml"
ALL = ["cloud-models", "c2rcc-snap", "ladder-crop-distill"]


def _select(tmp_path, event, *, changed=None, ref="refs/pull/1/merge", requested=""):
    args = [sys.executable, str(SCRIPT), "--event", event, "--ref", ref,
            "--requested", requested]
    if changed is not None:
        listing = tmp_path / "changed.txt"
        listing.write_text("".join(f"{path}\n" for path in changed))
        args += ["--changed-files", str(listing)]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    values = dict(line.split("=", 1) for line in out.splitlines())
    assert set(values) == {"matrix", "build"}
    names = [entry["image"] for entry in json.loads(values["matrix"])["include"]]
    assert values["build"] == ("true" if names else "false")
    return names


@pytest.mark.parametrize("changed,expected", [
    ([], []),
    (["README.md", "scripts/infer_tiles.py", "k8s/api-deployment.yaml"], []),
    (["docker/era5-smoke/Dockerfile", "docker/sen2cor/build.sh"], []),
    (["docker/c2rcc-snap/graph.xml"], ["c2rcc-snap"]),
    (["docker/cloud-models/Dockerfile", "docker/ladder-crop-distill/x.lock"],
     ["cloud-models", "ladder-crop-distill"]),
    (["docker/cloud-models-old/Dockerfile"], []),
    ([".github/workflows/build-pipeline-images.yml"], ALL),
    ([".github/scripts/select_pipeline_images.py"], ALL),
    ([".github/workflows/test.yml"], []),
])
def test_pull_request_builds_only_images_whose_paths_changed(tmp_path, changed, expected):
    assert _select(tmp_path, "pull_request", changed=changed) == expected


def test_pull_request_requires_the_changed_file_list(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        _select(tmp_path, "pull_request")


@pytest.mark.parametrize("event,ref,requested,expected", [
    ("push", "refs/heads/main", "", ALL),
    ("workflow_dispatch", "refs/heads/main", "", ALL),
    ("workflow_dispatch", "refs/heads/feature", "", ["ladder-crop-distill"]),
    ("workflow_dispatch", "refs/heads/feature", "all", ALL),
    ("workflow_dispatch", "refs/heads/main", "ladder-crop-distill", ["ladder-crop-distill"]),
])
def test_non_pr_events_keep_existing_selection(tmp_path, event, ref, requested, expected):
    assert _select(tmp_path, event, ref=ref, requested=requested) == expected


def test_selector_matrix_matches_dockerfiles():
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("select_pipeline_images", SCRIPT)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, entry in module.IMAGES.items():
        assert entry["dockerfile"] == f"docker/{name}/Dockerfile"
        assert (REPO / entry["dockerfile"]).is_file()


def test_build_summary_is_a_stable_check_on_every_pull_request():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    triggers = workflow[True]  # YAML 1.1 reads the bare key `on` as True.
    assert triggers["pull_request"] is None or "paths" not in triggers["pull_request"]
    summary = workflow["jobs"]["summary"]
    assert summary["name"] == "Build summary"
    assert "strategy" not in summary and summary["if"] == "always()"
    assert set(summary["needs"]) == {"select_images", "pr_build", "publish"}
    assert "needs.select_images.outputs.build == 'true'" in workflow["jobs"]["pr_build"]["if"]
