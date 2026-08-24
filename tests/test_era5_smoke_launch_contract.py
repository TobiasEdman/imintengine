"""Focused tests for the ERA5 smoke Job launch contract."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.create_era5_smoke_configmap import (
    BASE_GIT_SHA,
    RUNTIME_REPOSITORY,
    _current_runtime_context_sha256,
    _runtime_reference_from_manifest,
    _verify_runtime_image_parity,
    _verify_job_manifest,
    create_immutable_configmap,
    build_configmap,
)


BUNDLE_SHA256 = "b" * 64
CONFIGMAP_NAME = f"era5-smoke-code-{BUNDLE_SHA256[:12]}"
RUNTIME_IMAGE = (
    "ghcr.io/tobiasedman/imint-era5-smoke@sha256:" + "a" * 64
)


def _job(arm: str = "control") -> dict:
    suffix = BUNDLE_SHA256[:12]
    return {
        "metadata": {"name": f"era5-p600m-{arm}-{suffix}"},
        "spec": {
            "template": {
                "metadata": {"labels": {"arm": arm}},
                "spec": {
                    "imagePullSecrets": [{"name": "ghcr-push"}],
                    "containers": [{
                        "name": "trainer",
                        "image": RUNTIME_IMAGE,
                        "command": ["/bin/bash", "/patches/run_arm.sh"],
                        "env": [
                            {"name": "ARM", "value": arm},
                            {
                                "name": "RUN_ID",
                                "value": f"era5-p600m-20260821-{suffix}",
                            },
                            {"name": "BASE_GIT_SHA", "value": BASE_GIT_SHA},
                            {"name": "CONTAINER_IMAGE", "value": RUNTIME_IMAGE},
                        ],
                        "volumeMounts": [
                            {"name": "cephfs", "mountPath": "/cephfs", "readOnly": True},
                            {"name": "checkpoints", "mountPath": "/checkpoints"},
                            {"name": "training-data", "mountPath": "/data", "readOnly": True},
                            {"name": "patches", "mountPath": "/patches", "readOnly": True},
                        ],
                    }],
                    "volumes": [
                        {"name": "cephfs", "persistentVolumeClaim": {"claimName": "training-data-cephfs"}},
                        {"name": "checkpoints", "persistentVolumeClaim": {"claimName": "training-checkpoints"}},
                        {"name": "training-data", "persistentVolumeClaim": {"claimName": "training-data"}},
                        {"name": "patches", "configMap": {"name": CONFIGMAP_NAME}},
                    ],
                },
            },
        },
    }


def _validate(job: dict, arm: str = "control") -> None:
    _verify_job_manifest(
        job,
        arm=arm,
        source="synthetic.yaml",
        configmap_name=CONFIGMAP_NAME,
        bundle_sha256=BUNDLE_SHA256,
    )


def _set_env(job: dict, name: str, value: str) -> None:
    environment = job["spec"]["template"]["spec"]["containers"][0]["env"]
    next(item for item in environment if item["name"] == name)["value"] = value


def test_valid_job_satisfies_launch_contract():
    _validate(_job())
    _validate(_job("treatment"), arm="treatment")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("command", ["/bin/bash", "-lc", "/patches/run_arm.sh"], "command"),
        ("image", "python@sha256:" + "a" * 64, "digest-pinned"),
    ],
)
def test_rejects_unpinned_runtime_contract(field, value, message):
    job = _job()
    job["spec"]["template"]["spec"]["containers"][0][field] = value
    with pytest.raises(ValueError, match=message):
        _validate(job)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (
            "CONTAINER_IMAGE",
            "ghcr.io/tobiasedman/imint-era5-smoke@sha256:" + "c" * 64,
            "exactly equal",
        ),
        ("BASE_GIT_SHA", "f" * 40, "immutable BASE_GIT_SHA"),
        ("ARM", "treatment", "ARM=control"),
        ("RUN_ID", "wrong-run", "RUN_ID"),
    ],
)
def test_rejects_environment_drift(name, value, message):
    job = _job()
    _set_env(job, name, value)
    with pytest.raises(ValueError, match=message):
        _validate(job)


def test_rejects_pull_secret_or_configmap_drift():
    job = _job()
    job["spec"]["template"]["spec"]["imagePullSecrets"] = []
    with pytest.raises(ValueError, match="imagePullSecret ghcr-push"):
        _validate(job)

    job = deepcopy(_job())
    volumes = job["spec"]["template"]["spec"]["volumes"]
    next(item for item in volumes if item["name"] == "patches")[
        "configMap"
    ]["name"] = (
        "era5-smoke-code-wrong"
    )
    with pytest.raises(ValueError, match=f"ConfigMap {CONFIGMAP_NAME}"):
        _validate(job)


def test_rejects_checkpoint_or_source_mount_drift():
    job = _job()
    mounts = job["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
    next(item for item in mounts if item["name"] == "cephfs")["readOnly"] = False
    with pytest.raises(ValueError, match="cephfs must mount /cephfs read-only"):
        _validate(job)

    job = _job()
    volumes = job["spec"]["template"]["spec"]["volumes"]
    next(item for item in volumes if item["name"] == "checkpoints")[
        "persistentVolumeClaim"
    ]["claimName"] = "wrong"
    with pytest.raises(ValueError, match="checkpoints must use PVC"):
        _validate(job)


def _runtime_manifest(digest: str = "a" * 64) -> dict:
    context_digest = _current_runtime_context_sha256()
    return {
        "schema": "imint-pipeline-image-v1",
        "image": (
            f"{RUNTIME_REPOSITORY}:20260821-{context_digest[:12]}"
        ),
        "image_digest": f"sha256:{digest}",
        "build_context_sha256": context_digest,
    }


def test_runtime_manifest_binds_both_arms_to_one_digest():
    manifest = _runtime_manifest()
    expected = _runtime_reference_from_manifest(manifest)
    _verify_runtime_image_parity(
        {"control": expected, "treatment": expected}, manifest,
    )

    with pytest.raises(ValueError, match="same MANIFEST-pinned runtime"):
        _verify_runtime_image_parity(
            {
                "control": expected,
                "treatment": (
                    f"{RUNTIME_REPOSITORY}@sha256:" + "c" * 64
                ),
            },
            manifest,
        )


def test_runtime_manifest_rejects_stale_context_or_unsealed_digest():
    manifest = _runtime_manifest()
    manifest["build_context_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="build context is stale"):
        _runtime_reference_from_manifest(manifest)

    manifest = _runtime_manifest()
    manifest["image_digest"] = "TBD-after-cluster-build"
    with pytest.raises(ValueError, match="image_digest"):
        _runtime_reference_from_manifest(manifest)


def test_immutable_configmap_is_created_without_last_applied_annotation(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(
        "scripts.create_era5_smoke_configmap.subprocess.run", fake_run,
    )
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "era5-smoke-code-test"},
        "immutable": True,
        "data": {"large.py": "x" * 300_000},
    }

    create_immutable_configmap(configmap, context="icekube")

    assert calls == [(
        ["kubectl", "--context", "icekube", "create", "-f", "-"],
        {
            "input": json.dumps(configmap),
            "text": True,
            "check": True,
        },
    )]


def test_configmap_foundation_fetcher_imports_provenance_module(tmp_path):
    configmap, _ = build_configmap("test-namespace")
    patches = tmp_path / "patches"
    patches.mkdir()
    for name, contents in configmap["data"].items():
        (patches / name).write_text(contents)

    assert (patches / "era5_smoke_provenance.py").is_file()
    result = subprocess.run(
        [sys.executable, str(patches / "fetch_foundation.py"), "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: fetch_foundation.py" in result.stdout
    assert "--target TARGET" in result.stdout
