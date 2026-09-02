"""Security boundaries emitted for the crop-distill Kubernetes Jobs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts import gen_ladder_manifests as manifests

SOURCE_SHA = "a" * 40
IMAGE_REF = "ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:" + "b" * 64
SPLIT_SHA = "c" * 64


@pytest.fixture
def render_identity(monkeypatch):
    monkeypatch.setattr(manifests, "CROP_DISTILL_SOURCE_GIT_SHA", SOURCE_SHA)
    monkeypatch.setattr(manifests, "CROP_DISTILL_IMAGE", IMAGE_REF)
    monkeypatch.setattr(
        manifests,
        "CROP_DISTILL_SPLIT_MANIFEST_SHA256",
        SPLIT_SHA,
    )


def _pod_and_container(text: str) -> tuple[dict, dict]:
    document = yaml.safe_load(text)
    pod = document["spec"]["template"]["spec"]
    return pod, pod["containers"][0]


def _assert_common_hardening(pod: dict, container: dict) -> None:
    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}
    security = container["securityContext"]
    assert security["allowPrivilegeEscalation"] is False
    assert security["readOnlyRootFilesystem"] is True
    assert security["capabilities"]["drop"] == ["ALL"]


def _assert_ice_resources(container: dict) -> None:
    """ICE admission requires guaranteed QoS for every declared resource."""
    assert container["resources"]["requests"] == container["resources"]["limits"]


def test_crop_render_requires_external_split_authority(monkeypatch):
    monkeypatch.setattr(manifests, "CROP_DISTILL_SOURCE_GIT_SHA", SOURCE_SHA)
    monkeypatch.setattr(manifests, "CROP_DISTILL_IMAGE", IMAGE_REF)
    monkeypatch.setattr(
        manifests,
        "CROP_DISTILL_SPLIT_MANIFEST_SHA256",
        "0" * 64,
    )

    with pytest.raises(ValueError, match="split freeze"):
        manifests.render_crop_distill("clay")

    # The producer must remain renderable before the digest it creates exists.
    manifests.render_lucas_crop_split()
    manifests.render_crop_storage_prep()


def test_crop_job_is_nonroot_and_receives_reviewed_split_digest(render_identity):
    pod, container = _pod_and_container(manifests.render_crop_distill("croma"))
    _assert_common_hardening(pod, container)
    _assert_ice_resources(container)
    assert pod["securityContext"] == {
        "runAsNonRoot": True,
        "runAsUser": manifests.CROP_MODEL_UIDS["croma"],
        "runAsGroup": 2000,
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    assert container["securityContext"]["runAsNonRoot"] is True
    assert (
        container["securityContext"]["runAsUser"]
        == (manifests.CROP_MODEL_UIDS["croma"])
    )
    assert container["securityContext"]["runAsGroup"] == 2000

    env = {item["name"]: item for item in container["env"]}
    assert env["CROP_DISTILL_SPLIT_MANIFEST_SHA256"]["value"] == SPLIT_SHA
    mounts = {item["name"]: item for item in container["volumeMounts"]}
    assert mounts["split"] == {
        "name": "split",
        "mountPath": "/cephfs/distill/crop_split",
        "subPath": "distill/crop_split/crop_consumer",
        "readOnly": True,
    }
    assert mounts["tiles"]["readOnly"] is True
    assert mounts["checkpoint"]["readOnly"] is True
    assert mounts["work"]["mountPath"] == "/work"
    assert container["resources"]["requests"]["ephemeral-storage"] == "8Gi"
    assert container["resources"]["limits"]["ephemeral-storage"] == "8Gi"
    work_volume = next(
        volume for volume in pod["volumes"] if volume["name"] == "work"
    )
    assert work_volume["emptyDir"] == {"sizeLimit": "8Gi"}
    assert "validator" not in manifests.render_crop_distill("croma").lower()


def test_split_job_is_nonroot_and_has_only_required_pvc_subpaths(render_identity):
    pod, container = _pod_and_container(manifests.render_lucas_crop_split())
    _assert_common_hardening(pod, container)
    _assert_ice_resources(container)
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["securityContext"]["runAsUser"] == 2000
    assert pod["securityContext"]["runAsGroup"] == 2000
    mounts = {item["name"]: item for item in container["volumeMounts"]}
    assert mounts == {
        "tiles": {
            "name": "tiles",
            "mountPath": "/cephfs/unified_v2_512",
            "subPath": "unified_v2_512",
            "readOnly": True,
        },
        "lucas": {
            "name": "lucas",
            "mountPath": "/cephfs/lucas",
            "subPath": "lucas",
            "readOnly": True,
        },
        "distill": {
            "name": "distill",
            "mountPath": "/cephfs/distill/crop_split",
            "subPath": "distill/crop_split",
        },
        "ops": {
            "name": "ops",
            "mountPath": "/cephfs/ops/crop-distill",
            "subPath": "ops/crop-distill/split",
        },
        "work": {"name": "work", "mountPath": "/work"},
    }


def test_crop_columns_have_distinct_fixed_uids():
    assert set(manifests.CROP_MODEL_UIDS) == set(manifests.CROP_MODELS)
    assert len(set(manifests.CROP_MODEL_UIDS.values())) == len(
        manifests.CROP_MODEL_UIDS
    )
    assert set(manifests.CROP_MODEL_UIDS.values()) == set(range(2001, 2007))


@pytest.mark.parametrize("model", manifests.CROP_MODELS)
def test_crop_mounts_only_its_preowned_output_directories(render_identity, model):
    pod, container = _pod_and_container(manifests.render_crop_distill(model))
    mounts = {item["name"]: item for item in container["volumeMounts"]}

    assert mounts["heads"]["subPath"] == (
        f"distill/crop_heads/{model}_r2_crop_runs"
    )
    assert mounts["records"]["subPath"] == f"ops/crop-distill/{model}"
    assert all(
        other not in mounts["heads"]["subPath"]
        and other not in mounts["records"]["subPath"]
        for other in manifests.CROP_MODELS
        if other != model
    )
    assert pod["securityContext"]["runAsUser"] == manifests.CROP_MODEL_UIDS[model]


@pytest.mark.parametrize(
    "bad_uids, message",
    (
        ({model: 2001 for model in manifests.CROP_MODELS}, "unique"),
        (
            {
                **manifests.CROP_MODEL_UIDS,
                "tessera": 2007,
            },
            "2001..2006",
        ),
        (
            {
                model: uid
                for model, uid in manifests.CROP_MODEL_UIDS.items()
                if model != "tessera"
            },
            "exactly",
        ),
    ),
)
def test_crop_render_refuses_invalid_model_uid_map(
    render_identity, monkeypatch, bad_uids, message
):
    monkeypatch.setattr(manifests, "CROP_MODEL_UIDS", bad_uids)
    with pytest.raises(ValueError, match=message):
        manifests.render_crop_distill("clay")


def test_bootstrap_refuses_preexisting_consumer_manifest(
    render_identity, monkeypatch, tmp_path, capsys
):
    output_dir = tmp_path / "k8s" / "ladder"
    output_dir.mkdir(parents=True)
    stale = output_dir / "crop-distill-clay-job.yaml"
    stale.write_text("stale runnable consumer\n")
    monkeypatch.setattr(manifests, "REPO", tmp_path)
    monkeypatch.setattr(manifests, "OUT_DIR", output_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gen_ladder_manifests.py", "--crop-bootstrap-only"],
    )

    assert manifests.main() == 2
    assert "stale consumer manifests" in capsys.readouterr().err
    assert stale.read_text() == "stale runnable consumer\n"
    assert not (output_dir / "lucas-crop-split-job.yaml").exists()


def test_storage_prep_is_the_only_root_job_and_has_one_capability(render_identity):
    pod, container = _pod_and_container(manifests.render_crop_storage_prep())
    _assert_common_hardening(pod, container)
    _assert_ice_resources(container)
    assert pod["securityContext"]["runAsUser"] == 0
    assert pod["securityContext"]["runAsGroup"] == 2000
    security = container["securityContext"]
    assert security["runAsUser"] == 0
    assert security["runAsGroup"] == 2000
    assert security["capabilities"] == {
        "drop": ["ALL"],
        "add": ["CHOWN", "FOWNER"],
    }
    assert container["args"] == [
        "/opt/imintengine/scripts/prepare_crop_distill_storage.py"
    ]
    env = {item["name"]: item for item in container["env"]}
    assert env["CROP_DISTILL_SOURCE_GIT_SHA"]["value"] == SOURCE_SHA
    assert env["CROP_DISTILL_IMAGE"]["value"] == IMAGE_REF
    assert env["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == (
        "metadata.uid"
    )
    assert container["volumeMounts"] == [
        {
            "name": "distill",
            "mountPath": "/cephfs/distill",
            "subPath": "distill",
        },
        {
            "name": "ops",
            "mountPath": "/cephfs/ops",
            "subPath": "ops",
        },
    ]


def test_crop_runtime_network_policy_denies_all_egress():
    policy = yaml.safe_load(manifests.render_crop_deny_egress())
    assert policy["apiVersion"] == "networking.k8s.io/v1"
    assert policy["kind"] == "NetworkPolicy"
    assert policy["spec"] == {
        "podSelector": {
            "matchExpressions": [
                {
                    "key": "purpose",
                    "operator": "In",
                    "values": [
                        "ladder-crop-distill",
                        "ladder-crop-distill-storage",
                    ],
                }
            ]
        },
        "policyTypes": ["Egress"],
        "egress": [],
    }
