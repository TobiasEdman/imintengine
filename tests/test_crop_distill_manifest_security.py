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
from scripts import crop_source_freeze as source_freeze

SOURCE_SHA = "a" * 40
IMAGE_REF = "ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:" + "b" * 64
SPLIT_SHA = "c" * 64
PLAN_SHA = "d" * 64
PLAN_POD_UID = "plan-pod-uid"
COMPLETION_SHA = "e" * 64
COMPLETION_POD_UID = "apply-pod-uid"


@pytest.fixture
def render_identity(monkeypatch):
    monkeypatch.setattr(manifests, "CROP_DISTILL_SOURCE_GIT_SHA", SOURCE_SHA)
    monkeypatch.setattr(manifests, "CROP_DISTILL_IMAGE", IMAGE_REF)
    monkeypatch.setattr(
        manifests, "CROP_SOURCE_ACCESS_SOURCE_GIT_SHA", SOURCE_SHA
    )
    monkeypatch.setattr(manifests, "CROP_SOURCE_ACCESS_IMAGE", IMAGE_REF)
    monkeypatch.setattr(
        manifests, "CROP_DISTILL_SPLIT_SOURCE_GIT_SHA", SOURCE_SHA
    )
    monkeypatch.setattr(manifests, "CROP_SOURCE_ACCESS_PLAN_SHA256", PLAN_SHA)
    monkeypatch.setattr(manifests, "CROP_SOURCE_ACCESS_PLAN_POD_UID", PLAN_POD_UID)
    monkeypatch.setattr(
        manifests, "CROP_SOURCE_ACCESS_COMPLETION_SHA256", COMPLETION_SHA
    )
    monkeypatch.setattr(
        manifests, "CROP_SOURCE_ACCESS_COMPLETION_POD_UID", COMPLETION_POD_UID
    )
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


def _mounts_by_path(container: dict) -> dict[str, dict]:
    mounts = container["volumeMounts"]
    by_path = {mount["mountPath"]: mount for mount in mounts}
    assert len(by_path) == len(mounts)
    return by_path


def _assert_no_duplicate_pvc_claims(document: dict) -> None:
    pod = document["spec"]["template"]["spec"]
    claims = [
        volume["persistentVolumeClaim"]["claimName"]
        for volume in pod["volumes"]
        if "persistentVolumeClaim" in volume
    ]
    assert claims
    assert len(claims) == len(set(claims)), claims


def _assert_live_freeze_lease(pod: dict, container: dict) -> None:
    env = {entry["name"]: entry for entry in container["env"]}
    assert env["CROP_SOURCE_FREEZE_LEASE_PATH"]["value"] == (
        "/var/run/crop-source-freeze/lease.json"
    )
    mount = next(
        item
        for item in container["volumeMounts"]
        if item["name"] == "crop-source-freeze-lease"
    )
    assert mount == {
        "name": "crop-source-freeze-lease",
        "mountPath": "/var/run/crop-source-freeze",
        "readOnly": True,
    }
    assert "subPath" not in mount
    volume = next(
        item for item in pod["volumes"] if item["name"] == "crop-source-freeze-lease"
    )
    assert volume == {
        "name": "crop-source-freeze-lease",
        "configMap": {
            "name": "crop-source-freeze-lease",
            "optional": False,
            "items": [{"key": "lease.json", "path": "lease.json"}],
        },
    }


def test_crop_render_requires_external_split_authority(monkeypatch):
    monkeypatch.setattr(manifests, "CROP_DISTILL_SOURCE_GIT_SHA", SOURCE_SHA)
    monkeypatch.setattr(manifests, "CROP_DISTILL_IMAGE", IMAGE_REF)
    monkeypatch.setattr(
        manifests, "CROP_SOURCE_ACCESS_SOURCE_GIT_SHA", SOURCE_SHA
    )
    monkeypatch.setattr(manifests, "CROP_SOURCE_ACCESS_IMAGE", IMAGE_REF)
    monkeypatch.setattr(
        manifests, "CROP_DISTILL_SPLIT_SOURCE_GIT_SHA", SOURCE_SHA
    )
    monkeypatch.setattr(manifests, "CROP_SOURCE_ACCESS_PLAN_SHA256", PLAN_SHA)
    monkeypatch.setattr(manifests, "CROP_SOURCE_ACCESS_PLAN_POD_UID", PLAN_POD_UID)
    monkeypatch.setattr(
        manifests, "CROP_SOURCE_ACCESS_COMPLETION_SHA256", COMPLETION_SHA
    )
    monkeypatch.setattr(
        manifests, "CROP_SOURCE_ACCESS_COMPLETION_POD_UID", COMPLETION_POD_UID
    )
    monkeypatch.setattr(
        manifests,
        "CROP_DISTILL_SPLIT_MANIFEST_SHA256",
        "0" * 64,
    )

    with pytest.raises(ValueError, match="split freeze"):
        manifests.render_crop_distill("clay")

    # Upstream phases remain renderable before the split digest exists.
    manifests.render_lucas_crop_split()
    manifests.render_crop_storage_prep()


def test_split_render_requires_git_pinned_plan_pod_uid(
    render_identity,
    monkeypatch,
):
    monkeypatch.setattr(
        manifests,
        "CROP_SOURCE_ACCESS_PLAN_POD_UID",
        "<pending>",
    )

    with pytest.raises(ValueError, match="CROP_SOURCE_ACCESS_PLAN_POD_UID"):
        manifests.render_lucas_crop_split()


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
    mounts = _mounts_by_path(container)
    assert mounts["/cephfs/distill/crop_split"] == {
        "name": "training-data-cephfs",
        "mountPath": "/cephfs/distill/crop_split",
        "subPath": "distill/crop_split/crop_consumer",
        "readOnly": True,
    }
    assert mounts["/cephfs/unified_v2_512"]["readOnly"] is True
    assert mounts["/cephfs/checkpoints/ladder/croma_r2"]["readOnly"] is True
    assert mounts["/work"] == {"name": "work", "mountPath": "/work"}
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
    env = {item["name"]: item for item in container["env"]}
    assert env["CROP_SOURCE_ACCESS_PLAN_SHA256"]["value"] == PLAN_SHA
    assert env["CROP_SOURCE_ACCESS_PLAN_POD_UID"]["value"] == PLAN_POD_UID
    _assert_live_freeze_lease(pod, container)
    assert container["volumeMounts"] == [
        {
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/unified_v2_512",
            "subPath": "unified_v2_512",
            "readOnly": True,
        },
        {
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/lucas",
            "subPath": "lucas",
            "readOnly": True,
        },
        {
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/distill/crop_split",
            "subPath": "distill/crop_split",
            "readOnly": False,
        },
        {
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/ops/crop-distill",
            "subPath": "ops/crop-distill/split",
        },
        {
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/source-access-completion/completion.json",
            "subPath": (
                "ops/crop-distill/source-access/apply/"
                f"{COMPLETION_POD_UID}/completion.json"
            ),
            "readOnly": True,
        },
        {
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/source-access-lock",
            "subPath": "ops/crop-distill/source-access/locks",
        },
        {"name": "work", "mountPath": "/work"},
        {
            "name": "crop-source-freeze-lease",
            "mountPath": "/var/run/crop-source-freeze",
            "readOnly": True,
        },
    ]
    assert pod["volumes"] == [
        {
            "name": "training-data-cephfs",
            "persistentVolumeClaim": {"claimName": "training-data-cephfs"},
        },
        {"name": "work", "emptyDir": {}},
        {
            "name": "crop-source-freeze-lease",
            "configMap": {
                "name": "crop-source-freeze-lease",
                "optional": False,
                "items": [{"key": "lease.json", "path": "lease.json"}],
            },
        },
    ]


def test_crop_columns_have_distinct_fixed_uids():
    assert set(manifests.CROP_MODEL_UIDS) == set(manifests.CROP_MODELS)
    assert len(set(manifests.CROP_MODEL_UIDS.values())) == len(
        manifests.CROP_MODEL_UIDS
    )
    assert set(manifests.CROP_MODEL_UIDS.values()) == set(range(2001, 2007))


@pytest.mark.parametrize("model", manifests.CROP_MODELS)
def test_crop_mounts_only_its_preowned_output_directories(render_identity, model):
    pod, container = _pod_and_container(manifests.render_crop_distill(model))
    mounts = _mounts_by_path(container)
    heads = mounts["/cephfs/crop-heads"]
    records = mounts["/cephfs/crop-records"]

    assert heads["subPath"] == f"distill/crop_heads/{model}_r2_crop_runs"
    assert records["subPath"] == f"ops/crop-distill/{model}"
    assert all(
        other not in heads["subPath"] and other not in records["subPath"]
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
    assert "stale downstream manifests" in capsys.readouterr().err
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
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/distill",
            "subPath": "distill",
        },
        {
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/ops",
            "subPath": "ops",
        },
    ]


def test_storage_prep_uses_one_pvc_volume(render_identity):
    document = yaml.safe_load(manifests.render_crop_storage_prep())
    pod = document["spec"]["template"]["spec"]

    assert pod["volumes"] == [
        {
            "name": "training-data-cephfs",
            "persistentVolumeClaim": {"claimName": "training-data-cephfs"},
        }
    ]
    mounts = pod["containers"][0]["volumeMounts"]
    assert [mount["name"] for mount in mounts] == [
        "training-data-cephfs",
        "training-data-cephfs",
    ]
    assert [mount["subPath"] for mount in mounts] == ["distill", "ops"]


def test_crop_jobs_use_unique_pvcs_and_pod_scoped_deadlines(render_identity):
    jobs = [
        ("storage-prep", manifests.render_crop_storage_prep(), 600),
        ("source-plan", manifests.render_crop_source_access_plan(), 7200),
        ("source-apply", manifests.render_crop_source_access_apply(), 7200),
        ("split", manifests.render_lucas_crop_split(), 21600),
        *[
            (model, manifests.render_crop_distill(model), 43200)
            for model in manifests.CROP_MODELS
        ],
    ]
    assert {name for name, _, _ in jobs} == {
        "storage-prep",
        "source-plan",
        "source-apply",
        "split",
        *manifests.CROP_MODELS,
    }

    for _, text, deadline in jobs:
        document = yaml.safe_load(text)
        job_spec = document["spec"]
        pod = job_spec["template"]["spec"]
        assert "activeDeadlineSeconds" not in job_spec
        assert pod["activeDeadlineSeconds"] == deadline
        _assert_no_duplicate_pvc_claims(document)

    source_phase_deadlines = [deadline for name, _, deadline in jobs if name in {
        "source-plan",
        "source-apply",
        "split",
    }]
    assert source_freeze.PHASE_REQUEST_SECONDS >= (
        max(source_phase_deadlines) + source_freeze.LEASE_SECONDS
    )


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
                        "ladder-crop-source-access-plan",
                        "ladder-crop-source-access-apply",
                    ],
                }
            ]
        },
        "policyTypes": ["Egress"],
        "egress": [],
    }


def test_source_access_plan_is_root_read_only_and_drop_all(render_identity):
    pod, container = _pod_and_container(manifests.render_crop_source_access_plan())
    _assert_common_hardening(pod, container)
    assert container["securityContext"]["capabilities"] == {"drop": ["ALL"]}
    assert pod["securityContext"]["runAsUser"] == 0
    _assert_live_freeze_lease(pod, container)
    mounts = _mounts_by_path(container)
    assert mounts["/cephfs/unified_v2_512"]["readOnly"] is True
    assert mounts["/cephfs/lucas/lucas_tile_index.parquet"]["readOnly"] is True
    assert all(mount["mountPath"] != "/cephfs" for mount in container["volumeMounts"])


def test_source_access_apply_has_exact_caps_and_dataset_subpath(render_identity):
    pod, container = _pod_and_container(manifests.render_crop_source_access_apply())
    _assert_common_hardening(pod, container)
    assert container["securityContext"]["capabilities"] == {
        "drop": ["ALL"],
        "add": ["CHOWN", "FOWNER"],
    }
    assert pod["securityContext"]["runAsUser"] == 0
    _assert_live_freeze_lease(pod, container)
    mounts = _mounts_by_path(container)
    dataset = mounts["/cephfs/unified_v2_512"]
    assert dataset == {
        "name": "training-data-cephfs",
        "mountPath": "/cephfs/unified_v2_512",
        "subPath": "unified_v2_512",
    }
    plan = mounts["/cephfs/source-access-plan/plan.json"]
    assert plan["subPath"].endswith(f"/{PLAN_POD_UID}/plan.json")
    assert plan["readOnly"] is True
    assert all(mount["mountPath"] != "/cephfs" for mount in container["volumeMounts"])
