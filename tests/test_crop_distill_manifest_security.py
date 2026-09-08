"""Security boundaries emitted for the crop-distill Kubernetes Jobs."""

from __future__ import annotations

import os
import sys
import types
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
        manifests, "CROP_SOURCE_FREEZE_OPERATOR_SOURCE_GIT_SHA", SOURCE_SHA
    )
    monkeypatch.setattr(
        manifests, "CROP_SOURCE_FREEZE_OPERATOR_IMAGE", IMAGE_REF
    )
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


def _operator_documents(text: str) -> dict[str, dict]:
    documents = list(yaml.safe_load_all(text))
    by_kind = {document["kind"]: document for document in documents}
    assert len(documents) == len(by_kind) == 4
    return by_kind


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
    assert set(manifests.CROP_MODEL_UIDS.values()) == set(range(2001, 2008))


@pytest.mark.parametrize("model", manifests.CROP_MODELS)
def test_crop_mounts_only_its_preowned_output_directories(render_identity, model):
    pod, container = _pod_and_container(manifests.render_crop_distill(model))
    mounts = _mounts_by_path(container)
    heads = mounts["/cephfs/crop-heads"]
    records = mounts["/cephfs/crop-records"]

    assert heads["subPath"] == f"distill/crop_heads/{model}_r2_crop_runs"
    assert records["subPath"] == f"ops/crop-distill/{model}"
    # Boundary-aware: plain substring matching false-alarms on name
    # prefixes (prithvi300m is a prefix of prithvi300m4f) — compare the
    # exact foreign path forms instead.
    assert all(
        heads["subPath"] != f"distill/crop_heads/{other}_r2_crop_runs"
        and records["subPath"] != f"ops/crop-distill/{other}"
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
                "tessera": 2008,
            },
            "2001..2007",
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


def test_freeze_operator_has_narrow_resumable_ice_authority(render_identity):
    documents = _operator_documents(
        manifests.render_crop_source_freeze_operator()
    )
    account = documents["ServiceAccount"]
    role = documents["Role"]
    binding = documents["RoleBinding"]
    job = documents["Job"]

    assert account["automountServiceAccountToken"] is False
    assert binding["subjects"] == [
        {
            "kind": "ServiceAccount",
            "name": "ladder-crop-source-freeze-operator",
            "namespace": "prithvi-training-default",
        }
    ]
    assert binding["roleRef"] == {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "Role",
        "name": "ladder-crop-source-freeze-operator",
    }
    assert role["rules"] == [
        {
            "apiGroups": [""],
            "resources": ["pods", "replicationcontrollers"],
            "verbs": ["get", "list"],
        },
        {
            "apiGroups": [""],
            "resources": ["configmaps"],
            "resourceNames": ["crop-source-freeze-lease"],
            "verbs": ["get", "update"],
        },
        {
            "apiGroups": ["batch"],
            "resources": ["jobs", "cronjobs"],
            "verbs": ["get", "list"],
        },
        {
            "apiGroups": ["batch"],
            "resources": ["cronjobs"],
            "resourceNames": [
                "ladder-queue",
                "gpu-reaper",
                "campaign-orchestrator",
            ],
            "verbs": ["update"],
        },
        {
            "apiGroups": ["apps"],
            "resources": [
                "deployments",
                "statefulsets",
                "daemonsets",
                "replicasets",
            ],
            "verbs": ["get", "list"],
        },
    ]
    assert all(
        verb not in {"delete", "patch"}
        for rule in role["rules"]
        for verb in rule["verbs"]
    )
    assert all(
        not ("configmaps" in rule["resources"] and "create" in rule["verbs"])
        for rule in role["rules"]
    )

    job_spec = job["spec"]
    pod = job_spec["template"]["spec"]
    assert "ttlSecondsAfterFinished" not in job_spec
    assert job_spec["backoffLimit"] == 6
    assert pod["activeDeadlineSeconds"] == 43200
    assert pod["automountServiceAccountToken"] is False
    assert pod["serviceAccountName"] == "ladder-crop-source-freeze-operator"
    assert pod["restartPolicy"] == "OnFailure"
    assert pod["securityContext"] == {
        "seccompProfile": {"type": "RuntimeDefault"}
    }
    assert pod["imagePullSecrets"] == [{"name": "ghcr-push"}]

    prepare = pod["initContainers"][0]
    operator = pod["containers"][0]
    assert prepare["image"] == IMAGE_REF == operator["image"]
    assert prepare["command"] == ["/usr/local/bin/python"]
    assert prepare["args"] == [
        "/opt/imintengine/scripts/crop_source_freeze_operator.py",
        "prepare",
        "--state-parent",
        "/state-parent",
    ]
    for container in (prepare, operator):
        env = {item["name"]: item for item in container["env"]}
        assert env["CROP_DISTILL_SOURCE_GIT_SHA"]["value"] == SOURCE_SHA
        assert env["CROP_DISTILL_IMAGE"]["value"] == IMAGE_REF
        assert env["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == (
            "metadata.uid"
        )
        assert "CROP_SOURCE_FREEZE_OPERATOR_SOURCE_GIT_SHA" not in env
        assert "CROP_SOURCE_FREEZE_OPERATOR_IMAGE" not in env
    assert prepare["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "capabilities": {
            "drop": ["ALL"],
            "add": ["CHOWN", "FOWNER"],
        },
        "readOnlyRootFilesystem": True,
        "runAsUser": 0,
        "runAsGroup": 2000,
    }
    assert operator["command"] == ["/usr/local/bin/python"]
    assert operator["args"] == [
        "/opt/imintengine/scripts/crop_source_freeze_operator.py",
        "serve",
        "--state-dir",
        "/state",
        "--run-id",
        manifests.CROP_SOURCE_FREEZE_OPERATOR_RUN_ID,
        "--namespace",
        "prithvi-training-default",
    ]
    assert operator["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
        "readOnlyRootFilesystem": True,
        "runAsNonRoot": True,
        "runAsUser": 2000,
        "runAsGroup": 2000,
    }
    for container in (prepare, operator):
        _assert_ice_resources(container)

    assert prepare["volumeMounts"] == [
        {
            "name": "training-data-cephfs",
            "mountPath": "/state-parent",
            "subPath": "ops/crop-distill/source-access",
        }
    ]
    assert operator["volumeMounts"] == [
        {
            "name": "training-data-cephfs",
            "mountPath": "/state",
            "subPath": (
                "ops/crop-distill/source-access/crop-source-freeze"
            ),
        },
        {"name": "tmp", "mountPath": "/tmp"},
        {
            "name": "kube-api-access",
            "mountPath": "/var/run/secrets/kubernetes.io/serviceaccount",
            "readOnly": True,
        },
    ]
    assert pod["volumes"] == [
        {
            "name": "training-data-cephfs",
            "persistentVolumeClaim": {"claimName": "training-data-cephfs"},
        },
        {"name": "tmp", "emptyDir": {"sizeLimit": "128Mi"}},
        {
            "name": "kube-api-access",
            "projected": {
                "defaultMode": 420,
                "sources": [
                    {
                        "serviceAccountToken": {
                            "expirationSeconds": 3600,
                            "path": "token",
                        }
                    },
                    {
                        "configMap": {
                            "name": "kube-root-ca.crt",
                            "items": [{"key": "ca.crt", "path": "ca.crt"}],
                        }
                    },
                    {
                        "downwardAPI": {
                            "items": [
                                {
                                    "path": "namespace",
                                    "fieldRef": {
                                        "apiVersion": "v1",
                                        "fieldPath": "metadata.namespace",
                                    },
                                }
                            ]
                        }
                    },
                ],
            },
        },
    ]


def test_freeze_recovery_is_one_shot_exact_restore_only(render_identity):
    assert manifests.CROP_SOURCE_FREEZE_OPERATOR_RUN_ID == (
        "lucas-crop-attempt-16-verify"
    )
    assert manifests.CROP_SOURCE_FREEZE_RECOVERY_RUN_ID == (
        "lucas-crop-attempt-15-verify"
    )
    job = yaml.safe_load(manifests.render_crop_source_freeze_recovery())
    assert job["apiVersion"] == "batch/v1"
    assert job["kind"] == "Job"
    assert job["metadata"]["name"] == (
        "ladder-crop-source-freeze-recovery-attempt-15"
    )
    assert job["metadata"]["namespace"] == "prithvi-training-default"

    job_spec = job["spec"]
    pod = job_spec["template"]["spec"]
    assert job_spec["backoffLimit"] == 0
    assert "ttlSecondsAfterFinished" not in job_spec
    assert pod["activeDeadlineSeconds"] == 900
    assert pod["automountServiceAccountToken"] is False
    assert pod["serviceAccountName"] == "ladder-crop-source-freeze-operator"
    assert pod["restartPolicy"] == "Never"

    container = pod["containers"][0]
    assert container["name"] == "exact-restore"
    assert container["image"] == IMAGE_REF
    assert container["command"] == ["/usr/local/bin/python"]
    assert container["args"][0] == "-c"
    recovery_code = container["args"][1]
    assert "(os.geteuid(), os.getegid()) != (2000, 2000)" in recovery_code
    assert "operator._verify_runtime_identity()" in recovery_code
    assert "operator._require_state_root(Path(\"/state\"))" in recovery_code
    assert "operator._install_in_cluster_kubeconfig()" in recovery_code
    assert "old_operator_objects" in recovery_code
    assert 'labels.get("job-name")' in recovery_code
    assert 'reference.get("controller") is True' in recovery_code
    assert '.startswith(' in recovery_code
    assert '== "ladder-crop-source-freeze-operator"' in recovery_code
    assert "freeze.restore(" in recovery_code
    assert 'run_dir=Path("/state") / "lucas-crop-attempt-15-verify"' in (
        recovery_code
    )
    assert all(
        forbidden not in recovery_code
        for forbidden in ("freeze.hold(", "freeze.watch(", "freeze.gate_phase(")
    )

    env = {item["name"]: item for item in container["env"]}
    assert env["CROP_DISTILL_SOURCE_GIT_SHA"]["value"] == SOURCE_SHA
    assert env["CROP_DISTILL_IMAGE"]["value"] == IMAGE_REF
    assert env["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == (
        "metadata.uid"
    )
    assert container["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
        "readOnlyRootFilesystem": True,
        "runAsNonRoot": True,
        "runAsUser": 2000,
        "runAsGroup": 2000,
    }
    _assert_ice_resources(container)
    mounts = _mounts_by_path(container)
    assert mounts["/state"]["subPath"] == (
        "ops/crop-distill/source-access/crop-source-freeze"
    )
    assert mounts["/var/run/secrets/kubernetes.io/serviceaccount"][
        "readOnly"
    ] is True
    projected = next(
        volume["projected"]
        for volume in pod["volumes"]
        if volume["name"] == "kube-api-access"
    )
    token = projected["sources"][0]["serviceAccountToken"]
    assert token == {"expirationSeconds": 600, "path": "token"}


def test_freeze_recovery_executes_guard_and_exact_restore(
    render_identity,
    monkeypatch,
):
    recovery_code = yaml.safe_load(
        manifests.render_crop_source_freeze_recovery()
    )["spec"]["template"]["spec"]["containers"][0]["args"][1]
    calls = []

    class FakeKubectl:
        def __init__(self, *, context, namespace):
            calls.append(("client", context, namespace))

        def inventory(self):
            return []

    fake_freeze = types.SimpleNamespace(
        Kubectl=FakeKubectl,
        restore=lambda client, *, run_dir, timeout_seconds: calls.append(
            ("restore", client, run_dir, timeout_seconds)
        ),
    )
    fake_operator = types.SimpleNamespace(
        _verify_runtime_identity=lambda: calls.append(("identity",)),
        _require_state_root=lambda path: calls.append(("state", path)),
        _install_in_cluster_kubeconfig=lambda: calls.append(("kubeconfig",)),
    )
    monkeypatch.setitem(sys.modules, "crop_source_freeze", fake_freeze)
    monkeypatch.setitem(
        sys.modules,
        "crop_source_freeze_operator",
        fake_operator,
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(os, "geteuid", lambda: 2000)
    monkeypatch.setattr(os, "getegid", lambda: 2000)

    exec(compile(recovery_code, "<freeze-recovery>", "exec"), {})

    assert calls[0:4] == [
        ("identity",),
        ("state", Path("/state")),
        ("kubeconfig",),
        ("client", "", "prithvi-training-default"),
    ]
    restore_call = calls[4]
    assert restore_call[0] == "restore"
    assert restore_call[2:] == (
        Path("/state/lucas-crop-attempt-15-verify"),
        60.0,
    )


def test_freeze_recovery_refuses_old_owner_reference(
    render_identity,
    monkeypatch,
):
    recovery_code = yaml.safe_load(
        manifests.render_crop_source_freeze_recovery()
    )["spec"]["template"]["spec"]["containers"][0]["args"][1]
    restored = []

    class FakeKubectl:
        def __init__(self, **_kwargs):
            pass

        def inventory(self):
            return [{
                "kind": "Pod",
                "metadata": {
                    "name": "renamed-pod",
                    "labels": {},
                    "ownerReferences": [{
                        "kind": "Job",
                        "name": "ladder-crop-source-freeze-operator",
                        "controller": True,
                    }],
                },
            }]

    fake_freeze = types.SimpleNamespace(
        Kubectl=FakeKubectl,
        restore=lambda *_args, **_kwargs: restored.append(True),
    )
    fake_operator = types.SimpleNamespace(
        _verify_runtime_identity=lambda: None,
        _require_state_root=lambda _path: None,
        _install_in_cluster_kubeconfig=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "crop_source_freeze", fake_freeze)
    monkeypatch.setitem(
        sys.modules,
        "crop_source_freeze_operator",
        fake_operator,
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(os, "geteuid", lambda: 2000)
    monkeypatch.setattr(os, "getegid", lambda: 2000)

    with pytest.raises(RuntimeError, match="old freeze operator still exists"):
        exec(compile(recovery_code, "<freeze-recovery>", "exec"), {})
    assert restored == []


def test_freeze_recovery_refuses_wrong_runtime_identity(
    render_identity,
    monkeypatch,
):
    recovery_code = yaml.safe_load(
        manifests.render_crop_source_freeze_recovery()
    )["spec"]["template"]["spec"]["containers"][0]["args"][1]
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(os, "getegid", lambda: 2000)

    with pytest.raises(RuntimeError, match="recovery requires UID:GID 2000:2000"):
        exec(compile(recovery_code, "<freeze-recovery>", "exec"), {})


def test_freeze_operator_refuses_unpublished_image(render_identity, monkeypatch):
    monkeypatch.setattr(
        manifests,
        "CROP_SOURCE_FREEZE_OPERATOR_IMAGE",
        "ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:" + "0" * 64,
    )

    with pytest.raises(ValueError, match="CROP_SOURCE_FREEZE_OPERATOR_IMAGE"):
        manifests.render_crop_source_freeze_operator()


def test_operator_only_writes_no_other_manifest(
    render_identity,
    monkeypatch,
    tmp_path,
):
    output_dir = tmp_path / "k8s" / "ladder"
    monkeypatch.setattr(manifests, "REPO", tmp_path)
    monkeypatch.setattr(manifests, "OUT_DIR", output_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gen_ladder_manifests.py", "--crop-operator-only"],
    )

    assert manifests.main() == 0
    assert [path.name for path in output_dir.iterdir()] == [
        "crop-source-freeze-operator-job.yaml"
    ]


def test_recovery_only_writes_no_other_manifest(
    render_identity,
    monkeypatch,
    tmp_path,
):
    output_dir = tmp_path / "k8s" / "ladder"
    monkeypatch.setattr(manifests, "REPO", tmp_path)
    monkeypatch.setattr(manifests, "OUT_DIR", output_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gen_ladder_manifests.py", "--crop-recovery-only"],
    )

    assert manifests.main() == 0
    assert [path.name for path in output_dir.iterdir()] == [
        "crop-source-freeze-recovery-job.yaml"
    ]


def test_committed_operator_manifest_is_current(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["gen_ladder_manifests.py", "--crop-operator-only", "--check"],
    )

    assert manifests.main() == 0


def test_committed_recovery_manifest_is_current(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["gen_ladder_manifests.py", "--crop-recovery-only", "--check"],
    )

    assert manifests.main() == 0


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
