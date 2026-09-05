"""Structural and lease gates for the operator-side crop source freeze."""

from __future__ import annotations

import copy
import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import crop_source_freeze as freeze
from scripts import crop_source_access as access


def _pod_spec(
    *,
    sub_path: str | None = "unified_v2_512",
    read_only: bool = False,
    sub_path_expr: str | None = None,
    container_field: str = "containers",
    container_name: str = "worker",
    volume_name: str = "data",
) -> dict:
    mount = {
        "name": volume_name,
        "mountPath": "/data",
        "readOnly": read_only,
    }
    if sub_path is not None:
        mount["subPath"] = sub_path
    if sub_path_expr is not None:
        mount["subPathExpr"] = sub_path_expr
    return {
        "volumes": [{
            "name": volume_name,
            "persistentVolumeClaim": {"claimName": freeze.PVC_CLAIM},
        }],
        container_field: [{"name": container_name, "volumeMounts": [mount]}],
    }


@pytest.mark.parametrize(
    "sub_path",
    [None, "", "unified_v2_512", "unified_v2_512/child", "."],
)
def test_rw_mount_overlap_includes_full_exact_and_descendant(sub_path):
    assert freeze.pod_spec_rw_overlaps(_pod_spec(sub_path=sub_path))


def test_rw_mount_overlap_includes_ancestor_and_unresolved_expression():
    assert freeze.pod_spec_rw_overlaps(_pod_spec(sub_path="unified_v2_512/.."))
    overlaps = freeze.pod_spec_rw_overlaps(
        _pod_spec(sub_path=None, sub_path_expr="$(DATASET_SUBPATH)")
    )
    assert overlaps[0]["access"] == "rw-unresolved-subPathExpr"


def test_read_only_or_disjoint_mount_is_not_an_overlap():
    assert not freeze.pod_spec_rw_overlaps(
        _pod_spec(sub_path="unified_v2_512", read_only=True)
    )
    assert not freeze.pod_spec_rw_overlaps(_pod_spec(sub_path="distill/crop_split"))


@pytest.mark.parametrize("container_field", ["initContainers", "ephemeralContainers"])
def test_scan_includes_init_and_ephemeral_containers(container_field):
    assert freeze.pod_spec_rw_overlaps(
        _pod_spec(container_field=container_field)
    )


def test_scan_treats_rw_volume_device_as_full_claim_overlap():
    pod_spec = {
        "volumes": [{
            "name": "data",
            "persistentVolumeClaim": {"claimName": freeze.PVC_CLAIM},
        }],
        "containers": [{
            "name": "worker",
            "volumeDevices": [{"name": "data", "devicePath": "/dev/source"}],
        }],
    }

    overlaps = freeze.pod_spec_rw_overlaps(pod_spec)

    assert overlaps[0]["access"] == "rw-volume-device"


def _apply_source_pod_spec(**overrides) -> dict:
    options = {
        "sub_path": "unified_v2_512",
        "container_name": "source-access-apply",
        "volume_name": "training-data-cephfs",
    }
    options.update(overrides)
    return _pod_spec(**options)


def _job(
    name: str,
    purpose: str,
    uid: str = "job-uid",
    *,
    pod_spec: dict | None = None,
) -> dict:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "uid": uid, "labels": {"purpose": purpose}},
        "spec": {"template": {"spec": pod_spec or _pod_spec()}},
        "status": {},
    }


def _pod(
    name: str,
    purpose: str,
    job_name: str,
    job_uid: str,
    *,
    pod_spec: dict | None = None,
) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "uid": "pod-uid",
            "labels": {"purpose": purpose},
            "ownerReferences": [{
                "kind": "Job",
                "name": job_name,
                "uid": job_uid,
                "controller": True,
            }],
        },
        "spec": pod_spec or _pod_spec(),
        "status": {"phase": "Running"},
    }


def test_only_exact_phase_job_and_uid_owned_pod_are_allowed():
    apply_spec = _apply_source_pod_spec()
    job = _job(
        "ladder-crop-source-access-apply",
        "ladder-crop-source-access-apply",
        pod_spec=copy.deepcopy(apply_spec),
    )
    pod = _pod(
        "apply-pod",
        "ladder-crop-source-access-apply",
        "ladder-crop-source-access-apply",
        "job-uid",
        pod_spec=copy.deepcopy(apply_spec),
    )
    impostor = _pod(
        "impostor",
        "ladder-crop-source-access-apply",
        "ladder-crop-source-access-apply",
        "wrong-job-uid",
        pod_spec=copy.deepcopy(apply_spec),
    )

    violations = freeze.find_rw_overlap_violations(
        [job, pod, impostor],
        phase="apply",
        held_controllers={},
    )

    assert [(item["kind"], item["name"]) for item in violations] == [
        ("Pod", "impostor")
    ]


def test_allowed_job_reference_must_be_the_controlling_owner():
    apply_spec = _apply_source_pod_spec()
    job = _job(
        "ladder-crop-source-access-apply",
        "ladder-crop-source-access-apply",
        pod_spec=copy.deepcopy(apply_spec),
    )
    pod = _pod(
        "non-controlling-ref",
        "ladder-crop-source-access-apply",
        "ladder-crop-source-access-apply",
        "job-uid",
        pod_spec=copy.deepcopy(apply_spec),
    )
    pod["metadata"]["ownerReferences"][0]["controller"] = False

    violations = freeze.find_rw_overlap_violations(
        [job, pod],
        phase="apply",
        held_controllers={},
    )

    assert [(item["kind"], item["name"]) for item in violations] == [
        ("Pod", "non-controlling-ref")
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"sub_path": None},
        {"sub_path": "unified_v2_512/child"},
        {"container_name": "wrong-apply-container"},
        {"volume_name": "wrong-pvc-volume"},
    ],
)
def test_apply_exemption_rejects_broader_or_wrong_overlap_shape(overrides):
    job = _job(
        "ladder-crop-source-access-apply",
        "ladder-crop-source-access-apply",
        pod_spec=_apply_source_pod_spec(**overrides),
    )

    violations = freeze.find_rw_overlap_violations(
        [job],
        phase="apply",
        held_controllers={},
    )

    assert [(item["kind"], item["name"]) for item in violations] == [
        ("Job", "ladder-crop-source-access-apply")
    ]


def test_full_pvc_apply_job_and_owned_pod_are_both_violations():
    full_pvc = _apply_source_pod_spec(sub_path=None)
    job = _job(
        "ladder-crop-source-access-apply",
        "ladder-crop-source-access-apply",
        pod_spec=copy.deepcopy(full_pvc),
    )
    pod = _pod(
        "apply-pod",
        "ladder-crop-source-access-apply",
        "ladder-crop-source-access-apply",
        "job-uid",
        pod_spec=copy.deepcopy(full_pvc),
    )

    violations = freeze.find_rw_overlap_violations(
        [job, pod],
        phase="apply",
        held_controllers={},
    )

    assert [(item["kind"], item["name"]) for item in violations] == [
        ("Job", "ladder-crop-source-access-apply"),
        ("Pod", "apply-pod"),
    ]


@pytest.mark.parametrize(
    ("phase", "job_name", "purpose"),
    [
        (
            "plan",
            "ladder-crop-source-access-plan",
            "ladder-crop-source-access-plan",
        ),
        ("split", "ladder-lucas-crop-split", "ladder-crop-distill"),
    ],
)
def test_plan_and_split_receive_no_rw_source_exemption(
    phase,
    job_name,
    purpose,
):
    full_pvc = _pod_spec(sub_path=None)
    job = _job(
        job_name,
        purpose,
        pod_spec=copy.deepcopy(full_pvc),
    )
    pod = _pod(
        f"{phase}-pod",
        purpose,
        job_name,
        "job-uid",
        pod_spec=copy.deepcopy(full_pvc),
    )

    violations = freeze.find_rw_overlap_violations(
        [job, pod],
        phase=phase,
        held_controllers={},
    )

    assert [(item["kind"], item["name"]) for item in violations] == [
        ("Job", job_name),
        ("Pod", f"{phase}-pod"),
    ]


def test_nonterminal_latent_job_is_blocked_but_terminal_job_is_ignored():
    latent = _job("latent", "other")
    latent["spec"]["suspend"] = True
    terminal = _job("finished", "other", uid="finished-uid")
    terminal["status"] = {
        "conditions": [{"type": "Complete", "status": "True"}]
    }

    violations = freeze.find_rw_overlap_violations(
        [latent, terminal],
        phase="idle",
        held_controllers={},
    )

    assert [item["name"] for item in violations] == ["latent"]


def test_write_once_snapshot_is_canonical_and_immutable(tmp_path):
    path = tmp_path / "snapshot.json"
    digest = freeze.write_once_json(path, {"z": 1, "a": 2})

    assert path.read_bytes() == b'{"a":2,"z":1}\n'
    assert len(digest) == 64
    with pytest.raises(FileExistsError):
        freeze.write_once_json(path, {"different": True})


def test_write_once_snapshot_fsync_failure_never_leaves_partial_final(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "snapshot.json"

    def fail_fsync(_fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(freeze.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="fsync failure"):
        freeze.write_once_json(path, {"complete": False})

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


class _FakeClient:
    namespace = freeze.NAMESPACE

    def __init__(self) -> None:
        self.calls = []
        self.objects = {
            name: {
                "apiVersion": "batch/v1",
                "kind": "CronJob",
                "metadata": {
                    "name": name,
                    "uid": f"uid-{name}",
                    "resourceVersion": "1",
                },
                "spec": {
                    **({"suspend": True} if name == "campaign-orchestrator" else {}),
                    "jobTemplate": {
                        "spec": {
                            "template": {
                                "spec": _pod_spec(sub_path="distill")
                            }
                        }
                    },
                },
            }
            for name in freeze.ALL_CONTROLLERS
        }
        self.configmap = None

    def get(self, resource, name=None):
        if resource == "cronjob":
            return self.objects[name]
        if resource == "configmap":
            if self.configmap is None:
                raise freeze.FreezeError("configmaps not found (NotFound)")
            return self.configmap
        raise AssertionError((resource, name))

    def replace(self, value):
        if value["kind"] == "CronJob":
            name = value["metadata"]["name"]
            self.calls.append(("replace-cronjob", name))
            assert value["metadata"]["resourceVersion"] == (
                self.objects[name]["metadata"]["resourceVersion"]
            )
            result = {**value, "metadata": dict(value["metadata"])}
            result["metadata"]["resourceVersion"] = str(
                int(value["metadata"]["resourceVersion"]) + 1
            )
            self.objects[name] = result
            return result
        assert value["kind"] == "ConfigMap"
        self.calls.append(("replace-configmap", value["metadata"]["name"]))
        assert self.configmap is not None
        assert value["metadata"]["resourceVersion"] == (
            self.configmap["metadata"]["resourceVersion"]
        )
        self.configmap = value
        return value

    def create(self, value):
        assert self.configmap is None
        self.calls.append(("create-configmap", value["metadata"]["name"]))
        self.configmap = {**value, "metadata": dict(value["metadata"])}
        self.configmap["metadata"]["resourceVersion"] = "1"
        return self.configmap

    def inventory(self):
        return list(self.objects.values())


def _request_clean_restore_stop(client: _FakeClient, run_dir: Path) -> None:
    freeze.write_once_json(
        run_dir / "stop-requested.json",
        {"schema": freeze.FREEZE_SCHEMA, "run_id": run_dir.name},
    )
    assert freeze.watch(
        client,
        run_dir=run_dir,
        interval_seconds=0.01,
    ) == 0


def test_hold_captures_prior_suspend_shape_and_cas_suspends(tmp_path):
    client = _FakeClient()

    run_dir = freeze.hold(
        client,
        state_dir=tmp_path,
        run_id="attempt-3",
    )

    before = freeze._read_json(run_dir / "controllers-before.json")
    held = freeze._read_json(run_dir / "controllers-held.json")
    before_by_name = {entry["name"]: entry for entry in before["controllers"]}
    held_by_name = {entry["name"]: entry for entry in held["controllers"]}
    assert before_by_name["ladder-queue"]["prior_suspend_present"] is False
    assert before_by_name["ladder-queue"]["prior_suspend"] is None
    assert before_by_name["campaign-orchestrator"]["prior_suspend"] is True
    for name in freeze.SUSPEND_CONTROLLERS:
        assert held_by_name[name]["uid"] == before_by_name[name]["uid"]
        assert held_by_name[name]["resource_version"] == "2"
        assert held_by_name[name]["object"]["spec"]["suspend"] is True
    assert client.configmap["data"]["lease.json"]
    assert client.calls.index(("create-configmap", freeze.LEASE_CONFIGMAP)) < (
        client.calls.index(("replace-cronjob", "ladder-queue"))
    )


def test_freeze_lease_bytes_are_accepted_by_source_access(tmp_path):
    lease = freeze._lease_record(
        run_id="producer-consumer-contract",
        phase="plan",
        sequence=7,
        snapshot_sha256="a" * 64,
        controller_snapshot_sha256="b" * 64,
        now_ns=1_000_000_000,
    )
    lease_path = tmp_path / "lease.json"
    lease_path.write_text(
        freeze._lease_configmap(lease)["data"]["lease.json"],
        encoding="utf-8",
    )

    accepted = access.require_fresh_freeze_lease(
        lease_path,
        expected_phase="plan",
        now_ns=2_000_000_000,
    )

    assert accepted == lease


def test_kubectl_retries_only_transient_reads(monkeypatch):
    responses = iter([
        SimpleNamespace(
            returncode=1,
            stdout=b"",
            stderr=b"read: connection reset by peer",
        ),
        SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"kind": "Pod"}).encode("utf-8"),
            stderr=b"",
        ),
    ])
    calls = []
    sleeps = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return next(responses)

    monkeypatch.setattr(freeze.subprocess, "run", fake_run)
    monkeypatch.setattr(freeze.time, "sleep", sleeps.append)

    result = freeze.Kubectl(context="icekube", namespace="ns").get(
        "pods,jobs"
    )

    assert result == {"kind": "Pod"}
    assert len(calls) == 2
    assert sleeps == [freeze.KUBECTL_READ_RETRY_BASE_SECONDS]


def test_kubectl_bounds_server_request_and_client_process(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b'{"kind":"Pod"}',
            stderr=b"",
        )

    monkeypatch.setattr(freeze.subprocess, "run", fake_run)

    freeze.Kubectl(context="icekube", namespace="ns").get("pods")

    assert calls[0][0] == [
        "kubectl",
        "--context",
        "icekube",
        f"--request-timeout={freeze.KUBECTL_REQUEST_TIMEOUT_SECONDS}s",
        "-n",
        "ns",
        "get",
        "pods",
        "-o",
        "json",
    ]
    assert calls[0][1]["timeout"] == freeze.KUBECTL_PROCESS_TIMEOUT_SECONDS


def test_kubectl_retries_one_timed_out_read(monkeypatch):
    calls = []
    sleeps = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b'{"kind":"Pod"}',
            stderr=b"",
        )

    monkeypatch.setattr(freeze.subprocess, "run", fake_run)
    monkeypatch.setattr(freeze.time, "sleep", sleeps.append)

    result = freeze.Kubectl(context="icekube", namespace="ns").get("pods")

    assert result == {"kind": "Pod"}
    assert len(calls) == 2
    assert sleeps == [freeze.KUBECTL_READ_RETRY_BASE_SECONDS]


@pytest.mark.parametrize(
    "stderr",
    [
        b"Error from server (Timeout): context deadline exceeded",
        b"request did not complete within requested timeout",
    ],
)
def test_kubectl_retries_api_request_timeout_for_reads(monkeypatch, stderr):
    responses = iter([
        subprocess.CompletedProcess([], 1, stdout=b"", stderr=stderr),
        subprocess.CompletedProcess(
            [],
            0,
            stdout=b'{"kind":"Pod"}',
            stderr=b"",
        ),
    ])
    calls = []
    sleeps = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return next(responses)

    monkeypatch.setattr(freeze.subprocess, "run", fake_run)
    monkeypatch.setattr(freeze.time, "sleep", sleeps.append)

    result = freeze.Kubectl(context="icekube", namespace="ns").get("pods")

    assert result == {"kind": "Pod"}
    assert len(calls) == 2
    assert sleeps == [freeze.KUBECTL_READ_RETRY_BASE_SECONDS]


def test_kubectl_exhausted_timed_out_reads_fail_closed(monkeypatch):
    calls = []
    sleeps = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(freeze.subprocess, "run", fake_run)
    monkeypatch.setattr(freeze.time, "sleep", sleeps.append)

    with pytest.raises(freeze.FreezeError, match="timed out after 12s"):
        freeze.Kubectl(context="icekube", namespace="ns").get("pods")

    assert len(calls) == freeze.KUBECTL_READ_ATTEMPTS
    assert sleeps == [freeze.KUBECTL_READ_RETRY_BASE_SECONDS]


def test_kubectl_permanent_read_failure_stops_without_retry(monkeypatch):
    calls = []
    sleeps = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=b"",
            stderr=b'Error from server (Forbidden): pods is forbidden',
        )

    monkeypatch.setattr(freeze.subprocess, "run", fake_run)
    monkeypatch.setattr(freeze.time, "sleep", sleeps.append)

    with pytest.raises(freeze.FreezeError, match="Forbidden"):
        freeze.Kubectl(context="icekube", namespace="ns").get("pods")

    assert len(calls) == 1
    assert sleeps == []


def test_kubectl_exhausted_transient_reads_fail_closed(monkeypatch):
    calls = []
    sleeps = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=b"",
            stderr=b"read: connection reset by peer",
        )

    monkeypatch.setattr(freeze, "KUBECTL_READ_ATTEMPTS", 3)
    monkeypatch.setattr(freeze.subprocess, "run", fake_run)
    monkeypatch.setattr(freeze.time, "sleep", sleeps.append)

    with pytest.raises(freeze.FreezeError, match="connection reset"):
        freeze.Kubectl(context="icekube", namespace="ns").get("pods")

    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_kubectl_writes_disable_client_openapi_validation(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b'{"kind":"ConfigMap"}',
            stderr=b"",
        )

    monkeypatch.setattr(freeze.subprocess, "run", fake_run)
    client = freeze.Kubectl(context="icekube", namespace="ns")

    client.create({"kind": "ConfigMap"})
    client.replace({"kind": "ConfigMap"})

    create_index = calls[0][0].index("create")
    replace_index = calls[1][0].index("replace")
    assert calls[0][0][create_index : create_index + 2] == [
        "create",
        "--validate=false",
    ]
    assert calls[1][0][replace_index : replace_index + 2] == [
        "replace",
        "--validate=false",
    ]


@pytest.mark.parametrize("operation", ["create", "replace"])
def test_kubectl_never_retries_ambiguous_write_failure(
    monkeypatch,
    operation,
):
    calls = []
    sleeps = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=b"",
            stderr=b"read: connection reset by peer",
        )

    monkeypatch.setattr(freeze.subprocess, "run", fake_run)
    monkeypatch.setattr(freeze.time, "sleep", sleeps.append)
    client = freeze.Kubectl(context="icekube", namespace="ns")

    with pytest.raises(freeze.FreezeError, match="connection reset"):
        getattr(client, operation)({"kind": "ConfigMap"})

    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.parametrize("operation", ["create", "replace"])
def test_kubectl_never_retries_timed_out_write(monkeypatch, operation):
    calls = []
    sleeps = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(freeze.subprocess, "run", fake_run)
    monkeypatch.setattr(freeze.time, "sleep", sleeps.append)
    client = freeze.Kubectl(context="icekube", namespace="ns")

    with pytest.raises(freeze.FreezeError, match="timed out after 12s"):
        getattr(client, operation)({"kind": "ConfigMap"})

    assert len(calls) == 1
    assert sleeps == []


def test_watchdog_external_io_budget_fits_lease():
    expected = (
        freeze.WATCHDOG_READS_PER_HEARTBEAT
        * (
            freeze.KUBECTL_READ_ATTEMPTS
            * freeze.KUBECTL_PROCESS_TIMEOUT_SECONDS
            + freeze.KUBECTL_READ_RETRY_BASE_SECONDS
        )
        + freeze.WATCHDOG_WRITES_PER_HEARTBEAT
        * freeze.KUBECTL_PROCESS_TIMEOUT_SECONDS
        + freeze.KUBECTL_PROCESS_TIMEOUT_SECONDS
        + freeze.DEFAULT_INTERVAL_SECONDS
    )

    assert freeze.WATCHDOG_READS_PER_HEARTBEAT == 5
    assert freeze.WATCHDOG_WRITES_PER_HEARTBEAT == 1
    assert freeze.WATCHDOG_PUBLICATION_SAFETY_SECONDS == 4
    assert freeze.WATCHDOG_EXTERNAL_IO_BUDGET_SECONDS == expected == 164
    assert freeze.WATCHDOG_EXTERNAL_IO_BUDGET_SECONDS < freeze.LEASE_SECONDS


def test_watchdog_heartbeat_budget_matches_actual_operations(tmp_path):
    class CountingClient(_FakeClient):
        def __init__(self):
            super().__init__()
            self.reads = 0
            self.writes = 0

        def get(self, resource, name=None):
            self.reads += 1
            return super().get(resource, name)

        def inventory(self):
            self.reads += 1
            return super().inventory()

        def replace(self, value):
            self.writes += 1
            return super().replace(value)

    client = CountingClient()
    run_dir = freeze.hold(client, state_dir=tmp_path, run_id="count-heartbeat")
    client.reads = 0
    client.writes = 0

    assert freeze.watch(
        client,
        run_dir=run_dir,
        interval_seconds=freeze.DEFAULT_INTERVAL_SECONDS,
        once=True,
    ) == 0

    assert client.reads == freeze.WATCHDOG_READS_PER_HEARTBEAT
    assert client.writes == freeze.WATCHDOG_WRITES_PER_HEARTBEAT


def test_held_lease_timestamp_is_sampled_after_cas_read(tmp_path, monkeypatch):
    client = _FakeClient()
    freeze.hold(client, state_dir=tmp_path, run_id="late-timestamp")
    clock = [1_000_000_000]
    real_get = client.get

    def advancing_get(resource, name=None):
        value = real_get(resource, name)
        clock[0] = 2_000_000_000
        return value

    client.get = advancing_get
    monkeypatch.setattr(freeze.time, "time_ns", lambda: clock[0])
    candidate = freeze._lease_record(
        run_id="late-timestamp",
        phase="idle",
        sequence=1,
        snapshot_sha256="a" * 64,
        controller_snapshot_sha256="b" * 64,
    )

    published = freeze._publish_held_lease(
        client,
        candidate,
        require_fresh_prior=True,
    )

    live = json.loads(client.configmap["data"]["lease.json"])
    assert published["heartbeat_unix_ns"] == 2_000_000_000
    assert live == published
    assert live["valid_until_unix_ns"] == (
        2_000_000_000 + freeze.LEASE_SECONDS * 1_000_000_000
    )


def test_exhausted_scan_budget_fails_closed_without_held_heartbeat(tmp_path):
    client = _FakeClient()
    run_dir = freeze.hold(client, state_dir=tmp_path, run_id="stale-scan")
    prior = json.loads(client.configmap["data"]["lease.json"])
    prior["valid_until_unix_ns"] = time.time_ns() + (
        freeze.KUBECTL_PROCESS_TIMEOUT_SECONDS
        + freeze.WATCHDOG_PUBLICATION_SAFETY_SECONDS
        - 1
    ) * 1_000_000_000
    client.configmap["data"]["lease.json"] = (
        freeze.workload_canonical_json_bytes(prior).decode("utf-8")
    )

    with pytest.raises(freeze.FreezeError, match="exhausted"):
        freeze.watch(
            client,
            run_dir=run_dir,
            interval_seconds=freeze.DEFAULT_INTERVAL_SECONDS,
        )

    live = json.loads(client.configmap["data"]["lease.json"])
    assert live["status"] == "failed"
    assert live["valid_until_unix_ns"] == live["heartbeat_unix_ns"]
    assert not (run_dir / "heartbeat.json").exists()


@pytest.mark.parametrize("interval", [0, -1, 15.000001, float("inf"), float("nan")])
def test_watchdog_rejects_interval_outside_budget(tmp_path, interval):
    with pytest.raises(freeze.FreezeError, match="at most 15 seconds"):
        freeze.watch(
            _FakeClient(),
            run_dir=tmp_path / "unused",
            interval_seconds=interval,
            once=True,
        )


def test_restore_cannot_enter_while_live_hold_is_initializing(tmp_path):
    class MidHoldRestoreClient(_FakeClient):
        attempted_restore = False

        def replace(self, value):
            if (
                value["kind"] == "CronJob"
                and value["metadata"]["name"] == "ladder-queue"
                and not self.attempted_restore
            ):
                self.attempted_restore = True
                run_dir = tmp_path / "mid-hold"
                assert freeze._live_lease(
                    self,
                    run_id="mid-hold",
                )["status"] == "initializing"
                before_specs = {
                    name: copy.deepcopy(item["spec"])
                    for name, item in self.objects.items()
                }
                with pytest.raises(freeze.CoordinationBusy, match="another hold"):
                    freeze.restore(
                        self,
                        run_dir=run_dir,
                        timeout_seconds=0.01,
                    )
                assert not (run_dir / "restore-in-progress.json").exists()
                assert {
                    name: item["spec"] for name, item in self.objects.items()
                } == before_specs
            return super().replace(value)

    client = MidHoldRestoreClient()

    freeze.hold(client, state_dir=tmp_path, run_id="mid-hold")

    assert client.attempted_restore is True
    for name in freeze.SUSPEND_CONTROLLERS:
        assert client.objects[name]["spec"]["suspend"] is True


def test_hold_rejects_observed_controller_spec_race(tmp_path):
    class CampaignRaceClient(_FakeClient):
        def __init__(self):
            super().__init__()
            self.campaign_reads = 0

        def get(self, resource, name=None):
            if resource == "cronjob" and name == "campaign-orchestrator":
                self.campaign_reads += 1
                result = copy.deepcopy(self.objects[name])
                if self.campaign_reads == 2:
                    result["spec"]["unexpected"] = True
                return result
            return super().get(resource, name)

    client = CampaignRaceClient()

    with pytest.raises(freeze.FreezeError, match="beyond exact suspension"):
        freeze.hold(client, state_dir=tmp_path, run_id="campaign-race")

    assert freeze._live_lease(client, run_id="campaign-race")["status"] == "failed"


def test_partial_hold_is_fail_closed_and_exactly_recoverable(tmp_path):
    class PartialHoldClient(_FakeClient):
        def replace(self, value):
            if (
                value["kind"] == "CronJob"
                and value["metadata"]["name"] == "gpu-reaper"
                and value["spec"].get("suspend") is True
            ):
                raise freeze.FreezeError("simulated second suspension CAS failure")
            return super().replace(value)

    client = PartialHoldClient()
    with pytest.raises(freeze.FreezeError, match="second suspension"):
        freeze.hold(client, state_dir=tmp_path, run_id="partial-hold")

    failed_lease = freeze._live_lease(client, run_id="partial-hold")
    assert failed_lease["status"] == "failed"
    assert client.objects["ladder-queue"]["spec"]["suspend"] is True
    assert "suspend" not in client.objects["gpu-reaper"]["spec"]

    freeze.restore(
        client,
        run_dir=tmp_path / "partial-hold",
        timeout_seconds=0.01,
    )

    for name in freeze.SUSPEND_CONTROLLERS:
        assert "suspend" not in client.objects[name]["spec"]
    assert freeze._live_lease(client, run_id="partial-hold")["status"] == "released"
    with pytest.raises(freeze.FreezeError, match="restore has begun"):
        freeze.watch(
            client,
            run_dir=tmp_path / "partial-hold",
            interval_seconds=0.01,
            once=True,
        )


def test_held_lease_before_hold_complete_is_exactly_recoverable(tmp_path):
    client = _FakeClient()
    run_dir = freeze.hold(client, state_dir=tmp_path, run_id="held-window")
    (run_dir / "hold-complete.json").unlink()

    assert freeze._live_lease(client, run_id=run_dir.name)["status"] == "held"
    freeze.restore(client, run_dir=run_dir, timeout_seconds=0.01)

    for name in freeze.SUSPEND_CONTROLLERS:
        assert "suspend" not in client.objects[name]["spec"]
    assert freeze._live_lease(client, run_id=run_dir.name)["status"] == "released"


def test_restore_resumes_after_partial_controller_cas(tmp_path):
    client = _FakeClient()
    run_dir = freeze.hold(client, state_dir=tmp_path, run_id="restore-resume")
    _request_clean_restore_stop(client, run_dir)

    real_replace = client.replace
    failed = False

    def fail_second_restore(value):
        nonlocal failed
        if (
            not failed
            and value["kind"] == "CronJob"
            and value["metadata"]["name"] == "gpu-reaper"
            and "suspend" not in value["spec"]
        ):
            failed = True
            raise freeze.FreezeError("simulated restore CAS failure")
        return real_replace(value)

    client.replace = fail_second_restore
    with pytest.raises(freeze.FreezeError, match="restore CAS failure"):
        freeze.restore(client, run_dir=run_dir, timeout_seconds=0.01)
    assert "suspend" not in client.objects["ladder-queue"]["spec"]
    assert client.objects["gpu-reaper"]["spec"]["suspend"] is True

    freeze.restore(client, run_dir=run_dir, timeout_seconds=0.01)

    for name in freeze.SUSPEND_CONTROLLERS:
        assert "suspend" not in client.objects[name]["spec"]
    assert freeze._live_lease(client, run_id=run_dir.name)["status"] == "released"


def test_restore_final_reread_blocks_post_cas_controller_drift(tmp_path):
    client = _FakeClient()
    run_dir = freeze.hold(client, state_dir=tmp_path, run_id="restore-drift")
    _request_clean_restore_stop(client, run_dir)
    real_replace = client.replace

    def drift_after_last_restore(value):
        result = real_replace(value)
        if (
            value["kind"] == "CronJob"
            and value["metadata"]["name"] == "gpu-reaper"
            and "suspend" not in value["spec"]
        ):
            client.objects["ladder-queue"]["spec"]["unexpected"] = True
        return result

    client.replace = drift_after_last_restore

    with pytest.raises(freeze.FreezeError, match="drifted after exact restore"):
        freeze.restore(client, run_dir=run_dir, timeout_seconds=0.01)

    assert freeze._live_lease(client, run_id=run_dir.name)["status"] == "closed"
    assert not (run_dir / "controllers-restored.json").exists()


def test_restore_preserves_explicit_false_suspend_state(tmp_path):
    client = _FakeClient()
    client.objects["gpu-reaper"]["spec"]["suspend"] = False
    run_dir = freeze.hold(client, state_dir=tmp_path, run_id="explicit-false")
    _request_clean_restore_stop(client, run_dir)

    freeze.restore(client, run_dir=run_dir, timeout_seconds=0.01)

    assert client.objects["gpu-reaper"]["spec"]["suspend"] is False


def test_watch_once_closes_lease_without_granting_restore_authority(tmp_path):
    client = _FakeClient()
    run_dir = freeze.hold(client, state_dir=tmp_path, run_id="once")
    apply_spec = _apply_source_pod_spec()
    allowed_job = _job(
        "ladder-crop-source-access-apply",
        "ladder-crop-source-access-apply",
        pod_spec=copy.deepcopy(apply_spec),
    )
    allowed_pod = _pod(
        "apply-pod",
        "ladder-crop-source-access-apply",
        "ladder-crop-source-access-apply",
        "job-uid",
        pod_spec=copy.deepcopy(apply_spec),
    )
    client.inventory = lambda: [*client.objects.values(), allowed_job, allowed_pod]
    freeze.replace_json(
        run_dir / "phase.json",
        freeze._new_phase_state(run_id=run_dir.name, phase="apply"),
    )

    assert freeze.watch(
        client,
        run_dir=run_dir,
        interval_seconds=0.01,
        once=True,
    ) == 0

    assert freeze._live_lease(client, run_id=run_dir.name)["status"] == "closed"
    assert not (run_dir / "watchdog-stopped.json").exists()
    assert list((run_dir / "watchdog-interrupted").glob("*.json"))

    with pytest.raises(freeze.FreezeError, match="did not publish"):
        freeze.restore(client, run_dir=run_dir, timeout_seconds=0.01)
    for name in freeze.SUSPEND_CONTROLLERS:
        assert client.objects[name]["spec"]["suspend"] is True


def test_watchdog_owner_is_singleton(tmp_path):
    client = _FakeClient()
    run_dir = freeze.hold(client, state_dir=tmp_path, run_id="singleton")

    with freeze._watchdog_owner(run_dir):
        with pytest.raises(freeze.FreezeError, match="another watchdog"):
            freeze.watch(
                client,
                run_dir=run_dir,
                interval_seconds=0.01,
                once=True,
            )


def test_phase_gate_is_singleton_and_refuses_shutdown(tmp_path):
    client = _FakeClient()
    run_dir = freeze.hold(client, state_dir=tmp_path, run_id="phase-singleton")

    with freeze._phase_owner(run_dir):
        with pytest.raises(freeze.FreezeError, match="another phase gate"):
            freeze.gate_phase(
                client,
                run_dir=run_dir,
                phase="apply",
                timeout_seconds=0.01,
            )

    before = (run_dir / "phase.json").read_bytes()
    freeze.write_once_json(
        run_dir / "stop-requested.json",
        {"schema": freeze.FREEZE_SCHEMA, "run_id": run_dir.name},
    )
    with pytest.raises(freeze.FreezeError, match="shutdown has begun"):
        freeze.gate_phase(
            client,
            run_dir=run_dir,
            phase="apply",
            timeout_seconds=0.01,
        )
    assert (run_dir / "phase.json").read_bytes() == before


def test_restore_holds_phase_owner_for_its_entire_operation(tmp_path, monkeypatch):
    run_dir = tmp_path / "restore-owner"
    run_dir.mkdir()
    client = _FakeClient()

    def assert_gate_excluded(_client, *, run_dir, timeout_seconds):
        del timeout_seconds
        with pytest.raises(freeze.FreezeError, match="another phase gate"):
            freeze.gate_phase(
                _client,
                run_dir=run_dir,
                phase="apply",
                timeout_seconds=0.01,
            )

    monkeypatch.setattr(freeze, "_restore_owned", assert_gate_excluded)

    freeze.restore(client, run_dir=run_dir, timeout_seconds=0.01)


def test_expired_phase_request_fails_closed(tmp_path):
    client = _FakeClient()
    run_dir = freeze.hold(client, state_dir=tmp_path, run_id="expired-phase")
    expired = freeze._new_phase_state(run_id=run_dir.name, phase="apply")
    expired["valid_until_unix_ns"] = time.time_ns() - 1
    freeze.replace_json(run_dir / "phase.json", expired)

    with pytest.raises(freeze.FreezeError, match="phase request expired"):
        freeze.watch(
            client,
            run_dir=run_dir,
            interval_seconds=0.01,
            once=True,
        )

    assert freeze._live_lease(client, run_id=run_dir.name)["status"] == "failed"
    stopped = freeze._read_json(run_dir / "watchdog-stopped.json")
    assert stopped["reason"] == "phase-expired-clean-idle-scan"

    freeze.restore(client, run_dir=run_dir, timeout_seconds=0.01)

    assert "suspend" not in client.objects["ladder-queue"]["spec"]
    assert "suspend" not in client.objects["gpu-reaper"]["spec"]
    assert client.objects["campaign-orchestrator"]["spec"]["suspend"] is True
    assert freeze._live_lease(client, run_id=run_dir.name)["status"] == "released"


def test_expiry_terminal_commit_rechecks_a_concurrent_gate_nonce(
    tmp_path,
    monkeypatch,
):
    client = _FakeClient()
    run_dir = freeze.hold(client, state_dir=tmp_path, run_id="expiry-gate-race")
    expired = freeze._new_phase_state(run_id=run_dir.name, phase="apply")
    expired["valid_until_unix_ns"] = time.time_ns() - 1
    freeze.replace_json(run_dir / "phase.json", expired)
    replacement = freeze._new_phase_state(run_id=run_dir.name, phase="apply")
    real_phase_owner = freeze._phase_owner
    raced = False

    @contextmanager
    def gate_wins_before_terminal_commit(candidate_run_dir):
        nonlocal raced
        if not raced:
            raced = True
            freeze.replace_json(candidate_run_dir / "phase.json", replacement)
        with real_phase_owner(candidate_run_dir):
            yield

    monkeypatch.setattr(freeze, "_phase_owner", gate_wins_before_terminal_commit)

    assert freeze.watch(
        client,
        run_dir=run_dir,
        interval_seconds=0.01,
        once=True,
    ) == 0

    assert raced is True
    assert freeze._read_json(run_dir / "phase.json") == replacement
    assert not (run_dir / "watchdog-stopped.json").exists()
    assert not (run_dir / "heartbeat.json").exists()
    lease = freeze._live_lease(client, run_id=run_dir.name)
    assert lease["status"] == "closed"
    assert lease["phase"] == "apply"


def test_watch_rejects_tampered_immutable_hold_authority(tmp_path):
    client = _FakeClient()
    run_dir = freeze.hold(client, state_dir=tmp_path, run_id="tampered")
    held_path = run_dir / "controllers-held.json"
    held = freeze._read_json(held_path)
    held["tampered"] = True
    freeze.replace_json(held_path, held)

    with pytest.raises(freeze.FreezeError, match="authority hash mismatch"):
        freeze.watch(
            client,
            run_dir=run_dir,
            interval_seconds=0.01,
            once=True,
        )


def test_live_lease_rejects_truncated_schema(tmp_path):
    client = _FakeClient()
    run_dir = freeze.hold(client, state_dir=tmp_path, run_id="truncated-lease")
    client.configmap["data"]["lease.json"] = "{}"

    with pytest.raises(freeze.FreezeError, match="schema is malformed"):
        freeze._live_lease(client, run_id=run_dir.name)


def test_gate_refuses_when_watchdog_never_publishes_heartbeat(tmp_path):
    client = _FakeClient()
    run_dir = tmp_path / "attempt-3"
    run_dir.mkdir()
    initial_phase = freeze._new_phase_state(
        run_id="attempt-3",
        phase="idle",
    )
    freeze.replace_json(run_dir / "phase.json", initial_phase)

    with pytest.raises(freeze.FreezeError, match="did not publish"):
        freeze.gate_phase(
            client,
            run_dir=run_dir,
            phase="apply",
            timeout_seconds=0.01,
        )
    assert freeze._read_json(run_dir / "phase.json") == initial_phase


def test_gate_timeout_expires_a_lease_published_before_rollback(
    tmp_path,
    monkeypatch,
):
    client = _FakeClient()
    run_dir = freeze.hold(client, state_dir=tmp_path, run_id="late-publish")
    initial_phase = freeze._read_json(run_dir / "phase.json")
    real_publication_owner = freeze._phase_publication_owner
    acquisitions = 0

    @contextmanager
    def publish_just_before_rollback(candidate_run_dir):
        nonlocal acquisitions
        acquisitions += 1
        with real_publication_owner(candidate_run_dir):
            if acquisitions == 2:
                requested = freeze._read_json(candidate_run_dir / "phase.json")
                late = freeze._lease_record(
                    run_id=candidate_run_dir.name,
                    phase=str(requested["phase"]),
                    sequence=9,
                    snapshot_sha256="a" * 64,
                    controller_snapshot_sha256="b" * 64,
                )
                freeze._replace_lease(client, late)
            yield

    monkeypatch.setattr(
        freeze,
        "_phase_publication_owner",
        publish_just_before_rollback,
    )

    with pytest.raises(freeze.FreezeError, match="did not publish"):
        freeze.gate_phase(
            client,
            run_dir=run_dir,
            phase="apply",
            timeout_seconds=0,
        )

    assert acquisitions == 2
    assert freeze._read_json(run_dir / "phase.json") == initial_phase
    lease = freeze._live_lease(client, run_id=run_dir.name)
    assert lease["status"] == "failed"
    assert lease["valid_until_unix_ns"] == lease["heartbeat_unix_ns"]
