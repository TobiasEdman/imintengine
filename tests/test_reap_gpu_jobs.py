"""Evidence-before-deletion gates for the GPU reaper.

The reaper destroyed its own diagnostic trail twice (ERA5 control 2026-08-27,
wave-1 retrain 2026-08-29): deleting a finished Job cascades to its pods and
their logs. These tests pin the contract that makes that impossible:

1. No deletion ever happens without a successfully written evidence archive.
2. A successful sweep leaves the job spec, pod specs, pod logs and events on
   disk, plus a run report that survives the reaper pod's own TTL.
3. Dry-run mode touches nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.reap_gpu_jobs as reap

DUE = [{"job": "train-x", "job_uid": "uid-train-x", "pod": "train-x-abc12", "phase": "Failed",
        "gpus": 1, "age_min": 120.0}]


def _fake_kubectl_factory(calls: list[list[str]], *, fail_on_logs: bool = False):
    deleted = False

    def fake(args: list[str], context: str, namespace: str) -> str:
        nonlocal deleted
        calls.append(args)
        if args[:2] == ["get", "job"]:
            if deleted and "--ignore-not-found=true" in args:
                return ""
            if args[-1] == "json":
                return json.dumps({
                    "metadata": {
                        "name": "train-x",
                        "uid": "uid-train-x",
                        "labels": {"purpose": "test"},
                    },
                    "status": {
                        "failed": 1,
                        "conditions": [{"type": "Failed", "status": "True"}],
                    },
                })
            return "kind: Job\nmetadata:\n  name: train-x\n"
        if args[:2] == ["get", "jobs"]:
            if args[-1] == "yaml":
                return "kind: Job\nmetadata:\n  name: train-x\n"
            if deleted:
                return json.dumps({"items": []})
            return json.dumps({"items": [{
                "metadata": {"name": "train-x", "uid": "uid-train-x"},
                "status": {
                    "failed": 1,
                    "conditions": [{"type": "Failed", "status": "True"}],
                },
            }]})
        if args[:2] == ["get", "pods"]:
            return json.dumps({"items": [{"metadata": {
                "name": "train-x-abc12",
                "uid": "uid-pod-train-x",
                "ownerReferences": [{
                    "kind": "Job",
                    "name": "train-x",
                    "uid": "uid-train-x",
                }],
            }}]})
        if args[0] == "logs":
            if "--previous" in args:
                raise RuntimeError("previous terminated container not found")
            if fail_on_logs:
                raise RuntimeError("simulated: pods/log forbidden")
            return "Traceback: the actual failure evidence\n"
        if args[:2] == ["get", "events"]:
            return "LAST SEEN   REASON   MESSAGE\n"
        if args[0] == "delete":
            deleted = True
            return "job.batch/train-x deleted\n"
        if args[0] == "patch":
            return "job.batch/train-x patched\n"
        raise AssertionError(f"unexpected kubectl call: {args}")
    return fake


def _run_main(monkeypatch, tmp_path: Path, *, apply: bool, fail_on_logs: bool = False):
    calls: list[list[str]] = []
    monkeypatch.setattr(reap, "_kubectl", _fake_kubectl_factory(calls, fail_on_logs=fail_on_logs))
    monkeypatch.setattr(reap, "collect", lambda *a: (list(DUE), [], 1))
    argv = ["reap_gpu_jobs.py", "--grace-minutes", "60",
            "--archive-dir", str(tmp_path / "archive")]
    if apply:
        argv.append("--apply")
    monkeypatch.setattr("sys.argv", argv)
    assert reap.main() == 0
    return calls


def _deletes(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if c and c[0] == "delete"]


def test_archive_failure_forbids_delete(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, apply=True, fail_on_logs=True)
    assert _deletes(calls) == [], "deleted a job whose evidence could not be archived"


def test_unwritable_archive_root_forbids_delete(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(reap, "_kubectl", _fake_kubectl_factory(calls))
    monkeypatch.setattr(reap, "collect", lambda *a: (list(DUE), [], 1))
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, not a directory — mkdir must fail")
    monkeypatch.setattr("sys.argv", ["reap_gpu_jobs.py", "--apply",
                                     "--archive-dir", str(blocked)])
    assert reap.main() == 0
    assert _deletes(calls) == []


def test_successful_archive_then_delete_and_run_report(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, apply=True)
    assert _deletes(calls) == [[
        "delete", "jobs", "-l",
        f"{reap.REAPER_CLAIM_LABEL}=uid-train-x", "--wait=false",
    ]]

    root = tmp_path / "archive"
    job_dirs = [d for d in root.iterdir() if d.name.startswith("train-x-")]
    assert len(job_dirs) == 1
    d = job_dirs[0]
    assert (d / "job.yaml").read_text().startswith("kind: Job")
    assert json.loads((d / "pods.json").read_text())[0]["metadata"]["name"] == "train-x-abc12"
    assert "actual failure evidence" in (d / "train-x-abc12.log").read_text()
    assert (d / "events.txt").exists()
    # --previous had no restarted container: absent is correct, not an error.
    assert not (d / "train-x-abc12.previous.log").exists()

    reports = list((root / "runs").iterdir())
    assert len(reports) == 1
    assert "deleted train-x" in reports[0].read_text()


def test_job_scope_excludes_other_due_jobs(monkeypatch, tmp_path):
    """--job restricts deletion to exactly the named Jobs."""
    calls: list[list[str]] = []
    monkeypatch.setattr(reap, "_kubectl", _fake_kubectl_factory(calls))
    other = {"job": "train-other", "pod": "train-other-zzz99", "phase": "Failed",
             "job_uid": "uid-train-other", "gpus": 1, "age_min": 500.0}
    monkeypatch.setattr(reap, "collect", lambda *a: ([dict(DUE[0]), other], [], 2))
    monkeypatch.setattr("sys.argv", ["reap_gpu_jobs.py", "--apply",
                                     "--grace-minutes", "0",
                                     "--job", "train-x",
                                     "--archive-dir", str(tmp_path / "archive")])
    assert reap.main() == 0
    assert _deletes(calls) == [[
        "delete", "jobs", "-l",
        f"{reap.REAPER_CLAIM_LABEL}=uid-train-x", "--wait=false",
    ]]


def test_dry_run_touches_nothing(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, apply=False)
    assert _deletes(calls) == []
    assert not [c for c in calls if c[0] == "logs"], "dry run must not read logs"
    assert not (tmp_path / "archive").exists()


def test_collect_binds_job_uid_age_and_deduplicates_pods(monkeypatch):
    job_stamp = "2026-09-08T12:00:00Z"
    pod_stamp = "2026-09-08T13:00:00Z"

    def fake(args: list[str], context: str, namespace: str) -> str:
        if args[:2] == ["get", "jobs"]:
            return json.dumps({"items": [{
                "metadata": {
                    "name": "train-x",
                    "uid": "uid-train-x",
                    "creationTimestamp": job_stamp,
                },
                "status": {
                    "failed": 1,
                    "conditions": [{"type": "Failed", "status": "True"}],
                },
            }]})
        if args[:2] == ["get", "pods"]:
            return json.dumps({"items": [
                {
                    "metadata": {
                        "name": f"train-x-{suffix}",
                        "creationTimestamp": pod_stamp,
                        "ownerReferences": [{
                            "kind": "Job",
                            "name": "train-x",
                            "uid": "uid-train-x",
                        }],
                    },
                    "spec": {"containers": [{
                        "resources": {"limits": {"nvidia.com/gpu": 1}},
                    }]},
                    "status": {"phase": "Failed"},
                }
                for suffix in ("aaa", "bbb")
            ]})
        raise AssertionError(args)

    seen_stamps: list[str] = []
    monkeypatch.setattr(reap, "_kubectl", fake)
    monkeypatch.setattr(
        reap,
        "_age_minutes",
        lambda stamp: seen_stamps.append(stamp) or 120.0,
    )
    records, stuck, held = reap.collect("icekube", "ns", None)
    assert stuck == []
    assert held == 2
    assert len(records) == 1
    assert records[0]["job_uid"] == "uid-train-x"
    assert records[0]["gpus"] == 2
    assert seen_stamps == [job_stamp]


def test_claim_refuses_same_name_replacement(monkeypatch):
    calls: list[list[str]] = []

    def fake(args: list[str], context: str, namespace: str) -> str:
        calls.append(args)
        return json.dumps({
            "metadata": {"name": "train-x", "uid": "replacement-uid"},
            "status": {"active": 1},
        })

    monkeypatch.setattr(reap, "_kubectl", fake)
    with pytest.raises(RuntimeError, match="UID changed"):
        reap._claim_terminal_job("train-x", "uid-train-x", "icekube", "ns")
    assert not [call for call in calls if call[0] in {"patch", "delete"}]


def test_claim_refuses_active_retry_with_failed_pod_counter(monkeypatch):
    calls: list[list[str]] = []

    def fake(args: list[str], context: str, namespace: str) -> str:
        calls.append(args)
        return json.dumps({
            "metadata": {"name": "train-x", "uid": "uid-train-x"},
            "status": {"active": 1, "failed": 1},
        })

    monkeypatch.setattr(reap, "_kubectl", fake)
    with pytest.raises(RuntimeError, match="no longer terminal"):
        reap._claim_terminal_job("train-x", "uid-train-x", "icekube", "ns")
    assert not [call for call in calls if call[0] in {"patch", "delete"}]


def test_successful_job_complete_condition_is_terminal():
    assert reap._job_is_terminal({
        "status": {
            "succeeded": 1,
            "conditions": [{"type": "Complete", "status": "True"}],
        }
    }) is True


def test_claim_uses_uid_test_and_selector_delete(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, apply=True)
    patch = next(call for call in calls if call[0] == "patch")
    operations = json.loads(patch[patch.index("-p") + 1])
    assert operations[0] == {
        "op": "test",
        "path": "/metadata/uid",
        "value": "uid-train-x",
    }
    assert _deletes(calls)[0][2:] == [
        "-l", f"{reap.REAPER_CLAIM_LABEL}=uid-train-x", "--wait=false",
    ]


def test_delete_requires_claimed_uid_to_enter_deletion(monkeypatch):
    calls: list[list[str]] = []

    def fake(args: list[str], context: str, namespace: str) -> str:
        calls.append(args)
        if args[0] == "delete":
            return "no resources found\n"
        if args[:2] == ["get", "job"]:
            return json.dumps({
                "metadata": {"name": "train-x", "uid": "uid-train-x"},
                "status": {
                    "failed": 1,
                    "conditions": [{"type": "Failed", "status": "True"}],
                },
            })
        raise AssertionError(args)

    monkeypatch.setattr(reap, "_kubectl", fake)
    with pytest.raises(RuntimeError, match="deletion was not accepted"):
        reap._delete_claimed_job(
            f"{reap.REAPER_CLAIM_LABEL}=uid-train-x",
            "train-x",
            "uid-train-x",
            "icekube",
            "ns",
        )


def test_hold_selector_preserves_matching_job(monkeypatch, tmp_path, capsys):
    calls: list[list[str]] = []
    monkeypatch.setattr(reap, "_kubectl", _fake_kubectl_factory(calls))
    monkeypatch.setattr(reap, "collect", lambda *a: (list(DUE), [], 1))
    monkeypatch.setattr(
        reap,
        "_matching_job_uids",
        lambda *a: {"uid-train-x"},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "reap_gpu_jobs.py",
            "--apply",
            "--grace-minutes",
            "60",
            "--hold-selector",
            "purpose=ladder-crop-distill",
            "--hold-minutes",
            "1440",
            "--archive-dir",
            str(tmp_path / "archive"),
        ],
    )

    assert reap.main() == 0
    assert _deletes(calls) == []
    archive_root = tmp_path / "archive"
    assert not [
        path for path in archive_root.iterdir() if path.name.startswith("train-x-")
    ]
    reports = list((archive_root / "runs").iterdir())
    assert len(reports) == 1
    assert "0 due, 1 preserved" in reports[0].read_text()
    assert "HOLD(1320m remaining)" in capsys.readouterr().out


def test_hold_selector_expires_to_archive_then_delete(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    expired = [{**DUE[0], "age_min": 1500.0}]
    monkeypatch.setattr(reap, "_kubectl", _fake_kubectl_factory(calls))
    monkeypatch.setattr(reap, "collect", lambda *a: (expired, [], 1))
    monkeypatch.setattr(
        reap,
        "_matching_job_uids",
        lambda *a: {"uid-train-x"},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "reap_gpu_jobs.py",
            "--apply",
            "--hold-selector",
            "purpose=ladder-crop-distill",
            "--hold-minutes",
            "1440",
            "--archive-dir",
            str(tmp_path / "archive"),
        ],
    )

    assert reap.main() == 0
    assert _deletes(calls) == [[
        "delete", "jobs", "-l",
        f"{reap.REAPER_CLAIM_LABEL}=uid-train-x", "--wait=false",
    ]]


def test_hold_selector_does_not_preserve_nonmatching_job(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(reap, "_kubectl", _fake_kubectl_factory(calls))
    monkeypatch.setattr(reap, "collect", lambda *a: (list(DUE), [], 1))
    monkeypatch.setattr(reap, "_matching_job_uids", lambda *a: set())
    monkeypatch.setattr(
        "sys.argv",
        [
            "reap_gpu_jobs.py",
            "--apply",
            "--hold-selector",
            "purpose=ladder-crop-distill",
            "--hold-minutes",
            "1440",
            "--archive-dir",
            str(tmp_path / "archive"),
        ],
    )
    assert reap.main() == 0
    assert _deletes(calls) == [[
        "delete", "jobs", "-l",
        f"{reap.REAPER_CLAIM_LABEL}=uid-train-x", "--wait=false",
    ]]


def test_hold_lookup_failure_refuses_all_deletion(monkeypatch, tmp_path, capsys):
    calls: list[list[str]] = []
    monkeypatch.setattr(reap, "collect", lambda *a: (list(DUE), [], 1))
    monkeypatch.setattr(
        reap,
        "_matching_job_uids",
        lambda *a: (_ for _ in ()).throw(RuntimeError("forbidden")),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "reap_gpu_jobs.py",
            "--apply",
            "--hold-selector",
            "purpose=ladder-crop-distill",
            "--hold-minutes",
            "1440",
            "--archive-dir",
            str(tmp_path / "archive"),
        ],
    )

    assert reap.main() == 1
    assert _deletes(calls) == []
    assert "HOLD LOOKUP FAILED" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [
        ["--hold-selector", "purpose=ladder-crop-distill"],
        ["--hold-selector", "", "--hold-minutes", "1440"],
        ["--hold-minutes", "1440"],
        [
            "--hold-selector",
            "purpose=ladder-crop-distill",
            "--hold-minutes",
            "0",
        ],
        [
            "--hold-selector",
            "purpose=ladder-crop-distill",
            "--hold-minutes",
            "nan",
        ],
        [
            "--hold-selector",
            "purpose=ladder-crop-distill",
            "--hold-minutes",
            "inf",
        ],
    ],
)
def test_hold_arguments_fail_closed_as_a_pair(monkeypatch, extra):
    monkeypatch.setattr("sys.argv", ["reap_gpu_jobs.py", *extra])
    with pytest.raises(SystemExit) as exc:
        reap.main()
    assert exc.value.code == 2
