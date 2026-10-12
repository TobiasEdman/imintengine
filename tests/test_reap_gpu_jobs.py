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
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import scripts.reap_gpu_jobs as reap

DUE = [{"job": "train-x", "job_uid": "uid-train-x", "pod": "train-x-abc12", "phase": "Failed",
        "gpus": 1, "age_min": 120.0}]


def _fake_kubectl_factory(calls: list[list[str]], *, fail_on_logs: bool = False):
    def fake(args: list[str], context: str, namespace: str) -> str:
        calls.append(args)
        if args[:2] == ["get", "job"]:
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
        if args[0] == "patch":
            return "job.batch/train-x patched\n"
        raise AssertionError(f"unexpected kubectl call: {args}")
    return fake


def _run_main(monkeypatch, tmp_path: Path, *, apply: bool, fail_on_logs: bool = False):
    calls: list[list[str]] = []
    monkeypatch.setattr(reap, "_kubectl", _fake_kubectl_factory(calls, fail_on_logs=fail_on_logs))
    monkeypatch.setattr(
        reap,
        "_delete_job_uid_precondition",
        lambda job, uid, context, namespace: calls.append(
            ["api-delete", job, uid, context, namespace]
        ),
    )
    monkeypatch.setattr(reap, "collect", lambda *a: (list(DUE), [], 1))
    argv = ["reap_gpu_jobs.py", "--grace-minutes", "60",
            "--archive-dir", str(tmp_path / "archive")]
    if apply:
        argv.append("--apply")
    monkeypatch.setattr("sys.argv", argv)
    assert reap.main() == 0
    return calls


def _install_fake_delete(monkeypatch, calls: list[list[str]]) -> None:
    monkeypatch.setattr(
        reap,
        "_delete_job_uid_precondition",
        lambda job, uid, context, namespace: calls.append(
            ["api-delete", job, uid, context, namespace]
        ),
    )


def _deletes(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if c and c[0] == "api-delete"]


def _assert_uid_delete(calls: list[list[str]]) -> None:
    deletes = _deletes(calls)
    assert deletes == [[
        "api-delete",
        "train-x",
        "uid-train-x",
        "icekube",
        "prithvi-training-default",
    ]]


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
    # Non-zero: an archive root that cannot be written is an operational
    # fault, and the run report cannot land there either. A broken mount must
    # surface to the scheduler rather than look like a clean sweep forever.
    assert reap.main() == 1
    assert _deletes(calls) == []


def test_successful_archive_then_delete_and_run_report(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, apply=True)
    _assert_uid_delete(calls)

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
    _install_fake_delete(monkeypatch, calls)
    other = {"job": "train-other", "pod": "train-other-zzz99", "phase": "Failed",
             "job_uid": "uid-train-other", "gpus": 1, "age_min": 500.0}
    monkeypatch.setattr(reap, "collect", lambda *a: ([dict(DUE[0]), other], [], 2))
    monkeypatch.setattr("sys.argv", ["reap_gpu_jobs.py", "--apply",
                                     "--grace-minutes", "0",
                                     "--job", "train-x",
                                     "--archive-dir", str(tmp_path / "archive")])
    assert reap.main() == 0
    _assert_uid_delete(calls)


def test_dry_run_touches_nothing(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, apply=False)
    assert _deletes(calls) == []
    assert not [c for c in calls if c[0] == "logs"], "dry run must not read logs"
    assert not (tmp_path / "archive").exists()


def test_collect_binds_job_uid_age_and_deduplicates_pods(monkeypatch):
    job_stamp = "2026-09-08T12:00:00Z"
    pod_stamp = "2026-09-08T13:00:00Z"
    finished_stamp = "2026-09-08T14:00:00Z"

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
                    "conditions": [{
                        "type": "Failed",
                        "status": "True",
                        "lastTransitionTime": finished_stamp,
                    }],
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
    # Only the terminal-transition stamp is read: both grace and hold are
    # measured from when the job finished, never from when it was created.
    assert seen_stamps == [finished_stamp]
    assert "job_age_min" not in records[0]


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


def test_claim_uses_uid_test_before_precondition_delete(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, apply=True)
    patch = next(call for call in calls if call[0] == "patch")
    operations = json.loads(patch[patch.index("-p") + 1])
    assert operations[0] == {
        "op": "test",
        "path": "/metadata/uid",
        "value": "uid-train-x",
    }
    _assert_uid_delete(calls)


def test_delete_refuses_changed_claim_before_api_request(monkeypatch):
    calls: list[list[str]] = []

    def fake(args: list[str], context: str, namespace: str) -> str:
        calls.append(args)
        if args[:2] == ["get", "jobs"]:
            return json.dumps({"items": [{
                "metadata": {"name": "train-x", "uid": "replacement-uid"},
                "status": {"active": 1},
            }]})
        raise AssertionError(args)

    api_calls: list[tuple] = []
    monkeypatch.setattr(reap, "_kubectl", fake)
    monkeypatch.setattr(
        reap,
        "_delete_job_uid_precondition",
        lambda *args: api_calls.append(args),
    )
    with pytest.raises(RuntimeError, match="changed before delete"):
        reap._delete_claimed_job(
            f"{reap.REAPER_CLAIM_LABEL}=uid-train-x",
            "train-x",
            "uid-train-x",
            "icekube",
            "ns",
        )
    assert api_calls == []


def test_uid_precondition_is_transmitted_and_replacement_gets_409(monkeypatch):
    received: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_DELETE(self):
            length = int(self.headers["Content-Length"])
            received.update({
                "method": self.command,
                "path": self.path,
                "content_type": self.headers["Content-Type"],
                "payload": json.loads(self.rfile.read(length)),
            })
            self.send_response(409)
            self.end_headers()
            self.wfile.write(b'{"message":"UID precondition failed"}')

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    @contextmanager
    def fake_api(context: str):
        assert context == "icekube"
        yield f"http://127.0.0.1:{server.server_port}", {}, None

    monkeypatch.setattr(reap, "_kubernetes_api", fake_api)
    try:
        with pytest.raises(RuntimeError, match="HTTP 409"):
            reap._delete_job_uid_precondition(
                "train-x", "uid-train-x", "icekube", "ns"
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert received == {
        "method": "DELETE",
        "path": "/apis/batch/v1/namespaces/ns/jobs/train-x",
        "content_type": "application/json",
        "payload": {
        "apiVersion": "v1",
        "kind": "DeleteOptions",
        "preconditions": {"uid": "uid-train-x"},
        "propagationPolicy": "Background",
        },
    }


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
    _install_fake_delete(monkeypatch, calls)
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
    _assert_uid_delete(calls)


def test_hold_window_does_not_shrink_with_job_runtime(monkeypatch, tmp_path, capsys):
    """The hold runs from when the job finished, whatever its runtime was.

    Regression: the hold used to be measured from creationTimestamp, so the
    effective window was hold_minutes MINUS the runtime — shortest for the
    long runs whose evidence is most expensive to reproduce, and zero for any
    job outliving the window. This job finished 5 minutes ago and must show
    very nearly the full 1440 m remaining; the record carries no creation-age
    field at all, so there is nothing left for the hold to regress onto.
    """
    calls: list[list[str]] = []
    just_finished = [{**DUE[0], "age_min": 5.0}]
    monkeypatch.setattr(reap, "_kubectl", _fake_kubectl_factory(calls))
    _install_fake_delete(monkeypatch, calls)
    monkeypatch.setattr(reap, "collect", lambda *a: (just_finished, [], 1))
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
    # Held for very nearly the full window, not 1440 − 1500 = already expired.
    assert "HOLD(1435m remaining)" in capsys.readouterr().out


def test_partial_sweep_still_records_what_it_deleted(monkeypatch, tmp_path):
    """A crash mid-sweep must not cost the run report.

    Regression: `archive_evidence` caught only (RuntimeError, OSError) and the
    per-job loop only RuntimeError, so a malformed kubectl response — which
    raises json.JSONDecodeError, a ValueError — escaped main() and skipped the
    report write. Jobs already deleted in that sweep became unrecorded, and
    the reaper pod's TTL then destroyed stdout: exactly the "no record of what
    was deleted" failure the report exists to prevent.
    """
    archive = tmp_path / "archive"
    deleted: list[str] = []
    due = [DUE[0], {**DUE[0], "job": "train-y", "job_uid": "uid-train-y"}]
    monkeypatch.setattr(reap, "collect", lambda *a: (due, [], 2))

    def claim(name, uid, ctx, ns):
        if name == "train-y":
            raise json.JSONDecodeError("Expecting value", "", 0)
        return f"sel={uid}"

    monkeypatch.setattr(reap, "_claim_terminal_job", claim)
    monkeypatch.setattr(
        reap, "archive_evidence",
        lambda job, uid, sel, ctx, ns, root: root / job,
    )
    monkeypatch.setattr(reap, "_claimed_terminal_job", lambda *a: True)
    monkeypatch.setattr(
        reap, "_delete_claimed_job",
        lambda sel, name, uid, ctx, ns: deleted.append(name),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["reap_gpu_jobs.py", "--apply", "--archive-dir", str(archive)],
    )

    # Non-zero: a broad guard must not turn a failing sweep into a silent
    # success, or the CronJob looks healthy while nothing worked.
    assert reap.main() == 1
    assert deleted == ["train-x"]
    reports = list((archive / "runs").glob("*.txt"))
    assert len(reports) == 1
    body = reports[0].read_text()
    assert "deleted train-x" in body
    assert "SKIPPED train-y" in body and "JSONDecodeError" in body
    assert "1 job(s) failed" in body


def test_unwritable_run_report_after_deletion_exits_nonzero(monkeypatch, tmp_path):
    """Deleting jobs and losing the record of it is not a successful sweep.

    Regression: `_write_run_report` swallowed OSError and `main()` keyed its
    exit code off per-job errors alone, so a sweep whose deletions all
    succeeded but whose report could not be written exited 0 — the same
    silent-success class the per-job guard was widened to avoid.
    """
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "runs").write_text("occupied by a file, so mkdir fails")
    deleted: list[str] = []
    monkeypatch.setattr(reap, "collect", lambda *a: (list(DUE), [], 1))
    monkeypatch.setattr(
        reap, "_claim_terminal_job", lambda name, uid, ctx, ns: f"sel={uid}"
    )
    monkeypatch.setattr(
        reap, "archive_evidence",
        lambda job, uid, sel, ctx, ns, root: root / job,
    )
    monkeypatch.setattr(reap, "_claimed_terminal_job", lambda *a: True)
    monkeypatch.setattr(
        reap, "_delete_claimed_job",
        lambda sel, name, uid, ctx, ns: deleted.append(name),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["reap_gpu_jobs.py", "--apply", "--archive-dir", str(archive)],
    )

    assert reap.main() == 1
    assert deleted == ["train-x"]          # the deletion really happened
    assert (archive / "runs").is_file()    # and left no report behind


def test_hold_minutes_help_states_terminal_transition(monkeypatch, capsys):
    """The operator-facing contract of a destructive tool must match the code.

    `--hold-minutes` documented "from creation" after the window had been
    changed to run from the terminal transition.
    """
    monkeypatch.setattr("sys.argv", ["reap_gpu_jobs.py", "--help"])
    with pytest.raises(SystemExit):
        reap.main()
    help_text = capsys.readouterr().out
    assert "terminal transition" in help_text
    assert "minutes from creation" not in help_text


def test_hold_selector_does_not_preserve_nonmatching_job(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(reap, "_kubectl", _fake_kubectl_factory(calls))
    _install_fake_delete(monkeypatch, calls)
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
    _assert_uid_delete(calls)


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
