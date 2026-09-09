#!/usr/bin/env python3
"""Reap finished GPU jobs and surface ones that never scheduled.

Two failure modes, both observed on 2026-08-27, both invisible to a status check.

**Squatting.** A Job that has Succeeded or Failed keeps its pod until
``ttlSecondsAfterFinished`` expires, and the iceguard admission webhook counts
that pod's ``nvidia.com/gpu`` request against the PROJECT quota the whole time.
Eight finished ERA5 arms — each done in ~135 min, each retained for a 48 h TTL —
accumulated into ``quota: 8, used: 8`` and blocked every H100 in the project for
about five hours. The oldest had been holding a slot for ~31 hours.

**Never-scheduled.** When the webhook denies the pod, no pod is created, so the
Job sits at ``0/1`` with no pod events. That is indistinguishable from "running"
to anything polling ``.status.succeeded``/``.status.failed`` — the blocked run
above retried 291 times over 4h49m while being reported as in progress.

So this reports what is actually held, deletes what is finished, and names what
is stuck. Dry-run unless ``--apply``.

    reap_gpu_jobs.py                      # report only
    reap_gpu_jobs.py --apply              # delete finished GPU jobs past grace
    reap_gpu_jobs.py --apply \
      --hold-selector purpose=ladder-crop-distill --hold-minutes 1440
    reap_gpu_jobs.py --grace-minutes 0 --apply --selector purpose=era5-prithvi600m-smoke

Never touches Running or Pending jobs, and only ever considers Jobs — a
long-lived Deployment such as the vllm-mistral chatbot cannot match.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import ssl
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

TERMINAL = ("Succeeded", "Failed")
TERMINAL_JOB_CONDITIONS = ("Complete", "Failed")
REAPER_CLAIM_LABEL = "imintengine.se/reaper-claim"


def _kubectl(args: list[str], context: str, namespace: str) -> str:
    # In-cluster there is no kubeconfig context — kubectl uses the pod's
    # ServiceAccount. Pass --context "" (or omit it) when running as a CronJob.
    cmd = ["kubectl", *(["--context", context] if context else []),
           "-n", namespace, *args]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}: {out.stderr.strip()[:200]}")
    return out.stdout


def _age_minutes(stamp: str | None) -> float:
    if not stamp:
        return 0.0
    t = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t).total_seconds() / 60.0


def _gpus(pod_spec: dict) -> int:
    return sum(
        int(c.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0) or 0)
        for c in pod_spec.get("containers", [])
    )


def _job_is_terminal(job: dict) -> bool:
    status = job.get("status", {})
    if status.get("active"):
        return False
    return any(
        condition.get("type") in TERMINAL_JOB_CONDITIONS
        and condition.get("status") == "True"
        for condition in status.get("conditions", [])
    )


def _terminal_transition_timestamp(job: dict) -> str | None:
    stamps = [
        condition.get("lastTransitionTime")
        for condition in job.get("status", {}).get("conditions", [])
        if condition.get("type") in TERMINAL_JOB_CONDITIONS
        and condition.get("status") == "True"
        and condition.get("lastTransitionTime")
    ]
    return max(stamps, default=None)


def _claim_terminal_job(
    job_name: str,
    job_uid: str,
    context: str,
    namespace: str,
) -> str:
    """Atomically bind this sweep to one terminal Job UID.

    Deletion later uses the unique claim label rather than the reusable Job
    name, so a delete/recreate recovery race cannot target the replacement.
    """
    current = json.loads(_kubectl(
        ["get", "job", job_name, "-o", "json"], context, namespace
    ))
    if current.get("metadata", {}).get("uid") != job_uid:
        raise RuntimeError(f"Job {job_name} UID changed before reaper claim")
    if not _job_is_terminal(current):
        raise RuntimeError(f"Job {job_name} is no longer terminal")

    labels = current.get("metadata", {}).get("labels")
    escaped_key = REAPER_CLAIM_LABEL.replace("~", "~0").replace("/", "~1")
    if labels is None:
        add_label = {
            "op": "add",
            "path": "/metadata/labels",
            "value": {REAPER_CLAIM_LABEL: job_uid},
        }
    elif isinstance(labels, dict):
        add_label = {
            "op": "add",
            "path": f"/metadata/labels/{escaped_key}",
            "value": job_uid,
        }
    else:
        raise RuntimeError(f"Job {job_name} has malformed metadata.labels")
    patch = [
        {"op": "test", "path": "/metadata/uid", "value": job_uid},
        add_label,
    ]
    _kubectl(
        ["patch", "job", job_name, "--type=json", "-p", json.dumps(patch)],
        context,
        namespace,
    )
    return f"{REAPER_CLAIM_LABEL}={job_uid}"


def _claimed_terminal_job(
    selector: str,
    job_name: str,
    job_uid: str,
    context: str,
    namespace: str,
) -> bool:
    jobs = json.loads(_kubectl(
        ["get", "jobs", "-l", selector, "-o", "json"], context, namespace
    ))["items"]
    return len(jobs) == 1 and (
        jobs[0].get("metadata", {}).get("name") == job_name
        and jobs[0].get("metadata", {}).get("uid") == job_uid
        and _job_is_terminal(jobs[0])
    )


@contextmanager
def _kubernetes_api(context: str):
    """Yield API base URL, headers, and TLS context for this execution mode."""
    if not context:
        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
        ca_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
        if not host or not token_path.is_file() or not ca_path.is_file():
            raise RuntimeError("in-cluster Kubernetes credentials are unavailable")
        token = token_path.read_text().strip()
        if not token:
            raise RuntimeError("in-cluster Kubernetes token is empty")
        yield (
            f"https://{host}:{port}",
            {"Authorization": f"Bearer {token}"},
            ssl.create_default_context(cafile=str(ca_path)),
        )
        return

    cmd = [
        "kubectl",
        "--context",
        context,
        "proxy",
        "--address=127.0.0.1",
        "--port=0",
        "--append-server-path",
    ]
    proxy = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if proxy.stdout is None:
            raise RuntimeError("kubectl proxy stdout was unavailable")
        line = proxy.stdout.readline().strip()
        match = re.search(r"127\.0\.0\.1:(\d+)", line)
        if match is None:
            if proxy.poll() is None:
                proxy.terminate()
            stderr = proxy.stderr.read(200) if proxy.stderr else ""
            raise RuntimeError(f"kubectl proxy failed: {line} {stderr}".strip())
        yield f"http://127.0.0.1:{match.group(1)}", {}, None
    finally:
        if proxy.poll() is None:
            proxy.terminate()
            try:
                proxy.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy.kill()
                proxy.wait(timeout=5)


def _http_delete_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    tls_context: ssl.SSLContext | None,
) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode()
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="DELETE",
    )
    try:
        with urlopen(request, context=tls_context, timeout=30) as response:
            if response.status >= 300:
                raise RuntimeError(f"Kubernetes DELETE returned HTTP {response.status}")
    except HTTPError as exc:
        detail = exc.read(200).decode(errors="replace")
        raise RuntimeError(
            f"Kubernetes DELETE returned HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Kubernetes DELETE transport failed: {exc.reason}") from exc


def _delete_job_uid_precondition(
    job_name: str,
    job_uid: str,
    context: str,
    namespace: str,
) -> None:
    delete_options = {
        "apiVersion": "v1",
        "kind": "DeleteOptions",
        "preconditions": {"uid": job_uid},
        "propagationPolicy": "Background",
    }
    namespace_path = quote(namespace, safe="")
    job_path = quote(job_name, safe="")
    path = f"/apis/batch/v1/namespaces/{namespace_path}/jobs/{job_path}"
    with _kubernetes_api(context) as (base_url, headers, tls_context):
        _http_delete_json(
            f"{base_url}{path}", delete_options, headers, tls_context
        )


def _delete_claimed_job(
    selector: str,
    job_name: str,
    job_uid: str,
    context: str,
    namespace: str,
) -> None:
    """Delete the exact UID through the API and verify acceptance."""
    if not _claimed_terminal_job(
        selector, job_name, job_uid, context, namespace
    ):
        raise RuntimeError(f"claimed Job {job_name} changed before delete")
    _delete_job_uid_precondition(job_name, job_uid, context, namespace)


def collect(context: str, namespace: str, selector: str | None) -> tuple[list, list, int]:
    """Return (reapable, stuck, gpus_held_by_terminal)."""
    jargs = ["get", "jobs", "-o", "json"]
    if selector:
        jargs += ["-l", selector]
    jobs = json.loads(_kubectl(jargs, context, namespace))["items"]
    jobs_by_uid = {
        job["metadata"]["uid"]: job
        for job in jobs
        if job.get("metadata", {}).get("uid")
    }

    args = ["get", "pods", "-o", "json"]
    if selector:
        args += ["-l", selector]
    pods = json.loads(_kubectl(args, context, namespace))["items"]

    reapable_by_uid: dict[str, dict] = {}
    held = 0
    for p in pods:
        g = _gpus(p["spec"])
        if not g:
            continue
        phase = p["status"].get("phase")
        owners = [
            owner
            for owner in p["metadata"].get("ownerReferences", [])
            if owner.get("kind") == "Job" and owner.get("uid")
        ]
        if phase not in TERMINAL or not owners:
            continue
        owner = owners[0]
        job = jobs_by_uid.get(owner["uid"])
        if (
            job is None
            or job["metadata"]["name"] != owner["name"]
            or not _job_is_terminal(job)
        ):
            continue
        held += g
        record = reapable_by_uid.get(owner["uid"])
        if record is None:
            record = {
                "job": owner["name"],
                "job_uid": owner["uid"],
                "pod": p["metadata"]["name"],
                "phase": phase,
                "gpus": 0,
                "age_min": _age_minutes(
                    _terminal_transition_timestamp(job)
                ),
            }
            reapable_by_uid[owner["uid"]] = record
        record["gpus"] += g

    # Jobs whose pod was never created: incomplete, zero pods, and the job
    # controller is emitting FailedCreate. Nothing else reveals these.
    stuck = []
    for j in jobs:
        st = j.get("status", {})
        if st.get("succeeded") or st.get("failed") or st.get("active"):
            continue
        name = j["metadata"]["name"]
        try:
            ev = json.loads(_kubectl(
                ["get", "events", "--field-selector",
                 f"involvedObject.name={name}", "-o", "json"], context, namespace))
        except RuntimeError:
            continue
        bad = [e for e in ev["items"] if e.get("reason") == "FailedCreate"]
        if bad:
            last = bad[-1]
            stuck.append({
                "job": name,
                "age_min": _age_minutes(j["metadata"].get("creationTimestamp")),
                "count": last.get("count", 1),
                "why": last.get("message", "")[:150],
            })
    return list(reapable_by_uid.values()), stuck, held


def _matching_job_uids(context: str, namespace: str, selector: str) -> set[str]:
    """Return Job UIDs selected by one cluster-visible preservation policy."""
    jobs = json.loads(
        _kubectl(
            ["get", "jobs", "-l", selector, "-o", "json"],
            context,
            namespace,
        )
    )["items"]
    return {job["metadata"]["uid"] for job in jobs}


def archive_evidence(
    job: str,
    job_uid: str,
    claim_selector: str,
    context: str,
    namespace: str,
    root: Path,
) -> Path | None:
    """Persist everything the cluster still knows about *job*, before deletion.

    Returns the archive directory on success, None on ANY failure — and the
    caller must then leave the Job alone. Deleting a Job cascades to its pods
    and their logs. Twice that cascade destroyed the only evidence of why a
    run ended: the ERA5 control's per-class IoU (2026-08-27) and the wave-1
    train-prithvi300m/train-croma-v3 failures (2026-08-29), where nothing
    remained but old checkpoint mtimes. Deletion without archived evidence is
    therefore forbidden, not merely discouraged.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = root / f"{job}-{stamp}"
    try:
        if not _claimed_terminal_job(
            claim_selector, job, job_uid, context, namespace
        ):
            raise RuntimeError("claimed Job UID/state changed before archive")
        dest.mkdir(parents=True, exist_ok=False)
        (dest / "job.yaml").write_text(
            _kubectl(
                ["get", "jobs", "-l", claim_selector, "-o", "yaml"],
                context,
                namespace,
            )
        )
        pods = json.loads(_kubectl(
            ["get", "pods", "-l", f"job-name={job}", "-o", "json"],
            context, namespace))["items"]
        pods = [
            pod
            for pod in pods
            if any(
                owner.get("kind") == "Job" and owner.get("uid") == job_uid
                for owner in pod.get("metadata", {}).get("ownerReferences", [])
            )
        ]
        (dest / "pods.json").write_text(json.dumps(pods, indent=1))
        events = [_kubectl(
            ["get", "events", "--field-selector", f"involvedObject.uid={job_uid}"],
            context, namespace)]
        for p in pods:
            name = p["metadata"]["name"]
            (dest / f"{name}.log").write_text(_kubectl(
                ["logs", name, "--all-containers", "--timestamps"],
                context, namespace))
            try:
                (dest / f"{name}.previous.log").write_text(_kubectl(
                    ["logs", name, "--all-containers", "--previous"],
                    context, namespace))
            except RuntimeError:
                pass  # no restarted container — the normal case
            events.append(_kubectl(
                [
                    "get",
                    "events",
                    "--field-selector",
                    f"involvedObject.uid={p['metadata']['uid']}",
                ],
                context, namespace))
        (dest / "events.txt").write_text("\n".join(events))
        if not _claimed_terminal_job(
            claim_selector, job, job_uid, context, namespace
        ):
            raise RuntimeError("claimed Job UID/state changed during archive")
        return dest
    except (RuntimeError, OSError) as exc:
        print(f"  EVIDENCE ARCHIVE FAILED for {job}: {exc}")
        return None


def _reap_due(due: list[dict], args, report: list[str]) -> tuple[int, int]:
    """Archive then delete each due Job. Returns (gpus freed, unexpected errors).

    Every per-job failure is caught and recorded rather than raised. Once one
    job in a sweep has been deleted, an escaping exception would cost the run
    report its only chance to record that deletion — the reaper pod's own TTL
    destroys stdout 30 min later. The guards are deliberately broad because a
    malformed kubectl response raises ValueError or KeyError, neither of which
    is a RuntimeError.

    Broad guards must not turn a failing sweep into a silent success, so the
    error count is returned and becomes a non-zero exit: a CronJob that exits
    0 while nothing worked is indistinguishable from one that had no work.
    """
    freed = 0
    errors = 0
    for r in due:
        try:
            claim_selector = _claim_terminal_job(
                r["job"], r["job_uid"], args.context, args.namespace
            )
        except Exception as exc:
            msg = f"SKIPPED {r['job']} ({r['phase']}) — claim failed: {exc!r}"
            print(f"  {msg}")
            report.append(msg)
            errors += 1
            continue
        try:
            dest = archive_evidence(
                r["job"],
                r["job_uid"],
                claim_selector,
                args.context,
                args.namespace,
                args.archive_dir,
            )
            if dest is None:
                msg = (f"SKIPPED {r['job']} ({r['phase']}) — refusing to "
                       f"delete without archived evidence")
                print(f"  {msg}")
                report.append(msg)
                continue
            if not _claimed_terminal_job(
                claim_selector, r["job"], r["job_uid"], args.context, args.namespace
            ):
                raise RuntimeError("claimed Job UID/state changed after archive")
            _delete_claimed_job(
                claim_selector, r["job"], r["job_uid"], args.context, args.namespace
            )
            freed += r["gpus"]
            msg = f"deleted {r['job']} ({r['phase']}) — evidence in {dest}"
            print(f"  {msg}")
            report.append(msg)
        except Exception as exc:
            msg = f"DELETE FAILED {r['job']}: {exc!r}"
            print(f"  {msg}")
            report.append(msg)
            errors += 1
    return freed, errors


def _write_run_report(archive_dir: Path, report: list[str]) -> None:
    """Persist what this sweep did, on every exit path.

    The reaper's own pod TTLs away 30 min after it runs, taking this stdout
    with it — which is how the overnight sweeps of 2026-08-29 left no record
    of WHAT they deleted. The run report therefore lives on the PVC too, and
    is written from a finally so no exit path can drop it.
    """
    try:
        runs = archive_dir / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (runs / f"{stamp}.txt").write_text("\n".join(report) + "\n")
    except OSError as exc:
        print(f"run-report write failed (deletions above were still "
              f"individually archived): {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--context", default="icekube")
    ap.add_argument("--namespace", default="prithvi-training-default")
    ap.add_argument("--selector", default=None, help="label selector to scope the sweep")
    ap.add_argument("--job", action="append", default=None, metavar="NAME",
                    help="restrict the sweep to exactly these Job names "
                         "(repeatable); anything else is reported but never "
                         "deleted")
    ap.add_argument("--grace-minutes", type=float, default=30.0,
                    help="leave finished jobs alone this long so logs stay readable")
    ap.add_argument(
        "--hold-selector",
        default=None,
        help=(
            "cluster-visible Job label selector whose matches receive a longer, "
            "bounded preservation window"
        ),
    )
    ap.add_argument(
        "--hold-minutes",
        type=float,
        default=None,
        help="preserve --hold-selector matches for this many minutes from creation",
    )
    ap.add_argument("--archive-dir", type=Path, default=Path("/cephfs/ops/reaper_archive"),
                    help="evidence archive root; no delete ever happens without a "
                         "successful archive here")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: report)")
    args = ap.parse_args()

    if (args.hold_selector is None) != (args.hold_minutes is None):
        ap.error("--hold-selector and --hold-minutes must be supplied together")
    if args.hold_selector is not None and not args.hold_selector.strip():
        ap.error("--hold-selector must not be empty")
    if args.hold_minutes is not None and (
        not math.isfinite(args.hold_minutes) or args.hold_minutes <= 0
    ):
        ap.error("--hold-minutes must be a finite value greater than zero")

    reapable, stuck, held = collect(args.context, args.namespace, args.selector)
    held_job_uids: set[str] = set()
    if args.hold_selector is not None:
        try:
            held_job_uids = _matching_job_uids(
                args.context,
                args.namespace,
                args.hold_selector,
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            print(
                f"HOLD LOOKUP FAILED — refusing all deletion: {exc}",
                file=sys.stderr,
            )
            return 1

    def under_hold(record: dict) -> bool:
        # Measured from the terminal transition, not creation. From creation
        # the effective window is hold_minutes MINUS however long the job ran,
        # so a job that fails near its activeDeadlineSeconds is preserved for
        # the shortest time exactly when it was the most expensive to produce
        # — and a job whose deadline exceeds hold_minutes gets no hold at all
        # while still reporting as held. The bound stays anti-squatter: a
        # finished job can squat its GPU for at most hold_minutes after it
        # finished.
        return (
            args.hold_minutes is not None
            and record["job_uid"] in held_job_uids
            and record["age_min"] < args.hold_minutes
        )

    print(f"=== GPU held by FINISHED jobs: {held} slot(s) ===")
    if not reapable:
        print("  (none — nothing squatting)")
    for r in sorted(reapable, key=lambda x: -x["age_min"]):
        if under_hold(r):
            remaining = args.hold_minutes - r["age_min"]
            due = f"HOLD({remaining:.0f}m remaining)"
        else:
            due = (
                "REAP"
                if r["age_min"] >= args.grace_minutes
                else f"grace({args.grace_minutes:g}m)"
            )
        print(f"  {r['gpus']} GPU  {r['phase']:9s} {r['age_min']/60:5.1f}h  "
              f"{r['job'][:48]:48s} {due}")

    if stuck:
        print(f"\n=== NEVER SCHEDULED — look running, are not ({len(stuck)}) ===")
        for s in sorted(stuck, key=lambda x: -x["age_min"]):
            print(f"  {s['age_min']/60:5.1f}h  x{s['count']:<4} {s['job'][:44]:44s}")
            print(f"         {s['why']}")

    preserved = [r for r in reapable if under_hold(r)]
    due = [
        r
        for r in reapable
        if r["age_min"] >= args.grace_minutes and not under_hold(r)
    ]
    if args.job is not None:
        allowed = set(args.job)
        excluded = [r["job"] for r in due if r["job"] not in allowed]
        if excluded:
            print(f"\n--job scope excludes {len(excluded)} otherwise-due job(s): "
                  f"{', '.join(sorted(excluded))}")
        due = [r for r in due if r["job"] in allowed]
    if not args.apply:
        print(f"\nDRY RUN — {len(due)} job(s) would be deleted, freeing "
              f"{sum(r['gpus'] for r in due)} GPU slot(s). Re-run with --apply.")
        return 0

    report = [
        (
            f"reap run {datetime.now(timezone.utc).isoformat()} — "
            f"{len(due)} due, {len(preserved)} preserved, "
            f"{held} GPU slot(s) held by finished jobs"
        )
    ]
    errors = 0
    try:
        freed, errors = _reap_due(due, args, report)
        summary = f"freed {freed} GPU slot(s)"
        if errors:
            summary += f" — {errors} job(s) failed"
        print(f"\n{summary}")
        report.append(summary)
    finally:
        _write_run_report(args.archive_dir, report)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
