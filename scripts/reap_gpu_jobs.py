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
    reap_gpu_jobs.py --grace-minutes 0 --apply --selector purpose=era5-prithvi600m-smoke

Never touches Running or Pending jobs, and only ever considers Jobs — a
long-lived Deployment such as the vllm-mistral chatbot cannot match.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TERMINAL = ("Succeeded", "Failed")


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


def collect(context: str, namespace: str, selector: str | None) -> tuple[list, list, int]:
    """Return (reapable, stuck, gpus_held_by_terminal)."""
    args = ["get", "pods", "-o", "json"]
    if selector:
        args += ["-l", selector]
    pods = json.loads(_kubectl(args, context, namespace))["items"]

    reapable, held = [], 0
    for p in pods:
        g = _gpus(p["spec"])
        if not g:
            continue
        phase = p["status"].get("phase")
        owners = [o for o in p["metadata"].get("ownerReferences", []) if o["kind"] == "Job"]
        if phase not in TERMINAL or not owners:
            continue
        held += g
        reapable.append({
            "job": owners[0]["name"],
            "pod": p["metadata"]["name"],
            "phase": phase,
            "gpus": g,
            "age_min": _age_minutes(p["metadata"].get("creationTimestamp")),
        })

    # Jobs whose pod was never created: incomplete, zero pods, and the job
    # controller is emitting FailedCreate. Nothing else reveals these.
    jargs = ["get", "jobs", "-o", "json"]
    if selector:
        jargs += ["-l", selector]
    jobs = json.loads(_kubectl(jargs, context, namespace))["items"]
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
    return reapable, stuck, held


def archive_evidence(job: str, context: str, namespace: str, root: Path) -> Path | None:
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
        dest.mkdir(parents=True, exist_ok=False)
        (dest / "job.yaml").write_text(
            _kubectl(["get", "job", job, "-o", "yaml"], context, namespace))
        pods = json.loads(_kubectl(
            ["get", "pods", "-l", f"job-name={job}", "-o", "json"],
            context, namespace))["items"]
        (dest / "pods.json").write_text(json.dumps(pods, indent=1))
        events = [_kubectl(
            ["get", "events", "--field-selector", f"involvedObject.name={job}"],
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
                ["get", "events", "--field-selector", f"involvedObject.name={name}"],
                context, namespace))
        (dest / "events.txt").write_text("\n".join(events))
        return dest
    except (RuntimeError, OSError) as exc:  # noqa: BLE001 — any miss forbids deletion
        print(f"  EVIDENCE ARCHIVE FAILED for {job}: {exc}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--context", default="icekube")
    ap.add_argument("--namespace", default="prithvi-training-default")
    ap.add_argument("--selector", default=None, help="label selector to scope the sweep")
    ap.add_argument("--grace-minutes", type=float, default=30.0,
                    help="leave finished jobs alone this long so logs stay readable")
    ap.add_argument("--archive-dir", type=Path, default=Path("/cephfs/ops/reaper_archive"),
                    help="evidence archive root; no delete ever happens without a "
                         "successful archive here")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: report)")
    args = ap.parse_args()

    reapable, stuck, held = collect(args.context, args.namespace, args.selector)

    print(f"=== GPU held by FINISHED jobs: {held} slot(s) ===")
    if not reapable:
        print("  (none — nothing squatting)")
    for r in sorted(reapable, key=lambda x: -x["age_min"]):
        due = "REAP" if r["age_min"] >= args.grace_minutes else f"grace({args.grace_minutes:g}m)"
        print(f"  {r['gpus']} GPU  {r['phase']:9s} {r['age_min']/60:5.1f}h  "
              f"{r['job'][:48]:48s} {due}")

    if stuck:
        print(f"\n=== NEVER SCHEDULED — look running, are not ({len(stuck)}) ===")
        for s in sorted(stuck, key=lambda x: -x["age_min"]):
            print(f"  {s['age_min']/60:5.1f}h  x{s['count']:<4} {s['job'][:44]:44s}")
            print(f"         {s['why']}")

    due = [r for r in reapable if r["age_min"] >= args.grace_minutes]
    if not args.apply:
        print(f"\nDRY RUN — {len(due)} job(s) would be deleted, freeing "
              f"{sum(r['gpus'] for r in due)} GPU slot(s). Re-run with --apply.")
        return 0

    freed = 0
    report = [f"reap run {datetime.now(timezone.utc).isoformat()} — "
              f"{len(due)} due, {held} GPU slot(s) held by finished jobs"]
    for r in due:
        dest = archive_evidence(r["job"], args.context, args.namespace,
                                args.archive_dir)
        if dest is None:
            msg = (f"SKIPPED {r['job']} ({r['phase']}) — refusing to delete "
                   f"without archived evidence")
            print(f"  {msg}")
            report.append(msg)
            continue
        try:
            _kubectl(["delete", "job", r["job"], "--wait=false"],
                     args.context, args.namespace)
            freed += r["gpus"]
            msg = f"deleted {r['job']} ({r['phase']}) — evidence in {dest}"
            print(f"  {msg}")
            report.append(msg)
        except RuntimeError as exc:  # noqa: BLE001 — one failure must not stop the sweep
            msg = f"DELETE FAILED {r['job']}: {exc}"
            print(f"  {msg}")
            report.append(msg)
    print(f"\nfreed {freed} GPU slot(s)")
    report.append(f"freed {freed} GPU slot(s)")

    # The reaper's own pod TTLs away 30 min after it runs, taking this stdout
    # with it — which is how the overnight sweeps of 2026-08-29 left no record
    # of WHAT they deleted. The run report therefore lives on the PVC too.
    try:
        runs = args.archive_dir / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (runs / f"{stamp}.txt").write_text("\n".join(report) + "\n")
    except OSError as exc:
        print(f"run-report write failed (deletions above were still "
              f"individually archived): {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
