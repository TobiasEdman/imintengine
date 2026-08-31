#!/usr/bin/env python3
"""Submit label-source ladder jobs as the memory quota frees. Runs in-cluster.

The ladder is 24 runs but the namespace fits only ~3 concurrently (250 Gi hard
cap on requests.memory, ~39 Gi held by standing services). Something has to
submit the next run when one finishes. That "something" must be automatic and
must live in the cluster: a laptop-side scheduled task missed every slot it was
given on 2026-08-29, so capacity sat idle overnight and each gap was only
noticed when a human asked for status.

Submits **rungs 1 and 2 only**. Rungs 3-4 read per-backbone distillation
sidecars that do not exist until their column's rung 2 finishes, so they are
deliberately out of scope here — see docs/experiments/label_source_ladder.md.

Order is fixed and largest-first within a rung, because greedy packing of an
80 Gi job after two 48 Gi jobs wastes a slot the 80 cannot fit into.

Never exceeds quota. A Job admitted past the cap creates NO pod and sits at
0/1 for ever, which is indistinguishable from running to anything polling
.status — the exact failure `reap_gpu_jobs.py --check-jobs` exists to surface.
So the walk stops at the first job that does not fit rather than skipping ahead
to a smaller one, which also keeps the submission order interpretable.

When all 12 exist it suspends its own CronJob, so it cannot become an orphan.

    ladder_queue.py --dry-run     # report only
    ladder_queue.py               # submit what fits, then maybe self-suspend
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Largest-first within each rung: 80 Gi backbones before 48 Gi ones.
MODEL_ORDER = ["prithvi600m", "prithvi300m", "tessera", "croma", "terramind", "clay"]
RUNGS = (1, 2)
# Rungs 3/4 are DISTILL-GATED: a column joins their queue only when its
# distill job has written the _GATE_OK marker (state on disk, never job
# existence — the reaper deletes finished Jobs). Both consume the same
# sidecars and neither warm-starts from the other, so they queue together
# the moment the gate opens; CROMA trails naturally because its gate
# lands last. [user-stated 2026-08-31: croma sist i egen fil, övriga
# två-och-två]
GATED_RUNGS = (3, 4)
DISTILL_ROOT = Path("/cephfs/distill")


def distill_gate_open(distill_root: Path, model: str) -> bool:
    return (distill_root / f"{model}_r2" / "_GATE_OK").exists()
# Headroom against the standing services drifting. Kept small: the quota is
# 250Gi and the biggest job wants 80, so an over-large margin strands an 80Gi
# slot that is arithmetically free (a 3Gi margin blocked exactly that for an
# hour on 2026-08-30).
MARGIN_GI = 1
LADDER_ROOT = Path("/cephfs/checkpoints/ladder")


def _kubectl(args: list[str], namespace: str) -> str:
    cmd = ["kubectl", "-n", namespace, *args]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}: {out.stderr.strip()[:200]}")
    return out.stdout


def _to_gi(value: str) -> float:
    """Kubernetes quantity → GiB. Handles the Mi/Gi/bare-bytes the API returns."""
    v = value.strip()
    if v.endswith("Mi"):
        return float(v[:-2]) / 1024
    if v.endswith("Gi"):
        return float(v[:-2])
    if v.endswith("Ti"):
        return float(v[:-2]) * 1024
    return float(v) / (1024 ** 3)


def manifest_path(repo: Path, rung: int, model: str) -> Path:
    return repo / "k8s" / "ladder" / f"ladder-r{rung}-{model}-job.yaml"


def requested_gi(path: Path) -> float:
    """Read the job's own memory request — never hard-code it here.

    The manifests are generated (scripts/gen_ladder_manifests.py); a number
    copied into this file would drift the moment a base manifest changes.
    """
    import yaml
    doc = yaml.safe_load(path.read_text())
    c = doc["spec"]["template"]["spec"]["containers"][0]
    return _to_gi(c["resources"]["requests"]["memory"])


def job_exists(name: str, namespace: str) -> bool:
    try:
        _kubectl(["get", "job", name, "-o", "name"], namespace)
        return True
    except RuntimeError:
        return False


def already_trained(ladder_root: Path, rung: int, model: str) -> bool:
    """Has this cell already produced a finished run?

    Job existence alone is NOT the answer: gpu-reaper archives and deletes a
    Job an hour after it finishes, so a completed run looks pending to a
    `kubectl get job` check. Without this the queue resubmits work that is
    already done — on 2026-08-30 both tessera_r1 (0.5654, 30/30) and
    prithvi600m_r1 had finished and been reaped, and the queue had them
    queued for resubmission.

    The durable record is the checkpoint dir, which the reaper never touches.
    A run counts as done when the trainer says `completed`, or when it logged
    its target epoch — "stopped" is the early-stop label and fires even on a
    run that reached the end.
    """
    log = ladder_root / f"{model}_r{rung}" / "training_log.json"
    try:
        d = json.loads(log.read_text())
    except (OSError, ValueError):
        return False
    if d.get("status") == "completed":
        return True
    epochs = d.get("epochs") or []
    target = (d.get("config") or {}).get("epochs")
    last = epochs[-1].get("epoch", 0) if epochs else 0
    return bool(target and last >= target)


def free_gi(namespace: str, quota: str) -> float:
    raw = json.loads(_kubectl(["get", "resourcequota", quota, "-o", "json"], namespace))
    used = _to_gi(raw["status"]["used"]["requests.memory"])
    hard = _to_gi(raw["status"]["hard"]["requests.memory"])
    return hard - used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--namespace", default="prithvi-training-default")
    ap.add_argument("--quota", default="default-9s779")
    ap.add_argument("--repo", type=Path, default=Path("/w"))
    ap.add_argument("--cronjob", default="ladder-queue",
                    help="suspend this CronJob once every job is submitted")
    ap.add_argument("--distill-root", type=Path, default=DISTILL_ROOT,
                    help="distill sidecar root; <model>_r2/_GATE_OK unlocks "
                         "that column's rungs 3/4")
    ap.add_argument("--ladder-root", type=Path, default=LADDER_ROOT,
                    help="checkpoint root; a finished run here counts as done "
                         "even after the reaper has deleted its Job")
    ap.add_argument("--log", type=Path, default=Path("/cephfs/ops/ladder_queue.log"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pending = [
        (r, m) for r in RUNGS for m in MODEL_ORDER
        if not job_exists(f"ladder-r{r}-{m}", args.namespace)
        and not already_trained(args.ladder_root, r, m)
    ]
    gate_locked = 0
    for r in GATED_RUNGS:
        for m in MODEL_ORDER:
            if job_exists(f"ladder-r{r}-{m}", args.namespace) or                     already_trained(args.ladder_root, r, m):
                continue
            if distill_gate_open(args.distill_root, m):
                pending.append((r, m))
            else:
                gate_locked += 1
    if gate_locked:
        print(f"  {gate_locked} rung-3/4 cells await their distill gate")

    if not pending and not gate_locked:
        n_total = (len(RUNGS) + len(GATED_RUNGS)) * len(MODEL_ORDER)
        line = f"{stamp} DONE — all {n_total} ladder jobs submitted"
        print(line)
        if not args.dry_run:
            try:
                _kubectl(["patch", "cronjob", args.cronjob, "-p",
                          '{"spec":{"suspend":true}}'], args.namespace)
                line += "; self-suspended"
                print("  self-suspended — nothing left to schedule")
            except RuntimeError as exc:
                print(f"  self-suspend FAILED (harmless, runs are no-ops): {exc}")
        _append(args.log, line)
        return 0

    free = free_gi(args.namespace, args.quota)
    submitted: list[str] = []
    for rung, model in pending:
        name = f"ladder-r{rung}-{model}"
        path = manifest_path(args.repo, rung, model)
        if not path.exists():
            print(f"  MISSING manifest {path} — skipping run")
            continue
        need = requested_gi(path)
        if free < need + MARGIN_GI:
            print(f"  stop: {name} needs {need:.1f}Gi (+{MARGIN_GI} margin), "
                  f"{free:.1f}Gi free")
            break
        if args.dry_run:
            print(f"  would submit {name} ({need:.0f}Gi)")
        else:
            try:
                _kubectl(["create", "-f", str(path)], args.namespace)
                print(f"  submitted {name} ({need:.0f}Gi)")
            except RuntimeError as exc:
                print(f"  FAILED {name}: {exc}")
                break
        submitted.append(name)
        free -= need

    line = (f"{stamp} free={free:.1f}Gi pending={len(pending)} "
            f"submitted={submitted or '[]'}")
    print(line)
    if not args.dry_run:
        _append(args.log, line)
    return 0


def _append(path: Path, line: str) -> None:
    """Log to the PVC — this pod's own stdout dies with its TTL."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        print(f"log write failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
