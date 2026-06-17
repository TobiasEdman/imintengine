#!/usr/bin/env python3
"""Resumable orchestrator for the post-Phase-1 dataset-completion + promote campaign.

The re-coreg campaign writes ``/data/unified_v2_512_recoreg``. Phase 1
(``campaign-phase1-des-recoreg``) fills the >=2018 slots; this orchestrator runs
the remaining stages — sen2cor pre-2018 frames, per-frame cloud QC (measure +
replace), label carry-forward, VPP backfill — then a final sample-verify, then
``promote_recoreg.py`` to swap ``_recoreg`` over the live ``unified_v2_512``.

ADVANCE-ONE-STAGE-PER-INVOCATION
--------------------------------
The orchestrator is designed to be fired repeatedly by a scheduled task (CLAUDE.md
§8). Each ``--advance`` invocation makes at most ONE transition and exits, so a cron
firing every N minutes walks the campaign forward without ever blocking a session:

  * skip stages already ``passed`` (idempotent resume);
  * if the current stage's k8s Job is still Running → exit 0 (wait for next firing);
  * if the Job is Complete → run that stage's GATE;
      - gate PASS → mark the stage passed + ``kubectl apply`` the NEXT stage's Job;
      - gate FAIL → set the campaign HALTED, write a sentinel, exit non-zero, and
        NEVER apply anything further. A halted campaign requires human inspection;
        re-running ``--advance`` on a halted state is a no-op that re-exits non-zero.

State (completed stages + the current stage + its status) is persisted to an ATOMIC
JSON file (``--state-file``, default ``/data/debug/campaign_orchestrator_state.json``;
when the orchestrator runs off-cluster, point it at a local path). Re-runs read this
file, so the campaign survives session close.

HOW GATES READ ``/data``
------------------------
The orchestrator runs WHERE kubectl is (a user machine / scheduled agent) — it has
NO PVC mount, so it cannot read ``/data`` tiles directly. Every gate that needs to
inspect tiles therefore runs its CHECK *inside the job's pod* (which does mount the
PVC) and the pod emits a single machine-parseable line to stdout::

    CAMPAIGN_GATE_VERDICT={"gate": "...", "ok": true, "metrics": {...}}

The orchestrator reads that line back via ``kubectl logs job/<name>`` after the Job
reaches Complete. This needs only ``get/logs`` RBAC, no ``exec`` (the pod is gone
after completion), and no shared filesystem. The campaign-authored jobs
(``campaign-verify``, ``campaign-promote``) emit the line directly; the pre-existing
jobs already run an in-pod sanity ``PYEOF`` block, and the gate parses that job's
report JSON via a tiny ``kubectl logs`` verdict line appended for this purpose (the
verify/promote jobs created here own that contract end-to-end).

This module is BUILD + UNIT-TEST ONLY. It shells out to ``kubectl`` but the tests
mock every subprocess call; nothing here touches a real cluster or ``/data``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Repo root — the k8s manifests live under <root>/k8s.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_K8S_DIR = _REPO_ROOT / "k8s"

# The sentinel line a job's pod prints so the orchestrator can read its in-pod
# gate verdict via `kubectl logs` (no PVC access needed on the orchestrator).
_VERDICT_PREFIX = "CAMPAIGN_GATE_VERDICT="

_DEFAULT_NAMESPACE = "prithvi-training-default"
_DEFAULT_STATE_FILE = "/data/debug/campaign_orchestrator_state.json"
# Written next to the state file when a gate halts the campaign — the next session
# / scheduled firing picks it up and the cron's failure-escalation (§8) fires.
_SENTINEL_NAME = "campaign_orchestrator_HALTED.json"

# Status values a stage can carry in the state file.
_PENDING = "pending"   # not yet started (no Job applied)
_RUNNING = "running"   # Job applied, not yet Complete
_PASSED = "passed"     # Job Complete AND its gate passed
_HALTED = "halted"     # gate FAILED — campaign frozen, human inspection required


# ── Stage model ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Stage:
    """One step of the campaign.

    Attributes:
        name: stable identifier used in the state file + CLI logging.
        job_name: the k8s Job's ``metadata.name`` (also the manifest stem). ``None``
            for a precondition stage (``wait-phase1``) that applies nothing and only
            waits on a job some OTHER process launched.
        manifest: path to the ``kubectl apply -f`` manifest. ``None`` for a
            precondition stage.
        gate: the gate function ``(JobRunner, Stage) -> GateResult`` run once the
            stage's Job reaches Complete.
        precondition: True for ``wait-phase1`` — the orchestrator waits for
            ``job_name`` to be Complete but never applies it (Phase 1 is launched
            separately and is running already).
    """

    name: str
    job_name: str | None
    manifest: str | None
    gate: "object"  # callable; typed loosely to keep the dataclass import-light
    precondition: bool = False


@dataclass
class GateResult:
    """Verdict of a stage gate."""

    ok: bool
    reason: str
    metrics: dict = field(default_factory=dict)


# ── kubectl wrapper ──────────────────────────────────────────────────────────


class JobRunner:
    """Thin, testable wrapper over the ``kubectl`` calls the orchestrator needs.

    Every method shells out via :meth:`_kubectl`; the tests monkeypatch that one
    seam. ``dry_run`` makes the mutating calls (``apply``) no-ops that only log.
    """

    def __init__(self, namespace: str, *, dry_run: bool = False,
                 poll_timeout_s: int = 30):
        self.namespace = namespace
        self.dry_run = dry_run
        self.poll_timeout_s = poll_timeout_s

    # -- low-level seam (the ONLY place subprocess is invoked) ----------------

    def _kubectl(self, args: list[str], *, timeout: int | None = None) -> tuple[int, str, str]:
        """Run ``kubectl -n <ns> <args>``; return ``(returncode, stdout, stderr)``.

        Raises ``RuntimeError`` only on a missing ``kubectl`` binary or a timeout —
        a non-zero return code is returned to the caller, which decides whether it
        is fatal (e.g. ``get job`` on a missing job is "not found", not a crash).
        """
        cmd = ["kubectl", "-n", self.namespace, *args]
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout if timeout is not None else self.poll_timeout_s,
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"kubectl not found on PATH: {e}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"kubectl timed out: {' '.join(cmd)}") from e
        return r.returncode, r.stdout, r.stderr

    # -- job status -----------------------------------------------------------

    def job_status(self, job_name: str) -> str:
        """Return one of ``complete`` / ``failed`` / ``running`` / ``not_found``.

        Reads ``.status.conditions`` via jsonpath. A Job is Complete when a
        ``Complete`` condition is ``True``, Failed when a ``Failed`` condition is
        ``True``; an existing-but-condition-less Job is still ``running``.
        """
        rc, out, err = self._kubectl([
            "get", "job", job_name,
            "-o", "jsonpath={range .status.conditions[*]}{.type}={.status};{end}",
        ])
        if rc != 0:
            low = (err or out).lower()
            if "notfound" in low.replace(" ", "") or "not found" in low:
                return "not_found"
            raise RuntimeError(f"kubectl get job {job_name} failed: {err.strip()}")
        conds = dict(
            part.split("=", 1)
            for part in out.strip().strip(";").split(";")
            if "=" in part
        )
        if conds.get("Failed") == "True":
            return "failed"
        if conds.get("Complete") == "True":
            return "complete"
        return "running"

    def apply(self, manifest: str) -> None:
        """``kubectl apply -f <manifest>`` (no-op under ``dry_run``)."""
        if self.dry_run:
            print(f"    [dry-run] would apply: kubectl -n {self.namespace} "
                  f"apply -f {manifest}")
            return
        if not os.path.exists(manifest):
            raise RuntimeError(f"manifest not found: {manifest}")
        rc, out, err = self._kubectl(["apply", "-f", manifest], timeout=60)
        if rc != 0:
            raise RuntimeError(f"kubectl apply -f {manifest} failed: {err.strip()}")
        print(f"    applied: {out.strip() or manifest}")

    def read_verdict(self, job_name: str) -> GateResult | None:
        """Parse the ``CAMPAIGN_GATE_VERDICT=`` line from a job's pod logs.

        Returns a :class:`GateResult` built from the JSON payload, or ``None`` when
        no verdict line is present (the caller treats a missing verdict as a HALT —
        a gate that can't read its own verdict must not pass silently).
        """
        rc, out, err = self._kubectl(["logs", f"job/{job_name}", "--tail", "-1"],
                                     timeout=60)
        if rc != 0:
            return None
        payload = _extract_verdict(out)
        if payload is None:
            return None
        return GateResult(
            ok=bool(payload.get("ok", False)),
            reason=str(payload.get("reason", payload.get("gate", "verdict"))),
            metrics=dict(payload.get("metrics", {})),
        )


def _extract_verdict(logs: str) -> dict | None:
    """Return the JSON dict from the LAST ``CAMPAIGN_GATE_VERDICT=`` line, or None."""
    found: dict | None = None
    for line in logs.splitlines():
        line = line.strip()
        if line.startswith(_VERDICT_PREFIX):
            try:
                found = json.loads(line[len(_VERDICT_PREFIX):])
            except json.JSONDecodeError:
                continue
    return found


# ── Gates ────────────────────────────────────────────────────────────────────
#
# Every gate is "conservative — halt on ANY doubt". The substantive checks (tile
# shapes, VPP counts, fail-rate sanity) run IN-POD and arrive as the verdict line;
# the orchestrator-side gate validates that the verdict exists, is well-formed, and
# satisfies the orchestrator-level bound (e.g. fail-rate < --max-fail-frac). A
# missing/garbled verdict ⇒ ok=False (never pass on absence of evidence).


def _verdict_gate(runner: JobRunner, stage: Stage) -> GateResult:
    """Generic gate: trust the in-pod verdict line, halt if it is absent/false."""
    v = runner.read_verdict(stage.job_name) if stage.job_name else None
    if v is None:
        return GateResult(False, f"{stage.name}: no CAMPAIGN_GATE_VERDICT line in "
                                 f"job logs (cannot confirm success)")
    if not v.ok:
        return GateResult(False, f"{stage.name}: in-pod verdict failed: {v.reason}",
                          v.metrics)
    return GateResult(True, f"{stage.name}: in-pod verdict ok: {v.reason}", v.metrics)


def _gate_wait_phase1(runner: JobRunner, stage: Stage) -> GateResult:
    """Precondition gate: Phase-1 reaching Complete IS the pass.

    Phase 1 has no orchestrator-authored verdict line (it predates this), so the
    gate is satisfied purely by the Complete condition the dispatcher already
    confirmed before calling the gate. No tile read needed here.
    """
    return GateResult(True, "phase1 Job is Complete — preconditions met")


def _gate_frameqc_measure(max_fail_frac: float):
    """Measure gate (closure over ``--max-fail-frac``): verdict ok AND the worst
    per-slot >20%-cloud fail-rate is within a sane bound. An absurd fail-rate
    (e.g. 0.9 of frames cloudy) means the metric or the data is wrong → halt."""

    def _gate(runner: JobRunner, stage: Stage) -> GateResult:
        base = _verdict_gate(runner, stage)
        if not base.ok:
            return base
        worst = float(base.metrics.get("worst_fail_rate", 1.0))
        if worst >= max_fail_frac:
            return GateResult(
                False,
                f"frameqc-measure: worst per-slot fail-rate {worst:.2f} >= "
                f"--max-fail-frac {max_fail_frac:.2f} (absurd → something is wrong)",
                base.metrics)
        return GateResult(True, f"frameqc-measure: worst fail-rate {worst:.2f} "
                                f"within bound", base.metrics)

    return _gate


# ── Stage list ───────────────────────────────────────────────────────────────


def build_stages(max_fail_frac: float) -> list[Stage]:
    """The campaign stages, IN ORDER. See module docstring for the model.

    Job names + manifest stems mirror the committed k8s/*.yaml files exactly:
      wait-phase1     precondition on campaign-phase1-des-recoreg (no apply)
      phase2-sen2cor  campaign-phase2-sen2cor-job.yaml
      frameqc-measure campaign-frame-qc-measure-job.yaml
      frameqc-replace campaign-frame-qc-replace-job.yaml
      label-restore   campaign-label-restore-job.yaml
      vpp-backfill    campaign-vpp-backfill-job.yaml
      verify          campaign-verify-job.yaml          (authored here)
      promote         campaign-promote-job.yaml         (authored here)
    """
    k = lambda name: str(_K8S_DIR / name)  # noqa: E731
    return [
        Stage("wait-phase1", "campaign-phase1-des-recoreg", None,
              _gate_wait_phase1, precondition=True),
        Stage("phase2-sen2cor", "campaign-phase2-sen2cor",
              k("campaign-phase2-sen2cor-job.yaml"), _verdict_gate),
        Stage("frameqc-measure", "campaign-frame-qc-measure",
              k("campaign-frame-qc-measure-job.yaml"),
              _gate_frameqc_measure(max_fail_frac)),
        Stage("frameqc-replace", "campaign-frame-qc-replace",
              k("campaign-frame-qc-replace-job.yaml"), _verdict_gate),
        Stage("label-restore", "campaign-label-restore",
              k("campaign-label-restore-job.yaml"), _verdict_gate),
        Stage("vpp-backfill", "campaign-vpp-backfill",
              k("campaign-vpp-backfill-job.yaml"), _verdict_gate),
        Stage("verify", "campaign-verify",
              k("campaign-verify-job.yaml"), _verdict_gate),
        Stage("promote", "campaign-promote",
              k("campaign-promote-job.yaml"), _verdict_gate),
    ]


# ── State persistence (atomic) ───────────────────────────────────────────────


def load_state(state_file: str) -> dict:
    """Read the orchestrator state, or a fresh ``pending`` state if absent."""
    try:
        with open(state_file) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"completed": [], "current": None, "status": _PENDING,
                "halted_stage": None, "halted_reason": None}


def save_state(state_file: str, state: dict) -> None:
    """Atomically persist the state (tmp in same dir + ``os.replace``)."""
    p = Path(state_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp, state_file)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_sentinel(state_file: str, stage: Stage, result: GateResult) -> str:
    """Write the HALTED sentinel next to the state file; return its path."""
    sentinel = str(Path(state_file).parent / _SENTINEL_NAME)
    save_state(sentinel, {
        "halted_stage": stage.name,
        "job": stage.job_name,
        "reason": result.reason,
        "metrics": result.metrics,
        "halted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": "campaign frozen — inspect the job's logs, fix, then clear this "
                  "sentinel + reset the stage's status to advance.",
    })
    return sentinel


# ── Orchestration core ───────────────────────────────────────────────────────


def _next_pending(stages: list[Stage], completed: set[str]) -> Stage | None:
    for s in stages:
        if s.name not in completed:
            return s
    return None


def advance(
    state_file: str,
    namespace: str,
    *,
    max_fail_frac: float,
    dry_run: bool = False,
    runner: JobRunner | None = None,
) -> int:
    """Advance the campaign by AT MOST one stage. Returns a process exit code.

    Exit codes:
        0  progress made, OR current job still running (wait for next firing),
           OR campaign already fully complete.
        2  the campaign is (or just became) HALTED — a gate failed.
    """
    stages = build_stages(max_fail_frac)
    by_name = {s.name: s for s in stages}
    state = load_state(state_file)

    if state.get("status") == _HALTED:
        print(f"=== HALTED at stage '{state.get('halted_stage')}': "
              f"{state.get('halted_reason')} ===")
        print("    Campaign is frozen. Inspect the job, fix, clear the sentinel + "
              "reset the stage status to resume. No further stages will run.")
        return 2

    runner = runner or JobRunner(namespace, dry_run=dry_run)
    completed = set(state.get("completed", []))

    stage = _next_pending(stages, completed)
    if stage is None:
        print("=== campaign complete — all stages passed (including promote) ===")
        state["status"] = _PASSED
        save_state(state_file, state)
        return 0

    print(f"=== campaign orchestrator: current stage '{stage.name}' "
          f"(completed: {len(completed)}/{len(stages)}) ===")

    # 1) For a non-precondition stage that hasn't been launched yet, APPLY its Job
    #    and exit (next firing polls it). The precondition stage applies nothing.
    if not stage.precondition and state.get("current") != stage.name:
        print(f"  stage '{stage.name}' not yet launched → applying its Job")
        runner.apply(stage.manifest)
        if dry_run:
            # dry-run mutates nothing — don't record a launch that didn't happen.
            print("  [dry-run] not recording launch; state unchanged")
            return 0
        state["current"] = stage.name
        state["status"] = _RUNNING
        save_state(state_file, state)
        print(f"  applied {stage.job_name}; exit 0, poll on next firing")
        return 0

    # 2) Poll the stage's Job.
    job_state = runner.job_status(stage.job_name)
    print(f"  job '{stage.job_name}' status: {job_state}")

    if job_state == "not_found":
        if stage.precondition:
            # Phase 1 must be launched by its own process; we only wait for it.
            return _halt(state_file, state, stage,
                         GateResult(False, f"{stage.job_name} not found — Phase 1 "
                                           f"must be launched + Complete first"))
        # A non-precondition job we believe we launched has vanished → re-apply.
        print(f"  '{stage.job_name}' missing though we launched it → re-applying")
        runner.apply(stage.manifest)
        return 0

    if job_state == "running":
        print(f"  '{stage.job_name}' still running → exit 0, wait for next firing")
        return 0

    if job_state == "failed":
        return _halt(state_file, state, stage,
                     GateResult(False, f"{stage.job_name} Job FAILED "
                                       f"(k8s Failed condition)"))

    # job_state == "complete" → run the gate.
    print(f"  '{stage.job_name}' Complete → running gate")
    result = stage.gate(runner, stage)
    print(f"  gate verdict: {'PASS' if result.ok else 'FAIL'} — {result.reason}")
    if result.metrics:
        print(f"  gate metrics: {json.dumps(result.metrics, sort_keys=True)}")

    if not result.ok:
        return _halt(state_file, state, stage, result)

    # PASS → mark this stage complete + apply the NEXT stage's Job (if any).
    completed.add(stage.name)
    state["completed"] = sorted(completed, key=lambda n: list(by_name).index(n))
    state["current"] = None
    state["status"] = _PASSED
    save_state(state_file, state)

    nxt = _next_pending(stages, completed)
    if nxt is None:
        print("=== final stage passed — campaign complete ===")
        return 0
    if nxt.precondition:
        # The only precondition is wait-phase1 (always first); never reached here.
        return 0
    print(f"  → advancing: applying next stage '{nxt.name}' ({nxt.job_name})")
    runner.apply(nxt.manifest)
    if not dry_run:
        state["current"] = nxt.name
        state["status"] = _RUNNING
        save_state(state_file, state)
    return 0


def _halt(state_file: str, state: dict, stage: Stage, result: GateResult) -> int:
    """Freeze the campaign: persist HALTED, write the sentinel, return exit 2."""
    state["status"] = _HALTED
    state["halted_stage"] = stage.name
    state["halted_reason"] = result.reason
    save_state(state_file, state)
    sentinel = write_sentinel(state_file, stage, result)
    print(f"=== HALT: gate '{stage.name}' failed — {result.reason} ===")
    print(f"    sentinel written: {sentinel}")
    print("    NO further stages will run. Campaign requires human inspection.")
    return 2


# ── Dry-run planner ──────────────────────────────────────────────────────────


def print_plan(max_fail_frac: float, state_file: str, namespace: str) -> int:
    """Print the stage plan + what the NEXT advance would apply. No kubectl writes."""
    stages = build_stages(max_fail_frac)
    state = load_state(state_file)
    completed = set(state.get("completed", []))
    print("=== campaign plan (dry-run) ===")
    print(f"  namespace : {namespace}")
    print(f"  state-file: {state_file}  (status={state.get('status')})")
    print(f"  max-fail-frac (frameqc-measure gate): {max_fail_frac}")
    print("  stages:")
    for i, s in enumerate(stages):
        mark = "x" if s.name in completed else " "
        kind = "precondition" if s.precondition else "apply+gate"
        man = os.path.basename(s.manifest) if s.manifest else "(no manifest)"
        print(f"    [{mark}] {i}. {s.name:16s} {kind:12s} job={s.job_name}  {man}")
    nxt = _next_pending(stages, completed)
    if nxt is None:
        print("  next action: none — all stages complete")
    elif nxt.precondition:
        print(f"  next action: WAIT for {nxt.job_name} to reach Complete "
              f"(no apply)")
    else:
        print(f"  next action: WOULD apply {os.path.basename(nxt.manifest)} "
              f"then poll/gate")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Resumable, advance-one-stage orchestrator for the "
                    "dataset-completion + promote campaign.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--advance", action="store_true", default=True,
                   help="Advance the campaign by one stage (default action).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan + what WOULD be applied; no kubectl writes.")
    p.add_argument("--state-file", default=_DEFAULT_STATE_FILE,
                   help=f"Atomic JSON state path (default {_DEFAULT_STATE_FILE}).")
    p.add_argument("--max-fail-frac", type=float, default=0.6,
                   help="frameqc-measure gate: halt if the worst per-slot "
                        ">20%%-cloud fail-rate is >= this (default 0.6).")
    p.add_argument("--namespace", default=_DEFAULT_NAMESPACE,
                   help=f"k8s namespace (default {_DEFAULT_NAMESPACE}).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        return print_plan(args.max_fail_frac, args.state_file, args.namespace)
    return advance(
        args.state_file, args.namespace,
        max_fail_frac=args.max_fail_frac, dry_run=False)


if __name__ == "__main__":
    raise SystemExit(main())
