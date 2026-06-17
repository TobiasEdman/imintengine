"""run_dataset_completion_campaign — advance-one-stage orchestrator.

The orchestrator only ever talks to the cluster through ``JobRunner._kubectl`` and
``JobRunner.read_verdict``; every test drives a ``FakeRunner`` that scripts job
statuses + gate verdicts per stage, so nothing here touches kubectl or ``/data``.

Covered:
  * stages run in their defined order (one transition per invocation);
  * a still-Running job → exit 0, no advance, no apply;
  * a passing gate → mark done + apply the NEXT stage's Job;
  * a FAILING gate → HALT (no further apply, exit 2, sentinel written);
  * idempotent resume (re-run skips completed stages);
  * dry-run applies nothing;
  * promote runs ONLY after every prior gate passed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_dataset_completion_campaign as orch  # noqa: E402

# Stage order the orchestrator must follow (sans args that don't change order).
STAGE_ORDER = [
    "wait-phase1", "phase2-sen2cor", "frameqc-measure", "frameqc-replace",
    "label-restore", "vpp-backfill", "verify", "promote",
]
# Job name per stage (mirrors build_stages / the k8s manifests).
JOB_OF = {
    "wait-phase1": "campaign-phase1-des-recoreg",
    "phase2-sen2cor": "campaign-phase2-sen2cor",
    "frameqc-measure": "campaign-frame-qc-measure",
    "frameqc-replace": "campaign-frame-qc-replace",
    "label-restore": "campaign-label-restore",
    "vpp-backfill": "campaign-vpp-backfill",
    "verify": "campaign-verify",
    "promote": "campaign-promote",
}


class FakeRunner:
    """Scriptable stand-in for ``JobRunner``.

    ``statuses[job_name]`` → the value ``job_status`` returns (default "complete").
    ``verdicts[job_name]`` → the ``GateResult`` ``read_verdict`` returns
        (default a generic ok=true with ``worst_fail_rate`` metric for the
        measure gate). Records every applied manifest in ``applied``.
    """

    def __init__(self, statuses=None, verdicts=None, namespace="ns", dry_run=False):
        self.statuses = statuses or {}
        self.verdicts = verdicts or {}
        self.namespace = namespace
        self.dry_run = dry_run
        self.applied: list[str] = []

    def job_status(self, job_name):
        return self.statuses.get(job_name, "complete")

    def apply(self, manifest):
        if self.dry_run:
            return
        self.applied.append(Path(manifest).name)

    def read_verdict(self, job_name):
        v = self.verdicts.get(job_name)
        if v is not None:
            return v
        # Default: pass, with a benign worst_fail_rate for the measure gate.
        return orch.GateResult(True, "ok", {"worst_fail_rate": 0.1})


def _state(tmp_path) -> str:
    return str(tmp_path / "state.json")


def _read_state(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _advance(state_file, runner, max_fail_frac=0.6, dry_run=False):
    return orch.advance(state_file, "ns", max_fail_frac=max_fail_frac,
                        dry_run=dry_run, runner=runner)


# ── precondition: wait-phase1 ────────────────────────────────────────────────


def test_phase1_not_found_halts(tmp_path):
    """If Phase 1's Job is absent, the precondition gate HALTS (exit 2)."""
    runner = FakeRunner(statuses={"campaign-phase1-des-recoreg": "not_found"})
    rc = _advance(_state(tmp_path), runner)
    assert rc == 2
    assert runner.applied == []  # never apply on a failed precondition
    assert _read_state(_state(tmp_path))["status"] == orch._HALTED


def test_phase1_running_waits(tmp_path):
    """Phase 1 still running → exit 0, no advance, nothing applied, nothing written.

    A pure wait must mutate NOTHING — not the cluster (no apply) and not the state
    file (a wait is not progress). So the state file is never created on this path.
    """
    sf = _state(tmp_path)
    runner = FakeRunner(statuses={"campaign-phase1-des-recoreg": "running"})
    rc = _advance(sf, runner)
    assert rc == 0
    assert runner.applied == []
    assert not Path(sf).exists()  # a wait writes nothing


def test_phase1_complete_advances_and_applies_phase2(tmp_path):
    """Phase 1 Complete → wait-phase1 passes + phase2-sen2cor Job applied."""
    runner = FakeRunner()  # all jobs "complete" by default
    rc = _advance(_state(tmp_path), runner)
    assert rc == 0
    st = _read_state(_state(tmp_path))
    assert "wait-phase1" in st["completed"]
    assert st["current"] == "phase2-sen2cor"
    assert runner.applied == ["campaign-phase2-sen2cor-job.yaml"]


# ── one transition per invocation, in order ──────────────────────────────────


def test_running_stage_job_waits_no_advance(tmp_path):
    """A launched stage whose Job is still Running → exit 0, no new apply."""
    sf = _state(tmp_path)
    # Pre-seed: wait-phase1 done, phase2 launched + running.
    orch.save_state(sf, {"completed": ["wait-phase1"], "current": "phase2-sen2cor",
                         "status": orch._RUNNING})
    runner = FakeRunner(statuses={"campaign-phase2-sen2cor": "running"})
    rc = _advance(sf, runner)
    assert rc == 0
    assert runner.applied == []
    st = _read_state(sf)
    assert "phase2-sen2cor" not in st["completed"]
    assert st["current"] == "phase2-sen2cor"


def test_passing_gate_advances_and_applies_next(tmp_path):
    """phase2 Complete + gate pass → phase2 marked done, frameqc-measure applied."""
    sf = _state(tmp_path)
    orch.save_state(sf, {"completed": ["wait-phase1"], "current": "phase2-sen2cor",
                         "status": orch._RUNNING})
    runner = FakeRunner()  # complete + default-pass verdict
    rc = _advance(sf, runner)
    assert rc == 0
    st = _read_state(sf)
    assert st["completed"] == ["wait-phase1", "phase2-sen2cor"]
    assert st["current"] == "frameqc-measure"
    assert runner.applied == ["campaign-frame-qc-measure-job.yaml"]


def test_full_walk_runs_stages_in_order(tmp_path):
    """Drive the whole campaign; assert the apply sequence matches STAGE_ORDER."""
    sf = _state(tmp_path)
    runner = FakeRunner()  # everything complete + passing
    applied_order: list[str] = []
    # Each advance does at most one transition; loop to completion (cap for safety).
    for _ in range(40):
        before = list(runner.applied)
        rc = _advance(sf, runner)
        applied_order += runner.applied[len(before):]
        assert rc == 0, "no gate should fail in the all-pass walk"
        if _read_state(sf).get("status") == orch._PASSED and \
                len(_read_state(sf)["completed"]) == len(STAGE_ORDER):
            break
    # Every non-precondition stage applied exactly once, in order.
    expected = [f"{JOB_OF[s]}-job.yaml"
                if s != "phase2-sen2cor" else "campaign-phase2-sen2cor-job.yaml"
                for s in STAGE_ORDER if s != "wait-phase1"]
    # job-name → manifest stem mapping (manifests are <job>-job.yaml; QC jobs use
    # the frame-qc spelling) — derive from build_stages to stay in sync.
    stage_man = {s.name: Path(s.manifest).name
                 for s in orch.build_stages(0.6) if s.manifest}
    expected = [stage_man[s] for s in STAGE_ORDER if s != "wait-phase1"]
    assert applied_order == expected
    st = _read_state(sf)
    assert st["completed"] == STAGE_ORDER
    assert st["status"] == orch._PASSED


# ── halting on gate failure ──────────────────────────────────────────────────


def test_failing_gate_halts_and_writes_sentinel(tmp_path):
    """label-restore gate fails → HALT: exit 2, sentinel written, no apply."""
    sf = _state(tmp_path)
    orch.save_state(sf, {"completed": ["wait-phase1", "phase2-sen2cor",
                                       "frameqc-measure", "frameqc-replace"],
                         "current": "label-restore", "status": orch._RUNNING})
    runner = FakeRunner(verdicts={
        "campaign-label-restore": orch.GateResult(False, "no label on sample")})
    rc = _advance(sf, runner)
    assert rc == 2
    assert runner.applied == []  # the NEXT stage (vpp-backfill) is never applied
    st = _read_state(sf)
    assert st["status"] == orch._HALTED
    assert st["halted_stage"] == "label-restore"
    sentinel = tmp_path / orch._SENTINEL_NAME
    assert sentinel.exists()
    assert json.loads(sentinel.read_text())["halted_stage"] == "label-restore"


def test_halted_state_is_sticky(tmp_path):
    """Re-running --advance on a HALTED state is a no-op that re-exits non-zero."""
    sf = _state(tmp_path)
    orch.save_state(sf, {"completed": [], "current": "label-restore",
                         "status": orch._HALTED, "halted_stage": "label-restore",
                         "halted_reason": "x"})
    runner = FakeRunner()
    rc = _advance(sf, runner)
    assert rc == 2
    assert runner.applied == []


def test_measure_gate_absurd_fail_rate_halts(tmp_path):
    """frameqc-measure: an absurd worst_fail_rate (>= max-fail-frac) halts."""
    sf = _state(tmp_path)
    orch.save_state(sf, {"completed": ["wait-phase1", "phase2-sen2cor"],
                         "current": "frameqc-measure", "status": orch._RUNNING})
    runner = FakeRunner(verdicts={
        "campaign-frame-qc-measure": orch.GateResult(
            True, "ok", {"worst_fail_rate": 0.95})})  # absurd
    rc = _advance(sf, runner, max_fail_frac=0.6)
    assert rc == 2
    assert _read_state(sf)["status"] == orch._HALTED
    assert runner.applied == []


def test_missing_verdict_line_halts(tmp_path):
    """A Complete job with NO verdict line → gate fails (no pass on absent evidence)."""
    sf = _state(tmp_path)
    orch.save_state(sf, {"completed": ["wait-phase1", "phase2-sen2cor",
                                       "frameqc-measure", "frameqc-replace",
                                       "label-restore"],
                         "current": "vpp-backfill", "status": orch._RUNNING})
    runner = FakeRunner(verdicts={"campaign-vpp-backfill": None})

    # Force read_verdict to genuinely return None (FakeRunner default would pass).
    def _none(_job):
        return None
    runner.read_verdict = _none  # type: ignore[assignment]

    rc = _advance(sf, runner)
    assert rc == 2
    assert _read_state(sf)["status"] == orch._HALTED


def test_failed_job_condition_halts(tmp_path):
    """A k8s Failed condition on the stage's Job → HALT before the gate."""
    sf = _state(tmp_path)
    orch.save_state(sf, {"completed": ["wait-phase1"], "current": "phase2-sen2cor",
                         "status": orch._RUNNING})
    runner = FakeRunner(statuses={"campaign-phase2-sen2cor": "failed"})
    rc = _advance(sf, runner)
    assert rc == 2
    assert _read_state(sf)["status"] == orch._HALTED


# ── idempotent resume ────────────────────────────────────────────────────────


def test_resume_skips_completed_stages(tmp_path):
    """A state with stages already passed resumes at the first pending stage."""
    sf = _state(tmp_path)
    orch.save_state(sf, {
        "completed": ["wait-phase1", "phase2-sen2cor", "frameqc-measure"],
        "current": None, "status": orch._PASSED})
    runner = FakeRunner()
    rc = _advance(sf, runner)
    assert rc == 0
    # Should launch the next pending stage (frameqc-replace), not re-run earlier ones.
    assert runner.applied == ["campaign-frame-qc-replace-job.yaml"]
    assert _read_state(sf)["current"] == "frameqc-replace"


def test_all_complete_is_terminal_noop(tmp_path):
    """When every stage is done, advance is a no-op that reports complete (exit 0)."""
    sf = _state(tmp_path)
    orch.save_state(sf, {"completed": list(STAGE_ORDER), "current": None,
                         "status": orch._PASSED})
    runner = FakeRunner()
    rc = _advance(sf, runner)
    assert rc == 0
    assert runner.applied == []


# ── dry-run ──────────────────────────────────────────────────────────────────


def test_dry_run_applies_nothing(tmp_path, capsys):
    """--dry-run plan prints stages + next action but applies nothing, writes nothing."""
    sf = _state(tmp_path)
    rc = orch.print_plan(0.6, sf, "ns")
    assert rc == 0
    out = capsys.readouterr().out
    assert "campaign plan (dry-run)" in out
    assert "phase2-sen2cor" in out
    # print_plan never writes the state file.
    assert not Path(sf).exists()


def test_dry_run_advance_records_no_launch(tmp_path):
    """advance(dry_run=True) on a fresh state: gate passes wait-phase1 but the
    next apply is a no-op and no RUNNING launch is recorded."""
    sf = _state(tmp_path)
    runner = FakeRunner(dry_run=True)
    rc = _advance(sf, runner, dry_run=True)
    assert rc == 0
    assert runner.applied == []  # dry-run apply is a no-op


# ── promote ordering ─────────────────────────────────────────────────────────


def test_promote_only_after_all_prior_gates_pass(tmp_path):
    """promote (campaign-promote) is applied ONLY once every prior stage passed."""
    sf = _state(tmp_path)
    # All stages up to and including verify are done; verify just passed.
    orch.save_state(sf, {
        "completed": ["wait-phase1", "phase2-sen2cor", "frameqc-measure",
                      "frameqc-replace", "label-restore", "vpp-backfill", "verify"],
        "current": None, "status": orch._PASSED})
    runner = FakeRunner()
    rc = _advance(sf, runner)
    assert rc == 0
    assert runner.applied == ["campaign-promote-job.yaml"]
    assert _read_state(sf)["current"] == "promote"


def test_promote_not_applied_if_earlier_stage_pending(tmp_path):
    """If any earlier stage is still pending, the campaign never reaches promote."""
    sf = _state(tmp_path)
    # vpp-backfill not done yet → the next apply must be vpp-backfill, NOT promote.
    orch.save_state(sf, {
        "completed": ["wait-phase1", "phase2-sen2cor", "frameqc-measure",
                      "frameqc-replace", "label-restore"],
        "current": None, "status": orch._PASSED})
    runner = FakeRunner()
    rc = _advance(sf, runner)
    assert rc == 0
    assert "campaign-promote-job.yaml" not in runner.applied
    assert runner.applied == ["campaign-vpp-backfill-job.yaml"]


def test_promote_gate_failure_halts_without_swap(tmp_path):
    """If the promote job's verdict is ok=false, the campaign HALTS (exit 2)."""
    sf = _state(tmp_path)
    orch.save_state(sf, {
        "completed": ["wait-phase1", "phase2-sen2cor", "frameqc-measure",
                      "frameqc-replace", "label-restore", "vpp-backfill", "verify"],
        "current": "promote", "status": orch._RUNNING})
    runner = FakeRunner(verdicts={
        "campaign-promote": orch.GateResult(False, "promote rc=1")})
    rc = _advance(sf, runner)
    assert rc == 2
    assert _read_state(sf)["status"] == orch._HALTED


# ── verdict-line parsing (the /data-read mechanism) ──────────────────────────


def test_extract_verdict_picks_last_line():
    logs = (
        "noise\n"
        'CAMPAIGN_GATE_VERDICT={"gate":"verify","ok":false,"reason":"early"}\n'
        "more noise\n"
        'CAMPAIGN_GATE_VERDICT={"gate":"verify","ok":true,"reason":"final"}\n'
    )
    v = orch._extract_verdict(logs)
    assert v is not None and v["ok"] is True and v["reason"] == "final"


def test_extract_verdict_none_when_absent():
    assert orch._extract_verdict("just regular pod logs\nno verdict here") is None


def test_extract_verdict_skips_malformed_json():
    logs = "CAMPAIGN_GATE_VERDICT={not json}\n"
    assert orch._extract_verdict(logs) is None
