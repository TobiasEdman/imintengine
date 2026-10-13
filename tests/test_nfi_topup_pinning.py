"""Step-3 Jobs run the reviewed source, and the re-plan's no-HR-VPP list is explicit (decision 2026-10-13)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PIN = "d4d24f9ab1581ecdc8db739d4271905806401b6a"
PINNED = sorted(
    [*(f"k8s/nfi-topup-{kind}-{y}-job.yaml" for kind in ("fetch", "retry") for y in (2018, 2019, 2021, 2022)),
     "k8s/nfi-topup-vpp-prefetch-job.yaml", "k8s/nfi-topup-labels-job.yaml", "k8s/nfi-topup-enrich-job.yaml",
     "k8s/nfi-topup-tessera-job.yaml", "k8s/nfi-topup-vpp-stamp-job.yaml", "k8s/sen2cor-frame2016-topup-job.yaml",
     "k8s/nfi-topup-mark-frame-gaps-s1-2016-job.yaml"])


def _container(rel: str) -> dict:
    return yaml.safe_load((REPO / rel).read_text())["spec"]["template"]["spec"]["containers"][0]


def _script(rel: str) -> str:
    c = _container(rel)
    return (c.get("args") or c["command"])[-1]


def _env(rel: str) -> dict:
    return {e["name"]: e.get("value") for e in _container(rel)["env"]}


def _blob(path: str) -> bytes | None:
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{PIN}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


@pytest.mark.parametrize("rel", PINNED)
def test_every_step3_job_checks_out_the_pin_before_running_a_script(rel):
    env = _env(rel)
    assert env["SCRIPTS_COMMIT"] == PIN
    script = _script(rel)
    checkout = script.index('git checkout -q "$SCRIPTS_COMMIT"')
    head_guard = script.index('test "$(git rev-parse HEAD)" = "$SCRIPTS_COMMIT"')
    sha_guard = script.index("sha256sum -c -")
    first_python = min(i for i in (script.find("python3 scripts/"), script.find("python scripts/"),
                                   script.find("python - "), script.find("python3 - ")) if i >= 0)
    assert checkout < head_guard < sha_guard < first_python
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


@pytest.mark.parametrize("rel", PINNED)
def test_every_guarded_hash_is_the_pinned_commits_bytes_and_covers_what_runs(rel):
    env, script = _env(rel), _script(rel)
    guarded = {}
    for line in env["SCRIPTS_SHA256"].splitlines():
        digest, path = line.split("  ", 1)
        guarded[path] = digest
    for path, digest in guarded.items():
        blob = _blob(path)
        if blob is None:
            pytest.skip("pinned commit not in this clone")
        assert hashlib.sha256(blob).hexdigest() == digest, path
    for word in script.split():
        if word.startswith("scripts/") and word.endswith(".py"):
            assert word in guarded, (rel, word)


def test_rerunnable_jobs_use_generated_names():
    for rel in ("k8s/nfi-topup-plan-job.yaml", "k8s/nfi-topup-vpp-prefetch-job.yaml"):
        meta = yaml.safe_load((REPO / rel).read_text())["metadata"]
        assert "name" not in meta and meta["generateName"].endswith("-")


def test_planner_and_census_clone_main_not_a_deletable_branch():
    assert _env("k8s/nfi-topup-plan-job.yaml")["PLAN_BRANCH"] == "main"
    assert _env("k8s/nfi-topup-census-job.yaml")["CENSUS_BRANCH"] == "main"


# ---- the planner Job's no-HR-VPP heredoc, executed offline ----

PLAN = "k8s/nfi-topup-plan-job.yaml"
SCRIPT = _script(PLAN)
NO_VPP_BODY = SCRIPT.split("NO_VPP=$(python - <<'PYEOF'\n", 1)[1].split("\nPYEOF", 1)[0]


def _no_vpp(tmp_path, monkeypatch, missing: dict, cached: set, *, years="2018 2021"):
    man = tmp_path / "manifests"; man.mkdir(exist_ok=True)
    for y, cells in missing.items():
        (man / f"nfi{y}_topup_vpp_gaps.json").write_text(json.dumps({"year": int(y), "needed": cells, "missing": cells}))
    env = _env(PLAN)
    monkeypatch.setenv("YEARS", years)
    monkeypatch.setenv("NO_VPP_PROVEN", env["NO_VPP_PROVEN"])
    monkeypatch.setenv("NO_VPP_IF_STILL_MISSING", env["NO_VPP_IF_STILL_MISSING"])
    fake = types.ModuleType("derive_missing_vpp_mgrs"); fake.cached_pairs = lambda path: cached
    body = NO_VPP_BODY.replace('"/data/', json.dumps(str(tmp_path) + "/")[:-1])
    code, out = 0, []
    with patch.dict(sys.modules, {"derive_missing_vpp_mgrs": fake}), \
            patch("builtins.print", lambda *a, **k: out.append(" ".join(map(str, a))) if "file" not in k else None):
        try:
            exec(compile(body, "no-vpp-heredoc", "exec"), {})
        except SystemExit as exc:
            code = exc.code
    return code, out


def test_proven_cells_always_apply_and_a_candidate_still_missing_is_confirmed(tmp_path, monkeypatch):
    code, out = _no_vpp(tmp_path, monkeypatch,
                        {"2018": ["33UVC", "34WCA", "34WCU"], "2021": ["34WCT"]}, cached=set())
    tokens = out[-1].split()
    assert code == 0
    assert tokens[:5] == ["33UVC", "34WCS", "34WCT", "34WCU", "34WCV"] and tokens[5:] == ["2018:34WCA"]


def test_a_candidate_the_retry_filled_is_not_excluded(tmp_path, monkeypatch):
    code, out = _no_vpp(tmp_path, monkeypatch, {"2018": ["33UVC"], "2021": []}, cached={("34WCA", 2018)})
    assert code == 0 and "2018:34WCA" not in out[-1].split()


def test_an_unexpected_missing_cell_refuses_rather_than_dropping_tiles(tmp_path, monkeypatch):
    code, _ = _no_vpp(tmp_path, monkeypatch, {"2018": ["33VWE"], "2021": []}, cached=set())
    assert isinstance(code, str) and "neither proven nor a named retry candidate" in code


def test_a_candidate_is_year_scoped(tmp_path, monkeypatch):
    # 34WCA is a retry candidate for 2018 only; missing in 2021 it is unexpected
    code, _ = _no_vpp(tmp_path, monkeypatch, {"2018": [], "2021": ["34WCA"]}, cached=set())
    assert isinstance(code, str) and "2021" in code


def test_a_cached_pair_is_never_excluded_even_if_the_report_is_stale(tmp_path, monkeypatch):
    code, out = _no_vpp(tmp_path, monkeypatch, {"2018": ["34WDR"], "2021": []}, cached={("34WDR", 2018)})
    assert code == 0 and "2018:34WDR" not in out[-1].split()


def test_replan_refuses_once_a_staging_root_holds_tiles(tmp_path):
    guard = SCRIPT.split('for Y in $YEARS; do\n  S=/data/unified_v2_512_nfi${Y}_staging', 1)
    assert len(guard) == 2
    block = "for Y in $YEARS; do\n  S=/data/unified_v2_512_nfi${Y}_staging" + guard[1].split("\ndone\n", 1)[0] + "\ndone\necho PASSED"
    block = block.replace("/data/", f"{tmp_path}/")
    (tmp_path / "unified_v2_512_nfi2018_staging").mkdir()
    ok = subprocess.run(["bash", "-c", block], env={"YEARS": "2018 2021", "PATH": "/usr/bin:/bin"},
                        capture_output=True, text=True)
    assert ok.returncode == 0 and "PASSED" in ok.stdout          # empty root: re-plan allowed
    (tmp_path / "unified_v2_512_nfi2018_staging/nfi2018_1_2.npz").write_bytes(b"x")
    refused = subprocess.run(["bash", "-c", block], env={"YEARS": "2018 2021", "PATH": "/usr/bin:/bin"},
                             capture_output=True, text=True)
    assert refused.returncode != 0 and "REFUSING" in refused.stdout
