"""In-cluster chain for the Prithvi-600M b28 runs (task 0171).

k8s/prithvi-b28-chain-cronjob.yaml queues im023-prithvi600m-b28 and -control
once im023-prithvi600m-smoke has succeeded, from job specs pinned in its
ConfigMap. The specs must stay the ones queued with the smoke at 2108630,
whatever k8s/im023/prithvi600m/ is re-pinned to. chain.sh is run from the
ConfigMap against a fake kubectl and date; nothing here contacts a cluster.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "k8s" / "prithvi-b28-chain-cronjob.yaml"
NAME = "prithvi-b28-chain"
SMOKE = "im023-prithvi600m-smoke"
SOURCE_SHA = "21086305d34a1f88cd737bc9c894b8b5db905860"
# sha256 of the specs queued with the smoke on 2026-11-14 (task 0167).
PINNED = {
    "train-b28-job.yaml": ("im023-prithvi600m-b28",
                           "d1cd2b418a6f58585a6841a0d5caf1c1474a7fa03713878bfad2af0db201f91e"),
    "train-control-job.yaml": ("im023-prithvi600m-control",
                               "20f4b1d6b7cad543fb8e71846bd3a2662f48167ae9abce201cf325cb2d2d9ec2"),
}
BEFORE, EXPIRY = "20261128195959", "20261128200000"

FAKE_KUBECTL = r'''#!/usr/bin/env python3
import json, os, sys
argv = sys.argv[1:]
state = json.load(open(os.environ["FAKE_K8S"]))
with open(os.environ["FAKE_CALLS"], "a") as f:
    f.write(json.dumps(argv) + "\n")
line = " ".join(argv)
if any(s in line for s in state.get("fail", [])):
    sys.exit("Error from server: refused (fake)")
verb = argv[2]
if verb == "get" and argv[3] == "cronjob":
    print(state.get("outcome", ""), end="")
elif verb == "get":
    jobs = state.get("jobs", {})
    name = argv[4]
    if name not in jobs:
        sys.exit(0 if "--ignore-not-found" in argv else 'Error from server (NotFound)')
    if argv[argv.index("-o") + 1] == "name":
        print(f"job.batch/{name}")
    elif "reason" in line:
        print("BackoffLimitExceeded: Job has reached the specified backoff limit")
    else:
        c = jobs[name]
        print(f"{name}:c={c.get('Complete', '')}:f={c.get('Failed', '')}", end="")
else:
    print(f"{verb} ok")
'''

FAKE_DATE = r'''#!/usr/bin/env python3
import datetime as d, os, sys
now = d.datetime.strptime(os.environ["FAKE_NOW"], "%Y%m%d%H%M%S").replace(tzinfo=d.timezone.utc)
fmt = next(a for a in sys.argv[1:] if a.startswith("+"))[1:]
print(int(now.timestamp()) if fmt == "%s" else now.strftime(fmt))
'''


def docs() -> dict[str, dict]:
    return {d["kind"]: d for d in yaml.safe_load_all(MANIFEST.read_text())}


def configmap() -> dict[str, str]:
    return docs()["ConfigMap"]["data"]


def test_configmap_carries_exactly_the_pinned_specs():
    data = configmap()
    assert set(data) == {"chain.sh", *PINNED}
    for key, (job, digest) in PINNED.items():
        assert hashlib.sha256(data[key].encode()).hexdigest() == digest, key
        spec = yaml.safe_load(data[key])
        assert spec["kind"] == "Job" and spec["metadata"]["name"] == job
        assert re.findall(r"^\s*SHA=(\S+)$", data[key], re.M) == [SOURCE_SHA]
    # chain.sh re-checks the same digests in the pod and creates these names.
    script = data["chain.sh"]
    for key, (job, digest) in PINNED.items():
        assert f"{digest}  {key}" in script and job in script


def test_cronjob_is_a_small_pinned_forbid_schedule():
    cron = docs()["CronJob"]
    spec = cron["spec"]
    minutes = [int(m) for m in spec["schedule"].split()[0].split(",")]
    assert spec["schedule"].split()[1:] == ["*"] * 4
    assert [b - a for a, b in zip(minutes, minutes[1:] + [minutes[0] + 60])] == [15] * 4
    assert spec["concurrencyPolicy"] == "Forbid" and spec["suspend"] is False
    assert spec["successfulJobsHistoryLimit"] <= 3 and spec["failedJobsHistoryLimit"] <= 3
    pod = spec["jobTemplate"]["spec"]["template"]["spec"]
    assert pod["serviceAccountName"] == NAME and pod["restartPolicy"] == "Never"
    (container,) = pod["containers"]
    assert re.fullmatch(r"[\w./-]+@sha256:[0-9a-f]{64}", container["image"])
    assert container["resources"]["requests"] == container["resources"]["limits"]
    assert container["command"] == ["sh", "/chain/chain.sh"]
    (mount,) = container["volumeMounts"]
    (volume,) = pod["volumes"]
    assert mount == {"name": volume["name"], "mountPath": "/chain", "readOnly": True}
    assert volume["configMap"]["name"] == docs()["ConfigMap"]["metadata"]["name"]


def test_rbac_cannot_delete_or_touch_other_cronjobs():
    d = docs()
    assert d["ServiceAccount"]["metadata"]["name"] == NAME
    rules = {r["resources"][0]: r for r in d["Role"]["rules"]}
    assert set(rules) == {"jobs", "cronjobs"}
    assert sorted(rules["jobs"]["verbs"]) == ["create", "get", "list"]
    assert sorted(rules["cronjobs"]["verbs"]) == ["get", "patch"]
    assert rules["cronjobs"]["resourceNames"] == [NAME] == [d["CronJob"]["metadata"]["name"]]
    binding = d["RoleBinding"]
    assert binding["roleRef"]["name"] == d["Role"]["metadata"]["name"]
    assert binding["subjects"] == [{"kind": "ServiceAccount", "name": NAME,
                                    "namespace": "prithvi-training-default"}]


def test_status_templates_are_the_ones_checked_on_ice():
    # The fake kubectl does not evaluate jsonpath, so the templates are pinned
    # to the forms run against ICE on 2026-11-16: a pending job gave ":c=:f=",
    # a completed one ":c=True:f=", a failed one ":c=:f=True", and the outcome
    # key read back from an annotated object.
    script = configmap()["chain.sh"]
    assert ("-o jsonpath='{.metadata.name}:c={.status.conditions[?(@.type==\"Complete\")].status}"
            ":f={.status.conditions[?(@.type==\"Failed\")].status}'") in script
    assert '-o jsonpath="{.metadata.annotations.$SELF/outcome}"' in script


def tick(tmp_path, *, jobs, now=BEFORE, fail=(), tamper=False, outcome=""):
    """Run chain.sh once from a copy of the ConfigMap; return (rc, stdout, calls, patches)."""
    chain = tmp_path / "chain"
    chain.mkdir()
    for key, text in configmap().items():
        (chain / key).write_text(text)
    if tamper:
        (chain / "train-control-job.yaml").write_text(configmap()["train-control-job.yaml"] + "# edited\n")
    fake = tmp_path / "bin"
    fake.mkdir()
    for name, body in (("kubectl", FAKE_KUBECTL), ("date", FAKE_DATE)):
        (fake / name).write_text(body.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
        (fake / name).chmod(0o755)
    (tmp_path / "state.json").write_text(json.dumps({"jobs": jobs, "fail": list(fail), "outcome": outcome}))
    calls = tmp_path / "calls.jsonl"
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "KUBECONFIG": os.devnull,
           "FAKE_K8S": str(tmp_path / "state.json"), "FAKE_CALLS": str(calls), "FAKE_NOW": now}
    result = subprocess.run(["sh", str(chain / "chain.sh")], env=env, capture_output=True, text=True)
    log = [json.loads(l) for l in calls.read_text().splitlines()] if calls.exists() else []
    assert all(a[:2] == ["-n", "prithvi-training-default"] for a in log)
    assert not any(v in a for a in log for v in ("delete", "apply", "replace", "rollout", "scale"))
    patches = [json.loads(a[a.index("-p") + 1]) for a in log if a[2] == "patch"]
    assert all(a[3:5] == ["cronjob", NAME] for a in log if a[2] == "patch")
    return result.returncode, result.stdout, [a[2:] for a in log], patches


def creates(calls):
    return [(c[1] == "--dry-run=server", c[c.index("-f") + 1]) for c in calls if c[0] == "create"]


def suspended_with(patches, phrase):
    (patch,) = patches
    assert patch["spec"] == {"suspend": True}
    assert phrase in patch["metadata"]["annotations"][f"{NAME}/outcome"]
    return True


@pytest.mark.parametrize("conditions", [{}, {"Complete": "False"}], ids=["pending", "running"])
def test_an_unfinished_smoke_changes_nothing(tmp_path, conditions):
    rc, out, calls, patches = tick(tmp_path, jobs={SMOKE: conditions})
    assert rc == 0 and "smoke not finished" in out
    assert creates(calls) == [] and patches == []


def test_a_passed_smoke_dry_runs_both_then_creates_both_then_suspends(tmp_path):
    rc, out, calls, patches = tick(tmp_path, jobs={SMOKE: {"Complete": "True"}})
    assert rc == 0
    assert creates(calls) == [(True, "train-b28-job.yaml"), (True, "train-control-job.yaml"),
                              (False, "train-b28-job.yaml"), (False, "train-control-job.yaml")]
    assert calls[-1][0] == "patch" and suspended_with(patches, f"{SMOKE} succeeded")


def test_a_recorded_outcome_is_never_re_armed(tmp_path):
    rc, out, calls, patches = tick(tmp_path, jobs={SMOKE: {"Complete": "True"}},
                                   outcome=f"2026-11-20T08:07:30Z {SMOKE} succeeded; ... created")
    assert rc == 0 and "chain already ended" in out
    assert creates(calls) == [] and not any(c[:2] == ["get", "job"] for c in calls)
    assert patches == [{"spec": {"suspend": True}}]


def test_a_failed_create_after_both_dry_runs_suspends_with_the_file(tmp_path):
    rc, out, calls, patches = tick(tmp_path, jobs={SMOKE: {"Complete": "True"}},
                                   fail=["create -f train-control-job.yaml"])
    assert rc == 0 and creates(calls)[-2:] == [(False, "train-b28-job.yaml"),
                                               (False, "train-control-job.yaml")]
    assert suspended_with(patches, "creating train-control-job.yaml failed")


def test_a_failed_smoke_creates_nothing_and_suspends(tmp_path):
    rc, out, calls, patches = tick(tmp_path, jobs={SMOKE: {"Failed": "True"}})
    assert rc == 0 and "BackoffLimitExceeded" in out
    assert creates(calls) == [] and suspended_with(patches, f"{SMOKE} FAILED")


@pytest.mark.parametrize("existing", [job for job, _ in PINNED.values()])
def test_an_existing_full_run_suspends_without_reading_the_smoke(tmp_path, existing):
    rc, out, calls, patches = tick(tmp_path, jobs={SMOKE: {"Complete": "True"}, existing: {}})
    assert rc == 0 and creates(calls) == []
    assert not any(SMOKE in c for c in calls)
    assert suspended_with(patches, f"{existing} already exists")


def test_a_missing_smoke_creates_nothing_and_suspends(tmp_path):
    rc, out, calls, patches = tick(tmp_path, jobs={})
    assert rc == 0 and creates(calls) == [] and suspended_with(patches, "not found")


@pytest.mark.parametrize("fail", [(), ("get",)], ids=["reads-ok", "reads-failing"])
def test_the_hard_stop_holds_even_for_a_passed_smoke(tmp_path, fail):
    rc, out, calls, patches = tick(tmp_path, jobs={SMOKE: {"Complete": "True"}}, now=EXPIRY, fail=fail)
    assert rc == 0 and [c[0] for c in calls] == ["patch"] and suspended_with(patches, "expired")


def test_one_refused_dry_run_creates_neither(tmp_path):
    rc, out, calls, patches = tick(tmp_path, jobs={SMOKE: {"Complete": "True"}},
                                   fail=["create --dry-run=server -f train-control-job.yaml"])
    assert rc == 0 and all(dry for dry, _ in creates(calls))
    assert suspended_with(patches, "dry-run refused train-control-job.yaml")


@pytest.mark.parametrize("conditions", [{}, {"Complete": "True"}], ids=["pending", "passed"])
def test_an_edited_spec_stops_the_chain_on_the_next_tick(tmp_path, conditions):
    rc, out, calls, patches = tick(tmp_path, jobs={SMOKE: conditions}, tamper=True)
    assert rc == 0 and creates(calls) == [] and not any(SMOKE in c for c in calls)
    assert suspended_with(patches, "differ from the pinned copies")


@pytest.mark.parametrize("read", [f"get cronjob {NAME}", "get job im023-prithvi600m-b28",
                                  f"get job {SMOKE}"])
def test_a_read_error_fails_the_tick_without_suspending(tmp_path, read):
    rc, out, calls, patches = tick(tmp_path, jobs={SMOKE: {"Complete": "True"}}, fail=[read])
    assert rc == 1 and creates(calls) == [] and patches == []


def test_a_failed_suspend_fails_the_tick(tmp_path):
    rc, out, calls, patches = tick(tmp_path, jobs={SMOKE: {"Failed": "True"}}, fail=["patch"])
    assert rc == 1 and creates(calls) == [] and "could not suspend" in out
