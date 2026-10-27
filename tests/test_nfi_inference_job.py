"""The frozen-set inference Jobs publish safely and never show a score.

The Jobs' own shell scripts (k8s/nfi-inference-job.template.yaml and
k8s/nfi-inference-probe-job.yaml) run here against a fake validator in a
temporary directory: the validator command, the scratch path and the image's
Python are the only substitutions. The template must publish good cells
(parquet before metadata, 0644), refuse an existing output without touching
it, keep going after a failed rung, refuse a copy whose bytes differ from
prediction_sha256 without publishing its metadata, and exit nonzero if any
cell failed. Neither Job may let the validator's results JSON reach the Pod
log, including when the validator fails after printing it (Codex, #113).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "k8s" / "nfi-inference-job.template.yaml"
PROBE = REPO / "k8s" / "nfi-inference-probe-job.yaml"
MODELS = ("clay", "croma", "prithvi300m", "prithvi300m4f", "prithvi600m", "terramind", "tessera")
VALIDATOR = "/opt/venvs/model/bin/python -u /opt/imintengine/scripts/validate_against_nfi.py"
# Distinctive numbers in the fake report; none may appear in a Pod log.
SECRETS = ("0.8137", "4711", "0.9373", "0.6421", "38211")

FAKE_VALIDATOR = '''
import hashlib, json, os, sys
args = sys.argv[1:]
get = lambda k: args[args.index(k) + 1]
cell, dump = get("--cell"), get("--dump-per-plot")
print("plot index: 5,194 co-located plots on 1269 tiles")
print('{"overall_accuracy": 0.8137}')
if os.environ.get("PARTIAL_REPORT"):
    # As validate_against_nfi.py does: the pretty results JSON first, then
    # the writes, which fail here.
    print(json.dumps({"confusion": [[4711, 12], [3, 38211]], "per_class": {
        "1": {"auroc": 0.9373, "aupr": 0.6421}}}, indent=2))
    raise OSError("No space left on device")
if cell.endswith(os.environ.get("FAIL_RUNG", "_none")):
    print("Traceback: simulated failure")
    sys.exit(1)
data = f"predictions for {cell}".encode()
open(dump, "wb").write(data)
digest = "0" * 64 if os.environ.get("BAD_SHA") else hashlib.sha256(data).hexdigest()
open(dump + ".meta.json", "w").write(json.dumps(
    {"cell": cell, "status": "success", "prediction_sha256": digest}))
print(f"wrote {dump} (5194 plots)")
'''


def render(path: Path, model: str = "clay") -> dict:
    return yaml.safe_load(path.read_text().replace("__MODEL__", model))


def container(path: Path) -> dict:
    return render(path)["spec"]["template"]["spec"]["containers"][0]


def run_job(tmp_path: Path, path: Path = TEMPLATE, **env: str) -> tuple[int, str, Path]:
    spec = container(path)
    fake = tmp_path / "fake_validator.py"
    fake.write_text(FAKE_VALIDATOR)
    scratch, out = tmp_path / "scr", tmp_path / "out"
    scratch.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    script = spec["args"][0]
    assert script.count(VALIDATOR) == 1
    script = (script.replace("/scratch", str(scratch))
                    .replace(VALIDATOR, f"python3 -u {fake}")
                    .replace("/usr/local/bin/python", "python3"))
    variables = {e["name"]: e["value"] for e in spec["env"]}
    keep = ("MODEL", "CELL", "SAFE_DIAGNOSTICS")
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=120,
        env={"PATH": os.environ["PATH"], "OUT": str(out),
             "MANIFEST": "/frozen/manifest.json", "MANIFEST_SHA256": "a" * 64,
             **{k: variables[k] for k in keep if k in variables}, **env})
    return result.returncode, result.stdout + result.stderr, out


def test_rendered_jobs_are_distinct_and_pinned() -> None:
    names = set()
    for model in MODELS:
        doc = render(TEMPLATE, model)
        spec = doc["spec"]["template"]["spec"]["containers"][0]
        names.add(doc["metadata"]["name"])
        assert "@sha256:" in spec["image"]
        assert {e["name"]: e["value"] for e in spec["env"]}["MODEL"] == model
        writable = [m for m in spec["volumeMounts"] if m["name"] == "data" and not m.get("readOnly")]
        assert [m["subPath"] for m in writable] == ["nfi/nfi_inference_2026-10-27"]
    assert len(names) == len(MODELS)


def test_failure_output_is_an_allowlist_shared_by_both_jobs() -> None:
    patterns = []
    for path in (TEMPLATE, PROBE):
        spec = container(path)
        assert "grep -v" not in spec["args"][0], path.name
        patterns.append({e["name"]: e["value"] for e in spec["env"]}["SAFE_DIAGNOSTICS"])
    assert patterns[0] == patterns[1]


def test_publishes_refuses_and_continues(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "nfi-per-plot-clay_r2.parquet").write_text("pre-existing")
    code, log, out = run_job(tmp_path, FAIL_RUNG="_r3")
    assert code != 0
    assert "clay_r2: output already exists; refusing to overwrite" in log
    assert "failed: clay_r2 clay_r3" in log
    assert (out / "nfi-per-plot-clay_r2.parquet").read_text() == "pre-existing"
    for cell in ("clay_r1", "clay_r4"):
        dump = out / f"nfi-per-plot-{cell}.parquet"
        meta = json.loads((out / f"nfi-per-plot-{cell}.parquet.meta.json").read_text())
        assert hashlib.sha256(dump.read_bytes()).hexdigest() == meta["prediction_sha256"]
        assert dump.stat().st_mode & 0o777 == 0o644
    assert not (out / "nfi-per-plot-clay_r3.parquet").exists()
    assert not any(secret in log for secret in SECRETS)


def test_all_cells_published_exits_zero(tmp_path: Path) -> None:
    code, log, out = run_job(tmp_path)
    assert code == 0, log
    assert len(list(out.glob("*.meta.json"))) == 4
    assert "failed: none" in log


def test_hash_mismatch_publishes_no_metadata(tmp_path: Path) -> None:
    code, log, out = run_job(tmp_path, BAD_SHA="1")
    assert code != 0
    assert "copied bytes differ from prediction_sha256" in log
    assert not list(out.glob("*.meta.json"))


@pytest.mark.parametrize("path", [TEMPLATE, PROBE], ids=["template", "probe"])
def test_failure_after_partial_report_shows_no_score(tmp_path: Path, path: Path) -> None:
    code, log, out = run_job(tmp_path, path, PARTIAL_REPORT="1")
    assert code != 0
    assert "Traceback (most recent call last):" in log
    assert "OSError: No space left on device" in log
    leaked = [secret for secret in SECRETS if secret in log]
    assert not leaked, (leaked, log)
    assert not list(out.glob("*.meta.json"))
