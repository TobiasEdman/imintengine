"""The IM-023 amendment and LUCAS Jobs: one sealed image, sealed data, no overwrite.

Parses k8s/im023/eval/ and runs each Job's own publish code against a
temporary directory. Nothing here contacts a cluster.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "k8s" / "im023" / "eval"
JOBS = {"prepare": "nfi-amendment-prepare-job.yaml", "inference": "nfi-amendment-inference-job.yaml",
        "scoring": "nfi-amendment-scoring-job.yaml", "lucas": "eval-lucas-job.yaml"}
AMEND = "/cephfs/nfi/im023_amendment_2026-11-06"
FROZEN_RESULTS = ("nfi/nfi_holdout_freeze_2026-10-27", "nfi/nfi_inference_2026-10-27",
                  "nfi/nfi_scoring_2026-10-28")
REPOSITORY = "ghcr.io/tobiasedman/imint-ladder-crop-distill"
#: Placeholders until the pin commit after the merge, then real identities.
IMAGE = re.compile(rf"{re.escape(REPOSITORY)}@sha256:(REPLACE_WITH_SEALED_IMAGE_DIGEST|[0-9a-f]{{64}})")
SOURCE_SHA = re.compile(r"REPLACE_WITH_MERGED_SHA|[0-9a-f]{40}")
AMENDMENT_SHA = re.compile(r"REPLACE_WITH_APPROVED_AMENDMENT_SHA256|[0-9a-f]{64}")
#: Job -> the sealed interpreter its script verifies (sealed_runtime's environment).
INTERPRETER = {"prepare": "scoring", "inference": "model", "scoring": "model", "lucas": "model"}
CALL = re.compile(r"/opt/venvs/(model|scoring)/bin/python -u /opt/imintengine/(scripts/\w+\.py)(.*)")


def load(name: str) -> dict:
    return yaml.safe_load((EVAL / JOBS[name]).read_text())


def container(doc: dict) -> dict:
    return doc["spec"]["template"]["spec"]["containers"][0]


def script(doc: dict) -> str:
    return container(doc)["args"][0]


def env(doc: dict) -> dict:
    return {e["name"]: e["value"] for e in container(doc)["env"]}


def invocations(text: str) -> list[tuple[str, str, list[str]]]:
    """(environment, script, argv) of every sealed-interpreter call."""
    joined = re.sub(r"\\\n\s*", " ", text)
    calls = []
    for line in joined.splitlines():
        match = CALL.search(line)
        if match:
            tail = re.split(r"[;>]|\|\||\bthen\b", match[3])[0]
            calls.append((match[1], match[2], shlex.split(tail)))
    return calls


def test_every_job_runs_one_sealed_image_and_installs_nothing():
    frozen = yaml.safe_load((REPO / "k8s" / "nfi-scoring-job.yaml").read_text())
    assert container(frozen)["image"].split("@")[0] == REPOSITORY
    images = {container(load(name))["image"] for name in JOBS}
    assert len(images) == 1 and IMAGE.fullmatch(images.pop())
    sources = {env(load(name))["SOURCE_GIT_SHA"] for name in JOBS if "SOURCE_GIT_SHA" in env(load(name))}
    assert len(sources) == 1 and SOURCE_SHA.fullmatch(sources.pop())
    for name in JOBS:
        doc = load(name)
        pod = doc["spec"]["template"]["spec"]
        assert pod["imagePullSecrets"] == [{"name": "ghcr-push"}]
        assert doc["metadata"]["namespace"] == "prithvi-training-default"
        assert doc["spec"]["backoffLimit"] == 0
        names = [e["name"] for e in container(doc)["env"]]
        assert len(names) == len(set(names)), name
        resources = container(doc)["resources"]
        assert resources["requests"] == resources["limits"], name   # ICE admission webhook
        assert container(doc)["securityContext"] == {
            "allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True,
            "capabilities": {"drop": ["ALL"]}}, name
        text = script(doc)
        for forbidden in ("pip install", "apt-get", "git ", "python3 "):
            assert forbidden not in text, (name, forbidden)
        if "IMAGE_REF" in env(doc):
            assert env(doc)["IMAGE_REF"] == container(doc)["image"]
            assert "SOURCE_GIT_SHA" in env(doc)


def test_pins_are_filled_in_order_never_half():
    """The image digest and its commit are pinned together after the merge; the
    amendment digest only after prepare, so never before them."""
    image = container(load("prepare"))["image"]
    source = env(load("prepare"))["SOURCE_GIT_SHA"]
    amendment = env(load("inference"))["MANIFEST_SHA256"]
    image_pinned = "REPLACE_WITH" not in image
    assert image_pinned == ("REPLACE_WITH" not in source)
    assert image_pinned or "REPLACE_WITH" in amendment


def test_lucas_log_filter_drops_the_checkpoint_miou():
    text = script(load("lucas"))
    expression = re.search(r"sed -E '([^']+)'", text)[1]
    lines = "  [load_model] epoch=27 ckpt_mIoU=0.5 native_img=504 x\n  [load_model] ckpt_mIoU=None\n"
    out = subprocess.run(["sed", "-E", expression], input=lines, text=True, capture_output=True,
                         check=True).stdout
    assert "mIoU" not in out and "epoch=27 native_img=504 x" in out


def test_each_step_runs_the_interpreter_its_script_verifies():
    for name in JOBS:
        calls = invocations(script(load(name)))
        assert calls, name
        assert {environment for environment, _, _ in calls} == {INTERPRETER[name]}, name
    sources = {rel: (REPO / rel).read_text() for rel in
               ("scripts/prepare_nfi_amendment.py", "scripts/infer_nfi_amendment.py",
                "scripts/score_nfi_amendment.py")}
    assert 'environment="scoring"' in sources["scripts/prepare_nfi_amendment.py"]
    assert 'verify_amendment_source(amendment, environment="model")' in sources["scripts/infer_nfi_amendment.py"]
    assert 'verify_amendment_source(amendment, environment="model")' in sources["scripts/score_nfi_amendment.py"]


def test_the_image_smoke_covers_every_script_the_jobs_run():
    smoke = (REPO / "docker" / "ladder-crop-distill" / "runtime_smoke.py").read_text()
    scripts = {rel for name in JOBS for _, rel, _ in invocations(script(load(name)))}
    for rel in scripts | {"scripts/nfi_amendment_checkpoints.py"}:
        assert f'"{rel}"' in smoke, rel


def test_lucas_verifies_the_runtime_before_any_cell():
    text = script(load("lucas"))
    preflight = text.index("/opt/imintengine/scripts/crop_distill_provenance.py verify-runtime")
    assert preflight < text.index("validate_against_lucas.py")
    joined = re.sub(r"\\\n\s*", " ", text[preflight:])
    argv = shlex.split(joined.splitlines()[0].split("verify-runtime", 1)[1])
    assert argv == ["--source-git-sha", "$SOURCE_GIT_SHA", "--image-ref", "$IMAGE_REF",
                    "--runtime-manifest", "/opt/provenance/runtime.json"]


def test_data_is_read_only_except_one_new_subpath():
    writable = {}
    for name in JOBS:
        doc = load(name)
        pod = doc["spec"]["template"]["spec"]
        assert pod["securityContext"]["runAsUser"] == 0
        assert pod["volumes"][0] == {"name": "data",
                                     "persistentVolumeClaim": {"claimName": "training-data-cephfs"}}
        mounts = [m for m in container(doc)["volumeMounts"] if m["name"] == "data"]
        assert mounts[0] == {"name": "data", "mountPath": "/cephfs", "readOnly": True}
        (out,) = mounts[1:]
        assert not out.get("readOnly") and out["mountPath"] == "/cephfs/" + out["subPath"]
        assert not any(out["subPath"].startswith(p) for p in FROZEN_RESULTS)
        writable[name] = out["mountPath"]
        assert re.search(rf"^OUT={re.escape(out['mountPath'])}(/\w+)?$", script(doc), re.M), name
    assert writable == {"prepare": AMEND, "inference": f"{AMEND}/inference",
                        "scoring": f"{AMEND}/scoring",
                        "lucas": "/cephfs/ladder_eval/im023/lucas_2026-11-06"}


def test_invoked_scripts_declare_every_flag_and_chain_the_steps():
    paths = {}
    for name in JOBS:
        for _, rel, argv in invocations(script(load(name))):
            source = (REPO / rel).read_text()
            for flag in (a for a in argv if a.startswith("--")):
                assert f'"{flag}"' in source, (name, rel, flag)
            paths[name] = argv
    prepare = paths["prepare"]
    assert prepare[prepare.index("--out-dir") + 1] == "$OUT"
    assert prepare[prepare.index("--runtime-image") + 1] == "$IMAGE_REF"
    assert prepare[prepare.index("--runtime-manifest") + 1] == "/opt/provenance/runtime.json"
    assert prepare[prepare.index("--source-git-sha") + 1] == "$SOURCE_GIT_SHA"
    assert f"OUT={AMEND}/freeze" in script(load("prepare"))
    digests = {env(load(name))["MANIFEST_SHA256"] for name in ("inference", "scoring")}
    assert len(digests) == 1 and AMENDMENT_SHA.fullmatch(digests.pop())
    for name in ("inference", "scoring"):
        assert env(load(name))["MANIFEST"] == f"{AMEND}/freeze/manifest.json"
    scoring = paths["scoring"]
    assert scoring[scoring.index("--dump-dir") + 1] == f"{AMEND}/inference"
    lucas = paths["lucas"]
    assert {"--fraction", "--fold-28"} <= set(lucas)
    assert lucas[lucas.index("--num-classes") + 1] == "28"


def heredoc(doc: dict) -> str:
    text = script(doc)
    return text[text.index("<<'PY'\n") + 7:text.index("\nPY\n")]


def publish(code: str, *args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-", *map(str, args)], input=code, text=True,
                          capture_output=True)


def test_inference_publish_is_exclusive_and_hash_checked(tmp_path):
    code = heredoc(load("inference"))
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir(), out.mkdir()
    name = "nfi-amendment-per-plot-r4.parquet"
    (src / name).write_bytes(b"predictions")
    meta = {"status": "success", "prediction_sha256": hashlib.sha256(b"predictions").hexdigest()}
    (src / (name + ".meta.json")).write_text(json.dumps(meta))
    assert publish(code, src, out, name).returncode == 0
    assert (out / name).read_bytes() == b"predictions" and (out / (name + ".meta.json")).exists()
    (src / name).write_bytes(b"other bytes")
    assert publish(code, src, out, name).returncode != 0
    assert (out / name).read_bytes() == b"predictions"
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (src / (name + ".meta.json")).write_text(json.dumps(dict(meta, prediction_sha256="0" * 64)))
    assert publish(code, src, fresh, name).returncode != 0
    assert not (fresh / (name + ".meta.json")).exists()


def test_lucas_publish_refuses_an_existing_report(tmp_path):
    code = heredoc(load("lucas"))
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir(), out.mkdir()
    (src / "r.json").write_text("{}")
    (src / "p.parquet").write_bytes(b"points")
    assert publish(code, src, out, "r.json", "p.parquet").returncode == 0
    (src / "r.json").write_text('{"new": 1}')
    assert publish(code, src, out, "r.json", "p.parquet").returncode != 0
    assert (out / "r.json").read_text() == "{}"
