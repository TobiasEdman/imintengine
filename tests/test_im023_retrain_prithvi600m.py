"""IM-023 Prithvi-600M recipe: R4's flags plus the planned deltas, the preflight, a CPU smoke, the jobs.

The cohort fixture of test_im023_retrain (R4-style NMD and TESSERA distilled
sidecars, relabelled by relabel_barrblandskog.run) gains a second
distillation, relabelled by relabel_distilled_b28 against the same NMD
directory, as Prithvi-600M's is on the cluster. The jobs in
k8s/im023/prithvi600m/ are parsed only; nothing here contacts a cluster.
"""
from pathlib import Path
import json
import re
import shlex
import subprocess
import sys

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import im023_retrain as tessera
import im023_retrain_prithvi600m as recipe
import relabel_barrblandskog as relabel
import relabel_distilled_b28 as reuse
from imint.fm.registry import MODEL_CONFIGS
from imint.training.config import TrainingConfig
from tests.test_im023_retrain import body, cohort, flags, invocations, load  # noqa: F401  (cohort is a fixture)

R4 = ROOT / "k8s" / "ladder" / "ladder-r4-prithvi600m-job.yaml"
R4_CHECKPOINTS = "/cephfs/checkpoints/ladder/prithvi600m_r4"


def manifest_flags(path: Path) -> dict[str, str | None]:
    """Every flag of the manifest's train_unified.py call, on whichever line it sits."""
    (call,) = invocations(body(load(path)), "/workspace/imintengine/scripts/train_unified.py")
    return call


def test_manifest_flags_reads_the_whole_call():
    r4 = manifest_flags(R4)
    assert r4["--backbone-name"] == "prithvi_600m" and r4["--frac-dir"] == "/cephfs/tradslag_fracs"
    assert "--enable-tradslag-head" not in r4                    # the R4 gap this retrain closes


def test_control_without_the_head_is_the_r4_recipe():
    assert flags(recipe.train_args("control", 30, R4_CHECKPOINTS, recipe.PATHS,
                                   fraction_head=False)) == manifest_flags(R4)


HEAD = {"--enable-tradslag-head": None, "--lambda-frac": "1.0"}


@pytest.mark.parametrize("variant, checkpoint_dir", [("main", "/cephfs/checkpoints/im023/prithvi600m_b28"),
                                                     ("control", "/cephfs/checkpoints/im023/prithvi600m_control")])
@pytest.mark.parametrize("fraction_head", [True, False])
def test_only_the_planned_deltas_from_r4(variant, checkpoint_dir, fraction_head):
    r4 = manifest_flags(R4)
    ours = flags(recipe.train_args(variant, 30, checkpoint_dir, recipe.PATHS, fraction_head=fraction_head))
    assert set(r4) <= set(ours)
    delta = {k: v for k, v in ours.items() if k not in r4 or r4[k] != v}
    planned = {"--checkpoint-dir": checkpoint_dir, **(HEAD if fraction_head else {})}
    if variant == "main":
        planned |= {"--label-dir": "/cephfs/distill/prithvi600m_r2_b28", "--num-classes": "29",
                    "--class-stats": f"{checkpoint_dir}/{tessera.B28_STATS}"}
    assert delta == planned


def test_main_differs_from_control_only_in_class_28():
    main = flags(recipe.train_args("main", 30, "c", recipe.PATHS, fraction_head=True))
    control = flags(recipe.train_args("control", 30, "c", recipe.PATHS, fraction_head=True))
    assert {k for k in main.keys() | control.keys() if main.get(k) != control.get(k)} == \
           {"--label-dir", "--num-classes", "--class-stats"}


def test_written_out_lambda_frac_is_the_trainer_default():
    """R4 ran with the default; writing it out changes nothing but the record."""
    assert TrainingConfig().lambda_frac == float(HEAD["--lambda-frac"])
    assert TrainingConfig().enable_tradslag_head is False


def test_paths_are_r4s_and_its_relabelled_copy():
    r4 = manifest_flags(R4)
    assert (r4["--data-dirs"], r4["--frac-dir"], r4["--label-dir"]) == \
           (recipe.PATHS["data_dir"], recipe.PATHS["frac_dir"], recipe.PATHS["r2_label_dir"])
    assert recipe.PATHS["b28_label_dir"] == recipe.PATHS["r2_label_dir"] + "_b28"
    assert {recipe.PATHS[k] for k in ("data_dir", "frac_dir")} == \
           {tessera.PATHS[k] for k in ("data_dir", "frac_dir")}


# ── synthetic chain ──────────────────────────────────────────────────────────

@pytest.fixture
def prithvi(cohort):
    """``cohort`` plus a second distillation of its NMD labels, relabelled via the shared NMD dir."""
    rng = np.random.default_rng(1)
    dirs = cohort | {k: cohort["tiles"].parent / k for k in ("p_r2", "p_b28")}
    dirs["p_r2"].mkdir()
    for path in sorted(cohort["nmd"].glob("*.npz")):
        nmd = relabel.load(path)
        forest = np.isin(nmd["label"], (1, 2, 3, 4))
        distilled = np.where(forest, rng.integers(1, 5, forest.shape), nmd["label"]).astype(np.uint8)
        np.savez_compressed(dirs["p_r2"] / path.name, **(nmd | {"label": distilled, "head_sha": np.str_("p")}))
    (dirs["p_r2"] / "_GATE_OK").touch()
    reuse.run(cohort["nmd"], cohort["nmd_b28"], dirs["p_r2"], dirs["p_b28"])
    return dirs


def paths(dirs) -> dict[str, str]:
    return {"data_dir": str(dirs["tiles"]), "frac_dir": str(dirs["fracs"]),
            "r2_label_dir": str(dirs["p_r2"]), "b28_label_dir": str(dirs["p_b28"])}


def path_args(dirs) -> list[str]:
    return [arg for key, value in paths(dirs).items() for arg in (f"--{key.replace('_', '-')}", value)]


def test_preflight_accepts_the_cohort(prithvi):
    assert recipe.preflight("main", paths(prithvi)) == []
    assert recipe.preflight("control", paths(prithvi)) == []


def test_preflight_is_tesseras_without_its_key_check(prithvi):
    """Same refusals on the same inputs, but for the TESSERA key and the relabel job named."""
    p = paths(prithvi)

    def same(variant: str, given: dict[str, str] = p) -> list[str]:
        ours = recipe.preflight(variant, given)
        theirs = [m.replace("k8s/im023/relabel-b28-job.yaml", recipe.RELABEL_JOB)
                  for m in tessera.preflight(variant, given)]
        assert ours == theirs
        return ours

    manifest = prithvi["p_b28"] / relabel.MANIFEST
    content = json.loads(manifest.read_text())
    manifest.write_text(json.dumps(content | {"rule": "other"}))
    assert any("expected " + relabel.RULE in m for m in same("main"))
    manifest.write_text(json.dumps(content | {"distilled_tiles": content["distilled_tiles"] + 1}))
    assert any("cohort differs" in m for m in same("main"))
    manifest.write_text(json.dumps(content))
    (prithvi["p_b28"] / "t0.npz").rename(prithvi["p_b28"] / "unexpected.npz")
    assert any("unexpected ['unexpected.npz']" in m for m in same("main"))
    (prithvi["p_b28"] / "unexpected.npz").rename(prithvi["p_b28"] / "t0.npz")
    manifest.unlink()
    assert any(f"run {recipe.RELABEL_JOB} first" in m for m in same("main"))
    assert any("label 28 with 28 classes" in m for m in same("control", p | {"r2_label_dir": p["b28_label_dir"]}))
    (prithvi["p_r2"] / "_GATE_OK").unlink()
    assert any("_GATE_OK" in m for m in same("control"))
    (prithvi["tiles"] / "class_stats.json").unlink()
    assert any("weight cannot be derived" in m for m in same("main"))
    assert not any("class_stats" in m for m in same("control"))
    # Prithvi reads spectral, not TESSERA embeddings: tiles without them are no refusal.
    for path in sorted(prithvi["tiles"].glob("*.npz")):
        np.savez_compressed(path, **{k: v for k, v in relabel.load(path).items() if k != "tessera"})
    assert any("tessera key" in m for m in tessera.preflight("control", p))
    assert not any("tessera key" in m for m in recipe.preflight("control", p))


def test_class_28_weight_does_not_depend_on_the_model(prithvi):
    """The share is taken in the shared NMD labels, so main's weights equal TESSERA main's."""
    tiles = str(prithvi["tiles"])
    assert tessera.b28_class_stats(tiles, str(prithvi["p_b28"])) == \
           tessera.b28_class_stats(tiles, str(prithvi["b28"]))


SMOKE_BACKBONE = "prithvi_300m"   # same family, loader and decoder; 600M's weights are 2.6 GB


def weights_cached(name: str) -> bool:
    from huggingface_hub import try_to_load_from_cache
    spec = MODEL_CONFIGS[name].loader_kwargs
    local = ROOT / "imint" / "fm" / "prithvi_mae" / spec["weights_filename"]
    return local.is_file() or isinstance(try_to_load_from_cache(spec["hf_repo"], spec["weights_filename"]), str)


CPU = ["--", "--backbone-name", SMOKE_BACKBONE, "--patch-size", "64", "--img-size", "64",
       "--batch-size", "2", "--num-workers", "0", "--device", "cpu"]


@pytest.mark.parametrize("variant, classes", [("main", 29), ("control", 28)])
def test_cpu_smoke_trains_each_variant_with_the_head(prithvi, tmp_path, variant, classes, capfd, monkeypatch):
    """The recipe's own flags but the backbone and the CPU overrides; capfd, as the trainer is a subprocess."""
    if not weights_cached(SMOKE_BACKBONE):
        pytest.skip(f"{SMOKE_BACKBONE} weights not in the Hugging Face cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    checkpoints = tmp_path / f"ckpt_{variant}"
    argv = ["--variant", variant, "--epochs", "1", "--checkpoint-dir", str(checkpoints), *path_args(prithvi), *CPU]
    code = recipe.main(argv)
    out = capfd.readouterr().out
    assert code == 0, out
    state = torch.load(checkpoints / "best_model.pt", map_location="cpu", weights_only=False)
    config = state["config"]
    assert (config["backbone_name"], config["num_classes"], config["num_temporal_frames"]) == \
           (SMOKE_BACKBONE, classes, 4)
    assert config["enable_tradslag_head"] and config["enable_multitemporal"]
    # UperNet's SegmentationHead keeps TerraTorch's layout: head.head.2 is the classifier conv.
    (classifier,) = [v for k, v in state["state_dict"].items() if k.endswith("head.head.2.weight")]
    (frac_head,) = [v for k, v in state["state_dict"].items() if k.endswith("frac_head.weight")]
    assert (classifier.shape[0], frac_head.shape[0]) == (classes, 4)
    assert "fraction head: on" in out and "Trädslag head: ENABLED" in out
    assert "L_frac=" in out                                  # the sidecars supervised the head
    assert f"--num-classes {classes}" in out
    weights = json.loads(re.search(r"Class weights: (\[.*\])", out)[1])
    assert len(weights) == classes
    if variant == "main":
        assert weights[28] < 5.0                             # its frequency, not the cap
        assert (checkpoints / tessera.B28_STATS).is_file()
    assert recipe.main(argv) == 1
    assert "pass --resume" in capfd.readouterr().out


def test_no_fraction_head_reaches_the_trainer(prithvi, tmp_path, capfd, monkeypatch):
    """The switch is the only way to drop the head; the command is checked, nothing trains."""
    monkeypatch.setattr(recipe.subprocess, "run", lambda command: type("Done", (), {"returncode": 0}))
    argv = ["--variant", "control", "--epochs", "1", "--checkpoint-dir", str(tmp_path / "c"), *path_args(prithvi)]
    for extra, head in (([], True), (["--no-fraction-head"], False)):
        assert recipe.main(argv + extra) == 0
        out = capfd.readouterr().out
        assert ("--enable-tradslag-head" in out) == head
        assert ("--lambda-frac" in out) == head
        assert ("fraction head: on" if head else "fraction head: off") in out


# ── manifests ────────────────────────────────────────────────────────────────

JOBS = ROOT / "k8s" / "im023" / "prithvi600m"
TESSERA_JOBS = ROOT / "k8s" / "im023"
RUNS = {
    "train-smoke-job.yaml": [("main", "1", "prithvi600m_smoke_main"),
                             ("control", "1", "prithvi600m_smoke_control")],
    "train-b28-job.yaml": [("main", "30", "prithvi600m_b28")],
    "train-control-job.yaml": [("control", "30", "prithvi600m_control")],
}
#: Placeholder until the pin commit after the push, then the commit itself.
SOURCE_SHA = re.compile(r"REPLACE_WITH_PUSHED_SHA|[0-9a-f]{40}")
SHA_LINE = re.compile(r"^\s*SHA=(\S+)$", re.M)
PINNED_SCRIPTS = ("scripts/im023_retrain.py", "scripts/relabel_barrblandskog.py",
                  "scripts/im023_retrain_prithvi600m.py", "scripts/relabel_distilled_b28.py")


def pod(doc: dict) -> dict:
    return doc["spec"]["template"]["spec"]


def pinned_sha() -> str:
    (sha,) = {m for p in sorted(JOBS.glob("*.yaml")) for m in SHA_LINE.findall(body(load(p)))}
    return sha


def test_jobs_pin_one_source_and_the_tessera_image():
    docs = [load(p) for p in sorted(JOBS.glob("*.yaml"))]
    assert len(docs) == 4
    assert SOURCE_SHA.fullmatch(pinned_sha())
    images = {pod(d)["containers"][0]["image"] for d in docs}
    tessera_images = {pod(load(p))["containers"][0]["image"] for p in sorted(TESSERA_JOBS.glob("*.yaml"))}
    assert images == tessera_images and "@sha256:" in next(iter(images))
    for doc in docs:
        text = body(doc)
        assert 'test "$(git rev-parse HEAD)" = "$SHA"' in text
        assert text.index('[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]') < text.index("apt-get")
        assert doc["metadata"]["namespace"] == "prithvi-training-default"
        assert doc["spec"]["backoffLimit"] == 0
        resources = pod(doc)["containers"][0]["resources"]
        assert resources["requests"] == resources["limits"]     # ICE admission webhook


def test_an_unpinned_source_refuses_before_any_work():
    guard = next(ln.strip() for ln in body(load(JOBS / "train-smoke-job.yaml")).splitlines() if "=~" in ln)
    for sha, code in (("REPLACE_WITH_PUSHED_SHA", 1), ("abc", 1), ("0" * 40, 0)):
        result = subprocess.run(["bash", "-c", f"set -euo pipefail; SHA={sha}; {guard}"],
                                capture_output=True, text=True)
        assert result.returncode == code, (sha, result.stdout)


def test_pinned_source_carries_the_scripts():
    sha = pinned_sha()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        pytest.skip("source is pinned after the push")
    git = ["git", "-C", str(ROOT)]
    if subprocess.run([*git, "cat-file", "-e", sha], capture_output=True).returncode:
        pytest.skip(f"{sha} not in this clone")
    assert subprocess.run([*git, "merge-base", "--is-ancestor", sha, "HEAD"]).returncode == 0
    changed = subprocess.run([*git, "diff", "--name-only", sha, "HEAD", "--", *PINNED_SCRIPTS],
                             capture_output=True, text=True, check=True).stdout.split()
    assert changed == [], f"pinned SHA predates changes to {changed}: bump SHA in k8s/im023/prithvi600m/"


def setup_as_tessera(doc: dict, script: str) -> str:
    """The setup half with the job name, source pin and recipe script normalised."""
    setup = body(doc).split("# --- run ---")[0].replace(doc["metadata"]["name"], "NAME")
    setup = "".join(ln for ln in setup.splitlines(keepends=True) if "=~" not in ln)
    return SHA_LINE.sub("SHA=PINNED", setup).replace(script, "RECIPE")


def test_training_jobs_call_the_recipe_on_r4_hardware():
    r4 = pod(load(R4))
    tessera_smoke = load(TESSERA_JOBS / "train-tessera-smoke-job.yaml")
    assert "affinity" not in r4                                  # R4 Prithvi-600M had no srv02 exclusion
    setups = set()
    for name, expected in RUNS.items():
        doc = load(JOBS / name)
        setup, run = body(doc).split("# --- run ---")
        setups.add(setup_as_tessera(doc, "im023_retrain_prithvi600m.py"))
        full = name != "train-smoke-job.yaml"
        calls = [shlex.split(ln) for ln in run.splitlines() if '"$RETRAIN"' in ln]
        assert calls == [["python3", "$RETRAIN", "--variant", v, "--epochs", e,
                          "--checkpoint-dir", f"/cephfs/checkpoints/im023/{d}",
                          *(["${RESUME_FLAG[@]}"] if full else [])] for v, e, d in expected]
        assert ("prithvi600m_smoke_$v/_SMOKE_OK" in run) == full
        spec, container = pod(doc), pod(doc)["containers"][0]
        for key in ("resources", "volumeMounts"):
            assert container[key] == r4["containers"][0][key], (name, key)
        resume = {"name": "RESUME", "value": "0"}
        assert [e for e in container["env"] if e != resume] == r4["containers"][0]["env"]
        assert (resume in container["env"]) == full             # resuming is an explicit edit
        assert {k: spec[k] for k in ("nodeSelector", "volumes")} == {k: r4[k] for k in ("nodeSelector", "volumes")}
        assert spec["affinity"] == pod(tessera_smoke)["affinity"]
        if name != "train-smoke-job.yaml":
            assert spec["activeDeadlineSeconds"] == r4["activeDeadlineSeconds"]
        assert "activeDeadlineSeconds" not in doc["spec"]       # queue time is free
    assert setups == {setup_as_tessera(tessera_smoke, "im023_retrain.py")}


def test_full_runs_need_a_smoke_that_passed_at_their_commit(tmp_path):
    """Re-pinning to a new commit invalidates the old smoke."""
    full = body(load(JOBS / "train-b28-job.yaml"))
    guard = full[full.index("for v in main control; do"):full.index("RESUME_FLAG=")]
    marks = [ln.strip() for ln in body(load(JOBS / "train-smoke-job.yaml")).splitlines()
             if ln.strip().startswith("echo") and "_SMOKE_OK" in ln]
    assert len(marks) == 2
    root = tmp_path / "im023"
    for v in ("main", "control"):
        (root / f"prithvi600m_smoke_{v}").mkdir(parents=True)

    def bash(sha: str, script: str) -> int:
        script = script.replace("/cephfs/checkpoints/im023", str(root))
        return subprocess.run(["bash", "-c", f"set -euo pipefail; SHA={sha}\n{script}"],
                              capture_output=True, text=True).returncode

    first, second = "a" * 40, "b" * 40
    assert bash(first, guard) == 1                               # no smoke yet
    assert all(bash(first, mark) == 0 for mark in marks)
    assert bash(first, guard) == 0
    assert bash(second, guard) == 1                              # re-pinned: first's smoke does not count


def _log(directory: Path, *, classes: int, head: bool, status: str = "running") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "last_checkpoint.pt").write_bytes(b"state")
    (directory / "training_log.json").write_text(json.dumps(
        {"status": status, "config": {"num_classes": classes, "enable_tradslag_head": head}}))


@pytest.mark.parametrize("held", recipe.CHECKPOINT_STATE)
def test_existing_state_is_refused_without_resume(tmp_path, capsys, held):
    (tmp_path / held).write_bytes(b"x")
    assert recipe.main(["--variant", "control", "--epochs", "1", "--checkpoint-dir", str(tmp_path)]) == 1
    assert "pass --resume" in capsys.readouterr().out


def test_resume_accepts_only_this_runs_interrupted_state(tmp_path):
    ok = tmp_path / "ok"
    _log(ok, classes=29, head=True)
    assert recipe.resume_problems(ok, 29, True) == []
    assert recipe.resume_problems(tmp_path / "empty", 29, True)  # nothing to resume
    for name, kwargs in {"done": dict(classes=29, head=True, status="completed"),
                         "classes": dict(classes=28, head=True),
                         "head": dict(classes=29, head=False)}.items():
        _log(tmp_path / name, **kwargs)
        assert recipe.resume_problems(tmp_path / name, 29, True), name
    (ok / "training_log.json").unlink()
    assert recipe.resume_problems(ok, 29, True)                   # cannot tell whose state it is


def test_resume_refusal_happens_before_any_work(tmp_path, capsys):
    _log(tmp_path, classes=28, head=True)
    assert recipe.main(["--variant", "main", "--epochs", "30", "--checkpoint-dir", str(tmp_path),
                        "--resume"]) == 1
    assert "REFUSED --resume" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [["--checkpoint-dir", "/x"], ["--label-dir=/x"],
                                   ["--num-classes", "23"], ["--warm-start-from", "/x/best_model.pt"]])
def test_extras_may_not_redirect_the_run(tmp_path, capsys, extra):
    assert recipe.main(["--variant", "control", "--epochs", "1", "--checkpoint-dir", str(tmp_path / "c"),
                        "--", *extra]) == 1
    assert "would redirect the run" in capsys.readouterr().out


def test_relabel_job_runs_the_chain_that_made_the_r2_labels():
    text = body(load(JOBS / "relabel-b28-job.yaml"))
    tessera_text = body(load(TESSERA_JOBS / "relabel-b28-job.yaml"))
    ladder = load(ROOT / "k8s" / "ladder" / "distill-prithvi600m-job.yaml")

    (built,) = invocations(text, "scripts/build_labels.py")
    assert [built] == invocations(tessera_text, "scripts/build_labels.py")

    *_, r2_dense = invocations(body(ladder), "scripts/distill_forest_labels.py")
    old, new = invocations(text, "scripts/distill_forest_labels.py")
    per_run = {"--data-dir", "--label-dir", "--out-dir"}
    for fresh in (old, new):
        assert {k: v for k, v in fresh.items() if k not in per_run} == \
               {k: v for k, v in r2_dense.items() if k not in per_run}
    assert old["--label-dir"] == r2_dense["--label-dir"] == "/cephfs/nmd2023_labels"
    assert new["--label-dir"] == built["--label-out-dir"]

    (sampled,) = invocations(text, "scripts/relabel_barrblandskog.py sample")
    (verified,) = invocations(text, "scripts/relabel_barrblandskog.py verify")
    assert sampled["--distill-in"] == verified["--distill-in"] == r2_dense["--out-dir"]
    assert verified["--canon-distill-old"] == old["--out-dir"]
    assert verified["--canon-distill-new"] == new["--out-dir"]

    (ran,) = invocations(text, "scripts/relabel_distilled_b28.py")
    (tessera_ran,) = invocations(tessera_text, "scripts/relabel_barrblandskog.py run")
    assert ran["--nmd-in"] == tessera_ran["--nmd-in"] == old["--label-dir"]
    assert ran["--nmd-b28"] == tessera_ran["--nmd-out"]
    assert ran["--distill-in"] == recipe.PATHS["r2_label_dir"] == r2_dense["--out-dir"]
    assert ran["--distill-out"] == recipe.PATHS["b28_label_dir"]
    assert text.index("relabel_barrblandskog.py verify") < text.index("relabel_distilled_b28.py")
    assert "relabel_barrblandskog.py run" not in text           # the NMD directory is only read

    ours, theirs = pod(load(JOBS / "relabel-b28-job.yaml")), pod(load(TESSERA_JOBS / "relabel-b28-job.yaml"))
    assert ours["nodeSelector"] == pod(ladder)["nodeSelector"] == theirs["nodeSelector"]
    for key in ("image", "resources", "volumeMounts"):
        assert ours["containers"][0][key] == theirs["containers"][0][key], key
