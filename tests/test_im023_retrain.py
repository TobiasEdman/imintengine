"""IM-023 retrain recipe: R4's flags, the preflight refusals, and a CPU smoke.

The smoke runs the real chain on synthetic tiles: R4-style NMD and distilled
sidecars → relabel_barrblandskog.run → im023_retrain.py → train_unified.py,
one epoch per variant, and checks the classifier width of each checkpoint.
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
import im023_retrain as retrain
import relabel_barrblandskog as relabel
from imint.schema.unified_schema import nmd2023_to_unified

R4 = ROOT / "k8s" / "ladder" / "ladder-r4-tessera-job.yaml"
R4_CHECKPOINTS = "/cephfs/checkpoints/ladder/tessera_r4"


def flags(tokens: list[str]) -> dict[str, str | None]:
    out, i = {}, 0
    while i < len(tokens):
        value = tokens[i + 1] if i + 1 < len(tokens) and not tokens[i + 1].startswith("--") else None
        out[tokens[i]] = value
        i += 1 if value is None else 2
    return out


def r4_flags() -> dict[str, str | None]:
    script = yaml.safe_load(R4.read_text())["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    lines = [ln.strip().rstrip("\\").strip() for ln in script.splitlines()]
    return flags(" ".join(ln for ln in lines if ln.startswith("--")).split())


def test_control_is_the_r4_recipe():
    assert flags(retrain.train_args("control", 30, R4_CHECKPOINTS, retrain.PATHS)) == r4_flags()


def test_main_differs_from_control_only_in_the_labels():
    main = flags(retrain.train_args("main", 30, "c", retrain.PATHS))
    control = flags(retrain.train_args("control", 30, "c", retrain.PATHS))
    assert {k for k in main if main[k] != control[k]} == {"--label-dir", "--num-classes"}
    assert (main["--num-classes"], main["--label-dir"]) == ("29", retrain.PATHS["b28_label_dir"])


# ── synthetic chain ──────────────────────────────────────────────────────────

H = 64


@pytest.fixture
def cohort(tmp_path):
    rng = np.random.default_rng(0)
    dirs = {k: tmp_path / k for k in ("tiles", "nmd", "r2", "fracs", "nmd_b28", "b28")}
    for key in ("tiles", "nmd", "r2", "fracs"):
        dirs[key].mkdir()
    names = [f"t{i}" for i in range(6)]
    codes = np.array([111, 112, 113, 114, 115, 123, 3, 61], dtype=np.uint16)
    for name in names:
        np.savez_compressed(dirs["tiles"] / f"{name}.npz",
                            spectral=(rng.random((24, H, H)) * 0.4).astype(np.float32),
                            tessera=rng.standard_normal((128, H, H)).astype(np.float16),
                            has_tessera=np.int32(1), tessera_year=np.int32(2022),
                            doy=np.array([260, 130, 190, 220], np.float32), year=np.int32(2022))
        raw = rng.choice(codes, (H, H))
        old = nmd2023_to_unified(raw)
        old[old == 28] = 3                                   # R4-era 113 → lövskog
        sidecar = dict(nmd_label_raw=raw, nmd_area_ha=np.ones((H, H), np.float32),
                       parcel_area_ha=np.zeros((H, H), np.float32))
        np.savez_compressed(dirs["nmd"] / f"{name}.npz", label=old, **sidecar)
        distilled = np.where(np.isin(old, (1, 2, 3, 4)), rng.integers(1, 5, (H, H)), old).astype(np.uint8)
        np.savez_compressed(dirs["r2"] / f"{name}.npz", label=distilled, head_sha=np.str_("h"), **sidecar)
        unreliable = np.zeros((H, H), bool)
        unreliable[:, :8] = True
        np.savez_compressed(dirs["fracs"] / f"{name}.npz",
                            frac=(rng.random((4, H, H)) * 100).astype(np.float32),
                            frac_unreliable=unreliable)
    (dirs["r2"] / "_GATE_OK").touch()
    (dirs["tiles"] / "split_train.txt").write_text("".join(f"{n}.npz\n" for n in names[:4]))
    (dirs["tiles"] / "split_val.txt").write_text("".join(f"{n}.npz\n" for n in names[4:]))
    relabel.run(dirs["nmd"], dirs["r2"], dirs["nmd_b28"], dirs["b28"])
    return dirs


def path_args(dirs) -> list[str]:
    return ["--data-dir", str(dirs["tiles"]), "--frac-dir", str(dirs["fracs"]),
            "--r2-label-dir", str(dirs["r2"]), "--b28-label-dir", str(dirs["b28"])]


def paths(dirs) -> dict[str, str]:
    return {"data_dir": str(dirs["tiles"]), "frac_dir": str(dirs["fracs"]),
            "r2_label_dir": str(dirs["r2"]), "b28_label_dir": str(dirs["b28"])}


CPU = ["--", "--patch-size", "64", "--img-size", "64", "--batch-size", "2",
       "--num-workers", "0", "--device", "cpu"]


@pytest.mark.parametrize("variant, classes", [("main", 29), ("control", 28)])
def test_cpu_smoke_trains_each_variant(cohort, tmp_path, variant, classes, capsys):
    checkpoints = tmp_path / f"ckpt_{variant}"
    code = retrain.main(["--variant", variant, "--epochs", "1", "--checkpoint-dir", str(checkpoints),
                         *path_args(cohort), *CPU])
    assert code == 0, capsys.readouterr().out
    state = torch.load(checkpoints / "best_model.pt", map_location="cpu", weights_only=False)
    assert tuple(state["state_dict"]["model.classifier.weight"].shape[:1]) == (classes,)
    assert state["config"]["num_classes"] == classes
    out = capsys.readouterr().out
    assert "class weights:" in out and f"--num-classes {classes}" in out
    # A finished model is never overwritten.
    assert retrain.main(["--variant", variant, "--epochs", "1", "--checkpoint-dir", str(checkpoints),
                         *path_args(cohort), *CPU]) == 1
    assert "already holds a model" in capsys.readouterr().out


def test_preflight_accepts_the_cohort(cohort):
    assert retrain.preflight("main", paths(cohort)) == []
    assert retrain.preflight("control", paths(cohort)) == []


def test_preflight_refuses_wrong_or_partial_labels(cohort):
    p = paths(cohort)
    manifest = cohort["b28"] / relabel.MANIFEST
    content = json.loads(manifest.read_text())
    manifest.write_text(json.dumps(content | {"rule": "other"}))
    assert any("expected " + relabel.RULE in m for m in retrain.preflight("main", p))
    manifest.write_text(json.dumps(content))
    (cohort["b28"] / "t0.npz").unlink()
    assert any("cohort differs" in m for m in retrain.preflight("main", p))
    manifest.unlink()
    assert any("run k8s/im023/relabel-b28-job.yaml first" in m for m in retrain.preflight("main", p))
    # Class 28 in the control's labels would crash a 28-class head.
    assert any("label 28 with 28 classes" in m
               for m in retrain.preflight("control", p | {"r2_label_dir": str(cohort["b28"])}))
    (cohort["r2"] / "_GATE_OK").unlink()
    assert any("_GATE_OK" in m for m in retrain.preflight("control", p))


def test_class_weight_source_names_class_28(tmp_path):
    assert "absent: uniform" in retrain.class_weight_source(str(tmp_path))
    (tmp_path / "class_stats.json").write_text(json.dumps({"pixel_counts": {"tallskog": 5}}))
    assert "class 28 absent" in retrain.class_weight_source(str(tmp_path))
    (tmp_path / "class_stats.json").write_text(json.dumps({"pixel_counts": {"barrblandskog": 5}}))
    assert "class 28 counted" in retrain.class_weight_source(str(tmp_path))


# ── manifests ────────────────────────────────────────────────────────────────

MANIFESTS = ROOT / "k8s" / "im023"
RUNS = {
    "train-tessera-smoke-job.yaml": [("main", "1", "smoke_main"), ("control", "1", "smoke_control")],
    "train-tessera-b28-job.yaml": [("main", "30", "tessera_b28")],
    "train-tessera-control-job.yaml": [("control", "30", "tessera_control")],
}


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def body(doc: dict) -> str:
    return doc["spec"]["template"]["spec"]["containers"][0]["command"][-1]


def invocations(text: str, script: str) -> list[dict[str, str | None]]:
    """Flags of every call to ``script``, with ``NAME=value`` assignments resolved."""
    for name, value in re.findall(r"^\s*([A-Z_0-9]+)=(\S+)$", text, re.M):
        text = text.replace(f'"${name}"', value).replace(f"${name}", value)
    joined = re.sub(r"\\\n\s*", " ", text)
    call = re.compile(rf"\bpython3?\s+{re.escape(script)}(.*)")
    return [flags(shlex.split(m[1])) for m in map(call.search, joined.splitlines()) if m]


def test_manifests_pin_one_source_and_image():
    docs = [load(p) for p in sorted(MANIFESTS.glob("*.yaml"))]
    assert len(docs) == 4
    shas = {m for d in docs for m in re.findall(r"^\s*SHA=([0-9a-f]{40})$", body(d), re.M)}
    images = {d["spec"]["template"]["spec"]["containers"][0]["image"] for d in docs}
    assert len(shas) == 1 and len(images) == 1 and "@sha256:" in next(iter(images))
    assert all('test "$(git rev-parse HEAD)" = "$SHA"' in body(d) for d in docs)


def test_pinned_source_carries_the_im023_scripts():
    sha = re.search(r"SHA=([0-9a-f]{40})", (MANIFESTS / "relabel-b28-job.yaml").read_text())[1]
    git = ["git", "-C", str(ROOT)]
    if subprocess.run([*git, "cat-file", "-e", sha], capture_output=True).returncode:
        pytest.skip(f"{sha} not in this clone")
    assert subprocess.run([*git, "merge-base", "--is-ancestor", sha, "HEAD"]).returncode == 0
    changed = subprocess.run([*git, "diff", "--name-only", sha, "HEAD", "--",
                              "scripts/im023_retrain.py", "scripts/relabel_barrblandskog.py"],
                             capture_output=True, text=True, check=True).stdout.split()
    assert changed == [], f"pinned SHA predates changes to {changed}: bump SHA in k8s/im023/"


def test_training_jobs_call_the_recipe_on_r4_hardware():
    r4 = load(R4)["spec"]["template"]["spec"]
    pods, setups = [], set()
    for name, expected in RUNS.items():
        doc = load(MANIFESTS / name)
        setup, run = body(doc).split("# --- run ---")
        setups.add(setup.replace(doc["metadata"]["name"], "NAME"))
        calls = [shlex.split(ln) for ln in run.splitlines() if '"$RETRAIN"' in ln]
        assert calls == [["python3", "$RETRAIN", "--variant", v, "--epochs", e,
                          "--checkpoint-dir", f"/cephfs/checkpoints/im023/{d}"] for v, e, d in expected]
        assert ("smoke_$v/best_model.pt" in run) == (name != "train-tessera-smoke-job.yaml")
        pod = doc["spec"]["template"]["spec"]
        container = pod["containers"][0]
        assert container["resources"] == r4["containers"][0]["resources"]
        assert container["volumeMounts"] == r4["containers"][0]["volumeMounts"]
        assert {k: pod[k] for k in ("nodeSelector", "affinity", "volumes")} == \
               {k: r4[k] for k in ("nodeSelector", "affinity", "volumes")}
    assert len(setups) == 1


def test_relabel_job_runs_the_chain_that_made_the_r4_labels():
    text = body(load(MANIFESTS / "relabel-b28-job.yaml"))
    ladder = body(load(ROOT / "k8s" / "ladder" / "distill-tessera-job.yaml"))
    original = body(load(ROOT / "k8s" / "build-labels-nmd2023-job.yaml"))

    (built,) = invocations(text, "scripts/build_labels.py")
    (made,) = invocations(original.replace("/data/", "/cephfs/"), "scripts/build_labels.py")
    per_run = {"--label-out-dir", "--workers", "--tile-ids"}
    assert {k: v for k, v in built.items() if k not in per_run} == \
           {k: v for k, v in made.items() if k not in per_run}

    *_, r4_dense = invocations(ladder, "scripts/distill_forest_labels.py")
    old, new = invocations(text, "scripts/distill_forest_labels.py")
    per_run = {"--data-dir", "--label-dir", "--out-dir"}
    for fresh in (old, new):
        assert {k: v for k, v in fresh.items() if k not in per_run} == \
               {k: v for k, v in r4_dense.items() if k not in per_run}
    assert old["--label-dir"] == r4_dense["--label-dir"] == "/cephfs/nmd2023_labels"
    assert new["--label-dir"] == built["--label-out-dir"]

    (verified,) = invocations(text, "scripts/relabel_barrblandskog.py verify")
    (ran,) = invocations(text, "scripts/relabel_barrblandskog.py run")
    assert verified["--canon-distill-old"] == old["--out-dir"]
    assert verified["--canon-distill-new"] == new["--out-dir"]
    assert ran["--distill-out"] == retrain.PATHS["b28_label_dir"]
    assert ran["--distill-in"] == retrain.PATHS["r2_label_dir"] == r4_dense["--out-dir"]
    assert text.index("relabel_barrblandskog.py verify") < text.index("relabel_barrblandskog.py run")
