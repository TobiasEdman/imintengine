"""IM-023 f5: a distillation with five forest types and the TESSERA retrain on it.

The truth rule is checked case by case against the NFI evaluation's
barrblandskog rule; the head trainer and the head CV run on a synthetic
features parquet with that truth; the distiller's forest set must match its
head; the recipe is main's with another label directory, and its preflight
and a CPU smoke run on the cohort fixture of test_im023_retrain with an f5
label directory beside it. The jobs in k8s/im023/f5/ are parsed only and
held to R4's TESSERA distillation; nothing here contacts a cluster.
"""
from pathlib import Path
import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

yaml = pytest.importorskip("yaml")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import distill_forest_labels as distill
import im023_retrain as retrain
import im023_retrain_f5 as recipe
import nfi_truth_f5 as truth
from imint.eval.nfi_amendment import AMENDMENT_PROTOCOL, derive_barrblandskog
from tests.test_im023_retrain import body, cohort, flags, invocations, load  # noqa: F401  (cohort is a fixture)

JOBS = ROOT / "k8s" / "im023" / "f5"
LADDER_DISTILL = ROOT / "k8s" / "ladder" / "distill-tessera-job.yaml"
VOLUMES = ("VolPine", "VolContorta", "VolSpruce", "VolBirch", "VolOtherDec")

# ── truth ────────────────────────────────────────────────────────────────────

#: (pine, contorta, spruce, birch, other deciduous) → (v1, f5)
CASES = [
    ((100, 0, 0, 0, 0), (1, 1)),
    ((0, 0, 100, 0, 0), (2, 2)),
    ((50, 0, 50, 0, 0), (1, 28)),      # v1 calls a tie pine; neither reaches 0.7 of conifer
    ((60, 0, 40, 0, 0), (1, 28)),
    ((75, 0, 25, 0, 0), (1, 1)),
    ((40, 40, 20, 0, 0), (1, 1)),      # contorta counts as pine
    ((30, 0, 40, 30, 0), (2, 28)),     # conifer dominance at exactly 0.7
    ((0, 0, 0, 100, 0), (3, 3)),
    ((50, 0, 0, 50, 0), (4, 4)),       # lövblandad barrskog stays blandskog
    ((0, 0, 0, 0, 0), (-1, -1)),
]


def index_of(cases) -> pd.DataFrame:
    rows = [dict(zip(VOLUMES, map(float, v)), TractID=i, PlotID=1, point_id=i, tile_name=f"t{i % 3}")
            for i, (v, _) in enumerate(cases)]
    return pd.DataFrame(rows)


def test_truth_is_v1_with_its_mixed_conifer_plots_as_28():
    v1, f5 = truth.truths(index_of(CASES))
    assert v1.tolist() == [c[1][0] for c in CASES]
    assert f5.tolist() == [c[1][1] for c in CASES]
    # The evaluation's own rule on the same plots, and nothing but 28 moves.
    index = index_of(CASES).assign(nfi_forest=v1)
    mixed = derive_barrblandskog(index, species_fraction=AMENDMENT_PROTOCOL["barrblandskog_species_fraction"])
    np.testing.assert_array_equal(f5 == 28, mixed)
    np.testing.assert_array_equal(f5[~mixed], v1[~mixed])


def test_derive_writes_a_new_index_and_reports_the_moves(tmp_path):
    src, out, report = tmp_path / "index.parquet", tmp_path / "f5.parquet", tmp_path / "f5.json"
    index_of(CASES).to_parquet(src)
    summary = truth.derive(src, out, report)
    written = pd.read_parquet(out)
    assert written[truth.TRUTH_COL].tolist() == [c[1][1] for c in CASES]
    assert written.drop(columns=truth.TRUTH_COL).equals(pd.read_parquet(src))
    assert summary["moves_v1_to_f5"] == {"1->28": 2, "2->28": 1}
    assert json.loads(report.read_text()) == summary
    with pytest.raises(SystemExit, match="exists"):
        truth.derive(src, out, report)
    with pytest.raises(SystemExit, match="already has"):
        truth.derive(out, tmp_path / "again.parquet", report)


def test_same_plots_refuses_another_plot_set(tmp_path):
    reference, ours = tmp_path / "r4.parquet", tmp_path / "f5.parquet"
    frame = index_of(CASES)[list(truth.PLOT_KEYS)]
    frame.to_parquet(reference)
    frame.sample(frac=1, random_state=0).to_parquet(ours)          # order does not matter
    truth.same_plots(ours, reference)
    pd.concat([frame, frame.iloc[:1].assign(TractID=99)]).to_parquet(ours)
    with pytest.raises(SystemExit, match="1 plot rows not in the reference"):
        truth.same_plots(ours, reference)


# ── head, CV and distiller ───────────────────────────────────────────────────

@pytest.fixture
def features(tmp_path):
    """A features parquet with the f5 truth: 30 tiles, every class separable."""
    rng = np.random.default_rng(3)
    classes = np.array([-1, 1, 2, 3, 4, 28])
    rows = []
    for i in range(360):
        c = classes[i % len(classes)]
        rows.append({"TractID": i, "PlotID": 1, "point_id": i, "Easting": 0.0, "Northing": 0.0,
                     "tile_name": f"t{i % 30}", truth.TRUTH_COL: int(c),
                     **{f"f{k:03d}": float(rng.normal(np.cos(k * (c + 2)), 0.1)) for k in range(16)}})
    path = tmp_path / "features.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_head_trains_on_the_f5_truth(features, tmp_path):
    head, split = tmp_path / "head.npz", tmp_path / "split.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train_distill_head.py"), "--features", str(features),
                    "--truth-col", truth.TRUTH_COL, "--out-head", str(head), "--out-split", str(split)],
                   check=True, capture_output=True, text=True)
    loaded = distill.load_head(str(head))
    assert loaded["classes_"].tolist() == [0, *recipe.F5_FOREST_CLASSES]
    with np.load(head) as z:
        assert str(z["truth_col"]) == truth.TRUTH_COL
    record = json.loads(split.read_text())
    assert record["truth_col"] == truth.TRUTH_COL and record["baselines"] is None
    assert "test_preview_accuracy" in record and "test_preview_accuracy_5class" not in record


def test_head_cv_scores_the_f5_truth_in_its_own_space(features, tmp_path):
    out = tmp_path / "cv.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "nfi_head_cv.py"), "--features", str(features),
                    "--truth-col", truth.TRUTH_COL, "--folds", "5", "--heads", "mlp", "--out", str(out)],
                   check=True, capture_output=True, text=True)
    result = json.loads(out.read_text())
    assert result["_meta"]["truth_col"] == truth.TRUTH_COL and result["_meta"]["baselines"] is None
    assert set(result["_meta"]["class_support"]) == {"-1", "1", "2", "3", "4", "28"}


def test_forest_classes_must_be_the_heads():
    f5_head, r4_head = {"classes_": np.array([0, 1, 2, 3, 4, 28])}, {"classes_": np.arange(5)}
    assert distill.check_forest_classes(f5_head, frozenset(recipe.F5_FOREST_CLASSES)) == {1, 2, 3, 4, 28}
    assert distill.check_forest_classes(r4_head, distill.R4_FOREST_CLASSES) == {1, 2, 3, 4}
    for head, classes in ((f5_head, distill.R4_FOREST_CLASSES), (r4_head, frozenset(recipe.F5_FOREST_CLASSES))):
        with pytest.raises(SystemExit, match="differ from the head's"):
            distill.check_forest_classes(head, classes)


def test_override_treats_28_like_the_other_forest_types():
    forest = frozenset(recipe.F5_FOREST_CLASSES)
    #             NMD 113 → head 1, 111 → head 28, sumpskog, head non-forest, crop
    orig = np.array([[28, 1, 5, 2, 11]], np.uint8)
    head = np.array([[1, 28, 28, 0, 28]])
    new, n_over, n_forest = distill.apply_override(orig, head, 0, 5, forest)
    assert new.tolist() == [[1, 28, 5, 2, 11]]
    assert (n_over, n_forest) == (2, 3)


# ── recipe ───────────────────────────────────────────────────────────────────

def test_h100_profile_is_mains_recipe_with_the_f5_labels():
    ours = flags(recipe.train_args("h100", 30, "c", recipe.PATHS))
    main = flags(retrain.train_args("main", 30, "c", retrain.PATHS))
    assert {k for k in ours.keys() | main.keys() if (k in ours, ours.get(k)) != (k in main, main.get(k))} == \
           {"--label-dir"}
    assert ours["--label-dir"] == recipe.PATHS["f5_label_dir"] == "/cephfs/distill/tessera_r2_f5"


def test_2080ti_profile_changes_only_precision_batch_and_workers():
    h100, ti = flags(recipe.train_args("h100", 30, "c", recipe.PATHS)), \
        flags(recipe.train_args("2080ti", 30, "c", recipe.PATHS))
    assert {k for k in h100.keys() | ti.keys() if (k in h100, h100.get(k)) != (k in ti, ti.get(k))} == \
           {"--batch-size", "--num-workers", "--disable-bf16"}
    assert (ti["--batch-size"], ti["--num-workers"], "--disable-bf16" in ti) == ("2", "6", True)


def test_recipe_writes_only_new_paths():
    written = {recipe.PATHS[k] for k in ("f5_label_dir", "f5_head")}
    read_only = {retrain.PATHS[k] for k in retrain.PATHS} | {"/cephfs/distill/heads/tessera_r2_head.npz"}
    assert not written & read_only


@pytest.fixture
def f5(cohort, tmp_path):
    """``cohort`` plus a five-type head and its labels beside R4's."""
    rng = np.random.default_rng(5)
    head_path = tmp_path / "head_f5.npz"
    classes = np.array([0, *recipe.F5_FOREST_CLASSES])
    np.savez(head_path, mean_=np.zeros(4, np.float32), scale_=np.ones(4, np.float32),
             W1=np.zeros((4, 2), np.float32), b1=np.zeros(2, np.float32),
             W2=np.zeros((2, classes.size), np.float32), b2=np.zeros(classes.size, np.float32), classes_=classes)
    stamp = distill.load_head(str(head_path))["head_sha"]
    out = tmp_path / "f5"
    out.mkdir()
    for path in sorted(cohort["b28"].glob("*.npz")):
        with np.load(path) as z:
            sidecar = dict(z)
        forest = np.isin(sidecar["label"], recipe.F5_FOREST_CLASSES)
        sidecar["label"] = np.where(forest, rng.choice(recipe.F5_FOREST_CLASSES, forest.shape),
                                    sidecar["label"]).astype(np.uint8)
        np.savez_compressed(out / path.name, **(sidecar | {"head_sha": np.str_(stamp)}))
    (out / "_GATE_OK").touch()
    return cohort | {"f5": out, "f5_head": head_path}


def f5_paths(dirs) -> dict[str, str]:
    return {"data_dir": str(dirs["tiles"]), "frac_dir": str(dirs["fracs"]), "r2_label_dir": str(dirs["r2"]),
            "b28_label_dir": str(dirs["b28"]), "f5_label_dir": str(dirs["f5"]), "f5_head": str(dirs["f5_head"])}


def f5_args(dirs) -> list[str]:
    return [arg for key, value in f5_paths(dirs).items() for arg in (f"--{key.replace('_', '-')}", value)]


def test_preflight_accepts_the_f5_labels(f5):
    assert recipe.preflight(f5_paths(f5)) == []


def _no_gate(d):
    (d["f5"] / "_GATE_OK").unlink()


def _other_head(d):
    np.savez(d["f5_head"], **(dict(np.load(d["f5_head"])) | {"b2": np.ones(6, np.float32)}))


def _r4_head(d):
    z = dict(np.load(d["f5_head"]))
    np.savez(d["f5_head"], **(z | {"W2": z["W2"][:, :5], "b2": z["b2"][:5], "classes_": np.arange(5)}))


def _missing_tile(d):
    sorted(d["f5"].glob("*.npz"))[0].unlink()


def _class_29(d):
    path = sorted(d["f5"].glob("*.npz"))[0]
    with np.load(path) as z:
        sidecar = dict(z)
    sidecar["label"][0, 0] = 29
    np.savez_compressed(path, **sidecar)


@pytest.mark.parametrize("damage, message", [
    (_no_gate, "has no _GATE_OK"),
    (_other_head, "was not written by"),
    (_r4_head, "expected [0, 1, 2, 3, 4, 28]"),
    (_missing_tile, "cohort differs"),
    (_class_29, "label 29 with 29 classes"),
])
def test_preflight_refuses_wrong_or_partial_labels(f5, damage, message):
    damage(f5)
    assert any(message in p for p in recipe.preflight(f5_paths(f5)))


CPU = ["--", "--patch-size", "64", "--img-size", "64", "--batch-size", "2", "--num-workers", "0", "--device", "cpu"]


def test_cpu_smoke_trains_29_classes_with_mains_weights(f5, tmp_path, capfd):
    checkpoints = tmp_path / "ckpt_f5"
    code = recipe.main(["--profile", "2080ti", "--epochs", "1", "--checkpoint-dir", str(checkpoints),
                        *f5_args(f5), *CPU])
    assert code == 0, capfd.readouterr().out
    state = torch.load(checkpoints / "best_model.pt", map_location="cpu", weights_only=False)
    assert tuple(state["state_dict"]["model.classifier.weight"].shape[:1]) == (29,)
    out = capfd.readouterr().out
    assert "--disable-bf16" in out and f"--label-dir {f5['f5']}" in out
    assert json.loads((checkpoints / retrain.B28_STATS).read_text()) == \
        retrain.b28_class_stats(str(f5["tiles"]), str(f5["b28"]))
    assert recipe.main(["--profile", "2080ti", "--epochs", "1", "--checkpoint-dir", str(checkpoints),
                        *f5_args(f5), *CPU]) == 1
    assert "already holds" in capfd.readouterr().out


def test_only_the_smoke_overrides_may_follow_the_separator(tmp_path, capsys):
    assert recipe.main(["--profile", "2080ti", "--epochs", "1", "--checkpoint-dir", str(tmp_path / "c"),
                        "--", "--label-dir", "/x"]) == 1
    assert "only --patch-size" in capsys.readouterr().out


# ── jobs ─────────────────────────────────────────────────────────────────────

SHA_LINE = re.compile(r"^\s*SHA=(\S+)$", re.M)
SOURCE_SHA = re.compile(r"REPLACE_WITH_PUSHED_SHA|[0-9a-f]{40}")
#: What the f5 jobs run or import; the pinned commit must carry them as at HEAD.
PINNED_SCRIPTS = ("scripts/nfi_truth_f5.py", "scripts/extract_plot_features.py", "scripts/nfi_head_cv.py",
                  "scripts/train_distill_head.py", "scripts/distill_forest_labels.py",
                  "scripts/validate_against_nfi.py", "imint/eval/nfi_amendment.py",
                  "scripts/im023_retrain_f5.py", "scripts/im023_retrain.py",
                  "scripts/im023_retrain_prithvi600m.py", "scripts/relabel_barrblandskog.py")


def pod(doc: dict) -> dict:
    return doc["spec"]["template"]["spec"]


def pinned_sha() -> str:
    (sha,) = {m for p in sorted(JOBS.glob("*.yaml")) for m in SHA_LINE.findall(body(load(p)))}
    return sha


def test_jobs_pin_one_source_on_a_2080ti():
    docs = [load(p) for p in sorted(JOBS.glob("*.yaml"))]
    assert [d["metadata"]["name"] for d in docs] == ["im023-f5-distill", "im023-tessera-f5"]
    assert SOURCE_SHA.fullmatch(pinned_sha())
    for doc in docs:
        text = body(doc)
        assert 'test "$(git rev-parse HEAD)" = "$SHA"' in text
        assert text.index('[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]') < text.index("apt-get")
        assert pod(doc)["nodeSelector"] == {"accelerator": "nvidia-gtx-2080ti"}
        assert doc["metadata"]["namespace"] == "prithvi-training-default"
        assert doc["spec"]["backoffLimit"] == 0
        resources = pod(doc)["containers"][0]["resources"]
        assert resources["requests"] == resources["limits"]     # ICE admission webhook
        assert "@sha256:" in pod(doc)["containers"][0]["image"]


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
    assert changed == [], f"pinned SHA predates changes to {changed}: bump SHA in k8s/im023/f5/"


def step(job: Path, script: str) -> dict[str, str | None]:
    (call,) = invocations(body(load(job)), script)
    return call


#: script → (flags whose value the f5 chain changes, flags it adds)
CHAIN = {
    "scripts/extract_plot_features.py": ({"--plot-index", "--out"}, {"--truth-col"}),
    "scripts/nfi_head_cv.py": ({"--features", "--out"}, {"--truth-col"}),
    "scripts/train_distill_head.py": ({"--features", "--out-head", "--out-split"}, {"--truth-col"}),
    "scripts/distill_forest_labels.py": ({"--head", "--label-dir", "--out-dir"}, {"--forest-classes"}),
}


@pytest.mark.parametrize("script", sorted(CHAIN))
def test_distill_steps_are_r4s_but_for_the_planned_changes(script):
    ours, r4 = step(JOBS / "distill-f5-job.yaml", script), step(LADDER_DISTILL, script)
    changed, added = CHAIN[script]
    assert {k for k in r4 if ours.get(k) != r4[k]} == changed
    assert set(ours) - set(r4) == added
    if "--truth-col" in added:
        assert ours["--truth-col"] == truth.TRUTH_COL


def test_distill_job_writes_new_paths_and_signals_last():
    text = body(load(JOBS / "distill-f5-job.yaml"))
    dense = step(JOBS / "distill-f5-job.yaml", "scripts/distill_forest_labels.py")
    assert dense["--label-dir"] == "/cephfs/nmd2023_labels_b28"
    assert dense["--out-dir"] == recipe.PATHS["f5_label_dir"]
    assert dense["--forest-classes"] == ",".join(map(str, recipe.F5_FOREST_CLASSES))
    assert f"HEAD={recipe.PATHS['f5_head']}" in text
    run = text.split("# --- run ---")[1]
    assert run.index('rm -f "$OUT/_GATE_OK"') < run.index("scripts/nfi_truth_f5.py derive")
    assert run.rstrip().splitlines()[-2].strip() == 'touch "$OUT/_GATE_OK"'
    assert run.index("same-plots") < run.index("scripts/train_distill_head.py")
    # R4's distillation outputs are only read.
    for r4_output in ("/cephfs/distill/tessera_r2\n", "heads/tessera_r2_head.npz", "heads/tessera_r2_split.json"):
        assert r4_output not in run


def test_training_job_calls_the_recipe_with_the_2080ti_profile():
    (call,) = invocations(body(load(JOBS / "train-f5-job.yaml")), "/workspace/imintengine/scripts/im023_retrain_f5.py")
    assert (call["--profile"], call["--epochs"], call["--checkpoint-dir"]) == \
           ("2080ti", "30", "/cephfs/checkpoints/im023/tessera_f5")
