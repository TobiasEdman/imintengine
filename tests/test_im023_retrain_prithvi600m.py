"""IM-023 Prithvi-600M recipe: R4's flags plus the planned deltas, the preflight, a CPU smoke.

The cohort fixture of test_im023_retrain (R4-style NMD and TESSERA distilled
sidecars, relabelled by relabel_barrblandskog.run) gains a second
distillation, relabelled by relabel_distilled_b28 against the same NMD
directory, as Prithvi-600M's is on the cluster.
"""
from pathlib import Path
import json
import re
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
from tests.test_im023_retrain import cohort, flags  # noqa: F401  (cohort is a fixture)

R4 = ROOT / "k8s" / "ladder" / "ladder-r4-prithvi600m-job.yaml"
R4_CHECKPOINTS = "/cephfs/checkpoints/ladder/prithvi600m_r4"


def manifest_flags(path: Path) -> dict[str, str | None]:
    script = yaml.safe_load(path.read_text())["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    lines = [ln.strip().rstrip("\\").strip() for ln in script.splitlines()]
    return flags(" ".join(ln for ln in lines if ln.startswith("--")).split())


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
    assert "already holds a model" in capfd.readouterr().out


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
