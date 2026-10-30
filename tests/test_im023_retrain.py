"""IM-023 retrain recipe: R4's flags, the preflight refusals, and a CPU smoke.

The smoke runs the real chain on synthetic tiles: R4-style NMD and distilled
sidecars → relabel_barrblandskog.run → im023_retrain.py → train_unified.py,
one epoch per variant, and checks the classifier width of each checkpoint.
"""
from pathlib import Path
import json
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
