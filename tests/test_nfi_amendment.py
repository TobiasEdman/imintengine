"""IM-023 amendment to the 2026-10-27 NFI freeze: binding, rules, scoring.

Synthetic freeze, tiles, checkpoints and plots only; no NFI data. The sealed
runtime is replaced by a fixture where noted; ``sealed_runtime``'s own test
runs against a temporary source tree, and the checkpoint child runs for real
on a synthetic checkpoint.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from imint.eval import nfi_amendment as na  # noqa: E402
from imint.eval.fieldtruth import NFI_PROTOCOL, load_frozen_holdout, sha256_file  # noqa: E402
from tests.nfi_freeze_fixtures import complete_manifest  # noqa: E402
import infer_nfi_amendment as infer  # noqa: E402
import prepare_nfi_amendment as prepare  # noqa: E402
import score_nfi_amendment as score  # noqa: E402
from prepare_nfi_holdout import file_identity  # noqa: E402
from validate_against_nfi import collapse_fractions_to_nfi_class, derive_nfi_forest_class  # noqa: E402

IMAGE = "registry/im023@sha256:" + "e" * 64
SOURCE_SHA = "f" * 40
RUNTIME_MANIFEST = Path("/fixture/runtime.json")
RUNTIME = {"image": {"ref": IMAGE, "digest": "sha256:" + "e" * 64},
           "source": {"git_sha": SOURCE_SHA, "payload_sha256": "c" * 64},
           "runtime_manifest": {"path": str(RUNTIME_MANIFEST), "sha256": "d" * 64},
           "environments": {name: {"python": {"path": sys.executable}}
                            for name in ("model", "scoring")}}
CONTRACT = {"enabled_aux_names": ["dem"], "n_aux_channels": 1}
VOLUMES = {  # VolPine, VolContorta, VolSpruce, VolBirch, VolOtherDec
    "pine": (100, 0, 0, 0, 0), "spruce": (0, 0, 100, 0, 0), "barr": (50, 0, 50, 0, 0),
    "bland": (50, 0, 0, 50, 0), "lov": (0, 0, 0, 100, 0), "none": (0, 0, 0, 0, 0),
    "contorta": (60, 40, 0, 0, 0),
}
SHARES = {  # tall, gran, trivial, adel -> share-rule label
    1: (0.9, 0.0, 0.0, 0.0), 2: (0.0, 0.9, 0.0, 0.0), 28: (0.45, 0.45, 0.0, 0.0),
    4: (0.45, 0.0, 0.45, 0.0), 3: (0.0, 0.0, 0.9, 0.0), 0: (0.01, 0.0, 0.0, 0.0),
}


def plots(kinds: list[str], *, blocks: int = 1, tiles: list[str] | None = None) -> pd.DataFrame:
    """``blocks`` identical replicas, each its own tract and 50 km block."""
    rows = []
    for b in range(blocks):
        for i, kind in enumerate(kinds):
            pine, contorta, spruce, birch, other = VOLUMES[kind]
            rows.append(dict(TractID=b + 1, PlotID=i + 1, Year=2022,
                             tile_name=(tiles or ["t0"])[i % len(tiles or ["t0"])],
                             row=1 + i // 3, col=1 + i % 3,
                             VolPine=float(pine), VolContorta=float(contorta),
                             VolSpruce=float(spruce), VolBirch=float(birch),
                             VolOtherDec=float(other),
                             Easting=500000.0 + 100000.0 * b, Northing=6500000.0))
    frame = pd.DataFrame(rows)
    frame["nfi_forest"] = [derive_nfi_forest_class(r) or 0 for _, r in frame.iterrows()]
    return frame


def cell(holdout: pd.DataFrame, hard: list[int], shares: list[int]) -> pd.DataFrame:
    """A dump-like frame: hard predictions, and fractions whose share rule gives ``shares``."""
    reps = len(holdout) // len(hard)
    frame = holdout[na.NFI_KEY + ["tile_name", "nfi_forest"]].copy()
    frame["model_pred"] = np.tile(hard, reps)
    values = np.array([SHARES[s] for s in shares] * reps)
    for k, column in enumerate(na.FRACTION_COLUMNS):
        frame[column] = values[:, k]
    return frame


# ── synthetic freeze and amendment ───────────────────────────────────────────

def write_tile(path: Path, fill: float) -> None:
    np.savez(path, spectral=np.full((24, 8, 8), .1, np.float32),
             tessera=np.full((128, 8, 8), fill, np.float32), year=2022)


def base_freeze(tmp_path: Path, held: pd.DataFrame) -> tuple[Path, dict]:
    root = tmp_path / "freeze"
    root.mkdir()
    tiles = {}
    for i, name in enumerate(sorted(held["tile_name"].unique())):
        path = tmp_path / f"{name}.npz"
        write_tile(path, (i + 1) / 10)
        tiles[name] = dict(file_identity(path), geometry={"height": 8, "width": 8})
    held = held.assign(tile_path=held["tile_name"].map(lambda n: tiles[n]["path"]),
                       tile_role="campaign")
    held.to_parquet(root / "holdout.parquet", index=False)
    held[na.NFI_KEY].iloc[:0].to_parquet(root / "training.parquet", index=False)
    (tmp_path / "r4.pt").write_bytes(b"r4 checkpoint")
    (tmp_path / "nmd2023.tif").write_bytes(b"nmd raster")
    template = {"checkpoint": file_identity(tmp_path / "r4.pt"), "backbone": "tessera_v1",
                "img_size": 8, "num_classes": 28, "head": "class", "input_contract": CONTRACT}
    manifest = complete_manifest({
        "schema": "nfi-fieldtruth-freeze-v1", "campaign": {"training_tiles": 0},
        "cells": {"tessera_r4": template}, "tiles": tiles,
        "baselines": {"NMD2023": file_identity(tmp_path / "nmd2023.tif")},
        "holdout": {"file": "holdout.parquet", "sha256": sha256_file(root / "holdout.parquet"),
                    "observations": len(held)},
        "training": {"file": "training.parquet", "sha256": sha256_file(root / "training.parquet"),
                     "observations": 0}})
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


@pytest.fixture
def freeze(tmp_path, monkeypatch):
    """A freeze, the three checkpoints and an amendment built by the prepare step."""
    held = plots(["pine", "spruce", "barr", "bland", "lov", "none"], tiles=["t0", "t1"])
    base_path, _ = base_freeze(tmp_path, held)
    digest = sha256_file(base_path)
    monkeypatch.setattr(na, "BASE_MANIFEST_SHA256", digest)
    monkeypatch.setattr(prepare, "BASE_MANIFEST_SHA256", digest)
    monkeypatch.setattr(prepare, "sealed_runtime", lambda root, manifest, **kw: deepcopy(RUNTIME))
    monkeypatch.setattr(na, "sealed_runtime", lambda root, manifest, **kw: deepcopy(RUNTIME))
    checkpoints = {"im023_main": tmp_path / "main.pt", "im023_control": tmp_path / "control.pt",
                   "r4": tmp_path / "r4.pt"}
    checkpoints["im023_main"].write_bytes(b"main checkpoint")
    checkpoints["im023_control"].write_bytes(b"control checkpoint")
    stub_checkpoint_metadata(monkeypatch, lambda name: dict(CONTRACT))
    return types.SimpleNamespace(tmp=tmp_path, base=base_path, held=held, checkpoints=checkpoints)


def stub_checkpoint_metadata(monkeypatch, contract) -> None:
    """The child's answer: each cell's own class count, a fraction head, ``contract(cell)``."""
    def metadata(cells, model_python):
        assert model_python == sys.executable
        return {name: {"num_classes": na.AMENDMENT_CELLS[name], "fraction_head": True,
                       "input_contract": contract(name)} for name in cells}
    monkeypatch.setattr(prepare, "checkpoint_metadata", metadata)


def build(freeze, checkpoints=None) -> dict:
    return prepare.build_manifest(freeze.base, checkpoints or freeze.checkpoints,
                                  runtime_manifest=RUNTIME_MANIFEST, runtime_image=IMAGE,
                                  source_git_sha=SOURCE_SHA)


def write_amendment(freeze, manifest: dict | None = None) -> tuple[Path, str]:
    if manifest is None:
        manifest = build(freeze)
    path = freeze.tmp / "amendment.json"
    path.write_text(json.dumps(manifest))
    return path, sha256_file(path)


def test_amendment_binds_the_same_frozen_holdout(freeze):
    before = {p.name: p.read_bytes() for p in freeze.base.parent.iterdir()}
    path, digest = write_amendment(freeze)
    holdout, amendment, base = na.load_amended_holdout(path, digest)
    frozen, _ = load_frozen_holdout(freeze.base, na.BASE_MANIFEST_SHA256)
    pd.testing.assert_frame_equal(holdout, frozen)
    assert amendment["base"]["holdout_sha256"] == base["holdout"]["sha256"]
    assert amendment["cells"]["r4"]["checkpoint"] == base["cells"]["tessera_r4"]["checkpoint"]
    assert {n: c["num_classes"] for n, c in amendment["cells"].items()} == na.AMENDMENT_CELLS
    assert amendment["protocol"] == na.AMENDMENT_PROTOCOL
    assert amendment["execution"] == "awaiting_user_go_no_go"
    # The freeze is read, never written.
    assert {p.name: p.read_bytes() for p in freeze.base.parent.iterdir()} == before
    assert na.verify_amendment_source(amendment, environment="model")["source_git_sha"] == SOURCE_SHA


def test_amendment_refuses_another_holdout_or_unapproved_bytes(freeze):
    manifest = build(freeze)
    path, digest = write_amendment(freeze, manifest)
    with pytest.raises(ValueError, match="approved SHA256"):
        na.load_amended_holdout(path, "0" * 64)
    other = deepcopy(manifest)
    other["base"]["holdout_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="frozen holdout"):
        na.load_amended_holdout(*write_amendment(freeze, other))
    other = deepcopy(manifest)
    other["base"]["manifest_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="incomplete amendment"):
        na.load_amended_holdout(*write_amendment(freeze, other))
    other = deepcopy(manifest)
    other["cells"]["r4"]["checkpoint"] = file_identity(freeze.checkpoints["im023_control"])
    other["cells"]["im023_control"]["checkpoint"] = file_identity(freeze.checkpoints["r4"])
    with pytest.raises(ValueError, match="frozen tessera_r4"):
        na.load_amended_holdout(*write_amendment(freeze, other))
    # A changed base manifest no longer matches the pinned digest.
    base = json.loads(freeze.base.read_text())
    base["execution"] = "edited"
    freeze.base.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="frozen manifest differs from the approved SHA256"):
        na.load_amended_holdout(*write_amendment(freeze, manifest))


def test_prepare_refuses_a_foreign_r4_and_an_existing_output(freeze, monkeypatch):
    foreign = dict(freeze.checkpoints, r4=freeze.checkpoints["im023_control"])
    with pytest.raises(ValueError, match="frozen tessera_r4"):
        build(freeze, foreign)
    out = freeze.tmp / "existing"
    out.mkdir()
    monkeypatch.setattr(sys, "argv", ["prepare", "--base-manifest", str(freeze.base),
                                      "--checkpoint-main", "m", "--checkpoint-control", "c",
                                      "--checkpoint-r4", "r", "--runtime-image", IMAGE,
                                      "--runtime-manifest", str(RUNTIME_MANIFEST),
                                      "--source-git-sha", SOURCE_SHA, "--out-dir", str(out)])
    with pytest.raises(ValueError, match="never overwrite"):
        prepare.main()


def test_prepare_refuses_changed_tiles_lists_every_one_and_writes_nothing(freeze, monkeypatch):
    """Tobias 2026-11-07: refuse, as the freeze's own consumers do, naming all."""
    write_tile(freeze.tmp / "t0.npz", 0.7)
    (freeze.tmp / "t1.npz").unlink()
    out = freeze.tmp / "amendment"
    c = freeze.checkpoints
    monkeypatch.setattr(sys, "argv", ["prepare", "--base-manifest", str(freeze.base),
                                      "--checkpoint-main", str(c["im023_main"]),
                                      "--checkpoint-control", str(c["im023_control"]),
                                      "--checkpoint-r4", str(c["r4"]), "--runtime-image", IMAGE,
                                      "--runtime-manifest", str(RUNTIME_MANIFEST),
                                      "--source-git-sha", SOURCE_SHA, "--out-dir", str(out)])
    with pytest.raises(ValueError, match="2 of 2 frozen tiles changed") as refused:
        prepare.main()
    assert "t0: frozen input content changed" in str(refused.value)
    assert "t1: " in str(refused.value)
    assert not out.exists()   # a refusal writes nothing, so prepare can run again


def test_prepare_refuses_aux_outside_the_freeze_audit(freeze, monkeypatch):
    stub_checkpoint_metadata(monkeypatch, lambda name: {"enabled_aux_names": ["dem", "markfukt"]})
    with pytest.raises(ValueError, match="outside the freeze's tile audit"):
        build(freeze)


def test_prepare_refuses_an_r4_contract_other_than_the_frozen_one(freeze, monkeypatch):
    stub_checkpoint_metadata(monkeypatch, lambda name: {"enabled_aux_names": [], "n_aux_channels": 0})
    with pytest.raises(ValueError, match="r4 input contract differs"):
        build(freeze)


def test_prepare_refuses_a_checkpoint_without_its_heads(freeze, monkeypatch):
    def metadata(cells, model_python):
        return {name: {"num_classes": 28, "fraction_head": True, "input_contract": dict(CONTRACT)}
                for name in cells}
    monkeypatch.setattr(prepare, "checkpoint_metadata", metadata)
    with pytest.raises(ValueError, match="im023_main: checkpoint lacks a 29-class head"):
        build(freeze)


def test_checkpoint_child_reads_heads_and_contract_without_a_model(tmp_path):
    """The real child process, as prepare calls it, on a synthetic checkpoint."""
    path = tmp_path / "main.pt"
    torch.save({"config": {"enabled_aux_names": ["dem"], "n_aux_channels": 1, "num_classes": 29,
                           "enable_tradslag_head": True, "num_tradslag": 4},
                "model_state_dict": {"lidar_branch.net.0.conv.weight": torch.ones(2, 1, 3, 3),
                                     "frac_head.weight": torch.ones(4, 2)}}, path)
    cells = {"im023_main": {"checkpoint": file_identity(path)}}
    (cell,) = prepare.checkpoint_metadata(cells, sys.executable).values()
    assert cell["num_classes"] == 29 and cell["fraction_head"] is True
    assert cell["input_contract"]["enabled_aux_names"] == ["dem"]
    assert cell["input_contract"]["n_aux_channels"] == 1
    torch.save({"config": {"num_classes": 29}, "model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="checkpoint metadata preparation failed"):
        prepare.checkpoint_metadata(cells, sys.executable)   # bytes differ from the identity


def test_sealed_runtime_binds_the_source_tree_and_the_interpreter(tmp_path, monkeypatch):
    from imint.crop_distill_runtime import snapshot_tree, tree_payload_sha256

    root = tmp_path / "source"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "a.py").write_text("x = 1\n")
    payload = tree_payload_sha256(snapshot_tree(root))
    calls = []

    def verify_runtime(manifest, *, source_git_sha, image_ref):
        calls.append((manifest, source_git_sha, image_ref))
        return dict(deepcopy(RUNTIME), source={"git_sha": source_git_sha, "payload_sha256": payload},
                    environments={"model": {"python": {"path": sys.executable}},
                                  "scoring": {"python": {"path": "/opt/venvs/scoring/bin/python"}}})

    monkeypatch.setattr(na, "verify_runtime", verify_runtime)
    runtime = na.sealed_runtime(root, RUNTIME_MANIFEST, source_git_sha=SOURCE_SHA,
                                image_ref=IMAGE, environment="model")
    assert runtime["source"]["payload_sha256"] == payload
    assert calls == [(RUNTIME_MANIFEST, SOURCE_SHA, IMAGE)]
    with pytest.raises(ValueError, match="verified scoring interpreter"):
        na.sealed_runtime(root, RUNTIME_MANIFEST, source_git_sha=SOURCE_SHA,
                          image_ref=IMAGE, environment="scoring")
    (root / "scripts" / "a.py").write_text("x = 2\n")
    with pytest.raises(ValueError, match="verified source tree"):
        na.sealed_runtime(root, RUNTIME_MANIFEST, source_git_sha=SOURCE_SHA,
                          image_ref=IMAGE, environment="model")


def test_changed_runtime_or_source_fails_consumers(freeze, monkeypatch):
    path, digest = write_amendment(freeze)
    _, amendment, _ = na.load_amended_holdout(path, digest)
    changed = deepcopy(RUNTIME)
    changed["runtime_manifest"]["sha256"] = "9" * 64
    monkeypatch.setattr(na, "sealed_runtime", lambda root, manifest, **kw: changed)
    with pytest.raises(ValueError, match="execution runtime differs"):
        na.verify_amendment_source(amendment, environment="model")
    edited = deepcopy(amendment)
    edited["source_sha256"]["scripts/score_nfi_amendment.py"] = "8" * 64
    with pytest.raises(ValueError, match="evaluation code differs"):
        na.verify_amendment_source(edited, environment="model")
    missing = deepcopy(amendment)
    del missing["preparation_runtime"]["runtime_manifest"]["path"]
    with pytest.raises(ValueError, match="incomplete amendment runtime identity"):
        na.verify_amendment_source(missing, environment="model")


# ── rules ────────────────────────────────────────────────────────────────────

def test_primary_rule_folds_28_by_tall_vs_gran_share():
    pred = np.array([28, 28, 28, 1, 2, 4, 0])
    tall = np.array([.6, .2, .3, .0, .9, .5, np.nan])
    gran = np.array([.2, .6, .3, .9, .0, .5, np.nan])
    assert na.fold_barrblandskog(pred, tall, gran).tolist() == [1, 2, 1, 1, 2, 4, 0]
    # The tie rule is collapse_fractions_to_nfi_class's.
    assert collapse_fractions_to_nfi_class(.3, .3, 0, 0) == 1
    with pytest.raises(ValueError, match="finite"):
        na.fold_barrblandskog(np.array([28]), np.array([np.nan]), np.array([.1]))


def test_derived_barrblandskog_truth_exclusion_and_contorta():
    held = pd.DataFrame([VOLUMES[k] for k in ("pine", "spruce", "barr", "bland", "none", "contorta")]
                        + [(40, 35, 25, 0, 0), (35, 0, 35, 30, 0), (34, 0, 35, 31, 0), (30, 0, 70, 0, 0)],
                        columns=["VolPine", "VolContorta", "VolSpruce", "VolBirch", "VolOtherDec"],
                        dtype=float)
    held["nfi_forest"] = [derive_nfi_forest_class(r) or 0 for _, r in held.iterrows()]
    barr = na.derive_barrblandskog(held, species_fraction=.7)
    # pine+contorta 0.75 of conifer is pure tall; conifer exactly 0.7 counts,
    # 0.69 does not; 0.7 spruce of conifer is pure gran.
    assert barr.tolist() == [False, False, True, False, False, False, False, True, False, False]
    assert (na.total_volume(held) > 0).tolist() == [True] * 4 + [False] + [True] * 5
    share = na.contorta_share(held)
    assert share[5] == pytest.approx(.4) and share[4] == 0 and share[1] == 0
    edge = pd.DataFrame({"VolPine": [70., 71.], "VolContorta": [30., 29.], "VolSpruce": [0., 0.],
                         "VolBirch": [0., 0.], "VolOtherDec": [0., 0.]})
    assert (na.contorta_share(edge) < .3).tolist() == [False, True]
    with pytest.raises(ValueError, match="nonnegative"):
        na.derive_barrblandskog(held.assign(VolBirch=-1.), species_fraction=.7)


def test_barrblandskog_dominance_is_truth_v1s_own_at_the_boundary():
    """Review 2026-11-07: a second sum of the same volumes rounds the other way."""
    plot = pd.DataFrame([dict(VolPine=15.4, VolContorta=0., VolSpruce=7.7, VolBirch=9.3,
                              VolOtherDec=.6)])
    conifer = 15.4 + 7.7
    assert conifer / (conifer + 9.3 + .6) < .7 <= conifer / (conifer + (9.3 + .6))
    plot["nfi_forest"] = [derive_nfi_forest_class(plot.iloc[0])]
    assert plot["nfi_forest"].item() == na.TALLSKOG          # v1: conifer-dominated
    assert na.derive_barrblandskog(plot, species_fraction=.7).tolist() == [True]   # pine 0.67


def test_shared_protocol_constants_are_the_frozen_ones():
    shared = na.AMENDMENT_PROTOCOL.keys() & NFI_PROTOCOL.keys()
    assert {"bootstrap_seed", "bootstrap_samples", "block_km", "sesoi",
            "truth_dominant_fraction"} <= shared
    assert all(na.AMENDMENT_PROTOCOL[k] == NFI_PROTOCOL[k] for k in shared)


def test_share_rule_one_level_down_keeps_the_collapse_rule_and_floor():
    protocol = na.AMENDMENT_PROTOCOL
    cases = pd.DataFrame([
        (.7, .3, 0, 0), (.3, .7, 0, 0), (.6, .4, 0, 0), (.4, .6, 0, 0),   # conifer
        (.4, .1, .5, 0), (0, 0, .5, .4), (.05, .04, 0, 0), (.1, 0, 0, 0),  # bland, löv, floor
    ], columns=na.FRACTION_COLUMNS)
    assert score.share_rule(cases, protocol).tolist() == [1, 2, 28, 28, 4, 3, 0, 1]
    rng = np.random.default_rng(3)
    random = pd.DataFrame(rng.uniform(0, .5, size=(500, 4)), columns=na.FRACTION_COLUMNS)
    collapsed = np.array([collapse_fractions_to_nfi_class(*r) for r in random.itertuples(index=False)])
    split = score.share_rule(random, protocol)
    assert (split[~np.isin(collapsed, [1, 2])] == collapsed[~np.isin(collapsed, [1, 2])]).all()
    assert set(split[np.isin(collapsed, [1, 2])]) <= {1, 2, 28}


# ── item 3 ───────────────────────────────────────────────────────────────────

def test_primary_report_folds_main_and_reports_28_as_its_own_column():
    held = plots(["pine", "spruce", "bland", "none"], blocks=2)
    cells = {"im023_main": cell(held, [28, 28, 4, 0], [1, 2, 4, 0]),
             "im023_control": cell(held, [1, 2, 4, 0], [1, 2, 4, 0]),
             "r4": cell(held, [1, 2, 3, 0], [1, 2, 4, 0])}
    raw = np.full(len(held), 111)
    baselines = {"NMD2023": (held["nfi_forest"].to_numpy(), raw)}
    protocol = dict(NFI_PROTOCOL, primary_population="all")
    report = score.primary_report(held, cells, baselines, protocol, na.AMENDMENT_PROTOCOL)
    assert report == score.primary_report(held, cells, baselines, protocol, na.AMENDMENT_PROTOCOL)
    scores = report["primary"]["scores"]
    assert scores["im023_main"]["overall_exact"] == scores["im023_control"]["overall_exact"] == 1.0
    assert scores["r4"]["overall_exact"] == .75
    assert set(report["primary"]["pairs"]) == {"NMD2023 vs im023_control", "NMD2023 vs im023_main",
                                               "NMD2023 vs r4", "im023_control vs im023_main",
                                               "im023_control vs r4", "im023_main vs r4"}
    main = np.array(report["secondary_class_28"]["confusion_matrix"]["im023_main"])
    assert main[:, -1].tolist() == [0, 2, 2, 0, 0]     # rows tall and gran, column 28
    assert np.array(report["secondary_class_28"]["confusion_matrix"]["r4"])[:, -1].sum() == 0
    assert report["fold_counts"]["im023_main"] == {"predicted_28": 4, "to_tallskog": 2, "to_granskog": 2}


def test_primary_contorta_sensitivity_is_descriptive_on_the_same_coverage():
    """Tobias 2026-11-07: item 3 also without plots where contorta >= 30 % of pine."""
    held = plots(["pine", "spruce", "contorta", "bland", "none"], blocks=3)
    cells = {"im023_main": cell(held, [28, 2, 2, 4, 0], [1, 2, 1, 4, 0]),
             "im023_control": cell(held, [1, 2, 2, 4, 0], [1, 2, 1, 4, 0]),
             "r4": cell(held, [1, 2, 1, 4, 0], [1, 2, 1, 4, 0])}
    raw = np.full(len(held), 111)
    raw[held["TractID"].to_numpy() == 3] = 0          # block 3 outside NMD coverage
    baselines = {"NMD2023": (held["nfi_forest"].to_numpy(), raw)}
    protocol = dict(NFI_PROTOCOL, primary_population="all")
    block = score.primary_report(held, cells, baselines, protocol,
                                 na.AMENDMENT_PROTOCOL)["sensitivity_without_contorta_plots"]
    assert block["contorta_fraction_of_pine_volume"] == .3
    assert block["excluded_contorta_observations"] == 2      # covered contorta plots only
    assert block["compared_observations"] == 8
    assert set(block["scores"]) == {"im023_main", "im023_control", "r4", "NMD2023"}
    assert "pairs" not in block and "verdict" not in block
    # Without the contorta plots, control's only error (contorta as granskog) is gone.
    assert block["scores"]["im023_control"]["overall_exact"] == 1.0
    assert block["scores"]["im023_main"]["overall_exact"] == 1.0   # 28 folded to tallskog
    assert np.array(block["scores"]["r4"]["confusion_matrix"]).sum() == 8


# ── item 5 ───────────────────────────────────────────────────────────────────

KINDS = ["pine", "spruce", "barr", "bland", "lov", "none", "contorta"]
HARD_PERFECT = [1, 2, 28, 4, 3, 0, 1]
SHARE_PERFECT = [1, 2, 28, 4, 3, 0, 1]


def mixed(main_hard, main_share, control_hard, control_share, *, kinds=KINDS, blocks=4):
    held = plots(kinds, blocks=blocks)
    cells = {"im023_main": cell(held, main_hard, main_share),
             "im023_control": cell(held, control_hard, control_share),
             "r4": cell(held, control_hard, control_share)}
    return score.mixed_report(held, cells, np.ones(len(held), bool), na.AMENDMENT_PROTOCOL)


def test_condition_1_holds_but_2_does_not_keeps_the_class():
    report = mixed(HARD_PERFECT, SHARE_PERFECT, [1, 2, 1, 4, 3, 0, 1], SHARE_PERFECT)
    barr = report["primary"]["conditions"]["barrblandskog"]
    assert barr["condition_1"]["holds"] is True and barr["condition_1"]["bound"] == "lower"
    assert barr["condition_2"]["tallskog"]["bound_value"] == 0.0
    assert barr["condition_2"]["holds"] is False
    assert report["verdict"] == {"barrblandskog": "keep_class",
                                 "blandskog": "derivable_from_shares_without_loss"}
    # Identical blocks: every draw equals the point estimate.
    assert barr["condition_1"]["interval"] == [0.0, 0.0]
    support = report["support"]
    assert support["excluded_without_volume"] == 4 and support["derived_barrblandskog"] == 4
    assert support["excluded_in_sensitivity_contorta"] == 4
    assert report["primary"]["observations"] == {"volume": 24, "pure": 20}
    assert report["sensitivity_without_contorta_plots"]["observations"] == {"volume": 20, "pure": 16}
    assert "verdict" not in report["sensitivity_without_contorta_plots"]
    assert set(report["primary"]["metrics"]["barrblandskog"]) == {"a", "c", "d", "c_r4"}


def test_both_conditions_recommend_derivation_and_a_failed_condition_1_keeps_it():
    # Main also predicts 28 on a pure pine stand where control predicts tallskog.
    kinds = KINDS + ["pine"]
    derive = mixed(HARD_PERFECT + [28], SHARE_PERFECT + [1], [1, 2, 1, 4, 3, 0, 1, 1],
                   SHARE_PERFECT + [1], kinds=kinds)
    barr = derive["primary"]["conditions"]["barrblandskog"]
    tall = barr["condition_2"]["tallskog"]
    # tallskog: a has tp 2 of 3 pure pine plots, b all 3: F1 0.8 vs 1.
    assert tall["delta_f1"] == pytest.approx(.8 - 1) and tall["holds"] is True
    assert barr["condition_1"]["holds"] is True
    assert derive["verdict"]["barrblandskog"] == "derive_from_shares"
    # The share rule on control misses barrblandskog: condition 1 fails.
    missed = SHARE_PERFECT[:2] + [1] + SHARE_PERFECT[3:] + [1]
    keep = mixed(HARD_PERFECT + [28], SHARE_PERFECT + [1], [1, 2, 1, 4, 3, 0, 1, 1], missed, kinds=kinds)
    assert keep["primary"]["conditions"]["barrblandskog"]["condition_1"]["holds"] is False
    assert keep["primary"]["conditions"]["barrblandskog"]["condition_2"]["holds"] is True
    assert keep["verdict"]["barrblandskog"] == "keep_class"


def test_blandskog_condition_is_within_control_and_an_undefined_bound_keeps_the_class():
    bad_bland = SHARE_PERFECT[:3] + [3] + SHARE_PERFECT[4:]
    report = mixed(HARD_PERFECT, SHARE_PERFECT, [1, 2, 1, 4, 3, 0, 1], bad_bland)
    assert report["primary"]["conditions"]["blandskog"]["condition_1"]["comparison"] == "F1(c) - F1(b)"
    assert report["verdict"]["blandskog"] == "not_established"
    # No granskog truth and no granskog prediction: its F1 is undefined.
    kinds = ["pine", "barr", "bland", "none"]
    report = mixed([1, 28, 4, 0], [1, 28, 4, 0], [1, 1, 4, 0], [1, 28, 4, 0], kinds=kinds)
    barr = report["primary"]["conditions"]["barrblandskog"]
    assert barr["condition_2"]["granskog"]["holds"] is None
    assert barr["condition_2"]["granskog"]["undefined_draws"] == na.AMENDMENT_PROTOCOL["bootstrap_samples"]
    assert barr["condition_2"]["holds"] is None
    # Tobias 2026-11-07: an undefined bound is a condition not met.
    assert report["verdict"]["barrblandskog"] == "keep_class"
    no_bland = mixed([1, 28, 0], [1, 28, 0], [1, 1, 0], [1, 28, 0], kinds=["pine", "barr", "none"])
    bland = no_bland["primary"]["conditions"]["blandskog"]["condition_1"]
    assert bland["holds"] is None and bland["undefined_draws"] > 0
    assert no_bland["verdict"] == {"barrblandskog": "keep_class", "blandskog": "not_established"}
    assert report["primary"]["descriptive_d_minus_a"]["barrblandskog"]["comparison"] == "F1(d) - F1(a)"
    assert "holds" not in report["primary"]["descriptive_d_minus_a"]["barrblandskog"]


def test_block_bootstrap_is_deterministic_and_resamples_whole_blocks():
    rng = np.random.default_rng(11)
    blocks = np.repeat(np.arange(9) * 7, 13)
    truth = rng.random(len(blocks)) < .3
    predicted = {"x": rng.random(len(blocks)) < .3, "y": truth ^ (rng.random(len(blocks)) < .1)}
    sample = np.random.default_rng(20260818).integers(0, 9, size=(200, 9))
    draws = score.block_f1_draws(truth, predicted, blocks, sample)
    again = score.block_f1_draws(truth, predicted, blocks, sample)
    assert all(np.array_equal(draws[k], again[k]) for k in draws)
    for draw in (0, 57):
        picked = np.concatenate([np.flatnonzero(blocks == np.unique(blocks)[b]) for b in sample[draw]])
        t, p = truth[picked], predicted["y"][picked]
        assert draws["y"][draw] == pytest.approx(2 * (t & p).sum() / (2 * (t & p).sum() + (~t & p).sum()
                                                                      + (t & ~p).sum()))
    held = plots(KINDS, blocks=5)
    control = cell(held, [1, 2, 1, 4, 3, 0, 2], [1, 2, 1, 4, 3, 0, 1])
    # The share rule finds barrblandskog in two of five blocks: unequal blocks.
    hit = (held["PlotID"] == 3) & held["TractID"].isin([1, 2])
    control.loc[hit, na.FRACTION_COLUMNS] = SHARES[28]
    cells = {"im023_main": cell(held, HARD_PERFECT, SHARE_PERFECT), "im023_control": control,
             "r4": cell(held, HARD_PERFECT, SHARE_PERFECT)}
    first = score.mixed_report(held, cells, np.ones(len(held), bool), na.AMENDMENT_PROTOCOL)
    assert first == score.mixed_report(held, cells, np.ones(len(held), bool), na.AMENDMENT_PROTOCOL)
    barr = first["primary"]["conditions"]["barrblandskog"]
    lower, upper = barr["condition_1"]["interval"]
    assert lower < upper and barr["condition_1"]["bound_value"] == lower
    tall = barr["condition_2"]["tallskog"]
    assert tall["bound_value"] == tall["interval"][1]
    assert barr["condition_1"]["interval_quantiles"] == [0.05, 0.95]


def test_mixed_report_aligns_truth_labels_and_blocks_on_partial_coverage():
    held = plots(KINDS, blocks=4)
    cells = {n: cell(held, HARD_PERFECT, SHARE_PERFECT) for n in na.AMENDMENT_CELLS}
    outside = held["TractID"].to_numpy() == 2
    for frame in cells.values():                          # every error is outside coverage
        frame.loc[outside, "model_pred"] = 0
        frame.loc[outside, na.FRACTION_COLUMNS] = SHARES[0]
    report = score.mixed_report(held, cells, ~outside, na.AMENDMENT_PROTOCOL)
    assert report["support"]["covered_frozen_observations"] == 21
    assert report["primary"]["observations"] == {"volume": 18, "pure": 15}
    metrics = report["primary"]["metrics"]
    assert all(m["f1"] == 1.0 for cls in metrics.values() for m in cls.values())


def test_mixed_truth_must_reproduce_frozen_v1():
    held = plots(KINDS, blocks=2)
    held.loc[0, "nfi_forest"] = 2
    cells = {n: cell(held, HARD_PERFECT, SHARE_PERFECT) for n in na.AMENDMENT_CELLS}
    with pytest.raises(ValueError, match="truth v1"):
        score.mixed_report(held, cells, np.ones(len(held), bool), na.AMENDMENT_PROTOCOL)


# ── inference and scoring CLIs on the synthetic freeze ──────────────────────

def fake_model(monkeypatch, cell_predictions: dict[int, dict[str, tuple[int, int]]]):
    """Tile (by tessera fill) -> cell -> (hard class at every pixel, share-rule label)."""
    state = {}

    def load_model(checkpoint, device, **kwargs):
        state["cell"] = {"main.pt": "im023_main", "control.pt": "im023_control",
                         "r4.pt": "r4"}[Path(checkpoint).name]
        state["classes"] = na.AMENDMENT_CELLS[state["cell"]]
        model = types.SimpleNamespace(fm_spec=types.SimpleNamespace(family="tessera"),
                                      ck_cfg={}, num_frames=1, frac_head=object())
        return model, 1, .5, 8

    def forward(model, inputs, device, return_fractions=False):
        assert return_fractions
        tile = round(float(inputs["img5d"].mean()) * 10)
        hard, share = cell_predictions[tile][state["cell"]]
        logits = torch.zeros((1, state["classes"], 8, 8))
        logits[0, hard] = 5.0
        shares = torch.tensor(SHARES[share]).clamp(1e-4, 1 - 1e-4)
        frac_logits = torch.log(shares / (1 - shares)).view(1, 4, 1, 1).expand(1, 4, 8, 8)
        return logits, frac_logits

    monkeypatch.setattr(infer.infcmp, "load_model", load_model)
    monkeypatch.setattr(infer.infcmp, "_forward_from_inputs", forward)


def test_inference_dumps_feed_both_scorers(freeze, monkeypatch, capsys):
    path, digest = write_amendment(freeze)
    # Tile t0 (fill 0.1) holds pine, barr, lov; t1 (0.2) spruce, bland, none.
    fake_model(monkeypatch, {
        1: {"im023_main": (28, 1), "im023_control": (1, 1), "r4": (1, 28)},
        2: {"im023_main": (2, 2), "im023_control": (2, 2), "r4": (2, 2)}})
    dumps = freeze.tmp / "dumps"
    for name, checkpoint in freeze.checkpoints.items():
        out = dumps / na.dump_name(name)
        monkeypatch.setattr(sys, "argv", ["infer", "--amendment-manifest", str(path),
                                          "--expected-amendment-sha256", digest, "--cell", name,
                                          "--checkpoint", str(checkpoint), "--out", str(out),
                                          "--device", "cpu"])
        infer.main()
        dump = pd.read_parquet(out)
        assert len(dump) == len(freeze.held)
        assert set(dump.columns) >= {"model_pred", *na.FRACTION_COLUMNS}
        with pytest.raises(ValueError, match="never overwrite"):
            infer.main()
    main = pd.read_parquet(dumps / na.dump_name("im023_main"))
    assert set(main.loc[main["tile_name"] == "t0", "model_pred"]) == {28}
    assert main.loc[main["tile_name"] == "t0", "frac_tall"].iloc[0] == pytest.approx(.9, abs=1e-4)
    meta = json.loads((dumps / (na.dump_name("r4") + ".meta.json")).read_text())
    assert meta["holdout_manifest_sha256"] == digest and meta["status"] == "success"
    assert "accuracy" not in capsys.readouterr().out
    monkeypatch.setattr(score, "sample_nmd_unified",
                        lambda tif, e, n: (np.zeros(len(e), int), np.full(len(e), 111)))
    for analysis in ("primary", "mixed"):
        out = freeze.tmp / f"{analysis}.json"
        monkeypatch.setattr(sys, "argv", ["score", analysis, "--amendment-manifest", str(path),
                                          "--expected-amendment-sha256", digest,
                                          "--dump-dir", str(dumps),
                                          "--nmd2023", str(freeze.tmp / "nmd2023.tif"),
                                          "--out", str(out)])
        score.main()
        report = json.loads(out.read_text())
        assert report["amendment_manifest_sha256"] == digest
        assert set(report["prediction_sha256"]) == set(na.AMENDMENT_CELLS)
        with pytest.raises(ValueError, match="already exists"):
            score.main()
    primary = json.loads((freeze.tmp / "primary.json").read_text())
    assert primary["fold_counts"]["im023_main"]["predicted_28"] == 3
    mixed_report = json.loads((freeze.tmp / "mixed.json").read_text())
    assert set(mixed_report["verdict"]) == {"barrblandskog", "blandskog"}


def test_dump_outside_the_cell_head_is_rejected(freeze, monkeypatch):
    path, digest = write_amendment(freeze)
    fake_model(monkeypatch, {1: {"r4": (1, 1)}, 2: {"r4": (2, 2)}})
    out = freeze.tmp / na.dump_name("r4")
    monkeypatch.setattr(sys, "argv", ["infer", "--amendment-manifest", str(path),
                                      "--expected-amendment-sha256", digest, "--cell", "r4",
                                      "--checkpoint", str(freeze.checkpoints["r4"]),
                                      "--out", str(out), "--device", "cpu"])
    infer.main()
    _, amendment, _ = na.load_amended_holdout(path, digest)
    assert len(na.verify_amendment_dump(out, "r4", amendment)) == len(freeze.held)
    with pytest.raises(ValueError, match="provenance"):
        na.verify_amendment_dump(out, "im023_main", amendment)
    meta_path = out.with_suffix(".parquet.meta.json")
    original = meta_path.read_text()
    for edit in (dict(amendment_protocol_id="other"), dict(base_manifest_sha256="0" * 64),
                 dict(observations=1)):
        meta_path.write_text(json.dumps(dict(json.loads(original), **edit)))
        with pytest.raises(ValueError, match="dump metadata does not match"):
            na.verify_amendment_dump(out, "r4", amendment)
    meta_path.write_text(original)
    good = pd.read_parquet(out)
    for edit, message in ((dict(model_pred=28), "outside the cell's head"),
                          (dict(frac_gran=1.5), r"outside \[0, 1\]")):
        good.assign(**edit).to_parquet(out, index=False)
        meta = json.loads(meta_path.read_text())
        meta_path.write_text(json.dumps(dict(meta, prediction_sha256=sha256_file(out))))
        with pytest.raises(ValueError, match=message):
            na.verify_amendment_dump(out, "r4", amendment)


def test_class_and_fraction_inference_feeds_both_heads_the_checkpoint_aux(monkeypatch, tmp_path):
    infcmp = infer.infcmp
    seen = {}

    def build(tile_path, device, img_size, aux_channel_names, **kwargs):
        seen["aux"] = aux_channel_names
        return {"family": "tessera", "data": {}}

    def forward(model, inputs, device, return_fractions=False):
        seen["return_fractions"] = return_fractions
        return torch.zeros((1, 29, 2, 2)), torch.zeros((1, 4, 2, 2))

    monkeypatch.setattr(infcmp, "_build_inference_inputs", build)
    monkeypatch.setattr(infcmp, "_forward_from_inputs", forward)
    monkeypatch.setattr(infcmp, "_close_inference_inputs", lambda inputs: None)
    model = types.SimpleNamespace(fm_spec=types.SimpleNamespace(family="tessera"), frac_head=object(),
                                  ck_cfg={"enabled_aux_names": ["dem", "markfukt"]}, n_aux_channels=2)
    probs, fracs = infcmp.run_class_and_fraction_inference(model, "tile", "cpu", img_size=2)
    assert seen == {"aux": ["dem", "markfukt"], "return_fractions": True}
    assert probs.shape == (29, 2, 2) and np.allclose(probs.sum(0), 1) and np.allclose(fracs, .5)
    with pytest.raises(ValueError, match="no fraction head"):
        infcmp.run_class_and_fraction_inference(types.SimpleNamespace(), "tile", "cpu")


def infer_argv(path, digest, cell, checkpoint, out):
    return ["infer", "--amendment-manifest", str(path), "--expected-amendment-sha256", digest,
            "--cell", cell, "--checkpoint", str(checkpoint), "--out", str(out), "--device", "cpu"]


def test_inference_refuses_changed_inputs_and_wrong_outputs_and_writes_nothing(freeze, monkeypatch):
    path, digest = write_amendment(freeze)
    fake_model(monkeypatch, {1: {"r4": (1, 1)}, 2: {"r4": (2, 2)}})
    out = freeze.tmp / "dumps" / na.dump_name("r4")
    other = freeze.tmp / "other.pt"
    other.write_bytes(b"not the frozen r4")
    monkeypatch.setattr(sys, "argv", infer_argv(path, digest, "r4", other, out))
    with pytest.raises(ValueError, match="checkpoint differs from the amendment cell"):
        infer.main()
    # A model whose head disagrees with the cell's class count.
    forward = infer.infcmp._forward_from_inputs
    monkeypatch.setattr(infer.infcmp, "_forward_from_inputs",
                        lambda *a, **k: (torch.zeros((1, 29, 8, 8)), forward(*a, **k)[1]))
    monkeypatch.setattr(sys, "argv", infer_argv(path, digest, "r4", freeze.checkpoints["r4"], out))
    with pytest.raises(ValueError, match="model outputs do not match the amendment cell"):
        infer.main()
    write_tile(freeze.tmp / "t1.npz", .9)                   # a frozen tile changed after prepare
    with pytest.raises(ValueError, match="frozen input content changed"):
        infer.main()
    assert not out.exists() and not out.with_suffix(".parquet.meta.json").exists()


def test_crop_refuses_a_frozen_plot_outside_the_model_crop():
    held = plots(["pine", "spruce"])
    tiles = {"t0": {"geometry": {"height": 8, "width": 8}}}
    assert len(infer.crop_plots(held, tiles, img_size=6)) == 2   # rows 1-2, cols 1-2 in [1, 6]
    with pytest.raises(ValueError, match="outside the model crop"):
        infer.crop_plots(held, tiles, img_size=4)                 # crop rows/cols 2-5


def test_cells_must_pin_distinct_checkpoints(freeze):
    manifest = build(freeze)
    manifest["cells"]["im023_main"]["checkpoint"] = deepcopy(
        manifest["cells"]["im023_control"]["checkpoint"])
    with pytest.raises(ValueError, match="distinct checkpoints"):
        na.validate_amendment_structure(manifest)
