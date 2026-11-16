"""scripts/hf_release.py: a release reproduces its checkpoint and fails closed."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from imint.schema.unified_schema import NUM_CLASSES_BEFORE_BARRBLANDSKOG, UNIFIED_CLASS_NAMES
from imint.training.unified_dataset import (
    AUX_LOG_TRANSFORM, AUX_NAN_NODATA_CHANNELS, AUX_NORM, AUX_YYDDD_DATE_CHANNELS,
    normalize_aux_channel,
)
from scripts import hf_release
from scripts.inference_comparison import CheckpointIdentityError, load_model

AUX = ["height", "volume", "basal_area", "diameter", "dem", "vpp_sosd",
       "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv", "markfukt"]
EMBEDDING = "geotessera-0.10.2"
METRICS = {"miou": np.float64(0.2468), "overall_accuracy": np.float64(0.8642),
           "per_class_iou": {"tallskog": np.float64(0.1357)}}
REPO = Path(__file__).resolve().parents[1]
CARD_DIR = REPO / "release/hf/imint-tessera-b28"
DOCUMENTS = {"README.md": (CARD_DIR / "README.md").read_bytes(),
             "NOTICE": (CARD_DIR / "NOTICE").read_bytes(),
             "LICENSE": (REPO / "LICENSE").read_bytes()}
SHA_A, SHA_B = "0" * 40, "1" * 40
CARD = "release/hf/test-model"


def _tessera_checkpoint(path: Path, num_classes: int, num_tradslag: int = 4) -> tuple[str, int]:
    """A best_model.pt shaped like im023/tessera_b28 (gated aux, fraction head)."""
    from imint.fm.registry import build_backbone
    from imint.fm.upernet import build_segmentation_from_spec

    torch.manual_seed(0)
    encoder, spec = build_backbone("tessera_v1", num_frames=1, img_size=504, pretrained=False)
    model = build_segmentation_from_spec(
        spec, encoder=encoder, num_classes=num_classes, img_size=504,
        decoder_channels=128, n_aux_channels=len(AUX), enable_tradslag_head=True,
        num_tradslag=num_tradslag, aux_fusion="gated", device="cpu")
    torch.save({
        "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
        "epoch": 7,
        "metric_version": "test-metric",
        "metrics": METRICS,
        "config": {"backbone_name": "tessera_v1", "num_classes": num_classes,
                   "n_aux_channels": len(AUX), "num_temporal_frames": 1,
                   "enable_multitemporal": False, "enable_tradslag_head": True,
                   "num_tradslag": num_tradslag, "img_size": 504, "aux_fusion": "gated",
                   "enabled_aux_names": AUX},
    }, path)
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _export(tmp_path: Path, num_classes: int = 29, num_tradslag: int = 4, **overrides):
    ckpt = tmp_path / "best_model.pt"
    sha, size = _tessera_checkpoint(ckpt, num_classes, num_tradslag)
    kwargs = dict(expected_sha256=sha, expected_size=size, model_id="test-model",
                  exporter_git_sha=SHA_A, training_git_sha=SHA_B, embedding=EMBEDDING,
                  documents=DOCUMENTS)
    kwargs.update(overrides)
    out = tmp_path / "release"
    manifest = hf_release.export_release(ckpt, out, **kwargs)
    return ckpt, sha, size, out, manifest


def _anchor(out: Path) -> str:
    return hashlib.sha256((out / hf_release.MANIFEST).read_bytes()).hexdigest()


def _load(out: Path):
    return hf_release.load_release(out, expected_manifest_sha256=_anchor(out))


def _resign(out: Path, name: str, data: bytes) -> None:
    """Replace a release file AND re-sign it in the manifest (a consistent forgery)."""
    (out / name).write_bytes(data)
    manifest = json.loads((out / hf_release.MANIFEST).read_text())
    section = "files" if name in manifest["files"] else "documents"
    manifest[section][name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    (out / hf_release.MANIFEST).write_text(json.dumps(manifest))


def _edit_manifest(out: Path, edit) -> None:
    manifest = json.loads((out / hf_release.MANIFEST).read_text())
    edit(manifest)
    (out / hf_release.MANIFEST).write_text(json.dumps(manifest))


@pytest.fixture
def exported(tmp_path):
    return _export(tmp_path)


# ── parity and contents ──────────────────────────────────────────────────────

def test_release_reproduces_checkpoint_outputs(exported):
    ckpt, sha, size, out, _ = exported
    ref, _, _, _ = load_model(str(ckpt), "cpu", expected_checkpoint_sha256=sha,
                              expected_checkpoint_size=size)
    rel, _ = _load(out)

    ref_state, rel_state = ref.state_dict(), rel.state_dict()
    assert ref_state.keys() == rel_state.keys()
    assert all(torch.equal(ref_state[k], rel_state[k]) for k in ref_state)

    g = torch.Generator().manual_seed(1)
    emb, aux = torch.randn(1, 128, 64, 64, generator=g), torch.randn(1, len(AUX), 64, 64, generator=g)
    with torch.no_grad():
        outputs = [m(emb, aux=aux, temporal_coords=None, location_coords=None,
                     return_fractions=True) for m in (ref, rel)]
    (ref_logits, ref_frac), (rel_logits, rel_frac) = outputs
    assert ref_logits.shape == (1, 29, 64, 64) and ref_frac.shape == (1, 4, 64, 64)
    assert torch.equal(ref_logits, rel_logits) and torch.equal(ref_frac, rel_frac)


def test_weights_are_the_checkpoint_state_verbatim(exported):
    from safetensors.torch import load_file

    ckpt, _, _, out, _ = exported
    original = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    written = load_file(out / hf_release.WEIGHTS)
    assert written.keys() == original.keys()
    assert all(torch.equal(written[k], original[k]) for k in original)


def test_config_records_the_input_output_contract(exported):
    _, _, _, out, _ = exported
    config = json.loads((out / hf_release.CONFIG).read_text())
    assert set(config) == hf_release.CONFIG_KEYS
    assert config["classes"] == UNIFIED_CLASS_NAMES and config["classes"][28] == "barrblandskog"
    assert config["fraction_species"] == ["tall", "gran", "trivial", "adel"]
    assert config["epoch"] == 7
    contract = config["input"]
    assert contract["family"] == "tessera" and contract["img_size"] == 504
    assert contract["embedding"] == EMBEDDING
    assert [a["name"] for a in contract["aux"]] == AUX
    assert all((a["mean"], a["std"]) == AUX_NORM[a["name"]] for a in contract["aux"])


def test_config_alone_reproduces_the_aux_normalization(exported):
    """A user who reads only config.json normalizes exactly as training did."""
    _, _, _, out, _ = exported
    config = json.loads((out / hf_release.CONFIG).read_text())

    def from_config(entry, x):
        x = np.asarray(x, dtype=np.float32)
        if entry["nan_to_mean"]:
            x = np.where(np.isnan(x), np.float32(entry["mean"]), x)
        if entry["date_yyddd"]:
            x = np.where(x > 0, np.mod(x, 1000.0), entry["mean"]).astype(np.float32)
        elif entry["log1p"]:
            x = np.log1p(np.maximum(x, 0.0))
        return (x - entry["mean"]) / max(entry["std"], 1e-6)

    probe = np.array([np.nan, 0.0, -3.0, 0.5, 7.0, 150.0, 23130.0, 24265.0], dtype=np.float32)
    for entry in config["input"]["aux"]:
        expected = normalize_aux_channel(entry["name"], probe.copy())
        np.testing.assert_array_equal(from_config(entry, probe.copy()), expected,
                                      err_msg=entry["name"])


def test_no_metric_leaves_with_the_weights(exported):
    from safetensors import safe_open

    _, _, _, out, _ = exported
    texts = [(out / name).read_text() for name in (hf_release.CONFIG, hf_release.MANIFEST)]
    with safe_open(str(out / hf_release.WEIGHTS), "pt") as f:
        texts.append(json.dumps(f.metadata()))
    forbidden = ["miou", "overall_accuracy", "per_class_iou", "metric_version", "test-metric",
                 "0.2468", "0.8642", "0.1357"]
    for text in texts:
        assert not [word for word in forbidden if word in text]


def test_manifest_binds_files_documents_and_source(exported):
    _, sha, size, out, manifest = exported
    assert manifest == json.loads((out / hf_release.MANIFEST).read_text())
    assert set(manifest) == hf_release.MANIFEST_KEYS
    assert manifest["source_checkpoint"] == {"file": "best_model.pt", "sha256": sha, "size": size}
    assert (manifest["exporter_git_sha"], manifest["training_git_sha"]) == (SHA_A, SHA_B)
    assert set(manifest["documents"]) == set(hf_release.DOCUMENTS)
    for section in ("files", "documents"):
        for name, identity in manifest[section].items():
            data = (out / name).read_bytes()
            assert identity == {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


# ── export refusals ──────────────────────────────────────────────────────────

def test_28_class_checkpoint_gets_the_names_before_barrblandskog(tmp_path):
    _, _, _, out, _ = _export(tmp_path, num_classes=NUM_CLASSES_BEFORE_BARRBLANDSKOG)
    config = json.loads((out / hf_release.CONFIG).read_text())
    assert config["classes"] == UNIFIED_CLASS_NAMES[:NUM_CLASSES_BEFORE_BARRBLANDSKOG]


@pytest.mark.parametrize("kwargs, message", [
    (dict(num_classes=23), "classes"),
    (dict(num_tradslag=3), "fraction head has 3 channels"),
    (dict(embedding=None), "embedding source"),
    (dict(exporter_git_sha="HEAD"), "full 40-hex"),
    (dict(documents={"README.md": b"x"}), "documents must be exactly"),
    (dict(documents={}), "documents must be exactly"),
])
def test_a_refused_export_writes_nothing(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _export(tmp_path, **kwargs)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best_model.pt"]


def test_export_refuses_an_existing_folder(tmp_path):
    (tmp_path / "release").mkdir()
    (tmp_path / "release" / "stale.bin").write_bytes(b"old")
    with pytest.raises(ValueError, match="already exists"):
        _export(tmp_path)


def test_wrong_checkpoint_identity_is_refused(tmp_path):
    with pytest.raises(CheckpointIdentityError):
        _export(tmp_path, expected_sha256="f" * 64)


# ── load refusals ────────────────────────────────────────────────────────────

def test_load_requires_the_anchor(exported):
    _, _, _, out, _ = exported
    with pytest.raises(hf_release.ReleaseIdentityError, match="expected_manifest_sha256"):
        hf_release.load_release(out, expected_manifest_sha256=None)
    with pytest.raises(hf_release.ReleaseIdentityError, match="expected manifest"):
        hf_release.load_release(out, expected_manifest_sha256="0" * 64)
    hf_release.load_release(out, expected_manifest_sha256=None, trust_unanchored=True)


def test_the_anchor_catches_a_forgery_no_config_check_can(exported):
    """Aux order reversed in both places: consistent, re-signed, and only the anchor sees it."""
    _, _, _, out, _ = exported
    anchor = _anchor(out)
    config = json.loads((out / hf_release.CONFIG).read_text())
    config["checkpoint_config"]["enabled_aux_names"].reverse()
    config["input"]["aux"].reverse()
    _resign(out, hf_release.CONFIG, json.dumps(config, indent=2, ensure_ascii=False).encode())
    hf_release.load_release(out, expected_manifest_sha256=None, trust_unanchored=True)
    with pytest.raises(hf_release.ReleaseIdentityError, match="expected manifest"):
        hf_release.load_release(out, expected_manifest_sha256=anchor)


@pytest.mark.parametrize("name", [hf_release.WEIGHTS, hf_release.CONFIG, *hf_release.DOCUMENTS])
def test_a_changed_file_is_refused(exported, name):
    _, _, _, out, _ = exported
    path = out / name
    data = bytearray(path.read_bytes())
    data[-2] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(hf_release.ReleaseIdentityError, match="does not match"):
        hf_release.load_release(out, expected_manifest_sha256=None, trust_unanchored=True)


@pytest.mark.parametrize("name", [*hf_release.GENERATED, *hf_release.DOCUMENTS])
def test_a_missing_file_is_refused(exported, name):
    _, _, _, out, _ = exported
    (out / name).unlink()
    with pytest.raises(hf_release.ReleaseIdentityError):
        hf_release.load_release(out, expected_manifest_sha256=None, trust_unanchored=True)


def test_unlisted_files_are_refused_but_hidden_ones_ignored(exported):
    _, _, _, out, _ = exported
    (out / ".gitattributes").write_text("*.safetensors filter=lfs\n")
    (out / ".cache").mkdir()
    _load(out)
    (out / "evil.py").write_text("print('hi')\n")
    with pytest.raises(hf_release.ReleaseIdentityError, match="does not list"):
        _load(out)


@pytest.mark.parametrize("edit", [
    lambda m: m.update(format="imint-hf-release/0"),
    lambda m: m["files"].update({"extra.bin": {"sha256": "0" * 64, "size": 0}}),
    lambda m: m.update(files=[]),
    lambda m: m["files"].update({hf_release.CONFIG: "not-an-entry"}),
    lambda m: m.update(documents="garbage"),
    lambda m: m.update(documents={}),                       # a release without its licence
    lambda m: m.update(documents={"README.md": m["documents"]["README.md"]}),
    lambda m: m.update(exporter_git_sha=None),
    lambda m: m.update(training_git_sha="HEAD"),
    lambda m: m.pop("produced_at"),
    lambda m: m.update(extra_key=1),
])
def test_a_malformed_manifest_is_refused(exported, edit):
    _, _, _, out, _ = exported
    _edit_manifest(out, edit)
    with pytest.raises(hf_release.ReleaseIdentityError):
        hf_release.load_release(out, expected_manifest_sha256=None, trust_unanchored=True)


@pytest.mark.parametrize("edit", [
    lambda c: c["input"]["aux"].reverse(),
    lambda c: c.update(classes=c["classes"][:-1]),
    lambda c: c["classes"].__setitem__(1, "granskog"),
    lambda c: c["input"]["aux"][0].update(mean=8.0),
    lambda c: c["input"]["aux"][1].update(log1p=False),
    lambda c: c["fraction_species"].reverse(),
    lambda c: c.update(fraction_species=[]),
    lambda c: c["input"].update(img_size=224),
    lambda c: c["input"].update(num_temporal_frames=4),
    lambda c: c["input"].update(aux_normalization="z = x"),
    lambda c: c["input"].update(extra=1),
    lambda c: c.update(ignore_index=255),
    lambda c: c.update(model_id="another-model"),
    lambda c: c.update(validation_metrics={"miou": 0.5}),
])
def test_a_config_that_does_not_describe_the_model_is_refused(exported, edit):
    """Re-signed, so only the comparison with the derived config can catch it."""
    _, _, _, out, _ = exported
    config = json.loads((out / hf_release.CONFIG).read_text())
    edit(config)
    _resign(out, hf_release.CONFIG, json.dumps(config).encode())
    with pytest.raises(hf_release.ReleaseIdentityError):
        hf_release.load_release(out, expected_manifest_sha256=None, trust_unanchored=True)


@pytest.mark.parametrize("forged", [
    [],                                                      # not a dict
    {"num_classes": "x"},
    {"backbone_name": "no_such_backbone"},
])
def test_a_forged_checkpoint_config_is_the_same_refusal(exported, forged):
    """Callers catch one error type, even when the forgery breaks the build itself."""
    _, _, _, out, _ = exported
    config = json.loads((out / hf_release.CONFIG).read_text())
    config["checkpoint_config"] = forged if isinstance(forged, list) else config["checkpoint_config"] | forged
    _resign(out, hf_release.CONFIG, json.dumps(config).encode())
    with pytest.raises(hf_release.ReleaseIdentityError):
        hf_release.load_release(out, expected_manifest_sha256=None, trust_unanchored=True)


def test_weights_metadata_must_name_this_release(exported):
    from safetensors.torch import load, save

    _, _, _, out, _ = exported
    state = load((out / hf_release.WEIGHTS).read_bytes())
    _resign(out, hf_release.WEIGHTS, save(state, metadata={"format": hf_release.FORMAT,
                                                           "model_id": "another-model"}))
    with pytest.raises(hf_release.ReleaseIdentityError, match="metadata"):
        hf_release.load_release(out, expected_manifest_sha256=None, trust_unanchored=True)


# ── the model card ───────────────────────────────────────────────────────────

def test_card_tables_match_the_schema_and_contract():
    card = (CARD_DIR / "README.md").read_text()
    classes = {int(i): name.split(" (")[0] for i, name in
               re.findall(r"\| (\d+) \| ([^|]+?) (?=\|)", card.split("## Classes")[1].split("##")[0])}
    assert [classes[i] for i in sorted(classes)] == UNIFIED_CLASS_NAMES
    rows = re.findall(r"\| (\d+) \| (\w+) \| [^|]+ \| ([^|]+) \| ([^|]+) \| ([\d.]+) \| ([\d.]+) \|", card)
    assert [name for _, name, *_ in rows] == AUX
    for _, name, missing, transform, mean, std in rows:
        assert (float(mean), float(std)) == AUX_NORM[name]
        assert (missing.strip() == "NaN → mean") == (name in AUX_NAN_NODATA_CHANNELS), name
        assert (missing.strip() == "≤ 0 → mean") == (name in AUX_YYDDD_DATE_CHANNELS), name
        assert transform.strip().startswith("log1p") == (name in AUX_LOG_TRANSFORM), name
        assert transform.strip().startswith("YYDDD") == (name in AUX_YYDDD_DATE_CHANNELS), name


# ── export_guard and documents_at_head ──────────────────────────────────────

def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.org")
    _git(root, "config", "user.name", "t")
    for rel in [*hf_release.CONTRACT_FILES, f"{CARD}/README.md", f"{CARD}/NOTICE", "LICENSE"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"{rel} v1\n")
    (root / "other.py").write_text("x\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "train")
    trained = _git(root, "rev-parse", "HEAD")
    (root / "other.py").write_text("y\n")
    _git(root, "commit", "-q", "-am", "unrelated change")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    return root, trained


def test_guard_accepts_a_clean_main_commit_with_the_contract_unchanged(repo):
    root, trained = repo
    assert hf_release.export_guard(root, trained) == _git(root, "rev-parse", "HEAD")


def test_guard_refuses_a_dirty_tree(repo):
    root, trained = repo
    (root / "other.py").write_text("dirty\n")
    with pytest.raises(SystemExit, match="dirty"):
        hf_release.export_guard(root, trained)


def test_guard_refuses_a_commit_not_on_origin_main(repo):
    root, trained = repo
    (root / "other.py").write_text("z\n")
    _git(root, "commit", "-q", "-am", "unpushed")
    with pytest.raises(SystemExit, match="origin/main"):
        hf_release.export_guard(root, trained)


def test_guard_refuses_contract_drift_since_training(repo):
    root, trained = repo
    (root / hf_release.CONTRACT_FILES[0]).write_text("v2\n")
    _git(root, "commit", "-q", "-am", "change normalization")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    with pytest.raises(SystemExit, match="input contract changed"):
        hf_release.export_guard(root, trained)


@pytest.mark.parametrize("sha, message", [("f" * 40, "not found"), ("HEAD", "full 40-hex"),
                                          ("main", "full 40-hex")])
def test_guard_refuses_an_unknown_or_symbolic_training_commit(repo, sha, message):
    root, _ = repo
    with pytest.raises(SystemExit, match=message):
        hf_release.export_guard(root, sha)


def test_guard_refuses_a_training_commit_head_does_not_descend_from(repo):
    root, trained = repo
    _git(root, "checkout", "-q", "-b", "side", trained)
    (root / "side.py").write_text("s\n")
    _git(root, "add", "side.py")
    _git(root, "commit", "-q", "-m", "side")
    side = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", "-")
    with pytest.raises(SystemExit, match="does not descend"):
        hf_release.export_guard(root, side)


def test_documents_are_this_models_card_at_head(repo):
    root, _ = repo
    (root / CARD / "NOTICE").write_text("edited, not committed\n")
    documents = hf_release.documents_at_head(root, "test-model")
    assert documents == {"README.md": f"{CARD}/README.md v1\n".encode(),
                         "NOTICE": f"{CARD}/NOTICE v1\n".encode(), "LICENSE": b"LICENSE v1\n"}
    with pytest.raises(SystemExit, match="not committed"):
        hf_release.documents_at_head(root, "another-model")      # no card of its own
    for bad in ("../test-model", "a/b", ""):
        with pytest.raises(SystemExit, match="plain model id"):
            hf_release.documents_at_head(root, bad)
