"""scripts/hf_release.py: a release reproduces its checkpoint and fails closed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from imint.schema.unified_schema import NUM_CLASSES_BEFORE_BARRBLANDSKOG, UNIFIED_CLASS_NAMES
from imint.training.unified_dataset import AUX_NORM
from scripts import hf_release
from scripts.inference_comparison import CheckpointIdentityError, load_model

AUX = ["height", "volume", "basal_area", "diameter", "dem", "vpp_sosd",
       "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv", "markfukt"]
EMBEDDING = "geotessera-0.10.2"


def _tessera_checkpoint(path: Path, num_classes: int) -> tuple[str, int]:
    """A best_model.pt shaped like im023/tessera_b28 (gated aux, fraction head)."""
    from imint.fm.registry import build_backbone
    from imint.fm.upernet import build_segmentation_from_spec

    torch.manual_seed(0)
    encoder, spec = build_backbone("tessera_v1", num_frames=1, img_size=504, pretrained=False)
    model = build_segmentation_from_spec(
        spec, encoder=encoder, num_classes=num_classes, img_size=504,
        decoder_channels=128, n_aux_channels=len(AUX), enable_tradslag_head=True,
        num_tradslag=4, aux_fusion="gated", device="cpu")
    torch.save({
        "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
        "epoch": 7,
        "metric_version": "test",
        "metrics": {"miou": np.float64(0.25)},
        "config": {"backbone_name": "tessera_v1", "num_classes": num_classes,
                   "n_aux_channels": len(AUX), "num_temporal_frames": 1,
                   "enable_multitemporal": False, "enable_tradslag_head": True,
                   "num_tradslag": 4, "img_size": 504, "aux_fusion": "gated",
                   "enabled_aux_names": AUX},
    }, path)
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _export(tmp_path: Path, num_classes: int = 29, **overrides):
    ckpt = tmp_path / "best_model.pt"
    sha, size = _tessera_checkpoint(ckpt, num_classes)
    kwargs = dict(expected_sha256=sha, expected_size=size, model_id="test-model",
                  exporter_git_sha="0" * 40, training_git_sha="1" * 40,
                  embedding=EMBEDDING)
    kwargs.update(overrides)
    out = tmp_path / "release"
    manifest = hf_release.export_release(ckpt, out, **kwargs)
    return ckpt, sha, size, out, manifest


@pytest.fixture
def exported(tmp_path):
    return _export(tmp_path)


def test_release_reproduces_checkpoint_outputs(exported):
    ckpt, sha, size, out, _ = exported
    ref, _, _, _ = load_model(str(ckpt), "cpu", expected_checkpoint_sha256=sha,
                              expected_checkpoint_size=size)
    rel, _ = hf_release.load_release(out)

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
    assert config["format"] == hf_release.FORMAT
    assert config["classes"] == UNIFIED_CLASS_NAMES
    assert config["classes"][28] == "barrblandskog"
    assert config["fraction_species"] == ["tall", "gran", "trivial", "adel"]
    assert config["epoch"] == 7
    # Results stay private until the paper: no metric leaves with the weights.
    assert "miou" not in json.dumps(config)
    contract = config["input"]
    assert contract["family"] == "tessera" and contract["img_size"] == 504
    assert contract["embedding"] == EMBEDDING
    assert [a["name"] for a in contract["aux"]] == AUX
    assert all((a["mean"], a["std"]) == AUX_NORM[a["name"]] for a in contract["aux"])
    by_name = {a["name"]: a for a in contract["aux"]}
    assert by_name["volume"]["log1p"] and not by_name["height"]["log1p"]
    assert by_name["vpp_sosd"]["yyddd_to_doy"]
    assert by_name["markfukt"]["nodata"] == "nan" and by_name["dem"]["nodata"] == "zero"


def test_28_class_checkpoint_gets_the_names_before_barrblandskog(tmp_path):
    _, _, _, out, _ = _export(tmp_path, num_classes=NUM_CLASSES_BEFORE_BARRBLANDSKOG)
    config = json.loads((out / hf_release.CONFIG).read_text())
    assert config["classes"] == UNIFIED_CLASS_NAMES[:NUM_CLASSES_BEFORE_BARRBLANDSKOG]
    assert "barrblandskog" not in config["classes"]


def test_manifest_binds_files_and_source(exported):
    _, sha, size, out, manifest = exported
    assert manifest == json.loads((out / hf_release.MANIFEST).read_text())
    assert manifest["source_checkpoint"] == {"file": "best_model.pt", "sha256": sha, "size": size}
    assert manifest["exporter_git_sha"] == "0" * 40 and manifest["training_git_sha"] == "1" * 40
    for name in (hf_release.WEIGHTS, hf_release.CONFIG):
        data = (out / name).read_bytes()
        assert manifest["files"][name] == {"sha256": hashlib.sha256(data).hexdigest(),
                                           "size": len(data)}


@pytest.mark.parametrize("name", [hf_release.WEIGHTS, hf_release.CONFIG])
def test_a_changed_file_is_refused(exported, name):
    _, _, _, out, _ = exported
    path = out / name
    data = bytearray(path.read_bytes())
    data[-2] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(hf_release.ReleaseIdentityError, match="does not match"):
        hf_release.load_release(out)


@pytest.mark.parametrize("name", hf_release.GENERATED)
def test_a_missing_file_is_refused(exported, name):
    _, _, _, out, _ = exported
    (out / name).unlink()
    with pytest.raises(hf_release.ReleaseIdentityError):
        hf_release.load_release(out)


def test_wrong_checkpoint_identity_is_refused(tmp_path):
    with pytest.raises(CheckpointIdentityError):
        _export(tmp_path, expected_sha256="f" * 64)


def test_tessera_release_requires_the_embedding_source(tmp_path):
    with pytest.raises(ValueError, match="embedding source"):
        _export(tmp_path, embedding=None)
