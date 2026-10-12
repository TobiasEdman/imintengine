"""Fail-closed checkpoint compatibility for inference model loading."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inference_comparison.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_checkpoint_state_compatibility", SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inference = _load_module()


def test_clean_checkpoint_is_accepted() -> None:
    inference._validate_checkpoint_state_keys(
        {"encoder.weight": object()},
        {"encoder.weight": object()},
    )


def test_missing_model_keys_fail_with_bounded_diagnostics() -> None:
    missing = tuple(f"decoder.layer.{index}.weight" for index in range(10))

    with pytest.raises(RuntimeError) as exc_info:
        inference._validate_checkpoint_state_keys(
            {key: object() for key in missing},
            {},
        )

    message = str(exc_info.value)
    assert "missing=10" in message
    assert "unexpected=0" in message
    assert missing[0] in message
    assert missing[7] in message
    assert missing[8] not in message
    assert "+2 more" in message
    assert "Refusing a partial state_dict load" in message


def test_arbitrary_unexpected_checkpoint_key_fails() -> None:
    with pytest.raises(RuntimeError, match=r"unexpected=1.*foreign\.weight"):
        inference._validate_checkpoint_state_keys(
            {},
            {"foreign.weight": object()},
        )


@pytest.mark.parametrize(
    "key",
    (
        "encoder.teacher.blocks.0.weight",
        "encoder.decoder.layers.0.bias",
        "encoder.proj.weight",
    ),
)
def test_clay_training_only_prefixes_are_accepted(key: str) -> None:
    state = {"encoder.weight": object(), key: object()}
    filtered = inference._checkpoint_state_for_inference(
        state, family="clay",
    )
    assert filtered == {"encoder.weight": state["encoder.weight"]}


@pytest.mark.parametrize(
    "family,key",
    (
        ("croma", "_norm_croma.s2_mean"),
        ("croma", "_norm_croma.s2_std"),
        ("croma", "_norm_croma.s1_mean"),
        ("croma", "_norm_croma.s1_std"),
        ("terramind", "_norm_terramind.s2_mean"),
        ("terramind", "_norm_terramind.s2_std"),
        ("terramind", "_norm_terramind.s1_mean"),
        ("terramind", "_norm_terramind.s1_std"),
    ),
)
def test_family_normalization_state_is_filtered(family: str, key: str) -> None:
    state = {"encoder.weight": object(), key: object()}
    filtered = inference._checkpoint_state_for_inference(state, family=family)
    assert filtered == {"encoder.weight": state["encoder.weight"]}


@pytest.mark.parametrize("family", ("croma", "prithvi", "terramind"))
def test_clay_training_prefixes_are_forbidden_for_other_families(
    family: str,
) -> None:
    state = {"encoder.teacher.weight": object()}
    filtered = inference._checkpoint_state_for_inference(state, family=family)
    assert filtered == state


@pytest.mark.parametrize(
    "family,key",
    (
        ("clay", "_norm_croma.s2_mean"),
        ("terramind", "_norm_croma.s2_mean"),
        ("croma", "_norm_terramind.s2_mean"),
        ("prithvi", "_norm_terramind.s2_mean"),
    ),
)
def test_normalization_keys_are_forbidden_for_other_families(
    family: str,
    key: str,
) -> None:
    state = {key: object()}
    filtered = inference._checkpoint_state_for_inference(state, family=family)
    assert filtered == state


@pytest.mark.parametrize(
    "key",
    (
        "encoder.teacherish.weight",
        "encoder.teacher",
        "xencoder.teacher.weight",
        "encoder.decoder_extra.weight",
        "encoder.projection.weight",
    ),
)
def test_clay_near_prefix_spoofs_are_rejected(key: str) -> None:
    state = {key: object()}
    filtered = inference._checkpoint_state_for_inference(state, family="clay")
    assert filtered == state


@pytest.mark.parametrize(
    "family,key",
    (
        ("croma", "_norm_croma.s2_mean.extra"),
        ("croma", "_norm_croma.s2_means"),
        ("croma", "x_norm_croma.s2_mean"),
        ("terramind", "_norm_terramind.s1_std.extra"),
    ),
)
def test_normalization_near_key_spoofs_are_rejected(
    family: str,
    key: str,
) -> None:
    state = {key: object()}
    filtered = inference._checkpoint_state_for_inference(state, family=family)
    assert filtered == state


def test_load_model_uses_fail_closed_validation_without_warning_fallback() -> None:
    source = inspect.getsource(inference.load_model)
    assert "_checkpoint_state_for_inference(" in source
    assert "_validate_checkpoint_state_keys(" in source
    assert "load_state_dict(inference_state, strict=True)" in source
    assert "strict=False" not in source
    assert "WARN state_dict mismatch" not in source
