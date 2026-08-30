"""The distill stage must BUILD its backbone at the resolution it will FEED.

``inference_comparison.load_model`` takes an optional ``img_size``. Clay and
CROMA carry no ``pos_embed`` and omit ``img_size`` from their minimal registry
config, so without the argument the backbone is built at 224 — wrong
``grid_size``, wrong PSP pool count — and is then handed 504 px tiles by
``run_inference``. Prithvi recovers its size from ``pos_embed`` and is
unaffected, which is exactly why this survived the Prithvi-only era: both
distill-stage scripts predate the six-backbone ladder.

The consequence is silent, not a crash. ``extract_plot_features.py`` captures
256-dim features off a wrongly-shaped head, and ``distill_forest_labels.py``
writes dense sidecars from one — so 2 of the ladder's 6 columns would distil
garbage while looking healthy. See docs/experiments/ladder_distill_stage.md.

Asserted at the AST level because exercising the real path needs GPU weights.
What regresses in practice is the keyword going missing under an edit, and
that is precisely what this catches.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Every script that loads a checkpoint and then runs inference at a chosen
# --img-size. infer_tiles.py is the reference call shape; the two distill
# scripts are the ones that were missing it.
SCRIPTS_REQUIRING_IMG_SIZE = [
    "extract_plot_features.py",
    "distill_forest_labels.py",
    "infer_tiles.py",
]


def _load_model_calls(path: Path) -> list[ast.Call]:
    """Every ``…load_model(...)`` call in the file, however it is qualified."""
    tree = ast.parse(path.read_text())
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "load_model")
            or (isinstance(node.func, ast.Name) and node.func.id == "load_model")
        )
    ]


@pytest.mark.parametrize("script", SCRIPTS_REQUIRING_IMG_SIZE)
def test_load_model_receives_img_size(script: str) -> None:
    path = ROOT / "scripts" / script
    calls = _load_model_calls(path)
    assert calls, f"{script}: no load_model call found — did the script move?"

    for call in calls:
        kwargs = {kw.arg for kw in call.keywords if kw.arg}
        assert "img_size" in kwargs, (
            f"{script}:{call.lineno} calls load_model without img_size. "
            f"Clay and CROMA will be built at a 224 grid and then fed "
            f"full-size tiles, silently. Pass img_size=args.img_size."
        )


@pytest.mark.parametrize("script", ["extract_plot_features.py",
                                    "distill_forest_labels.py"])
def test_img_size_is_a_cli_flag(script: str) -> None:
    """The value forwarded above must be operator-controlled, not hardcoded.

    Each column of the ladder carries its own crop (496 for Prithvi-300M and
    TerraMind, 504 for the rest), so a literal here would silently apply one
    column's regime to another.
    """
    src = (ROOT / "scripts" / script).read_text()
    assert '"--img-size"' in src or "'--img-size'" in src, (
        f"{script}: --img-size is not an argparse flag; the per-column crop "
        f"cannot be honoured."
    )
