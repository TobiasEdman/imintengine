#!/usr/bin/env python3
"""Class count, fraction head and input contract of each amendment cell, on CPU.

JSON cells with checkpoint identities enter stdin; metadata-only JSON exits
stdout. Called with the sealed model interpreter by prepare_nfi_amendment,
which runs under the scoring interpreter (no torch), as prepare_nfi_holdout
calls nfi_checkpoint_inputs. Never constructs a model.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imint.eval.nfi_amendment import NUM_FRACTIONS
from scripts.inference_comparison import _load_checkpoint_for_inference
from scripts.nfi_checkpoint_inputs import checkpoint_inputs


def checkpoint_heads(identity: dict) -> dict:
    """Class count and fraction head from the authenticated checkpoint."""
    checkpoint = _load_checkpoint_for_inference(
        identity["path"], map_location="meta",
        expected_sha256=identity["sha256"], expected_size=identity["bytes"])
    config = checkpoint.get("config", {})
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
    return {"num_classes": config.get("num_classes"),
            "fraction_head": (config.get("enable_tradslag_head") is True
                              and config.get("num_tradslag") == NUM_FRACTIONS
                              and any("frac_head." in key for key in state))}


def checkpoint_metadata(cells: dict) -> dict:
    contracts = checkpoint_inputs(cells)["cells"]
    return {name: dict(checkpoint_heads(cell["checkpoint"]), input_contract=contracts[name])
            for name, cell in cells.items()}


if __name__ == "__main__":
    json.dump(checkpoint_metadata(json.load(sys.stdin)), sys.stdout, allow_nan=False)
    sys.stdout.write("\n")
