#!/usr/bin/env python3
"""Extract authenticated input contracts on CPU; never construct a model.

JSON cells with checkpoint identities enter stdin; metadata-only JSON exits
stdout. Called with the sealed model interpreter by prepare_nfi_holdout.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inference_comparison import (
    _load_checkpoint_for_inference, checkpoint_aux_count, inference_aux_names,
)
from imint.training.unified_dataset import (
    AUX_COMPUTED_CHANNELS, AUX_NAN_NODATA_CHANNELS, ERA5_AUX_CHANNELS,
)


def checkpoint_inputs(cells: dict) -> dict:
    result = {}
    for cell, entry in cells.items():
        identity = entry["checkpoint"]
        checkpoint = _load_checkpoint_for_inference(
            identity["path"], map_location="meta",
            expected_sha256=identity["sha256"], expected_size=identity["bytes"])
        config = checkpoint.get("config", {})
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
        count, count_source = checkpoint_aux_count(config, state)
        names = inference_aux_names(config, count)
        if set(names) & ERA5_AUX_CHANNELS:
            raise ValueError(f"{cell}: ERA5 checkpoint is unsupported by this inference path")
        result[cell] = {"enabled_aux_names": names, "n_aux_channels": count,
                        "count_source": count_source,
                        "names_source": "checkpoint" if config.get("enabled_aux_names") else "canonical_default"}
    return {"cells": result, "computed_channels": sorted(AUX_COMPUTED_CHANNELS),
            "nan_nodata_channels": sorted(AUX_NAN_NODATA_CHANNELS)}


if __name__ == "__main__":
    json.dump(checkpoint_inputs(json.load(sys.stdin)), sys.stdout, allow_nan=False)
    sys.stdout.write("\n")
