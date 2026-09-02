#!/usr/bin/env python3
"""Build-time, offline smoke for the exact crop-distill model surface."""
from __future__ import annotations

import argparse
import ctypes
import gc
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _release_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _load_script(path: Path) -> None:
    module_name = f"_smoke_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load smoke target {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and some import helpers consult sys.modules while the module
    # body executes. Mirror normal import machinery, then leave no synthetic
    # smoke module behind for the next entrypoint check.
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules[module_name]
        else:
            sys.modules[module_name] = previous


def smoke_model() -> None:
    import numpy
    import terratorch
    import timm
    import torch
    import torchgeo
    import torchvision
    from claymodel.module import ClayMAEModule  # noqa: F401
    from use_croma import PretrainedCROMA  # noqa: F401

    from imint.fm.loaders.clay import load_clay
    from imint.fm.loaders.croma import load_croma
    from imint.fm.loaders.terramind import load_terramind

    expected_versions = {
        "numpy": "2.2.6",
        "terratorch": "1.2.11",
        "timm": "1.0.15",
        "torch": "2.5.1+cu121",
        "torchgeo": "0.8.1",
        "torchvision": "0.20.1+cu121",
    }
    actual_versions = {
        package: importlib.metadata.version(package)
        for package in expected_versions
    }
    assert actual_versions == expected_versions, actual_versions
    assert numpy.__version__ == "2.2.6"
    assert all(
        module is not None
        for module in (terratorch, timm, torchgeo, torchvision)
    )

    # This is the reconstruction path used immediately before the exact rung
    # checkpoint is restored. It must build the upstream graph without making
    # a hidden network call for upstream pretraining weights.
    croma = load_croma(pretrained=False, img_size=120)
    assert croma.modality == "both"
    assert croma.num_patches == 225
    for attribute in ("s1_encoder", "s2_encoder", "cross_encoder"):
        assert hasattr(croma, attribute), attribute
    del croma
    _release_memory()

    terramind = load_terramind(pretrained=False, img_size=16)
    assert isinstance(terramind, torch.nn.Module)
    del terramind
    _release_memory()

    clay = load_clay(pretrained=False, img_size=16)
    assert hasattr(clay, "encoder")
    assert getattr(clay.encoder, "mask_ratio", None) == 0.0
    del clay
    _release_memory()

    source_root = Path("/opt/imintengine")
    for relative in (
        "scripts/crop_distill_protocol.py",
        "scripts/crop_distill_provenance.py",
        "scripts/extract_plot_features.py",
        "scripts/run_crop_distill_job.py",
    ):
        path = source_root / relative
        compile(path.read_text(encoding="utf-8"), relative, "exec")

    print({"status": "ok", "environment": "model", **actual_versions})


def smoke_scoring() -> None:
    # Do not let a transitive import blur the environment boundary: this
    # runtime is intentionally CPU-only and contains no model framework.
    assert "torch" not in sys.modules
    import numpy
    import pandas
    import pyarrow
    import scipy
    import sklearn

    expected_versions = {
        "numpy": "1.26.4",
        "pandas": "2.2.2",
        "pyarrow": "17.0.0",
        "scikit-learn": "1.5.1",
        "scipy": "1.15.2",
    }
    actual_versions = {
        package: importlib.metadata.version(package)
        for package in expected_versions
    }
    assert actual_versions == expected_versions, actual_versions
    assert numpy.__version__ == "1.26.4"
    assert all(
        module is not None for module in (pandas, pyarrow, scipy, sklearn)
    )

    source_root = Path("/opt/imintengine")
    sys.path.insert(0, str(source_root / "scripts"))
    for relative in (
        "scripts/build_lucas_crop_split.py",
        "scripts/crop_distill_protocol.py",
        "scripts/crop_distill_provenance.py",
        "scripts/nfi_head_cv.py",
        "scripts/run_lucas_crop_split_job.py",
        "scripts/validate_against_nfi.py",
    ):
        _load_script(source_root / relative)
    assert "torch" not in sys.modules
    print({"status": "ok", "environment": "scoring", **actual_versions})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=("model", "scoring"))
    args = parser.parse_args()
    if args.environment == "model":
        smoke_model()
    else:
        smoke_scoring()


if __name__ == "__main__":
    main()
