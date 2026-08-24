#!/usr/bin/env python3
"""Clean-image import and deterministic-guard autograd smoke."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.cpu == args.cuda:
        parser.error("select exactly one of --cpu or --cuda")

    sys.path.insert(0, "/opt/imintengine")
    import einops  # noqa: F401
    import numpy  # noqa: F401
    import pyproj  # noqa: F401
    import requests  # noqa: F401
    import scipy  # noqa: F401
    import sklearn  # noqa: F401
    import timm  # noqa: F401
    import torch
    import torchvision  # noqa: F401
    from imint.fm.prithvi_mae.prithvi_mae import PrithviMAE  # noqa: F401
    from imint.fm.upernet import ViTUPerNetHead  # noqa: F401

    expected_sha = "602c86dee8d4dbdc191ab2750c261cf86b56ac50"
    assert Path("/opt/imintengine/.base_git_sha").read_text().strip() == expected_sha
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.cuda:
        assert torch.cuda.is_available(), "CUDA is unavailable"
    torch.use_deterministic_algorithms(True, warn_only=False)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    x = torch.randn(2, 64, 64, device=device, requires_grad=True)
    projection = torch.randn(64, 32, device=device)
    y = x @ projection
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print({
        "status": "ok",
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    })


if __name__ == "__main__":
    main()
