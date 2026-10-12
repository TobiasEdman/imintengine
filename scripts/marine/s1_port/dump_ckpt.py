#!/usr/bin/env python3
"""Checkpoint structure: top-level keys, tensor count/params, every key + shape (dossier e).

    python scripts/marine/s1_port/dump_ckpt.py imint/fm/ai2_vessels_s1/detect_best.ckpt detect_keys.txt
"""
from __future__ import annotations

import argparse
import collections

import _record as rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt", type=rec.Path)
    ap.add_argument("keys_out", type=rec.Path, help="tab-separated key, shape, dtype list")
    ap.add_argument("--out", type=rec.Path, help="summary JSON")
    a = ap.parse_args()

    import torch

    ck = torch.load(a.ckpt, map_location="cpu", mmap=True, weights_only=True)
    sd = ck["state_dict"]
    with open(a.keys_out, "w") as f:
        for k, t in sd.items():
            f.write(f"{k}\t{tuple(t.shape)}\t{t.dtype}\n")
    rec.write_json(a.out, {
        "top_level_keys": list(ck),
        "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
        "lightning_version": ck.get("pytorch-lightning_version"),
        "n_tensors": len(sd), "n_params": sum(t.numel() for t in sd.values()),
        "dtypes": dict(collections.Counter(str(t.dtype) for t in sd.values())),
    })


if __name__ == "__main__":
    main()
