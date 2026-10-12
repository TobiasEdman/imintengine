#!/usr/bin/env python3
"""Plain-torchvision S1 attribute model vs AI2's reference attributes (dossier f).

Loads the attribute checkpoint ``strict=True`` into torchvision's swin_v2_b
(2-channel stem) + six pooling heads, independent of the analyzer code, and
runs it on AI2's own 128 px attribute windows of all reference detections.

    python scripts/marine/s1_port/port_feasibility_attr.py --record <record> --out attr.json
"""
from __future__ import annotations

import argparse
import math

import numpy as np

import _record as rec

TYPES = ["cargo", "tanker", "passenger", "service", "tug", "pleasure", "fishing",
         "enforcement", "sar"]
HEADS = {"length": 1, "width": 1, "speed": 1, "heading_x": 1, "heading_y": 1, "ship_type": 9}


def build_model():
    import torch
    import torchvision

    class PoolingHead(torch.nn.Module):
        def __init__(self, out):
            super().__init__()
            self.conv_layers = torch.nn.Sequential(torch.nn.Sequential(
                torch.nn.Conv2d(1024, 128, 3, padding=1), torch.nn.ReLU()))
            self.fc_layers = torch.nn.Sequential(
                torch.nn.Sequential(torch.nn.Linear(128, 512), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.ReLU()))
            self.output_layer = torch.nn.Linear(512, out)

        def forward(self, f):
            return self.output_layer(self.fc_layers(torch.amax(self.conv_layers(f), dim=(2, 3))))

    class PlainAttr(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torchvision.models.swin_v2_b()
            self.backbone.features[0][0] = torch.nn.Conv2d(2, 128, kernel_size=4, stride=4)
            self.heads = torch.nn.ModuleDict({k: PoolingHead(v) for k, v in HEADS.items()})

        def forward(self, x):
            f = self.backbone.features(x).permute(0, 3, 1, 2)   # 1024 ch, stride 32
            return {k: h(f) for k, h in self.heads.items()}

    return PlainAttr().eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    rec.add_record_arg(ap)
    ap.add_argument("--ckpt", type=rec.Path,
                    default=rec.REPO / "imint" / "fm" / "ai2_vessels_s1" / "attribute_best.ckpt")
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    import rasterio
    import torch

    torch.set_grad_enabled(False)
    sd = torch.load(a.ckpt, map_location="cpu", mmap=True, weights_only=True)["state_dict"]
    new = {}
    for k, v in sd.items():
        if k.startswith("model.encoder.0.model."):
            new["backbone." + k[len("model.encoder.0.model."):]] = v
        else:
            new[f"heads.{k.split('.')[2]}." + k.split(".0.", 1)[1]] = v
    model = build_model()
    load = model.load_state_dict(new, strict=True)
    windows = rec.ai2_window_dir(a.record).parent.parent / "attribute_predict"
    errs = {k: [] for k in ("length", "width", "speed", "heading")}
    type_match = 0
    ref = rec.reference_properties(a.record)
    for p in ref:
        bands = []
        for pol in rec.POLS:
            path = windows / f"0_{p['col']}_{p['row']}" / "layers" / "sentinel1" / pol / "geotiff.tif"
            with rasterio.open(path) as s:
                bands.append(s.read(1).astype(np.float32))
        o = model(torch.from_numpy(np.stack(bands))[None] / 250.0)    # no clamp
        got = dict(length=o["length"].item() * 100, width=o["width"].item() * 100,
                   speed=o["speed"].item() * 100,
                   heading=90 - math.degrees(math.atan2(o["heading_y"].item(),
                                                        o["heading_x"].item())),
                   vessel_type=TYPES[int(torch.softmax(o["ship_type"], -1).argmax())])
        for k in errs:
            errs[k].append(abs(got[k] - p["attributes"][k]))
        type_match += got["vessel_type"] == p["attributes"]["vessel_type"]
    rec.write_json(a.out, {"strict_load": str(load), "n": len(ref), "type_matches": type_match,
                           **{f"max_abs_{k}_diff": max(v) for k, v in errs.items()}})


if __name__ == "__main__":
    main()
