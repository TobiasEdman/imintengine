#!/usr/bin/env python3
"""Plain-torchvision S1 detector vs rslearn's own model on one real patch (dossier d).

Builds the detector from torchvision classes only, loads the checkpoint with
``strict=True``, builds rslearn's ``MultiTaskModel`` the way the config does,
and runs both on the 512 px window patch with the most reference detections.

Needs AI2's environment (rslearn installed); see the dossier, section a.

    <ai2-venv>/bin/python scripts/marine/s1_port/port_feasibility.py --record <record>
"""
from __future__ import annotations

import argparse
import collections

import numpy as np

import _record as rec


def build_plain():
    import torch
    import torchvision
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
    from torchvision.models.detection.image_list import ImageList
    from torchvision.models.detection.roi_heads import RoIHeads
    from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
    from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign

    class PlainS1Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torchvision.models.swin_v2_b()
            self.backbone.features[0][0] = torch.nn.Conv2d(2, 128, kernel_size=4, stride=4)
            self.fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 256)
            anchors = AnchorGenerator(((32,), (64,), (128,), (256,)), ((0.5, 1.0, 2.0),) * 4)
            self.rpn = RegionProposalNetwork(anchors, RPNHead(256, 3), 0.7, 0.3, 256, 0.5,
                                             dict(training=2000, testing=2000),
                                             dict(training=2000, testing=2000), 0.7)
            names = [f"feat{i}" for i in range(4)]
            self.roi_heads = RoIHeads(MultiScaleRoIAlign(names, 7, 2), TwoMLPHead(256 * 49, 1024),
                                      FastRCNNPredictor(1024, 2), 0.5, 0.5, 512, 0.25, None,
                                      0.05, 0.5, 100)

        def _swin(self, x):
            outs = []
            for i, layer in enumerate(self.backbone.features):
                x = layer(x)
                if i in (1, 3, 5, 7):
                    outs.append(x.permute(0, 3, 1, 2))
            return outs

        def forward(self, target, historical):
            ft, fh = self._swin(target), self._swin(historical)
            feats = collections.OrderedDict(
                (f"feat{i}", torch.cat([a, b], dim=1)) for i, (a, b) in enumerate(zip(ft, fh)))
            feats = self.fpn(feats)
            images = ImageList(target, [tuple(target.shape[-2:])] * target.shape[0])
            proposals, _ = self.rpn(images, feats)
            dets, _ = self.roi_heads(feats, proposals, images.image_sizes)
            return dets

    return PlainS1Detector().eval()


def load_plain(sd):
    model = build_plain()
    remap = {"model.encoder.0.encoder.model.": "backbone.", "model.encoder.1.fpn.": "fpn.",
             "model.decoders.detect.0.rpn.": "rpn.",
             "model.decoders.detect.0.roi_heads.": "roi_heads."}
    new = {}
    for k, v in sd.items():
        for src, dst in remap.items():
            if k.startswith(src):
                new[dst + k[len(src):]] = v
                break
        else:
            raise KeyError(k)
    return model, model.load_state_dict(new, strict=True)


def load_rslearn(sd):
    from rslearn.models.faster_rcnn import FasterRCNN
    from rslearn.models.fpn import Fpn
    from rslearn.models.multitask import MultiTaskModel
    from rslearn.models.simple_time_series import SimpleTimeSeries
    from rslearn.models.swin import Swin

    enc = SimpleTimeSeries(Swin(pretrained=False, input_channels=2, output_layers=[1, 3, 5, 7]),
                           image_channels=2, groups=[[0], [1]])
    model = MultiTaskModel(
        encoder=[enc, Fpn([256, 512, 1024, 2048], 256)],
        decoders={"detect": [FasterRCNN([4, 8, 16, 32], 256, 2, [[32], [64], [128], [256]])]}
    ).eval()
    return model, model.load_state_dict({k[len("model."):]: v for k, v in sd.items()}, strict=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    rec.add_record_arg(ap)
    ap.add_argument("--ckpt", type=rec.Path,
                    default=rec.REPO / "imint" / "fm" / "ai2_vessels_s1" / "detect_best.ckpt")
    ap.add_argument("--out", type=rec.Path)
    a = ap.parse_args()

    import rasterio
    import torch
    from rasterio.windows import Window

    torch.set_grad_enabled(False)
    sd = torch.load(a.ckpt, map_location="cpu", mmap=True, weights_only=True)["state_dict"]
    plain, r_plain = load_plain(sd)
    rsl, r_rsl = load_rslearn(sd)

    c0, r0, c1, r1 = rec.ai2_window_bounds(a.record)
    pts = [(p["col"], p["row"]) for p in rec.reference_properties(a.record)]
    cc, rr = max(((c, r) for c in range(c0, c1 - 512, 461) for r in range(r0, r1 - 512, 461)),
                 key=lambda o: sum(o[0] + 32 <= x < o[0] + 480 and o[1] + 32 <= y < o[1] + 480
                                   for x, y in pts))

    def read_patch(scene):
        bands = []
        for pol in rec.POLS:
            with rasterio.open(rec.ai2_band_path(a.record, scene, pol)) as s:
                bands.append(s.read(1, window=Window(cc - c0, rr - r0, 512, 512)))
        return torch.clamp(torch.from_numpy(np.stack(bands).astype(np.float32)) / 250.0, 0, 4)

    tgt, hist = read_patch("target"), read_patch("historical1")
    d_plain = plain(tgt[None], hist[None])[0]
    from rslearn.train.model_context import ModelContext, RasterImage

    img = torch.stack([tgt, hist], dim=1)        # C=2, T=2 (target, historical), H, W
    out = list(rsl(ModelContext(inputs=[{"image": RasterImage(img)}], metadatas=[None])).outputs)[0]
    d_rsl = out["detect"] if "detect" in out else out
    keep_p, keep_r = d_plain["scores"] > 0.7, d_rsl["scores"] > 0.7
    same_n = len(d_plain["scores"]) == len(d_rsl["scores"])
    rec.write_json(a.out, {
        "plain_strict_load": str(r_plain), "rslearn_strict_load": str(r_rsl),
        "patch_window_offset": [cc - c0, rr - r0], "patch_abs_px": [cc, rr],
        "n_plain": int(len(d_plain["scores"])), "n_plain_gt_0.7": int(keep_p.sum()),
        "n_rslearn": int(len(d_rsl["scores"])), "n_rslearn_gt_0.7": int(keep_r.sum()),
        "max_abs_score_diff": float((d_plain["scores"] - d_rsl["scores"]).abs().max()) if same_n else None,
        "max_abs_box_diff_px": float((d_plain["boxes"] - d_rsl["boxes"]).abs().max()) if same_n else None,
        "reference_in_patch": sorted([x - cc, y - rr] for x, y in pts
                                     if cc <= x < cc + 512 and rr <= y < rr + 512),
        "plain_centres_gt_0.7": sorted(((d_plain["boxes"][keep_p, :2] + d_plain["boxes"][keep_p, 2:])
                                        / 2).int().tolist()),
    })


if __name__ == "__main__":
    main()
