"""Dump S2 + S1 vessel-model outputs on fixed random inputs, for before/after refactor parity.

Shared blocks: imint/analyzers/ai2_rslearn_blocks.py. Dump once per code tree
(the tree on PYTHONPATH is the one measured), then compare:

    PYTHONPATH=<code tree> python scripts/marine/vessel_model_parity.py before.npz \\
        --s2-dir <dir with the S2 detect_best.ckpt + attribute_best.ckpt>
    python scripts/marine/vessel_model_parity.py --compare before.npz after.npz

Checkpoints are loaded with strict=True after each module's own key remap. The
S2 attribute encoder (satlaspretrain) normally downloads pretrained weights in
its constructor; here the architecture is built with random weights instead,
which the strict checkpoint load then overwrites completely.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.manual_seed(0)


def s2_outputs(s2_dir: Path) -> dict:
    import satlaspretrain_models
    from satlaspretrain_models.utils import SatlasPretrain_weights

    from imint.analyzers import ai2_vessels as s2

    out = {}
    raw = torch.load(str(s2_dir / "detect_best.ckpt"), map_location="cpu", weights_only=False)
    remap = {"model.encoder.0.model.": "swin.model.", "model.encoder.1.fpn.": "fpn.fpn.",
             "model.decoders.detect.0.": "head."}
    state = {}
    for k, v in raw["state_dict"].items():
        for a, b in remap.items():
            if k.startswith(a):
                k = b + k[len(a):]
                break
        state[k] = v
    det = s2.AI2VesselDetector()
    det.load_state_dict(state, strict=True)
    det.eval()
    g = torch.Generator().manual_seed(1)
    x = torch.rand((2, 9, 512, 512), generator=g)
    feats = det.swin(x)
    for i, f in enumerate(feats):
        out[f"s2_swin{i}"] = f.numpy()
    fpn = det.fpn(feats)
    for i, f in enumerate(fpn):
        out[f"s2_fpn{i}"] = f.numpy()
    for thr in (0.5, 0.0):
        det.head.roi_heads.score_thresh = thr
        for j, d in enumerate(det.head(fpn, [x[i] for i in range(2)])):
            out[f"s2_det_thr{thr}_{j}_boxes"] = d["boxes"].numpy()
            out[f"s2_det_thr{thr}_{j}_scores"] = d["scores"].numpy()
    del det, raw, state

    def random_arch(self, model_identifier, fpn=False, head=None, num_categories=None,
                    device="cuda"):
        info = SatlasPretrain_weights[model_identifier]
        return satlaspretrain_models.Model(info["num_channels"], info["multi_image"],
                                           info["backbone"], fpn=fpn, head=head,
                                           num_categories=num_categories, weights=None)

    satlaspretrain_models.Weights.get_pretrained_model = random_arch
    raw = torch.load(str(s2_dir / "attribute_best.ckpt"), map_location="cpu", weights_only=False)
    remap = {"model.encoder.0.model.": "encoder."}
    remap.update({f"model.decoders.{h}.0.": f"decoders.{h}."
                  for h in s2.AI2AttributePredictor.HEAD_NAMES})
    state = {}
    for k, v in raw["state_dict"].items():
        for a, b in remap.items():
            if k.startswith(a):
                state[b + k[len(a):]] = v
                break
        else:
            raise KeyError(k)
    attr = s2.AI2AttributePredictor()
    attr.load_state_dict(state, strict=True)
    attr.eval()
    xa = torch.rand((3, 9, 128, 128), generator=g)
    for k, v in attr(xa).items():
        out[f"s2_attr_{k}"] = v.numpy()
    crop = getattr(s2, "_crop_centered", None) or getattr(s2, "crop_centered")
    img = np.random.default_rng(2).random((9, 200, 300)).astype(np.float32)
    for cy, cx in ((0, 0), (100, 150), (199, 299), (5, 290)):
        out[f"s2_crop_{cy}_{cx}"] = crop(img, cy, cx, 128)
    return out


def s1_outputs() -> dict:
    from imint.analyzers import ai2_vessels_s1 as s1

    out = {}
    det = s1.load_detector(s1.DEFAULT_DETECT_CKPT)
    g = torch.Generator().manual_seed(3)
    t = torch.rand((2, 2, 512, 512), generator=g) * 4
    h = torch.rand((2, 2, 512, 512), generator=g) * 4
    for thr in (0.05, 0.0):
        det["head"].roi_heads.score_thresh = thr
        for j, d in enumerate(s1.detector_forward(det, t, h)):
            out[f"s1_det_thr{thr}_{j}_boxes"] = d["boxes"].numpy()
            out[f"s1_det_thr{thr}_{j}_scores"] = d["scores"].numpy()
    del det
    attr = s1.load_attribute_model(s1.DEFAULT_ATTR_CKPT)
    xa = torch.rand((3, 2, 128, 128), generator=g) * 3
    for k, v in s1.attribute_forward(attr, xa).items():
        out[f"s1_attr_{k}"] = v.numpy()
    return out


def compare(a: Path, b: Path) -> int:
    za, zb = np.load(a), np.load(b)
    if sorted(za.files) != sorted(zb.files):
        print("key sets differ:", sorted(set(za.files) ^ set(zb.files)))
        return 1
    bad = [k for k in za.files if za[k].shape != zb[k].shape or za[k].tobytes() != zb[k].tobytes()]
    n_det = {k: int(za[k].shape[0]) for k in za.files if k.endswith("_scores")}
    print(f"{len(za.files)} arrays compared, bit-identical: {len(za.files) - len(bad)}, "
          f"differing: {bad}")
    print("detections per dumped head output:", n_det)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--s2-dir", type=Path,
                    help="directory with the S2 detect_best.ckpt and attribute_best.ckpt")
    ap.add_argument("--compare", nargs=2, type=Path)
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    if a.out is None or a.s2_dir is None:
        ap.error("a dump needs <out.npz> and --s2-dir")
    import imint
    print("code tree:", Path(imint.__file__).parent)
    out = {**s2_outputs(a.s2_dir), **s1_outputs()}
    np.savez(a.out, **out)
    print("saved", a.out, len(out), "arrays")
    return 0


if __name__ == "__main__":
    sys.exit(main())
