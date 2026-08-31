#!/usr/bin/env python3
"""scripts/render_endgame_frames.py — render model prediction vs NMD2023 label.

Produces the deck's example-frame pair for one tile: the (distilled) model's
28-class prediction and the NMD2023-based label sidecar, both colorized with the
unified palette over the SAME centre-crop extent, so the two PNGs are directly
comparable side by side.

    python scripts/render_endgame_frames.py \
        --checkpoint /data/checkpoints/v8b_nmd2023_distill/best_model.pt \
        --tile /data/unified_v2_512/tile_331280_6541280.npz \
        --label-sidecar /data/nmd2023_labels/tile_331280_6541280.npz \
        --img-size 504 --out-dir /data/nfi_eval/frames
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.unified_schema import UNIFIED_COLOR_LIST


def colorize(label: np.ndarray) -> np.ndarray:
    """(H, W) unified class ids → (H, W, 3) uint8 RGB via the 28-class palette."""
    palette = np.array(UNIFIED_COLOR_LIST, dtype=np.uint8)
    return palette[np.clip(label, 0, len(palette) - 1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tile", required=True)
    ap.add_argument("--label-sidecar", default=None,
                    help="required unless --skip-label")
    ap.add_argument("--img-size", type=int, default=504)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--backbone-name", default=None,
                    help="override the checkpoint backbone (needed for "
                    "non-Prithvi families e.g. tessera_v1)")
    ap.add_argument("--model-frame-name", default="model_frame",
                    help="basename for the model-prediction PNG (no ext)")
    ap.add_argument("--skip-label", action="store_true",
                    help="only render the model prediction, not the NMD label")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image

    spec = importlib.util.spec_from_file_location(
        "_infcmp", str(Path(__file__).resolve().parent / "inference_comparison.py"))
    infcmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infcmp)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # img_size must reach load_model too: clay/croma carry no pos_embed and
    # otherwise get BUILT at a 224 grid while run_inference feeds --img-size
    # tiles (the mirror of the omission fixed in the distill scripts).
    model, epoch, miou, _ = infcmp.load_model(
        args.checkpoint, device, backbone_name=args.backbone_name,
        img_size=args.img_size)
    print(f"[load_model] epoch={epoch} mIoU={miou}")
    pred = infcmp.run_inference(model, args.tile, device, img_size=args.img_size)
    pred = np.asarray(pred)
    print(f"pred {pred.shape}, classes {sorted(np.unique(pred).tolist())}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colorize(pred)).save(
        out / f"{args.model_frame_name}.png", optimize=True)
    print(f"wrote {out}/{args.model_frame_name}.png")

    if not args.skip_label:
        label = np.load(args.label_sidecar, allow_pickle=True)["label"]
        # Same centre-crop extent as run_inference so both cover identical ground.
        h = label.shape[0]
        cs = min(args.img_size, h)
        off = (h - cs) // 2
        label = label[off:off + cs, off:off + cs]
        Image.fromarray(colorize(label)).save(out / "nmd2023_frame.png", optimize=True)
        print(f"wrote {out}/nmd2023_frame.png")


if __name__ == "__main__":
    main()
