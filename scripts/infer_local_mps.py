"""
scripts/infer_local_mps.py — Lokal inference på M1 Mac (MPS eller CPU)

Kör pixel-klassificering på de 5 viz-tilesen och sparar resultaten som
JSON (base64+zlib RGB-arrays) för inbäddning i comparison.html.

Används av watch_and_infer_local.sh automatiskt när nytt checkpoint synkas.

Usage:
    python scripts/infer_local_mps.py \
        --checkpoint checkpoints/pixel_v1/best_model.pt \
        --tiles-dir data/viz_tiles \
        --out data/viz_tiles/col6_inference.json \
        --context-px 32 \
        --use-frame-2016 \
        --stride 1
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np


_CLASS_COLORS_RGB: list[tuple[int, int, int]] = [
    (  0,   0,   0), ( 26,  92,  53), ( 45, 138,  91), (123, 198, 126),
    ( 77, 179, 128), (107, 142,  90), (201, 223, 110), (155, 119,  34),
    (212, 180,  74), (192,  57,  43), ( 36, 113, 163), (232, 184,   0),
    (212, 120,  10), (240, 208,  96), (212, 198,   0), (145, 200,  76),
    (184, 222, 134), (155,  89, 182), (214,  51, 129), (224, 112,  32),
    (139,  32,  32), (220, 200,   0), (  0, 168, 198),
]


def _colorize(cls_map: np.ndarray) -> np.ndarray:
    H, W = cls_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for i, c in enumerate(_CLASS_COLORS_RGB):
        m = cls_map == i
        if m.any():
            rgb[m] = c
    return rgb


def _encode(arr: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(arr.tobytes())).decode()


def _infer(model, tile_path: Path, context_px: int, use_frame_2016: bool,
           use_aux: bool, device, stride: int) -> tuple[np.ndarray, list[int]]:
    """Return (grid_h, grid_w) int32 prediction and [gh, gw] shape."""
    import torch

    data = np.load(str(tile_path), allow_pickle=False)
    spectral = np.asarray(data.get("spectral", data.get("image")), dtype=np.float32)
    label = np.asarray(data["label"], dtype=np.int32)
    H, W = label.shape
    half = context_px // 2

    rows = np.arange(half, H - half, stride)
    cols = np.arange(half, W - half, stride)
    gh, gw = len(rows), len(cols)
    N = gh * gw
    n_frames = 5 if use_frame_2016 else 4

    patches = np.empty((N, n_frames * 6, context_px, context_px), dtype=np.float32)
    has_2016 = use_frame_2016 and int(data.get("has_frame_2016", 0)) == 1
    frame_2016 = np.asarray(data["frame_2016"], dtype=np.float32) if has_2016 else None

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            r0, r1, c0, c1 = r - half, r + half, c - half, c + half
            base = spectral[:, r0:r1, c0:c1]
            if use_frame_2016:
                bg = frame_2016[:, r0:r1, c0:c1] if has_2016 else \
                     np.zeros((6, context_px, context_px), dtype=np.float32)
                patches[i * gw + j] = np.concatenate([bg, base], axis=0)
            else:
                patches[i * gw + j] = base

    aux_all = None
    if use_aux:
        try:
            from imint.training.unified_dataset import AUX_CHANNEL_NAMES, AUX_LOG_TRANSFORM, AUX_NORM
            from imint.fm.pixel_head import N_AUX_DEFAULT
            aux_all = np.zeros((N, N_AUX_DEFAULT), dtype=np.float32)
            for k, ch in enumerate(AUX_CHANNEL_NAMES):
                if ch not in data:
                    continue
                arr = np.asarray(data[ch], dtype=np.float32)
                for ii, r in enumerate(rows):
                    for jj, c in enumerate(cols):
                        val = float(arr[min(r, arr.shape[0]-1), min(c, arr.shape[1]-1)])
                        if ch in AUX_LOG_TRANSFORM:
                            val = float(np.log1p(val))
                        mu, sigma = AUX_NORM.get(ch, (0.0, 1.0))
                        aux_all[ii * gw + jj, k] = (val - mu) / max(sigma, 1e-8)
        except Exception as e:
            print(f"  AUX unavailable ({e}) — running without aux")
            aux_all = None

    model.eval()
    preds: list[np.ndarray] = []
    # MPS doesn't support bfloat16 → use float16 or float32
    amp_ctx: contextlib.AbstractContextManager
    if device.type == "mps":
        amp_ctx = contextlib.nullcontext()
    elif device.type == "cuda":
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        amp_ctx = contextlib.nullcontext()

    batch_size = 512  # conservative for MPS memory
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            bp = torch.tensor(patches[start:end], dtype=torch.float32, device=device)
            ba = (torch.tensor(aux_all[start:end], dtype=torch.float32, device=device)
                  if aux_all is not None else None)
            with amp_ctx:
                logits = model(bp, ba)
            preds.append(logits.argmax(dim=1).cpu().numpy())

    pred_flat = np.concatenate(preds)
    return pred_flat.reshape(gh, gw).astype(np.int32), [gh, gw]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tiles-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--context-px", type=int, default=32)
    p.add_argument("--use-frame-2016", action="store_true")
    p.add_argument("--no-aux", action="store_true")
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    import torch

    # Device: MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Add repo to path
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Load model
    from imint.fm.pixel_head import PrithviPixelClassifier, N_AUX_DEFAULT
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    epoch = ckpt.get("epoch", "?")
    metric = ckpt.get("val_mIoU", ckpt.get("val_mean_acc", "?"))
    print(f"Checkpoint: epoch={epoch}, metric={metric}")

    use_aux = not args.no_aux
    model = PrithviPixelClassifier(
        num_classes=23,
        context_px=args.context_px,
        num_frames=5 if args.use_frame_2016 else 4,
        n_aux=N_AUX_DEFAULT if use_aux else 0,
        pretrained=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    print("Model loaded.")

    tiles = [
        "43983968", "45524456", "43983958",
        "tile_421280_7011280", "45563754",
    ]
    tiles_dir = Path(args.tiles_dir)
    out: dict = {"_epoch": epoch, "_metric": str(metric)}

    for tid in tiles:
        path = tiles_dir / f"{tid}.npz"
        if not path.exists():
            print(f"  SKIP {tid} — not found")
            continue
        print(f"  {tid} …", flush=True)
        try:
            pred, shape = _infer(
                model, path,
                context_px=args.context_px,
                use_frame_2016=args.use_frame_2016,
                use_aux=use_aux,
                device=device,
                stride=args.stride,
            )
            rgb = _colorize(pred)
            out[tid + "_pred"] = _encode(rgb)
            out[tid + "_shape"] = shape
            # dominant class
            counts = np.bincount(pred.ravel(), minlength=23)
            counts[0] = 0
            dom = int(counts.argmax())
            print(f"    done {pred.shape}, dominant={dom}")
        except Exception as exc:
            print(f"    ERROR: {exc}")
            out[tid + "_err"] = str(exc)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\nSparpad: {args.out}")


if __name__ == "__main__":
    main()
