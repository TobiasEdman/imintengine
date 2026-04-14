"""
scripts/viz_tile_predictions.py — Before/after schema comparison visualization

Loads a trained pixel classifier checkpoint, runs inference on 5 diverse val
tiles, and saves a 5-row × 3-col comparison PNG:
  Col 0: Model prediction (current checkpoint)
  Col 1: Ground-truth unified label  (what the model was trained on)
  Col 2: NMD-only label              (NMD baseline, no LPIS overlay)

Each cell is 128 × 128 px.  A class legend strip is appended at the bottom.

Usage (on the k8s pod):
    python scripts/viz_tile_predictions.py \
        --checkpoint /checkpoints/pixel_v1/best_model.pt \
        --data-dir /scratch \
        --split-dir /data/unified_v2 \
        --out /checkpoints/pixel_v1/tile_viz/before_schema_change.png \
        --n-tiles 5 \
        --device cuda \
        --use-frame-2016
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import numpy as np


# ── Color palette (must match unified_schema.py + train_pixel.py) ─────────────
_CLASS_COLORS_RGB: list[tuple[int, int, int]] = [
    (  0,   0,   0),   #  0 background
    ( 26,  92,  53),   #  1 tallskog
    ( 45, 138,  91),   #  2 granskog
    (123, 198, 126),   #  3 lövskog
    ( 77, 179, 128),   #  4 blandskog
    (107, 142,  90),   #  5 sumpskog
    (201, 223, 110),   #  6 tillfälligt ej skog
    (155, 119,  34),   #  7 våtmark
    (212, 180,  74),   #  8 öppen mark
    (192,  57,  43),   #  9 bebyggelse
    ( 36, 113, 163),   # 10 vatten
    (232, 184,   0),   # 11 vete
    (212, 120,  10),   # 12 korn
    (240, 208,  96),   # 13 havre
    (212, 198,   0),   # 14 oljeväxter
    (145, 200,  76),   # 15 slåttervall
    (184, 222, 134),   # 16 bete
    (155,  89, 182),   # 17 potatis
    (214,  51, 129),   # 18 sockerbetor
    (224, 112,  32),   # 19 trindsäd
    (139,  32,  32),   # 20 råg
    (220, 200,   0),   # 21 majs (was: övrig åker)
    (  0, 168, 198),   # 22 hygge
]

_CLASS_NAMES = [
    "background", "tallskog", "granskog", "lövskog", "blandskog", "sumpskog",
    "tillfälligt ej skog", "våtmark", "öppen mark", "bebyggelse", "vatten",
    "vete", "korn", "havre", "oljeväxter", "slåttervall", "bete",
    "potatis", "sockerbetor", "trindsäd", "råg", "majs", "hygge",
]

_NMD19_TO_UNIFIED = np.zeros(20, dtype=np.uint8)
_NMD19_TO_UNIFIED[0] = 0; _NMD19_TO_UNIFIED[1] = 1; _NMD19_TO_UNIFIED[2] = 2
_NMD19_TO_UNIFIED[3] = 3; _NMD19_TO_UNIFIED[4] = 4; _NMD19_TO_UNIFIED[5] = 6
_NMD19_TO_UNIFIED[6] = 5; _NMD19_TO_UNIFIED[7] = 5; _NMD19_TO_UNIFIED[8] = 5
_NMD19_TO_UNIFIED[9] = 5; _NMD19_TO_UNIFIED[10] = 5; _NMD19_TO_UNIFIED[11] = 7
_NMD19_TO_UNIFIED[12] = 0   # NMD cropland → background (new schema)
_NMD19_TO_UNIFIED[13] = 8; _NMD19_TO_UNIFIED[14] = 8
_NMD19_TO_UNIFIED[15] = 9; _NMD19_TO_UNIFIED[16] = 9; _NMD19_TO_UNIFIED[17] = 9
_NMD19_TO_UNIFIED[18] = 10; _NMD19_TO_UNIFIED[19] = 10


def _colorize(class_map: np.ndarray) -> np.ndarray:
    """(H, W) int → (H, W, 3) uint8 RGB."""
    H, W = class_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for i, color in enumerate(_CLASS_COLORS_RGB):
        mask = class_map == i
        if mask.any():
            rgb[mask] = color
    return rgb


def _resize_nearest(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbour upscale for (H, W, 3) uint8."""
    H, W, C = img.shape
    row_idx = (np.arange(target_h) * H // target_h).astype(np.int32)
    col_idx = (np.arange(target_w) * W // target_w).astype(np.int32)
    return img[row_idx[:, None], col_idx[None, :], :]


def _select_tiles(val_tiles: list[Path], n: int) -> list[Path]:
    """Pick n class-diverse tiles."""
    scored: list[tuple[int, Path]] = []
    for p in val_tiles:
        try:
            d = np.load(str(p), allow_pickle=False)
            lbl = np.asarray(d["label"], dtype=np.int32)
            counts = np.bincount(lbl.ravel(), minlength=23)
            counts[0] = 0
            dominant = int(counts.argmax()) if counts.max() > 0 else -1
            scored.append((dominant, p))
        except Exception:
            continue

    seen: set[int] = set()
    selected: list[Path] = []
    for cls, path in sorted(scored, key=lambda x: x[0]):
        if cls not in seen:
            selected.append(path)
            seen.add(cls)
        if len(selected) >= n:
            break

    if len(selected) < n:
        used = set(selected)
        for _, p in scored:
            if p not in used:
                selected.append(p)
            if len(selected) >= n:
                break

    return selected[:n]


def _infer_tile(
    model,
    tile_path: Path,
    context_px: int,
    use_frame_2016: bool,
    use_aux: bool,
    device,
    stride: int = 4,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a tile.

    Returns:
        pred_map: (gh, gw) int32  — predicted classes
        gt_map:   (gh, gw) int32  — GT labels
        nmd_map:  (gh, gw) int32  — NMD-only labels (no LPIS overlay)
    """
    import contextlib
    import torch

    try:
        from imint.training.unified_dataset import AUX_CHANNEL_NAMES, AUX_LOG_TRANSFORM, AUX_NORM
        _has_aux_meta = True
    except ImportError:
        _has_aux_meta = False

    data = np.load(str(tile_path), allow_pickle=False)
    spectral = np.asarray(data.get("spectral", data.get("image")), dtype=np.float32)
    label = np.asarray(data["label"], dtype=np.int32)
    H, W = label.shape
    half = context_px // 2

    rows = np.arange(half, H - half, stride)
    cols = np.arange(half, W - half, stride)
    gh, gw = len(rows), len(cols)
    N = gh * gw

    # Patches
    n_frames = (5 if use_frame_2016 else 4)
    patches = np.empty((N, n_frames * 6, context_px, context_px), dtype=np.float32)
    has_2016 = use_frame_2016 and int(data.get("has_frame_2016", 0)) == 1
    frame_2016 = np.asarray(data["frame_2016"], dtype=np.float32) if has_2016 else None

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            r0, r1, c0, c1 = r - half, r + half, c - half, c + half
            base = spectral[:, r0:r1, c0:c1]
            if use_frame_2016:
                bg = frame_2016[:, r0:r1, c0:c1] if has_2016 else np.zeros((6, context_px, context_px), dtype=np.float32)
                patches[i * gw + j] = np.concatenate([bg, base], axis=0)
            else:
                patches[i * gw + j] = base

    # Aux
    aux_all = None
    if use_aux and _has_aux_meta:
        from imint.fm.pixel_head import N_AUX_DEFAULT as N_AUX
        aux_all = np.zeros((N, N_AUX), dtype=np.float32)
        for k, ch_name in enumerate(AUX_CHANNEL_NAMES):
            if ch_name not in data:
                continue
            arr = np.asarray(data[ch_name], dtype=np.float32)
            for i, r in enumerate(rows):
                for j, c in enumerate(cols):
                    val = float(arr[min(r, arr.shape[0]-1), min(c, arr.shape[1]-1)])
                    if ch_name in AUX_LOG_TRANSFORM:
                        val = float(np.log1p(val))
                    mu, sigma = AUX_NORM.get(ch_name, (0.0, 1.0))
                    aux_all[i * gw + j, k] = (val - mu) / max(sigma, 1e-8)

    # Inference
    model.eval()
    preds: list[np.ndarray] = []
    amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if device.type == "cuda" else contextlib.nullcontext())
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
    gt_flat = label[rows[:, None], cols[None, :]].flatten()

    # NMD-only labels (apply _NMD19_TO_UNIFIED to raw nmd_label key)
    nmd_map = np.zeros((gh, gw), dtype=np.int32)
    if "nmd_label_raw" in data:
        nmd_raw = np.asarray(data["nmd_label_raw"], dtype=np.uint8)
        nmd_unified = _NMD19_TO_UNIFIED[np.clip(nmd_raw, 0, 19)]
        nmd_flat = nmd_unified[rows[:, None], cols[None, :]].flatten()
        nmd_map = nmd_flat.reshape(gh, gw).astype(np.int32)
    elif "nmd_label" in data:
        nmd_raw = np.asarray(data["nmd_label"], dtype=np.uint8)
        nmd_unified = _NMD19_TO_UNIFIED[np.clip(nmd_raw, 0, 19)]
        nmd_flat = nmd_unified[rows[:, None], cols[None, :]].flatten()
        nmd_map = nmd_flat.reshape(gh, gw).astype(np.int32)

    return (
        pred_flat.reshape(gh, gw).astype(np.int32),
        gt_flat.reshape(gh, gw).astype(np.int32),
        nmd_map,
    )


def _build_legend(cell_w: int, n_cols: int) -> np.ndarray:
    """Build a legend strip showing class colors and names.

    Returns (legend_h, total_w, 3) uint8 RGB.
    """
    from PIL import Image, ImageDraw, ImageFont

    total_w = cell_w * n_cols
    swatch = 18
    pad = 4
    font_size = 12
    cols_per_row = max(1, total_w // 140)
    n_classes = len(_CLASS_NAMES) - 1  # skip background
    n_rows = (n_classes + cols_per_row - 1) // cols_per_row
    legend_h = n_rows * (swatch + pad) + pad * 2

    img = Image.new("RGB", (total_w, legend_h), color=(248, 248, 248))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    col_w = total_w // cols_per_row
    for idx in range(1, len(_CLASS_NAMES)):
        i = idx - 1
        row_i = i // cols_per_row
        col_i = i % cols_per_row
        x = col_i * col_w + pad
        y = pad + row_i * (swatch + pad)
        color = _CLASS_COLORS_RGB[idx]
        draw.rectangle([x, y, x + swatch, y + swatch], fill=color)
        draw.text((x + swatch + 4, y + 2), f"{idx} {_CLASS_NAMES[idx]}", fill=(40, 40, 40), font=font)

    return np.array(img)


def _render_label_strip(cell_w: int) -> np.ndarray:
    """Render a narrow 'Prediction | GT Unified | NMD-only' header bar."""
    from PIL import Image, ImageDraw, ImageFont

    h = 28
    w = cell_w * 3
    img = Image.new("RGB", (w, h), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    labels = ["Prediction", "GT (unified label)", "NMD-only baseline"]
    for i, lbl in enumerate(labels):
        draw.text((i * cell_w + 8, 6), lbl, fill=(230, 230, 230), font=font)
    return np.array(img)


def main() -> None:
    p = argparse.ArgumentParser(description="Tile prediction comparison grid")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--data-dir", required=True, help="Directory with .npz tiles")
    p.add_argument("--split-dir", default=None,
                   help="Directory containing val.txt (defaults to data-dir)")
    p.add_argument("--out", default="/checkpoints/pixel_v1/tile_viz/comparison.png",
                   help="Output PNG path")
    p.add_argument("--n-tiles", type=int, default=5, help="Number of tile rows")
    p.add_argument("--context-px", type=int, default=32)
    p.add_argument("--use-frame-2016", action="store_true", default=False)
    p.add_argument("--no-aux", action="store_true", default=False,
                   help="Disable aux channels")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stride", type=int, default=4,
                   help="Pixel stride for grid sampling (lower=denser but slower)")
    p.add_argument("--cell-px", type=int, default=128,
                   help="Output cell size in pixels")
    args = p.parse_args()

    import torch

    # ── Resolve paths ────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    split_dir = Path(args.split_dir) if args.split_dir else data_dir
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Add repo to path if running from git clone
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # ── Load val split ────────────────────────────────────────────────────────
    val_txt = split_dir / "val.txt"
    if val_txt.exists():
        with open(val_txt) as f:
            raw_names = [l.strip() for l in f if l.strip()]
        val_tiles = []
        for n in raw_names:
            # Handle both "tile.npz" and bare "tile" (append .npz if needed)
            p = data_dir / n
            if not p.exists():
                p = data_dir / (n + ".npz")
            if p.exists():
                val_tiles.append(p)
        print(f"Val split: {len(val_tiles)} tiles from {val_txt}")
    else:
        val_tiles = sorted(data_dir.glob("*.npz"))
        print(f"No val.txt found — using all {len(val_tiles)} tiles")

    if not val_tiles:
        print("ERROR: no tiles found", file=sys.stderr)
        sys.exit(1)

    # ── Select diverse tiles ─────────────────────────────────────────────────
    selected = _select_tiles(val_tiles, args.n_tiles)
    print(f"Selected {len(selected)} tiles:")
    for t in selected:
        print(f"  {t.name}")

    # ── Load model ───────────────────────────────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # The checkpoint stores model_state_dict (or is the state dict directly)
    state_dict = ckpt.get("model_state_dict", ckpt)
    epoch_info = ckpt.get("epoch", "?")
    val_acc_info = ckpt.get("val_mean_acc", ckpt.get("val_mIoU", "?"))
    print(f"Checkpoint: epoch={epoch_info}, val_metric={val_acc_info}")

    # Build model — same config as train-pixel-job.yaml
    from imint.fm.pixel_head import PrithviPixelClassifier, N_AUX_DEFAULT

    use_aux = not args.no_aux
    n_aux = N_AUX_DEFAULT if use_aux else 0
    num_frames = 5 if args.use_frame_2016 else 4
    model = PrithviPixelClassifier(
        num_classes=23,
        context_px=args.context_px,
        num_frames=num_frames,
        n_aux=n_aux,
        pretrained=False,   # no pretrained weights needed — loading checkpoint
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # ── Render grid ───────────────────────────────────────────────────────────
    from PIL import Image

    cell_px = args.cell_px
    n_rows = len(selected)
    n_cols = 3

    col_labels_strip = _render_label_strip(cell_px)
    grid_rows: list[np.ndarray] = [col_labels_strip]

    for tile_path in selected:
        print(f"  Inferring {tile_path.name} …", flush=True)
        try:
            pred, gt, nmd = _infer_tile(
                model, tile_path,
                context_px=args.context_px,
                use_frame_2016=args.use_frame_2016,
                use_aux=use_aux,
                device=device,
                stride=args.stride,
            )
        except Exception as exc:
            print(f"    WARN: {exc}", file=sys.stderr)
            # Render blank row on failure
            blank = np.zeros((cell_px, cell_px * 3, 3), dtype=np.uint8)
            grid_rows.append(blank)
            continue

        row_cells = []
        for map_data in (pred, gt, nmd):
            cell_rgb = _colorize(map_data)
            cell_resized = _resize_nearest(cell_rgb, cell_px, cell_px)
            row_cells.append(cell_resized)

        # Thin white separator between columns
        sep = np.full((cell_px, 2, 3), 200, dtype=np.uint8)
        row = np.concatenate([
            row_cells[0], sep, row_cells[1], sep, row_cells[2]
        ], axis=1)

        # Add tile name bar
        tile_bar_h = 20
        bar = np.full((tile_bar_h, row.shape[1], 3), 245, dtype=np.uint8)
        try:
            from PIL import ImageDraw, ImageFont
            bar_img = Image.fromarray(bar)
            draw = ImageDraw.Draw(bar_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            except Exception:
                font = ImageFont.load_default()
            dominant_cls = int(np.bincount(gt.ravel(), minlength=23)[1:].argmax() + 1)
            label_text = f"{tile_path.stem}  (dominant GT: {_CLASS_NAMES[dominant_cls]})"
            draw.text((4, 3), label_text, fill=(60, 60, 60), font=font)
            bar = np.array(bar_img)
        except Exception:
            pass

        row_with_bar = np.concatenate([bar, row], axis=0)
        grid_rows.append(row_with_bar)
        print(f"    done (pred_map {pred.shape})")

    legend = _build_legend(cell_px, n_cols)

    # Pad all rows to same width before stacking
    max_w = max(r.shape[1] for r in grid_rows + [legend])
    def pad_w(arr: np.ndarray) -> np.ndarray:
        if arr.shape[1] < max_w:
            pad = np.full((arr.shape[0], max_w - arr.shape[1], 3), 255, dtype=np.uint8)
            return np.concatenate([arr, pad], axis=1)
        return arr

    grid = np.concatenate([pad_w(r) for r in grid_rows] + [pad_w(legend)], axis=0)

    out_img = Image.fromarray(grid)
    out_img.save(str(out_path))
    print(f"\nSaved: {out_path}  ({grid.shape[1]}×{grid.shape[0]} px)")


if __name__ == "__main__":
    main()
