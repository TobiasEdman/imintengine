"""Generate the Superresolution showcase — runs every open SR model
side-by-side on a single Sentinel-2 RGB tile and renders a comparison
grid for the dashboard.

Models compared (RGB only — see CLAUDE.md research notes for why):
  - bicubic    Floor every learned model must beat
  - sen2sr     ESA OpenSR (sen2sr+mlstac), 4× radiometrically-consistent CNN
  - ldsr       ESA OpenSR (opensr-model), 4× latent diffusion

Pipeline:
  1. Fetch one Sentinel-2 L2A scene over ``COORDS_WGS84`` for ``--date``
  2. Build float32 RGB
  3. For each model: predict 4× upsampled RGB
  4. Save per-model PNG + assembled comparison grid
  5. Mirror to ``docs/showcase/sr/`` so the dashboard serves them

Usage::

    .venv/bin/python scripts/generate_sr_showcase.py --date 2025-07-15

Models that fail to load (missing dep, missing checkpoint) are reported
with a clear error and skipped — the rest of the showcase still renders.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Stockholm — central, user-picked via docs/bbox_picker.html ──────
COORDS_WGS84 = {
    "west":  18.0169,
    "south": 59.2953,
    "east":  18.1244,
    "north": 59.3768,
}
PRIMARY_DATE = "2025-07-15"

# Default model set — order controls grid layout (left→right, top→bottom).
# DiffFuSR and SR4RS dropped: no installable open weights as of 2026-04.
# LDSR uses chunked inference (128×128 LR patches with Hann blend) so it
# fits comfortably on a 2080ti — see imint/analyzers/sr/ldsr.py.
DEFAULT_MODELS = ["bicubic", "sen2sr", "ldsr"]


def _build_grid(
    panels: list[tuple[str, np.ndarray]],
    cols: int = 3,
    pad: int = 8,
    label_h: int = 28,
) -> np.ndarray:
    """Tile per-model PNGs into a grid with model-name banners.

    ``panels`` is a list of (label, rgb) where rgb is float32 [0,1].
    All panels are resized to the first panel's shape so the grid is
    rectangular even if a wrapper returned a different scale.
    """
    if not panels:
        return np.zeros((label_h, label_h, 3), dtype=np.float32)

    from PIL import Image, ImageDraw, ImageFont

    h, w, _ = panels[0][1].shape
    cell_h, cell_w = h + label_h, w
    rows = (len(panels) + cols - 1) // cols
    grid_h = rows * cell_h + (rows + 1) * pad
    grid_w = cols * cell_w + (cols + 1) * pad
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.float32)

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", 18
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, (label, rgb) in enumerate(panels):
        r, c = divmod(i, cols)
        # Match shape if any wrapper returned different size
        if rgb.shape != (h, w, 3):
            from PIL import Image as _Im
            rgb = np.asarray(
                _Im.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))
                   .resize((w, h), _Im.BICUBIC)
            ).astype(np.float32) / 255.0

        y0 = pad + r * (cell_h + pad)
        x0 = pad + c * (cell_w + pad)

        # Banner with label
        banner = Image.new("RGB", (w, label_h), (24, 24, 28))
        draw = ImageDraw.Draw(banner)
        draw.text((8, 4), label.upper(), fill=(220, 220, 220), font=font)
        grid[y0:y0 + label_h, x0:x0 + w] = (
            np.asarray(banner).astype(np.float32) / 255.0
        )
        grid[y0 + label_h:y0 + label_h + h, x0:x0 + w] = rgb

    return np.clip(grid, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Superresolution comparison showcase"
    )
    parser.add_argument("--date", default=PRIMARY_DATE,
                        help=f"S2 scene date (default {PRIMARY_DATE})")
    parser.add_argument("--cloud-threshold", type=float, default=0.3,
                        help="Max cloud fraction (default 0.3)")
    parser.add_argument("--date-window", type=int, default=5,
                        help="±days around --date to search for cloud-free imagery (default 5)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Models to run (default {DEFAULT_MODELS})")
    parser.add_argument("--device", default="cuda",
                        help="Torch device for learned models (default cuda)")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default outputs/showcase/sr)")
    args = parser.parse_args()

    from imint.config.env import load_env
    load_env("dev")

    from imint.fetch import fetch_des_data, FetchError
    from imint.exporters.export import save_rgb_png
    from imint.analyzers.sr import MODEL_REGISTRY

    out_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "outputs" / "showcase" / "sr"
    )
    docs_dir = PROJECT_ROOT / "docs" / "showcase" / "sr"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Superresolution Showcase — central Stockholm")
    print(f"  Date:    {args.date}")
    print(f"  WGS84:   {COORDS_WGS84}")
    print(f"  Models:  {args.models}")
    print(f"  Out:     {out_dir}")
    print("=" * 60)

    # ── Fetch the LR tile from DES (Digital Earth Sweden) ────────────
    print("\n[1/4] Fetching Sentinel-2 L2A from DES...")
    try:
        fetched = fetch_des_data(
            date=args.date,
            coords=COORDS_WGS84,
            cloud_threshold=args.cloud_threshold,
            include_scl=True,
            date_window=args.date_window,
        )
    except FetchError as e:
        print(f"  Fetch failed: {e}")
        sys.exit(1)

    # Three input tensors:
    #   raw_10b:  10-channel L2A reflectance (B02,B03,B04,B05,B06,B07,
    #             B08,B8A,B11,B12) for SEN2SR which expects all 10
    #             bands per its training spec.
    #   raw_rgb:  3-channel RGB+G-as-NIR-proxy (4ch internally) for
    #             LDSR which trained on RGBN at 10m.
    #   rgb_lr:   percentile-stretched display version that the
    #             dashboard shows to the user. Saved to disk + used by
    #             bicubic so all panels share the same visual envelope.
    #
    # All three derive from the same DES fetch — no second network call.
    bands = fetched.bands
    band_order = ["B02", "B03", "B04", "B05", "B06", "B07",
                  "B08", "B8A", "B11", "B12"]
    missing = [b for b in band_order if b not in bands]
    if missing:
        print(f"  WARNING: missing bands {missing} — SEN2SR will skip.")
    raw_10b = np.stack(
        [bands[b] for b in band_order if b in bands], axis=0
    ).astype(np.float32) if not missing else None
    raw_rgb = np.stack(
        [bands["B04"], bands["B03"], bands["B02"]], axis=-1
    ).astype(np.float32)
    rgb_lr = fetched.rgb.astype(np.float32)
    print(f"  cloud_frac={fetched.cloud_fraction:.1%}")
    print(f"  raw refl range:    [{raw_rgb.min():.3f}, {raw_rgb.max():.3f}]")
    if raw_10b is not None:
        print(f"  10-band stack:     {raw_10b.shape}")
    print(f"  display rgb shape: {rgb_lr.shape}")

    save_rgb_png(rgb_lr, str(out_dir / "rgb_lr.png"))


    def _stretch_like_lr(sr: np.ndarray) -> np.ndarray:
        """Apply a 2/98 percentile stretch so all model panels share the
        same display envelope. Operates on raw reflectance output."""
        p2, p98 = np.percentile(sr, [2, 98])
        if p98 - p2 < 1e-3:
            return np.clip(sr, 0.0, 1.0)
        return np.clip((sr - p2) / (p98 - p2), 0.0, 1.0)

    def _sam_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Per-pixel Spectral Angle Mapper between two (H, W, C) tensors.

        Returns angles in degrees ∈ [0, 180]. Zero degrees means identical
        spectral signature (only magnitude differs); larger angles mean
        the model invented or distorted spectral content. This is the
        primary fidelity axis on the opensr-test benchmark.
        """
        eps = 1e-9
        dot = (a * b).sum(axis=-1)
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        cos = np.clip(dot / (na * nb + eps), -1.0, 1.0)
        return np.degrees(np.arccos(cos)).astype(np.float32)

    def _save_sam_png(sam_deg: np.ndarray, path: Path,
                      vmax: float = 10.0) -> None:
        """Render SAM array as a colour-mapped PNG using viridis-like
        ramp: dark blue (0°, faithful) → green/yellow (5–10°) →
        red (>vmax°, suspect hallucination)."""
        from PIL import Image
        # 0..vmax → 0..1
        x = np.clip(sam_deg / vmax, 0.0, 1.0)
        # Hand-rolled viridis-ish ramp without matplotlib dep:
        # blue → green → yellow → red
        r = np.clip(2.0 * x - 0.5, 0.0, 1.0)
        g = np.clip(1.0 - np.abs(2.0 * x - 1.0), 0.0, 1.0)
        b = np.clip(1.0 - 2.0 * x, 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
        img = (rgb * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)
        print(f"    saved: {path}")

    # ── Run each model ───────────────────────────────────────────────
    print("\n[2/4] Running SR models...")
    panels: list[tuple[str, np.ndarray]] = []
    timings: dict[str, float] = {}
    failures: dict[str, str] = {}
    raw_sr_outputs: dict[str, np.ndarray] = {}

    for model_id in args.models:
        if model_id not in MODEL_REGISTRY:
            failures[model_id] = f"unknown model id (known: {list(MODEL_REGISTRY)})"
            print(f"  {model_id}: SKIP — {failures[model_id]}")
            continue

        cls = MODEL_REGISTRY[model_id]
        model = cls(config={"device": args.device})
        # Per-model input shape (verified via local CPU smoke-test):
        #   bicubic:  raw 3ch RGB (scipy.ndimage.zoom on float reflectance)
        #   sen2sr:   raw 10-channel reflectance (B02..B12)
        #   ldsr:     raw 3ch RGB; wrapper internally pads NIR proxy
        if model_id == "sen2sr":
            if raw_10b is None:
                failures[model_id] = "missing required bands B05–B12"
                print(f"  {model_id}: SKIP — {failures[model_id]}")
                continue
            model_input = raw_10b
        else:
            model_input = raw_rgb
        t0 = time.time()
        result = model.predict(model_input)
        dt = time.time() - t0
        timings[model_id] = dt

        if not result.success:
            failures[model_id] = result.error or "unknown failure"
            print(f"  {model_id}: FAIL ({dt:.1f}s) — {failures[model_id]}")
            continue

        # All wrappers now return raw reflectance at SR resolution.
        # Apply a shared 2/98 stretch for the display PNG; keep raw for SAM.
        sr_raw = result.sr
        sr_display = _stretch_like_lr(sr_raw)
        out_png = out_dir / f"{model_id}.png"
        save_rgb_png(sr_display, str(out_png))
        panels.append((model_id, sr_display))
        raw_sr_outputs[model_id] = sr_raw
        print(f"  {model_id}: OK ({dt:.1f}s) → {sr_raw.shape}")

    # ── Spectral Angle Mapper analysis ────────────────────────────────
    # For each successful model, compute per-pixel SAM between the
    # nearest-neighbour-upsampled LR (preserves the LR's exact spectral
    # signature at every block) and the model's raw SR output. SAM
    # surfaces *where* a model has invented or distorted spectral
    # content — the colourmap maps 0° (blue, faithful) → 10° (red,
    # suspect hallucination).
    print("\n[3/4] Computing Spectral Angle Mapper (LR ↔ SR)...")
    s = 4
    H, W, _ = raw_rgb.shape
    lr_nn = np.repeat(np.repeat(raw_rgb, s, axis=0), s, axis=1)  # (H*4, W*4, 3)
    sam_means: dict[str, float] = {}
    for model_id, sr_raw in raw_sr_outputs.items():
        if sr_raw.shape != lr_nn.shape:
            print(f"  {model_id}: SAM skipped — shape mismatch "
                  f"{sr_raw.shape} vs {lr_nn.shape}")
            continue
        sam_deg = _sam_degrees(lr_nn, sr_raw)
        mean_sam = float(sam_deg.mean())
        sam_means[model_id] = mean_sam
        print(f"  {model_id}: mean SAM = {mean_sam:.2f}°  "
              f"(p95 = {np.percentile(sam_deg, 95):.2f}°)")
        _save_sam_png(sam_deg, out_dir / f"sam_{model_id}.png", vmax=10.0)

    # Write a small JSON summary the dashboard can load.
    import json
    summary = {
        "date": args.date,
        "coords_wgs84": COORDS_WGS84,
        "lr_shape": list(rgb_lr.shape),
        "scale": 4,
        "timings_s": timings,
        "sam_mean_deg": sam_means,
        "failures": failures,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Mirror to docs/ for the dashboard ────────────────────────────
    print("\n[4/4] Mirroring to docs/showcase/sr/...")
    for src in out_dir.iterdir():
        if src.is_file():
            shutil.copy2(src, docs_dir / src.name)
    print(f"  done — {sum(1 for _ in docs_dir.iterdir())} files in {docs_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
