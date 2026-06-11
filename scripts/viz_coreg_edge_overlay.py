"""Frozen edge-overlay panels for the coregistration showcase.

Renders, for each before/after flicker GIF in ``docs/showcase/coreg/``, a
static edge overlay that makes inter-frame drift legible in a single still
image (the flicker GIF shows it in motion; this freezes it). Colours are
drawn from the DES document palette on the forest-green card background:

    grey      = reference-frame edges   (Sobel magnitude, thresholded)
    brick-red = target-frame edges
    mint      = the two coincide  ->  aligned

Before MI coregistration the grey and brick-red edge bands separate (the
drift); after MI they collapse onto each other (mint). Residual
grey/brick-red in the *after* panel is seasonal content difference between
acquisition dates, not misregistration — which is exactly why mutual
information is needed over phase correlation.

Input is the committed GIFs themselves (frame 0 = reference, frame 1 =
target, B04, 360x360), so this is fully reproducible from the repo with no
external data and no DES processing units. Run:

    python scripts/viz_coreg_edge_overlay.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import sobel

COREG_DIR = Path(__file__).resolve().parent.parent / "docs" / "showcase" / "coreg"

EDGE_PERCENTILE = 85.0  # keep the strongest ~15% of gradient pixels as "edge"
UPSCALE = 2             # nearest-neighbour, for crisp pixel edges at any display size

# DES document palette (RGB) — every colour below appears in docs/css/styles.css
# or the showcase markup. Forest-green background matches the .gif-pane card;
# grey + brick-red read as two distinct channels, and mint (the ba-after "Efter"
# accent) is the "aligned" colour where the two edge sets coincide.
BG_COLOR = (0x14, 0x3A, 0x30)       # forest green — .gif-pane card background
REF_COLOR = (0x9C, 0xA3, 0xAF)      # grey — reference-frame edges
FRAME_COLOR = (0xC0, 0x39, 0x2B)    # brick red — target-frame edges
ALIGNED_COLOR = (0xCF, 0xF8, 0xE4)  # mint (ba-after accent) — coincident edges (aligned)


def _gif_ref_and_target(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (reference, target) frames from a 2-frame flicker GIF."""
    with Image.open(path) as im:
        im.seek(0)
        ref = np.asarray(im.convert("L"), dtype=np.float32)
        im.seek(1)
        target = np.asarray(im.convert("L"), dtype=np.float32)
    return ref, target


def _edge_mask(frame: np.ndarray, percentile: float = EDGE_PERCENTILE) -> np.ndarray:
    """Binary Sobel-magnitude edge mask, thresholded per-frame for balance."""
    grad = np.hypot(sobel(frame, 0, mode="nearest"), sobel(frame, 1, mode="nearest"))
    return grad > np.percentile(grad, percentile)


def _edge_overlay(ref: np.ndarray, target: np.ndarray, upscale: int = UPSCALE) -> Image.Image:
    """grey = ref edges, brick-red = target edges, mint = both -> aligned."""
    ref_edges = _edge_mask(ref)
    frame_edges = _edge_mask(target)
    rgb = np.empty((*ref_edges.shape, 3), dtype=np.uint8)
    rgb[:] = BG_COLOR
    rgb[frame_edges & ~ref_edges] = FRAME_COLOR
    rgb[ref_edges & ~frame_edges] = REF_COLOR
    rgb[ref_edges & frame_edges] = ALIGNED_COLOR
    im = Image.fromarray(rgb)
    if upscale != 1:
        im = im.resize((im.width * upscale, im.height * upscale), Image.NEAREST)
    return im


def main() -> None:
    before_gifs = sorted(COREG_DIR.glob("*_before.gif"))
    if not before_gifs:
        raise SystemExit(f"no *_before.gif found in {COREG_DIR}")

    for before_gif in before_gifs:
        base = before_gif.name[: -len("_before.gif")]  # e.g. "43983928_f1"
        after_gif = COREG_DIR / f"{base}_after.gif"
        if not after_gif.exists():
            print(f"  skip {base}: no matching _after.gif")
            continue

        ref_b, target_before = _gif_ref_and_target(before_gif)
        ref_a, target_after = _gif_ref_and_target(after_gif)
        assert np.array_equal(ref_b, ref_a), f"{base}: reference differs between before/after GIF"

        _edge_overlay(ref_b, target_before).save(COREG_DIR / f"{base}_edge_before.png")
        _edge_overlay(ref_a, target_after).save(COREG_DIR / f"{base}_edge_after.png")
        print(f"  {base}: wrote _edge_before.png + _edge_after.png")


if __name__ == "__main__":
    main()
