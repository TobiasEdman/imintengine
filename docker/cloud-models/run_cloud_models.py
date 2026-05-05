"""
docker/cloud-models/run_cloud_models.py

Container entrypoint: runs each open-source cloud detector on every tile
in /work/in/ and writes one .npz per (tile × model) to /work/out/.

Tile input format
-----------------
Each /work/in/<date>.npz contains key `arr` of shape (11, H, W), float32 BOA
reflectance in [0, 1], in S2L2A_SPECTRAL_BANDS order:
    ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

Output format
-------------
For each input file <date>.npz we write:
    /work/out/<date>__s2cloudless.npz       cloud_prob (H, W) float32 [0,1]
    /work/out/<date>__omnicloudmask.npz     classes    (H, W) uint8
                                            (0=clear, 1=thick, 2=thin, 3=shadow)

Why these two: s2cloudless is a different model family (LightGBM pixel
classifier) than the DES MLP5 ensemble we run on the host, and OmniCloudMask
is a recent transformer-based cloud + cloud-shadow segmenter — three
distinct architectures + training-data-mixes covering the same task.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

IN_DIR  = Path("/work/in")
OUT_DIR = Path("/work/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Band order for incoming arrays — must match host pipeline.
INPUT_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08",
               "B8A", "B09", "B11", "B12"]


# ── s2cloudless ──────────────────────────────────────────────────────────
#
# s2cloudless wants 10 specific TOA bands:
#   B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12
# We have 10 of those (no B10 — L2A drops the cirrus band) and we have
# B01 missing too (the host fetches the 11-band spectral set without B01).
# Substitute strategy:
#   - B01 → use B02 as a proxy (both are short-wave visible, B01 is
#           coastal-aerosol; the model is robust to small spectral
#           displacement here — this is the same fall-back used by
#           several public s2cloudless-on-L2A tutorials).
#   - B10 → predict on a zero band; this is what the s2cloudless authors
#           recommend for L2A input where the cirrus band is unavailable.
# We tag the result with these substitutions in the metadata sidecar.
S2CLOUDLESS_INPUT_BANDS = [
    "B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12",
]

def run_s2cloudless(arr: np.ndarray) -> tuple[np.ndarray, dict]:
    from s2cloudless import S2PixelCloudDetector

    band_idx = {b: i for i, b in enumerate(INPUT_BANDS)}

    chans = []
    notes = []
    for b in S2CLOUDLESS_INPUT_BANDS:
        if b == "B01":
            chans.append(arr[band_idx["B02"]])
            notes.append("B01<-B02")
        elif b == "B10":
            chans.append(np.zeros_like(arr[0]))
            notes.append("B10=0")
        else:
            chans.append(arr[band_idx[b]])

    stack = np.stack(chans, axis=-1)[None, ...]  # (1, H, W, 10)
    detector = S2PixelCloudDetector(threshold=0.4, all_bands=False)
    cloud_prob = detector.get_cloud_probability_maps(stack)[0]  # (H, W)
    return cloud_prob.astype(np.float32), {"substitutions": notes}


# ── OmniCloudMask ─────────────────────────────────────────────────────────
#
# OmniCloudMask predicts 4 classes per pixel:
#   0 = clear, 1 = thick cloud, 2 = thin cloud, 3 = cloud shadow
# Inputs: red (B04), green (B03), NIR (B08), SWIR (B11) at 10 m. The package
# auto-downloads its model weights on first use. Network needed once per
# container build → cache to a baked-in HF cache for offline reproducibility
# (handled by the install layer; on first container run it does the fetch).
def run_omnicloudmask(arr: np.ndarray) -> tuple[np.ndarray, dict]:
    """OmniCloudMask v1.7 expects 3 input bands (Red, Green, NIR) at 10 m,
    DN-style uint16 (0-10000)."""
    from omnicloudmask import predict_from_array

    band_idx = {b: i for i, b in enumerate(INPUT_BANDS)}
    stack = np.stack([
        arr[band_idx["B04"]],   # Red
        arr[band_idx["B03"]],   # Green
        arr[band_idx["B08"]],   # NIR
    ], axis=0)
    stack = (stack * 10000).astype(np.uint16)

    pred = predict_from_array(stack)
    # predict_from_array returns (C, H, W) where C may be 1 (class index)
    # or N classes (probability per class) depending on version.
    if pred.ndim == 3:
        if pred.shape[0] == 1:
            pred = pred[0]
        else:
            pred = pred.argmax(axis=0)
    return pred.astype(np.uint8), {}


# ── Driver ───────────────────────────────────────────────────────────────

MODELS = {
    "s2cloudless":    run_s2cloudless,
    "omnicloudmask":  run_omnicloudmask,
}


def main() -> int:
    inputs = sorted(IN_DIR.glob("*.npz"))
    if not inputs:
        print(f"No .npz files in {IN_DIR}", file=sys.stderr)
        return 1

    summary: dict[str, dict] = {}
    for npz_path in inputs:
        date = npz_path.stem
        arr = np.load(npz_path)["arr"]
        if arr.shape[0] != len(INPUT_BANDS):
            print(f"  {date}: skip (wrong band count {arr.shape[0]})")
            continue

        per_date = {}
        for model_name, fn in MODELS.items():
            t0 = time.time()
            try:
                pred, meta = fn(arr)
                out_path = OUT_DIR / f"{date}__{model_name}.npz"
                np.savez_compressed(out_path, arr=pred)
                elapsed = time.time() - t0
                per_date[model_name] = {
                    "elapsed_s": round(elapsed, 2),
                    "shape":     list(pred.shape),
                    "dtype":     str(pred.dtype),
                    "meta":      meta,
                }
                print(f"  {date} {model_name}: {elapsed:.1f}s, "
                      f"shape={pred.shape}, dtype={pred.dtype}")
            except Exception as e:
                per_date[model_name] = {"error": f"{type(e).__name__}: {e}"}
                print(f"  {date} {model_name}: FAIL {e}", file=sys.stderr)
        summary[date] = per_date

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote summary: {OUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
