"""
demos/cloud_models/run_comparison.py

Drives the full cloud-model comparison on every Kiruna scene:

    Host (Python 3.9):
        - DES MLP5 ensemble        (continuous COT, 11 bands)
        - DES MLP5 individual ×3   (variance within the ensemble)
        - Sen2Cor SCL              (categorical, from L2A)
        - Sen2Cor CLD              (probability 0-100, from L2A)

    Container (imint/cloud-models, Python 3.11):
        - s2cloudless              (Sinergise, LightGBM pixel classifier)
        - OmniCloudMask            (DPIRD, transformer 4-class segmenter)

For each scene we render a side-by-side comparison frame and emit a
manifest with per-model output paths + summary stats.

Run:
    python demos/cloud_models/run_comparison.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from imint.fetch import S2L2A_SPECTRAL_BANDS  # noqa: E402
from imint.analyzers.cot import (  # noqa: E402
    DEFAULT_MODEL_PATHS, _load_ensemble, MLP5, COT_MEANS, COT_STDS,
    COT_BAND_ORDER,
)
import torch  # noqa: E402

HERE         = Path(__file__).parent
SPEC_CACHE   = HERE / "cache_kiruna"
PRED_CACHE   = HERE / "preds"
PRED_CACHE.mkdir(exist_ok=True)
DOCS_OUT     = REPO_ROOT / "docs" / "showcase" / "cloud_models"
FRAMES_DIR   = DOCS_OUT / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

CONTAINER_IMG = "imint/cloud-models:latest"

# ── Color maps ─────────────────────────────────────────────────────────────
COT_CMAP = LinearSegmentedColormap.from_list("cot", [
    (0.00, "#0d4d2a"), (0.05, "#27ae60"), (0.10, "#f1c40f"),
    (0.30, "#e67e22"), (1.00, "#ecf0f1"),
])
PROB_CMAP = LinearSegmentedColormap.from_list("prob", [
    (0.0, "#0d4d2a"), (0.5, "#f1c40f"), (1.0, "#c0392b"),
])
# Sen2Cor SCL colour table (official ESA palette, abbreviated)
SCL_COLORS = {
    0: ("#000000", "no data"),
    1: ("#ff0000", "saturated/defective"),
    2: ("#404040", "casted shadows"),
    3: ("#833c0c", "cloud shadows"),
    4: ("#549e3f", "vegetation"),
    5: ("#fefe00", "non-vegetated"),
    6: ("#0000fe", "water"),
    7: ("#7e7e7e", "unclassified"),
    8: ("#bfbfbf", "cloud medium prob"),
    9: ("#ffffff", "cloud high prob"),
    10: ("#65cdee", "thin cirrus"),
    11: ("#ff6cff", "snow/ice"),
}
SCL_CMAP = ListedColormap([SCL_COLORS[i][0] for i in range(12)])
# OmniCloudMask classes: 0=clear, 1=thick, 2=thin, 3=shadow
OMNI_COLORS = ["#0d4d2a", "#ecf0f1", "#f1c40f", "#5d4037"]
OMNI_CMAP = ListedColormap(OMNI_COLORS)


# ── Host-side models (DES MLP5 + Sen2Cor parsing) ─────────────────────────

def cot_inference_mean(arr: np.ndarray, models: list[MLP5]) -> np.ndarray:
    """Return per-pixel mean COT across the ensemble."""
    band_idx = {b: i for i, b in enumerate(S2L2A_SPECTRAL_BANDS)}
    img = np.stack([arr[band_idx[b]] for b in COT_BAND_ORDER], axis=-1)
    h, w, _ = img.shape
    pix = torch.tensor(img.reshape(-1, len(COT_BAND_ORDER)),
                       dtype=torch.float32)
    pix = (pix - COT_MEANS) / COT_STDS
    sums = np.zeros(h * w, dtype=np.float32)
    with torch.no_grad():
        for m in models:
            preds = []
            for i in range(0, h * w, 65536):
                preds.append(m(pix[i:i + 65536])[:, 0].cpu().numpy())
            sums += np.concatenate(preds) / len(models)
    return sums.reshape(h, w)


def cot_inference_individual(arr: np.ndarray, models: list[MLP5]) -> list[np.ndarray]:
    """Return one COT map per ensemble member (for variance)."""
    band_idx = {b: i for i, b in enumerate(S2L2A_SPECTRAL_BANDS)}
    img = np.stack([arr[band_idx[b]] for b in COT_BAND_ORDER], axis=-1)
    h, w, _ = img.shape
    pix = torch.tensor(img.reshape(-1, len(COT_BAND_ORDER)),
                       dtype=torch.float32)
    pix = (pix - COT_MEANS) / COT_STDS
    out = []
    with torch.no_grad():
        for m in models:
            preds = []
            for i in range(0, h * w, 65536):
                preds.append(m(pix[i:i + 65536])[:, 0].cpu().numpy())
            out.append(np.concatenate(preds).reshape(h, w))
    return out


def scl_to_cloud_prob(scl: np.ndarray) -> np.ndarray:
    """Convert Sen2Cor SCL classes to a [0, 1] cloud-prob proxy.

    Mapping: cloud high = 1.0, cloud medium = 0.7, thin cirrus = 0.5,
             cloud shadow = 0.3, all else = 0.
    Used only for the heatmap panel so SCL can be compared on the same
    colour scale as the other models.
    """
    prob = np.zeros(scl.shape, dtype=np.float32)
    prob[scl == 8]  = 0.70   # medium probability
    prob[scl == 9]  = 1.00   # high probability
    prob[scl == 10] = 0.50   # thin cirrus
    prob[scl == 3]  = 0.30   # cloud shadow
    return prob


# ── Container runner ─────────────────────────────────────────────────────

def run_container(scene_dates: list[str]) -> dict:
    """Bind-mount cache → /work/in/, capture preds in /work/out/."""
    in_dir  = PRED_CACHE / "container_in"
    out_dir = PRED_CACHE / "container_out"
    if in_dir.exists():  shutil.rmtree(in_dir)
    if out_dir.exists(): shutil.rmtree(out_dir)
    in_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    # Copy just the spectral arr to /work/in/ — container only needs that
    for d in scene_dates:
        src = SPEC_CACHE / f"{d}.npz"
        data = np.load(src)
        # Save just `arr` (11 BOA bands) — container expects single key
        np.savez_compressed(in_dir / f"{d}.npz", arr=data["arr"])

    runner_dir = HERE.parent.parent / "docker" / "cloud-models"

    print(f"  Running container ({CONTAINER_IMG}) …")
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{in_dir}:/work/in:ro",
        "-v", f"{out_dir}:/work/out",
        "-v", f"{runner_dir}:/opt/runner:ro",
        CONTAINER_IMG,
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    print(f"  Container exit={res.returncode} elapsed={elapsed:.1f}s")
    if res.stdout:
        print(res.stdout)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise RuntimeError("Container failed")

    summary = {}
    sj = out_dir / "summary.json"
    if sj.exists():
        with open(sj) as f:
            summary = json.load(f)

    # Move predictions next to host preds
    preds_by_date: dict[str, dict[str, Path]] = {}
    for npz in out_dir.glob("*__*.npz"):
        date, _, model = npz.stem.partition("__")
        preds_by_date.setdefault(date, {})[model] = npz

    return {"summary": summary, "preds_by_date": preds_by_date,
            "wall_s": round(elapsed, 2)}


# ── Comparison frame rendering ────────────────────────────────────────────

def to_rgb(arr: np.ndarray) -> np.ndarray:
    band = {b: arr[i] for i, b in enumerate(S2L2A_SPECTRAL_BANDS)}
    rgb = np.stack([band["B04"], band["B03"], band["B02"]], axis=-1)
    valid = np.isfinite(rgb).all(axis=-1)
    if not valid.any():
        return np.zeros(rgb.shape, dtype=np.uint8)
    lo = np.percentile(rgb[valid], 2)
    hi = np.percentile(rgb[valid], 98)
    if hi <= lo: hi = lo + 1e-3
    rgb = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def cloud_frac(mask: np.ndarray, threshold: float) -> float:
    valid = np.isfinite(mask)
    return float((mask[valid] >= threshold).mean()) if valid.any() else float("nan")


def render_comparison(date: str, preds: dict, arr: np.ndarray,
                      scl: np.ndarray) -> dict:
    """One row of 5 panels: RGB | Sen2Cor SCL | s2cloudless | OmniCloudMask | DES MLP5."""
    rgb = to_rgb(arr)
    s2cl = preds.get("s2cloudless")
    omni = preds.get("omnicloudmask")
    mlp5 = preds["mlp5_mean"]

    panels = [
        ("RGB (B04/B03/B02)", rgb, None, None, None),
        ("Sen2Cor SCL",       scl, SCL_CMAP, 0, 11),
        ("s2cloudless",       s2cl if s2cl is not None else np.zeros(scl.shape, dtype=np.float32),
                              PROB_CMAP, 0, 1),
        ("OmniCloudMask",     omni if omni is not None else np.zeros(scl.shape, dtype=np.uint8),
                              OMNI_CMAP, 0, 3),
        ("DES MLP5 (COT)",    mlp5, COT_CMAP, 0, 0.3),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(17, 4.0), facecolor="#0f0f0f")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.04, wspace=0.04)
    for ax, (title, data, cmap, vmin, vmax) in zip(axes, panels):
        if cmap is None:
            ax.imshow(data, interpolation="nearest")
        else:
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.axis("off")

    stats = {
        "scl_cloud_frac":     round(float(np.isin(scl, [3, 8, 9, 10]).mean()), 4),
        "s2cloudless_frac":   round(cloud_frac(s2cl, 0.4), 4) if s2cl is not None else None,
        "omnicloudmask_thick_frac":
            round(float(((omni == 1) | (omni == 2)).mean()), 4) if omni is not None else None,
        "mlp5_mean_cot":      round(float(mlp5.mean()), 5),
        "mlp5_thick_frac":    round(float((mlp5 >= 0.025).mean()), 4),
    }

    fig.suptitle(
        f"Kiruna · {date}  |  "
        f"SCL clouds {stats['scl_cloud_frac']*100:.0f}% · "
        f"s2cl {stats['s2cloudless_frac']*100 if stats['s2cloudless_frac'] is not None else 0:.0f}% · "
        f"OCM thick+thin {stats['omnicloudmask_thick_frac']*100 if stats['omnicloudmask_thick_frac'] is not None else 0:.0f}% · "
        f"MLP5 thick {stats['mlp5_thick_frac']*100:.0f}% (mean COT {stats['mlp5_mean_cot']:.4f})",
        color="#27ae60", fontsize=11, fontweight="bold", y=0.96,
    )

    out = FRAMES_DIR / f"{date}.jpg"
    fig.savefig(out, dpi=110, facecolor="#0f0f0f", bbox_inches="tight")
    plt.close(fig)
    return {"date": date, "frame_path": str(out.relative_to(REPO_ROOT / "docs")),
            "stats": stats}


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    manifest_in = SPEC_CACHE / "manifest.json"
    if not manifest_in.exists():
        sys.exit("Run fetch_kiruna.py first.")
    with open(manifest_in) as f:
        fetch_manifest = json.load(f)

    dates = sorted(fetch_manifest["scl_fracs"].keys())
    print(f"Comparing {len(dates)} scenes: {dates}")

    print("\n[1/4] Loading DES MLP5 ensemble …")
    models = _load_ensemble(DEFAULT_MODEL_PATHS, device="cpu")

    print("\n[2/4] Running container (s2cloudless + OmniCloudMask) …")
    container = run_container(dates)
    container_preds = container["preds_by_date"]

    print("\n[3/4] Running host-side models (DES MLP5 + Sen2Cor parsing) …")
    frames = []
    for d in dates:
        npz = np.load(SPEC_CACHE / f"{d}.npz")
        arr  = npz["arr"]
        scl  = npz["scl"]

        t0 = time.time()
        mlp5_mean = cot_inference_mean(arr, models)
        elapsed_mlp5 = time.time() - t0
        print(f"  {d}  MLP5: {elapsed_mlp5:.1f}s")

        preds: dict = {"mlp5_mean": mlp5_mean}
        for model_name in ("s2cloudless", "omnicloudmask"):
            if model_name in container_preds.get(d, {}):
                preds[model_name] = np.load(container_preds[d][model_name])["arr"]

        frame = render_comparison(d, preds, arr, scl)
        frames.append(frame)
        print(f"    → {frame['frame_path']}")

    print("\n[4/4] Writing manifest …")
    manifest = {
        "aoi":     fetch_manifest["aoi"],
        "bbox":    fetch_manifest["bbox"],
        "period":  fetch_manifest["period"],
        "scl_fracs": fetch_manifest["scl_fracs"],
        "container_summary": container["summary"],
        "container_wall_s":  container["wall_s"],
        "frames": frames,
    }
    with open(DOCS_OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n✓ {len(frames)} comparison frames → {DOCS_OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
