#!/usr/bin/env python3
"""Per-plot hard class and fraction shares for one IM-023 amendment cell.

Run only after Tobias's go for the amendment. Verifies the amendment, the
frozen holdout it references, this process's source and runtime, the cell's
checkpoint and every frozen tile, then runs one forward per tile through both
heads. The dump keeps, per frozen plot-year, the frozen truth, the hard
class and the four fraction shares (tall, gran, trivial, adel); its sidecar
``.meta.json`` carries the provenance ``fieldtruth.verify_prediction_dump``
checks. Both files are created exclusively. Computes no score.

    /opt/venvs/model/bin/python /opt/imintengine/scripts/infer_nfi_amendment.py \
        --amendment-manifest /cephfs/nfi/im023_amendment_2026-11-06/freeze/manifest.json \
        --expected-amendment-sha256 SHA256 --cell im023_main \
        --checkpoint /cephfs/checkpoints/im023/tessera_b28/best_model.pt \
        --out /scratch/im023_main/nfi-amendment-per-plot-im023_main.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imint.eval.fieldtruth import NFI_KEY, sha256_file, verify_file_identity
from imint.eval.nfi_amendment import (
    AMENDMENT_CELLS, FRACTION_COLUMNS, load_amended_holdout, verify_amendment_source,
)
from scripts import inference_comparison as infcmp


def crop_plots(holdout: pd.DataFrame, tiles: dict, img_size: int) -> pd.DataFrame:
    """Plot pixels in each tile's centre crop; a frozen plot outside it is an error."""
    cropped = []
    for name, group in holdout.groupby("tile_name", sort=True):
        geometry = tiles[str(name)]["geometry"]
        height, width = geometry["height"], geometry["width"]
        size = min(img_size, height, width)
        y0, x0 = (height - size) // 2, (width - size) // 2
        if not (group["row"].between(y0, y0 + size - 1).all()
                and group["col"].between(x0, x0 + size - 1).all()):
            raise ValueError("frozen observation fell outside the model crop")
        cropped.append(group.assign(row=group["row"] - y0, col=group["col"] - x0))
    return pd.concat(cropped)


def make_predict_fn(cell: dict, checkpoint: str, device, tiles: dict):
    """``tile_path -> (probs (C, cs, cs), fractions (4, cs, cs))`` on authenticated bytes."""
    model, epoch, miou, native = infcmp.load_model(
        checkpoint, device, backbone_name=cell["backbone"], img_size=cell["img_size"],
        expected_checkpoint_sha256=cell["checkpoint"]["sha256"],
        expected_checkpoint_size=cell["checkpoint"]["bytes"])
    if not infcmp.model_has_frac_head(model):
        raise ValueError("amendment cell checkpoint has no fraction head")
    print(f"  [load_model] epoch={epoch} native_img={native}", flush=True)
    identities = {record["path"]: record for record in tiles.values()}

    def predict_fn(tile_path):
        identity = identities[str(Path(tile_path).resolve())]
        probs, fractions = infcmp.run_class_and_fraction_inference(
            model, tile_path, device, img_size=cell["img_size"],
            expected_tile_sha256=identity["sha256"], expected_tile_size=identity["bytes"])
        if (probs.shape[0] != cell["num_classes"]
                or fractions.shape != (len(FRACTION_COLUMNS), *probs.shape[1:])):
            raise ValueError("model outputs do not match the amendment cell")
        return probs, fractions

    return predict_fn


def predict_plots(plots: pd.DataFrame, predict_fn) -> pd.DataFrame:
    """One row per frozen plot-year: truth, hard class, four fraction shares."""
    frames = []
    for _, group in plots.groupby("tile_name", sort=True):
        probs, fractions = predict_fn(group["tile_path"].iloc[0])
        rows, cols = group["row"].to_numpy(dtype=int), group["col"].to_numpy(dtype=int)
        frame = group[NFI_KEY + ["tile_name", "nfi_forest"]].copy()
        frame["model_pred"] = probs.argmax(0)[rows, cols].astype(np.int64)
        for k, column in enumerate(FRACTION_COLUMNS):
            frame[column] = fractions[k, rows, cols].astype(np.float64)
        frames.append(frame)
    return pd.concat(frames).sort_values(NFI_KEY).reset_index(drop=True)


def write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amendment-manifest", type=Path, required=True)
    ap.add_argument("--expected-amendment-sha256", required=True,
                    help="amendment digest approved at go/no-go")
    ap.add_argument("--cell", choices=sorted(AMENDMENT_CELLS), required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", type=Path, required=True, help="new dump parquet; never overwritten")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    if args.out.exists() or meta_path.exists():
        raise ValueError("dump already exists; never overwrite it")
    import torch

    holdout, amendment, base = load_amended_holdout(args.amendment_manifest,
                                                    args.expected_amendment_sha256)
    runtime_identity = verify_amendment_source(amendment, environment="model")
    cell = amendment["cells"][args.cell]
    if sha256_file(args.checkpoint) != cell["checkpoint"]["sha256"]:
        raise ValueError("checkpoint differs from the amendment cell")
    for record in base["tiles"].values():
        verify_file_identity(record)
    if not holdout["tile_path"].map(os.path.exists).all():
        raise ValueError("frozen tile disappeared")
    plots = crop_plots(holdout, base["tiles"], cell["img_size"])
    print(f"{args.cell}: {len(plots)} frozen plot-years on {plots['tile_name'].nunique()} tiles",
          flush=True)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    dump = predict_plots(plots, make_predict_fn(cell, args.checkpoint, device, base["tiles"]))
    if len(dump) != len(holdout):
        raise ValueError("dump does not cover every frozen plot-year")
    buffer = io.BytesIO()
    dump.to_parquet(buffer, index=False)
    payload = buffer.getvalue()
    metadata = dict(runtime_identity, cell=args.cell, checkpoint=args.checkpoint,
                    checkpoint_sha256=cell["checkpoint"]["sha256"],
                    holdout_manifest_sha256=amendment["_manifest_sha256"],
                    amendment_protocol_id=amendment["protocol_id"],
                    base_manifest_sha256=amendment["base"]["manifest_sha256"],
                    img_size=cell["img_size"], observations=len(dump),
                    prediction_sha256=hashlib.sha256(payload).hexdigest(), status="success")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_exclusive(args.out, payload)
    write_exclusive(meta_path, (json.dumps(metadata, indent=2) + "\n").encode())
    print(f"wrote {args.out} ({len(dump)} plot-years)", flush=True)


if __name__ == "__main__":
    main()
