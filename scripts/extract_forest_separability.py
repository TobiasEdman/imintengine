#!/usr/bin/env python3
"""Bounded frozen R4 feature extraction from historical training tiles only.

No NFI index, NFI observation, sealed holdout, or new external data is read.
Selection is deterministic before any classification score is computed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_forest_separability import ROLE, digest, sample_pixels, write_json


def ensure_development_roots(data_dir: Path, label_dir: Path, distill_dir: Path) -> None:
    """Fail closed on evaluation or symlinked directories before reading data."""
    for p, expected in ((data_dir, "unified_v2_512"), (label_dir, "nmd2023_labels"),
                        (distill_dir, "tessera_r2")):
        p = p.absolute()
        if p.name != expected or p.resolve() != p:
            raise ValueError(f"requires literal historical development directory {expected}")
        if any(token in str(p).lower() for token in ("holdout", "nfi2024", "topup")):
            raise ValueError("evaluation inputs are prohibited")


def tile_candidates(data_dir: Path, label_dir: Path, *, max_tiles: int,
                    year: int, audit_path: Path | None = None) -> list[dict]:
    if not 5 <= max_tiles <= 24 or year not in (2022, 2023):
        raise ValueError("cap is 5–24 tiles; pre-specified feature year must be2022 or2023")
    paths = sorted(data_dir.glob("*.npz"), key=lambda p: hashlib.sha256(p.name.encode()).digest())
    selected, centres = [], []
    blocks = set()
    years = Counter()
    scanned = 0
    for path in paths:
        scanned += 1
        label = label_dir / path.name
        if not label.exists() or path.is_symlink() or label.is_symlink():
            continue
        with np.load(path, allow_pickle=False) as z:
            tile_year = int(z["tessera_year"]) if "tessera_year" in z.files else None
            years[str(tile_year)] += 1
            if audit_path is not None and scanned % 250 == 0:
                write_json(audit_path, {"scanned": scanned, "total": len(paths),
                    "tessera_year_counts": dict(years), "selected": len(selected),
                    "requested_year": year, "complete": False})
            if "tessera" not in z.files or tile_year != year:
                continue
            bbox = np.asarray(z["bbox_3006"], dtype=float)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ValueError("invalid training grid bbox")
        west, south, east, north = bbox
        if not np.allclose([east - west, north - south], [5120, 5120]):
            continue
        center = np.array([(west + east) / 2, (south + north) / 2])
        block = tuple(np.floor(center / 50000).astype(int))
        if (tuple(np.floor(np.array([west, south]) / 50000).astype(int)) != block
                or tuple(np.floor(np.array([east - 1, north - 1]) / 50000).astype(int)) != block):
            continue  # Keep every pixel of a tile in one spatial CV/bootstrap block.
        # One tile per50km cell plus >=15km centre distance gives >=9.8km
        # separation between square footprints, exceeding the5km CV buffer.
        if block in blocks or any(np.max(np.abs(center - c)) < 15000 for c in centres):
            continue
        with np.load(label, allow_pickle=False) as z:
            raw = z["nmd_label_raw"]
        if raw.shape != (512, 512) or raw.dtype != np.uint16:
            raise ValueError("NMD2023 sidecar must retain full raw uint16 codes")
        try:
            sample_pixels(raw[4:-4, 4:-4], np.ones((504, 504), bool), per_class=128, seed=42)
        except ValueError:
            continue
        selected.append({"path": str(path), "label_path": str(label),
                         "bbox_3006": bbox.tolist(), "year": year})
        centres.append(center)
        blocks.add(block)
        if len(selected) == max_tiles:
            break
    if audit_path is not None:
        write_json(audit_path, {"scanned": scanned, "total": len(paths),
            "tessera_year_counts": dict(years), "selected": len(selected),
            "requested_year": year, "complete": True})
    if len(selected) < 5:
        raise ValueError(f"only {len(selected)} qualified development tiles")
    return selected


def read_sidecar(path: Path, identity: dict) -> dict:
    """Authenticate the exact sidecar bytes used, even on a mutable PVC."""
    from scripts.inference_comparison import _load_npz_for_inference

    archive, buffer = _load_npz_for_inference(
        path, expected_sha256=identity["sha256"], expected_size=identity["size"])
    try:
        result = {key: archive[key].copy() for key in ("label", "nmd_label_raw")}
        if "head_sha" in archive.files:
            result["head_sha"] = str(archive["head_sha"])
        return result
    finally:
        archive.close()
        if buffer is not None:
            buffer.close()


def validate_selected_metadata(data, entry: dict) -> None:
    """Bind selection year/geometry to the authenticated bytes actually used."""
    if int(data["tessera_year"]) != entry["year"]:
        raise ValueError("authenticated feature year differs from selection")
    if not np.array_equal(np.asarray(data["bbox_3006"], dtype=float),
                          np.asarray(entry["bbox_3006"], dtype=float)):
        raise ValueError("authenticated pixel geometry differs from selection")


def extract(args) -> None:
    import torch
    from scripts.cache_predict import file_identity
    from scripts.extract_plot_features import register_preclassifier_hook, _sample_feature
    from scripts.inference_comparison import (
        load_model, inference_aux_names, _build_inference_inputs,
        _forward_from_inputs, _close_inference_inputs)

    ensure_development_roots(args.data_dir, args.label_dir, args.distill_dir)
    if not 8 <= args.per_class <= 128:
        raise ValueError("per-class cap must be8–128")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    selected = tile_candidates(args.data_dir, args.label_dir, max_tiles=args.max_tiles,
                               year=args.feature_year, audit_path=args.output / "availability.json")
    checkpoint = file_identity(args.checkpoint)
    contract = {"role": ROLE, "checkpoint": checkpoint, "selection": selected,
                "feature_year": args.feature_year, "reference_year": 2023,
                "reference_year_gap": 2023 - args.feature_year, "per_class_cap": args.per_class,
                "source_manifest_sha256": digest(ROOT / "SOURCE_MANIFEST.json")
                    if (ROOT / "SOURCE_MANIFEST.json").exists() else None,
                "sampling": "100m lattice;3x3 interiors;seed42;one tile per50km block",
                "runtime": {"torch": torch.__version__, "numpy": np.__version__, "threads": 2}}
    write_json(args.output / "selection.json", contract)
    model, epoch, _, native = load_model(
        str(args.checkpoint), torch.device("cpu"), backbone_name="tessera_v1", img_size=504,
        expected_checkpoint_sha256=checkpoint["sha256"], expected_checkpoint_size=checkpoint["size"])
    if native != 504 or model.num_classes != 28 or model.embed_dim != 128:
        raise ValueError("unexpected R4 architecture")
    model.eval()
    aux_names = inference_aux_names(model.ck_cfg, model.n_aux_channels)
    store = register_preclassifier_hook(model)
    shards, records = [], []
    started = time.monotonic()
    try:
        for entry in selected:
            path, label_path = Path(entry["path"]), Path(entry["label_path"])
            distill_path = args.distill_dir / path.name
            identities = {k: file_identity(p) for k, p in (
                ("tile", path), ("nmd2023", label_path), ("distill", distill_path))}
            base = read_sidecar(label_path, identities["nmd2023"])
            distill = read_sidecar(distill_path, identities["distill"])
            raw = base["nmd_label_raw"][4:-4, 4:-4].copy()
            base_labels = base["label"][4:-4, 4:-4].copy()
            distilled = distill["label"][4:-4, 4:-4].copy()
            head_sha = distill["head_sha"]
            if not np.array_equal(raw, distill["nmd_label_raw"][4:-4, 4:-4]):
                raise ValueError("raw labels changed between NMD and distilled sidecar")
            inp = _build_inference_inputs(
                str(path), torch.device("cpu"), 504, aux_names, family="tessera",
                expected_tile_sha256=identities["tile"]["sha256"],
                expected_tile_size=identities["tile"]["size"])
            try:
                validate_selected_metadata(inp["data"], entry)
                if (inp["crop_sz"], inp["y0"], inp["x0"]) != (504, 4, 4):
                    raise ValueError("unexpected pixel alignment")
                emb = inp["img5d"][0].numpy()
                valid = np.isfinite(emb).all(0) & np.any(emb != 0, axis=0)
                r, c = sample_pixels(raw, valid, per_class=args.per_class, seed=42)
                store["feat"] = None
                with torch.inference_mode():
                    logits = _forward_from_inputs(model, inp, torch.device("cpu"))
                if store["feat"] is None or not torch.isfinite(logits).all():
                    raise ValueError("invalid feature capture or logits")
                vectors = _sample_feature(store["feat"], r, c, 504)
                west, _, _, north = entry["bbox_3006"]
                sample = {"r4": vectors, "tessera": emb[:, r, c].T.copy(),
                          "raw_nmd": raw[r, c], "tile": np.full(len(r), path.stem),
                          "xy": np.column_stack((west + (c + 4.5) * 10,
                                                  north - (r + 4.5) * 10)),
                          "r4_prediction": logits.argmax(1)[0, r, c].numpy(),
                          "base_label": base_labels[r, c], "distilled_label": distilled[r, c]}
                if not np.isfinite(vectors).all():
                    raise ValueError("nonfinite frozen features")
            finally:
                _close_inference_inputs(inp)
                store["feat"] = None
            shard = args.output / (path.stem + ".npz")
            np.savez_compressed(shard, **sample)
            records.append({"identities": identities, "head_sha": head_sha,
                            "samples": len(r), "shard_sha256": digest(shard)})
            write_json(args.output / (path.stem + ".json"), records[-1])
            shards.append(sample)
            print(json.dumps({"completed": len(shards), "tiles": len(selected),
                              "elapsed_s": round(time.monotonic() - started, 2)}), flush=True)
    finally:
        store["handle"].remove()
    bundle = args.output / "features.npz"
    np.savez_compressed(bundle, **{k: np.concatenate([s[k] for s in shards]) for k in shards[0]})
    write_json(bundle.with_suffix(".json"), {**contract, "epoch": epoch,
        "aux_names": aux_names, "tiles": records, "bundle_sha256": digest(bundle),
        "elapsed_s": time.monotonic() - started})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--label-dir", required=True, type=Path)
    ap.add_argument("--distill-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--feature-year", required=True, choices=(2022, 2023), type=int)
    ap.add_argument("--max-tiles", default=24, type=int)
    ap.add_argument("--per-class", default=128, type=int)
    extract(ap.parse_args())


if __name__ == "__main__":
    main()
