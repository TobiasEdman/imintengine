"""scripts/infer_tiles.py — Stage A: GPU, batched, cached inference.

Produces a per-tile prediction cache ONCE per checkpoint, so every truth source
(NFI / LUCAS / …) scores from the same cache instead of re-inferring. This is
the GPU half of the two-stage architecture in
``docs/plans/faster_validation_architecture.md``; the CPU scoring half is
``score_against_truth.py``.

Correctness strategy — NO re-implementation of preprocessing or the model
forward. Each tile is preprocessed by the EXISTING
``inference_comparison._build_inference_inputs`` (identical spectral/aux
normalization, centre-crop and coord construction the validators use), and the
cached ``pred``/``probs``/``fracs`` are exactly what ``run_inference`` /
``run_fraction_inference`` return — argmax class map, ``softmax(logits)``,
``sigmoid(frac_logits)`` — after the SAME centre-crop, so the cache is
bit-identical modulo float16 storage (asserted in
``tests/test_cached_validation_parity.py``).

Speedups over the fused validators: a ``DataLoader`` (``num_workers`` parallel
``np.load`` + pinned prefetch), ``batch_size`` tiles per GPU forward, and a
SINGLE dual-head forward (``return_fractions=True``) that yields both the class
logits and the fraction logits — no second pass.

Cache layout: ``<cache-dir>/<ckpt_sha>/<tile_stem>.npz`` with
``pred`` (uint8, H×W), ``probs`` (float16, C×H×W), ``fracs`` (float16, 4×H×W),
plus a ``<cache-dir>/<ckpt_sha>/MANIFEST.json`` provenance sidecar. Idempotent:
a tile already cached for this checkpoint is skipped (like ``_valid_existing_tile``
in the fetchers).

    python scripts/infer_tiles.py \
        --checkpoint /data/checkpoints/tessera_gated/best_model.pt \
        --backbone-name tessera_v1 --img-size 504 --num-classes 28 \
        --data-dir /data/unified_v2_512 \
        --tile-list /data/lucas/lucas_tiles.txt \
        --cache-dir /cephfs/pred_cache --batch-size 16 --num-workers 8 \
        --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.tile_fetch import is_tile_tmp  # noqa: E402


def _load_infcmp():
    """Import inference_comparison by path (mirrors the validators' loader)."""
    spec = importlib.util.spec_from_file_location(
        "_infcmp", str(Path(__file__).resolve().parent / "inference_comparison.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ckpt_sha256(checkpoint: str) -> str:
    """First 16 hex of the checkpoint's sha256 — the cache subdir name.

    Keys the cache to the exact checkpoint bytes so two checkpoints never share
    a cache dir and a re-trained model invalidates its old predictions.
    """
    h = hashlib.sha256()
    with open(checkpoint, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[1]),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def resolve_tile_list(data_dir: str, tile_list: str | None,
                      allow_missing: bool = False) -> list[Path]:
    """Resolve the tiles to infer: an explicit list, or all ``*.npz`` in dir.

    ``tile_list`` may be a parquet with a ``tile_name`` column or a newline text
    file of tile names/paths. Names are resolved to ``<data-dir>/<name>.npz``
    (flat layout, matching the validators). Missing tiles raise — never a silent
    partial cache.
    """
    data_dir = Path(data_dir)
    if tile_list is None:
        tiles = sorted(p for p in data_dir.glob("*.npz") if not is_tile_tmp(p.name))
        if not tiles:
            raise FileNotFoundError(f"no *.npz tiles in {data_dir}")
        return tiles

    tl = Path(tile_list)
    if tl.suffix == ".parquet":
        import pandas as pd
        names = pd.read_parquet(tl)["tile_name"].unique().tolist()
    else:
        names = [ln.strip() for ln in tl.read_text().splitlines() if ln.strip()]

    tiles = []
    missing = []
    for name in names:
        stem = Path(str(name)).stem  # accept bare name or a full path
        p = data_dir / f"{stem}.npz"
        (tiles if p.exists() else missing).append(p)
    if missing:
        if allow_missing:
            # Truth indices can list tiles since removed from the dataset —
            # the scorers drop those plots/points anyway. Log loudly, skip.
            print(f"[infer_tiles] WARNING: skipping {len(missing)} tiles from "
                  f"{tile_list} not under {data_dir}, e.g. {missing[0].name}")
        else:
            raise FileNotFoundError(
                f"{len(missing)} tiles from {tile_list} not under {data_dir}, "
                f"e.g. {missing[0]} (pass --allow-missing to skip+log)")
    # De-dup preserving order.
    seen, uniq = set(), []
    for p in tiles:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


class TileInferenceDataset:
    """torch Dataset: per tile, run the EXISTING ``_build_inference_inputs``.

    Returns the model-ready tensors (batch dim squeezed off — the DataLoader
    re-adds it via collation) plus crop meta. Tiles are grouped by crop size
    upstream so a batch collates cleanly; the loader falls back to batch=1 for
    any odd-sized tile rather than mis-cropping.
    """

    def __init__(self, tile_paths, device_family, img_size, aux_channel_names,
                 num_frames=None):
        import torch  # noqa: F401  (import guarded to keep module import light)
        self._torch = __import__("torch")
        self.tile_paths = [str(p) for p in tile_paths]
        self.family = device_family
        self.img_size = img_size
        self.aux_channel_names = aux_channel_names
        # The model's frame count MUST reach _build_inference_inputs, exactly
        # as run_inference threads it. A single-frame checkpoint (Prithvi
        # ladder: num_temporal_frames=1) with num_frames=None here defaults to
        # the 4-frame branch, building 4x temporal_coords the model's
        # single-frame temporal_encoding cannot add (15376 vs 3844) — the
        # first ladder-eval wave died exactly so.
        self.num_frames = num_frames
        self._infcmp = _load_infcmp()

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, i):
        tile_path = self.tile_paths[i]
        # Build on CPU (device="cpu") — the DataLoader worker has no GPU;
        # tensors are moved to the GPU in the main loop after collation.
        inp = self._infcmp._build_inference_inputs(
            tile_path, self._torch.device("cpu"), self.img_size,
            self.aux_channel_names, family=self.family,
            num_frames=self.num_frames)
        item = {
            "img5d": inp["img5d"].squeeze(0),   # drop the (1, …) batch dim
            "aux": inp["aux"].squeeze(0),
            "stem": Path(tile_path).stem,
            "crop_sz": int(inp["crop_sz"]),
        }
        tc = inp["temporal_coords"]
        lc = inp["location_coords"]
        # Coords are per-tile and only present for Prithvi; carry them so the
        # collate can rebuild a batch tensor (all tiles in a batch share family).
        item["temporal_coords"] = tc.squeeze(0) if tc is not None else None
        item["location_coords"] = lc.squeeze(0) if lc is not None else None
        return item


def _collate(batch):
    """Stack same-size items into batch tensors; None coords stay None."""
    import torch
    out = {
        "img5d": torch.stack([b["img5d"] for b in batch], dim=0),
        "aux": torch.stack([b["aux"] for b in batch], dim=0),
        "stems": [b["stem"] for b in batch],
        "crop_sz": [b["crop_sz"] for b in batch],
    }
    tcs = [b["temporal_coords"] for b in batch]
    lcs = [b["location_coords"] for b in batch]
    out["temporal_coords"] = (
        torch.stack(tcs, dim=0) if all(t is not None for t in tcs) else None)
    out["location_coords"] = (
        torch.stack(lcs, dim=0) if all(t is not None for t in lcs) else None)
    return out


def _write_manifest(sha_dir: Path, *, checkpoint, ckpt_sha, backbone_name,
                    num_classes, img_size, n_tiles, git_sha, produced_at):
    """Provenance sidecar mirroring the repo's Docker-artifact MANIFEST rule."""
    manifest = {
        "checkpoint": str(checkpoint),
        "ckpt_sha": ckpt_sha,
        "backbone_name": backbone_name,
        "num_classes": int(num_classes),
        "img_size": int(img_size),
        "n_tiles": int(n_tiles),
        "git_sha": git_sha,
        "produced_at": produced_at,
    }
    (sha_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))


def infer_all(
    *, checkpoint, backbone_name, data_dir, tile_list, img_size, num_classes,
    cache_dir, batch_size, num_workers, device, shard, log_every, produced_at,
    allow_missing=False,
):
    """Load the model ONCE, batch-infer the tile set, write the cache.

    Returns the checkpoint sha and (n_written, n_skipped).
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    infcmp = _load_infcmp()
    dev = torch.device(device)

    sha = ckpt_sha256(checkpoint)
    sha_dir = Path(cache_dir) / sha
    sha_dir.mkdir(parents=True, exist_ok=True)

    tiles = resolve_tile_list(data_dir, tile_list, allow_missing=allow_missing)
    if shard is not None:
        i, k = shard
        tiles = tiles[i::k]
        print(f"shard {i}/{k}: {len(tiles)} tiles")

    print(f"[infer_tiles] checkpoint sha={sha} cache={sha_dir}")
    # Thread runtime img_size so clay/croma (no pos_embed, minimal config)
    # build their head at the exact resolution inference feeds — see
    # inference_comparison.load_model docstring.
    model, epoch, miou, native = infcmp.load_model(
        checkpoint, dev, backbone_name=backbone_name, img_size=img_size)
    family = getattr(getattr(model, "fm_spec", None), "family", "prithvi")
    print(f"  loaded epoch={epoch} ckpt_mIoU={miou} native_img={native} "
          f"family={family}")

    # The checkpoint KNOWS its aux set — reconstructing it from defaults is
    # how the ladder eval died on its first run: every ladder model is
    # 11-aux (markfukt on) and terramind is 13-aux (ΔSAR), while a None
    # here builds the canonical 10 and the aux_proj conv rejects the
    # tensor at the first forward. Same rule as extract_plot_features.
    try:
        ck_cfg = torch.load(checkpoint, map_location="cpu",
                            weights_only=False).get("config", {}) or {}
    except Exception:
        # This is an OPPORTUNISTIC config read — load_model above is the
        # authoritative loader and has already accepted the checkpoint
        # (tests monkeypatch it with non-torch fixtures). Unreadable here
        # ⇒ pre-config era ⇒ canonical aux fallback, as before.
        ck_cfg = {}
    aux_names = ck_cfg.get("enabled_aux_names")
    aux_names = list(aux_names) if aux_names else None
    if aux_names:
        print(f"  aux from checkpoint config: {len(aux_names)} channels "
              f"({aux_names[-3:]}...)")

    # load_model stamps the model with its trained frame count (single-frame
    # ladder columns = 1, Prithvi-600M = 4). Thread it exactly as
    # run_inference does — without it the dataset defaults to the 4-frame
    # branch and a 1-frame model's temporal_encoding rejects the 4x token
    # grid (15376 vs 3844), which killed the first ladder-eval wave.
    model_num_frames = getattr(model, "num_frames", None)
    ds = TileInferenceDataset(tiles, family, img_size, aux_names,
                              num_frames=model_num_frames)

    # Group tiles by crop size so a batch collates cleanly. Odd-sized tiles are
    # inferred at batch=1 rather than being mis-cropped or padded.
    size_of = {}
    for i, p in enumerate(tiles):
        cs = min(img_size, *_tile_hw(p))
        size_of.setdefault(cs, []).append(i)

    n_written = n_skipped = 0
    t0 = time.time()
    done = 0
    total = len(tiles)

    for cs, idxs in sorted(size_of.items()):
        # Skip tiles already cached (idempotent) BEFORE building the loader,
        # so a resumed run only re-infers the missing slice.
        pending = [
            i for i in idxs
            if not (sha_dir / f"{Path(ds.tile_paths[i]).stem}.npz").exists()
        ]
        n_skipped += len(idxs) - len(pending)
        done += len(idxs) - len(pending)
        if not pending:
            continue
        # All tiles in this group share crop size `cs`, so they collate cleanly
        # at the full batch size; distinct sizes are separate groups (batch=1
        # only when a size class has a single tile).
        loader = DataLoader(
            Subset(ds, pending), batch_size=batch_size, num_workers=num_workers,
            pin_memory=(dev.type == "cuda"), collate_fn=_collate, shuffle=False)
        for batch in loader:
            img5d = batch["img5d"].to(dev, non_blocking=True)
            aux = batch["aux"].to(dev, non_blocking=True)
            tc = batch["temporal_coords"]
            lc = batch["location_coords"]
            tc = tc.to(dev, non_blocking=True) if tc is not None else None
            lc = lc.to(dev, non_blocking=True) if lc is not None else None
            with torch.no_grad():
                logits, frac_logits = model(
                    img5d, aux=aux, temporal_coords=tc, location_coords=lc,
                    return_fractions=True)
                pred = logits.argmax(1).to(torch.uint8).cpu().numpy()
                probs = torch.softmax(logits, dim=1).to(
                    torch.float16).cpu().numpy()
                fracs = torch.sigmoid(frac_logits).to(
                    torch.float16).cpu().numpy()
            for j, stem in enumerate(batch["stems"]):
                cpath = sha_dir / f"{stem}.npz"
                np.savez_compressed(
                    cpath, pred=pred[j], probs=probs[j], fracs=fracs[j])
                n_written += 1
            done += len(batch["stems"])
            if log_every and (n_written % log_every < len(batch["stems"])):
                elapsed = time.time() - t0
                rate = n_written / elapsed if elapsed else 0.0
                remaining = total - done
                eta = remaining / rate if rate else float("nan")
                print(f"  [{done}/{total}] {rate:.1f} tiles/s "
                      f"ETA {eta/60:.1f} min", flush=True)

    _write_manifest(
        sha_dir, checkpoint=checkpoint, ckpt_sha=sha, backbone_name=backbone_name,
        num_classes=num_classes, img_size=img_size, n_tiles=total,
        git_sha=_git_sha(), produced_at=produced_at)
    dt = time.time() - t0
    print(f"[infer_tiles] wrote {n_written}, skipped {n_skipped} (already "
          f"cached) in {dt/60:.1f} min → {sha_dir}")
    return sha, (n_written, n_skipped)


def _tile_hw(path) -> tuple[int, int]:
    """(H, W) of a tile's raster, read from the .npz member HEADER only.

    ``np.load(path)[key].shape`` inflates the whole array (e.g. a
    128×512×512 tessera embedding) just to read two ints — pathological
    across thousands of tiles (it turned the size-grouping pre-pass into a
    full-dataset decompression before any GPU work). numpy stores each array
    as a ``.npy`` member whose shape lives in the first ~128-byte header, so
    stream just that header out of the zip and never touch the payload.
    """
    import zipfile
    from numpy.lib import format as npformat
    with zipfile.ZipFile(path) as zf:
        members = set(zf.namelist())
        for key in ("tessera.npy", "spectral.npy", "image.npy"):
            if key in members:
                with zf.open(key) as f:
                    version = npformat.read_magic(f)
                    if version == (1, 0):
                        shape, _f, _dt = npformat.read_array_header_1_0(f)
                    elif version == (2, 0):
                        shape, _f, _dt = npformat.read_array_header_2_0(f)
                    else:
                        shape, _f, _dt = npformat._read_array_header(f, version)
                return int(shape[-2]), int(shape[-1])
    raise KeyError(f"no tessera/spectral/image array in {path}")


def _parse_shard(s: str | None):
    if s is None:
        return None
    i, k = s.split("/")
    i, k = int(i), int(k)
    if not (0 <= i < k):
        raise ValueError(f"--shard i/K needs 0<=i<K, got {s}")
    return (i, k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backbone-name", default=None,
                    help="registry backbone override (tessera_v1, prithvi_600m…)")
    ap.add_argument("--data-dir", required=True, help="tile root (flat *.npz)")
    ap.add_argument("--tile-list", default=None,
                    help="parquet (tile_name col) or txt of tile names; "
                         "default = all *.npz in --data-dir")
    ap.add_argument("--img-size", type=int, default=504)
    ap.add_argument("--num-classes", type=int, default=28)
    ap.add_argument("--cache-dir", default="/cephfs/pred_cache")
    # Defaults sized for large 128-ch tessera embeddings (~130 MB/tile): the
    # DataLoader buffers batch×workers×prefetch tiles, so batch16/workers8 OOMs
    # a 24Gi pod. Raise these only for small-channel tiles with memory headroom.
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default=None,
                    help="i/K: infer only this shard of the tile list")
    ap.add_argument("--log-every", type=int, default=50,
                    help="log tiles/s + ETA every N written tiles")
    ap.add_argument("--allow-missing", action="store_true",
                    help="skip+log tiles in --tile-list absent from --data-dir "
                         "(truth indices may list removed tiles)")
    ap.add_argument("--produced-at", default=None,
                    help="ISO timestamp for MANIFEST provenance (deterministic; "
                         "no wall-clock read in library code)")
    args = ap.parse_args()

    infer_all(
        checkpoint=args.checkpoint, backbone_name=args.backbone_name,
        data_dir=args.data_dir, tile_list=args.tile_list, img_size=args.img_size,
        num_classes=args.num_classes, cache_dir=args.cache_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        device=args.device, shard=_parse_shard(args.shard),
        log_every=args.log_every, produced_at=args.produced_at,
        allow_missing=args.allow_missing)


if __name__ == "__main__":
    main()
