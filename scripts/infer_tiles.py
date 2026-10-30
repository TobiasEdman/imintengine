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
SINGLE family-routed forward (``return_fractions=True``) that yields both outputs when
a fraction head exists, or class logits alone otherwise — no second pass.
A present fraction head does not establish supervision; campaign scoring must
check training provenance separately before using fraction predictions.

Cache layout: ``<cache-dir>/<contract_sha256>/<tile_stem>.npz`` with
``pred`` (uint8, H×W), ``probs`` (float16, C×H×W), ``fracs`` (float16, 4×H×W),
with ``fracs`` omitted when the model has no fraction head,
plus an immutable MANIFEST.json containing the full recipe/input inventory.
Each output has a hash-bound JSON receipt. Cache hits require valid receipts,
content and shapes; changed inputs/code/options select a new cache directory.

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
from scripts import prediction_cache as cache  # noqa: E402


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
            ["git", "rev-parse", "HEAD"],
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
                 num_frames=None, tile_identities=None):
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
        self.tile_identities = tile_identities or {}
        self._infcmp = _load_infcmp()

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, i):
        tile_path = self.tile_paths[i]
        # Build on CPU (device="cpu") — the DataLoader worker has no GPU;
        # tensors are moved to the GPU in the main loop after collation.
        identity = self.tile_identities.get(Path(tile_path).stem)
        auth = ({
            "expected_tile_sha256": identity["sha256"],
            "expected_tile_size": identity["size_bytes"],
        } if identity else {})
        inp = self._infcmp._build_inference_inputs(
            tile_path, self._torch.device("cpu"), self.img_size,
            self.aux_channel_names, family=self.family,
            num_frames=self.num_frames, **auth)
        try:
            def unbatch(value):
                return value.squeeze(0) if value is not None else None

            routed_batch = inp.get("batch")
            return {
                "img5d": unbatch(inp["img5d"]),
                "batch": (
                    {key: unbatch(value) for key, value in routed_batch.items()}
                    if routed_batch is not None else None
                ),
                "family": self.family,
                "aux": unbatch(inp["aux"]),
                "temporal_coords": unbatch(inp["temporal_coords"]),
                "location_coords": unbatch(inp["location_coords"]),
                "stem": Path(tile_path).stem,
                "crop_sz": int(inp["crop_sz"]),
            }
        finally:
            self._infcmp._close_inference_inputs(inp)


def _collate(batch):
    """Stack one family's tensors, preserving routed inputs and absent coords."""
    import torch

    def stack_optional(values, label):
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"inconsistent {label} within inference batch")
        return torch.stack(values, dim=0)

    families = {item["family"] for item in batch}
    if len(families) != 1:
        raise ValueError("cannot collate different model families")
    routed = [item["batch"] for item in batch]
    model_batch = None
    if any(value is not None for value in routed):
        if any(value is None for value in routed):
            raise ValueError("inconsistent routed inputs within inference batch")
        keys = set(routed[0])
        if any(set(value) != keys for value in routed):
            raise ValueError("inconsistent routed keys within inference batch")
        model_batch = {
            key: torch.stack([value[key] for value in routed], dim=0)
            for key in sorted(keys)
        }
    return {
        **{
            key: stack_optional([item[key] for item in batch], key)
            for key in ("img5d", "aux", "temporal_coords", "location_coords")
        },
        "batch": model_batch,
        "family": next(iter(families)),
        "stems": [item["stem"] for item in batch],
        "crop_sz": [item["crop_sz"] for item in batch],
    }


def infer_all(
    *, checkpoint, backbone_name, data_dir, tile_list, img_size, num_classes,
    cache_dir, batch_size, num_workers, device, shard, log_every, produced_at,
    allow_missing=False,
):
    """Load the model ONCE, batch-infer the tile set, write the cache.

    Returns the full prediction-contract key and (n_written, n_skipped).
    """
    import torch
    from torch.utils.data import DataLoader

    infcmp = _load_infcmp()
    dev = torch.device(device)

    checkpoint_identity = cache.file_identity(checkpoint)
    all_tiles = resolve_tile_list(data_dir, tile_list, allow_missing=allow_missing)
    if not all_tiles:
        raise ValueError("empty inference tile inventory")
    tiles = all_tiles
    if shard is not None:
        i, k = shard
        if not 0 <= i < k:
            raise ValueError("invalid shard")
        tiles = all_tiles[i::k]
        print(f"shard {i}/{k}: {len(tiles)} tiles")

    # Thread runtime img_size so clay/croma (no pos_embed, minimal config)
    # build their head at the exact resolution inference feeds — see
    # inference_comparison.load_model docstring.
    model, epoch, miou, native = infcmp.load_model(
        checkpoint, dev, backbone_name=backbone_name, img_size=img_size,
        expected_checkpoint_sha256=checkpoint_identity["sha256"],
        expected_checkpoint_size=checkpoint_identity["size_bytes"])
    family = getattr(getattr(model, "fm_spec", None), "family", "prithvi")
    print(f"  loaded epoch={epoch} ckpt_mIoU={miou} native_img={native} "
          f"family={family}")

    # The checkpoint KNOWS its aux set — reconstructing it from defaults is
    # how the ladder eval died on its first run: every ladder model is
    # 11-aux (markfukt on) and terramind is 13-aux (ΔSAR), while a None
    # here builds the canonical 10 and the aux_proj conv rejects the
    # tensor at the first forward. Same rule as extract_plot_features.
    # load_model already read the config and derived the aux count from the
    # aux conv's in-channel dim, so read both off the model rather than
    # re-opening the checkpoint (a second torch.load peaks at ~2x its size).
    aux_names = getattr(model, "ck_cfg", {}).get("enabled_aux_names")
    aux_names = list(aux_names) if aux_names else None
    n_aux = getattr(model, "n_aux_channels", None)
    if aux_names:
        # The count is authoritative (read off the conv); the names are what
        # the trainer recorded. A disagreement means the config does not
        # describe this checkpoint, and feeding it would build a stack of the
        # right depth from the wrong channels — right shape, wrong content,
        # no exception. Refuse instead.
        if n_aux is not None and len(aux_names) != n_aux:
            raise ValueError(
                f"{checkpoint}: config lists {len(aux_names)} aux names "
                f"{aux_names} but the aux conv takes {n_aux} channels")
        print(f"  aux from checkpoint config: {len(aux_names)} channels "
              f"{aux_names}")

    # load_model stamps the model with its trained frame count (single-frame
    # ladder columns = 1, Prithvi-600M = 4). Thread it exactly as
    # run_inference does — without it the dataset defaults to the 4-frame
    # branch and a 1-frame model's temporal_encoding rejects the 4x token
    # grid (15376 vs 3844), which killed the first ladder-eval wave.
    model_num_frames = getattr(model, "num_frames", None)
    has_fraction_head = infcmp.model_has_frac_head(model)
    inventory = {}
    for path in all_tiles:
        identity = cache.file_identity(path)
        metadata = cache.tile_metadata(path, identity, family=family)
        inventory[path.stem] = {
            "input": identity,
            "crop_size": min(img_size, metadata["height"], metadata["width"]),
        }
    if len(inventory) != len(all_tiles):
        raise ValueError("duplicate tile stems in cache inventory")
    contract = {
        "schema": cache.SCHEMA, "checkpoint": checkpoint_identity,
        "source": cache.source_identity(Path(__file__).resolve().parents[1], _git_sha()),
        "runtime": cache.runtime_identity(torch, dev),
        "options": {
            "backbone_name": backbone_name, "img_size": img_size,
            "num_classes": num_classes, "batch_size": batch_size,
            "storage": "uint8-pred-float16-probs-fracs",
            "tile_order": [p.stem for p in all_tiles],
            "num_shards": shard[1] if shard is not None else 1,
            "allow_missing": allow_missing,
        },
        "model": {
            "family": family, "num_frames": model_num_frames,
            "aux_names": aux_names, "n_aux_channels": n_aux,
            # Capability only: supervision eligibility is a separate scoring gate.
            "fraction_head": has_fraction_head,
        },
        "tiles": inventory,
    }
    sha_dir = cache.initialize(cache_dir, contract)
    sha = sha_dir.name
    print(f"[infer_tiles] cache key={sha} cache={sha_dir}")
    ds = TileInferenceDataset(
        tiles, family, img_size, aux_names, num_frames=model_num_frames,
        tile_identities={stem: entry["input"] for stem, entry in inventory.items()})
    size_of = {}
    for i, path in enumerate(tiles):
        size_of.setdefault(inventory[path.stem]["crop_size"], []).append(i)

    n_written = n_skipped = 0
    t0 = time.time()
    done = 0
    total = len(tiles)

    for cs, idxs in sorted(size_of.items()):
        # Plan immutable batch membership BEFORE filtering hits. Recompute
        # a whole partial batch, publishing only its missing members, so resume
        # preserves the original numerical batch context.
        hits = {
            i for i in idxs
            if cache.has_prediction(sha_dir, contract, Path(ds.tile_paths[i]).stem)
        }
        hit_stems = {Path(ds.tile_paths[i]).stem for i in hits}
        planned = [idxs[j:j + batch_size] for j in range(0, len(idxs), batch_size)]
        pending_batches = [indices for indices in planned if not set(indices) <= hits]
        n_skipped += len(hits)
        done = n_written + n_skipped
        if not pending_batches:
            continue
        loader = DataLoader(
            ds, batch_sampler=pending_batches, num_workers=num_workers,
            pin_memory=(dev.type == "cuda"), collate_fn=_collate)
        for batch in loader:
            inputs = {
                key: (
                    batch[key].to(dev, non_blocking=True)
                    if batch[key] is not None else None
                )
                for key in ("img5d", "aux", "temporal_coords", "location_coords")
            }
            # family_forward owns device transfer and normalization of the
            # routed stacks, exactly as in direct inference and training.
            inputs.update(batch=batch["batch"], family=batch["family"])
            with torch.no_grad():
                output = infcmp._forward_from_inputs(
                    model, inputs, dev, return_fractions=has_fraction_head)
                if has_fraction_head:
                    logits, frac_logits = output
                    if frac_logits is None:
                        raise ValueError("fraction head returned no fraction logits")
                    fracs = torch.sigmoid(frac_logits).to(
                        torch.float16).cpu().numpy()
                else:
                    logits, fracs = output, None
                pred = logits.argmax(1).to(torch.uint8).cpu().numpy()
                probs = torch.softmax(logits, dim=1).to(
                    torch.float16).cpu().numpy()
            for j, stem in enumerate(batch["stems"]):
                if stem in hit_stems:
                    continue
                arrays = {"pred": pred[j], "probs": probs[j]}
                if fracs is not None:
                    arrays["fracs"] = fracs[j]
                if cache.publish_prediction(sha_dir, contract, stem, arrays, produced_at):
                    n_written += 1
                else:
                    n_skipped += 1
            done = n_written + n_skipped
            if log_every and (n_written % log_every < len(batch["stems"])):
                elapsed = time.time() - t0
                rate = n_written / elapsed if elapsed else 0.0
                remaining = total - done
                eta = remaining / rate if rate else float("nan")
                print(f"  [{done}/{total}] {rate:.1f} tiles/s "
                      f"ETA {eta/60:.1f} min", flush=True)

    for path in tiles:
        if not cache.has_prediction(sha_dir, contract, path.stem):
            raise cache.CacheError(f"missing prediction after inference: {path.stem}")
    dt = time.time() - t0
    print(f"[infer_tiles] wrote {n_written}, skipped {n_skipped} (already "
          f"cached) in {dt/60:.1f} min → {sha_dir}")
    return sha, (n_written, n_skipped)


def _tile_hw(path, *, family=None) -> tuple[int, int]:
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
        keys = (("tessera.npy",) if family == "tessera" else
                ("spectral.npy", "image.npy") if family is not None else
                ("tessera.npy", "spectral.npy", "image.npy"))
        for key in keys:
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
    ap.add_argument("--cache-key-out", help="write the full prediction-contract key here")
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

    key, _counts = infer_all(
        checkpoint=args.checkpoint, backbone_name=args.backbone_name,
        data_dir=args.data_dir, tile_list=args.tile_list, img_size=args.img_size,
        num_classes=args.num_classes, cache_dir=args.cache_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        device=args.device, shard=_parse_shard(args.shard),
        log_every=args.log_every, produced_at=args.produced_at,
        allow_missing=args.allow_missing)
    if args.cache_key_out:
        Path(args.cache_key_out).write_text(key + "\n")


if __name__ == "__main__":
    main()
