#!/usr/bin/env python3
"""Freeze the LUCAS crop distill/holdout split — ONCE, before any training.

LUCAS is the ladder's independent cross-validator ("never trained on by
any rung"). Distilling crop type from it burns that property unless the
data is split FIRST: a grouped-by-tile 70/30 freeze [user-approved
2026-08-31] keeps a holdout side that remains untouched-by-training, so
the cross-check survives as "LUCAS holdout never trained on".

Non-negotiables encoded here:
- **Grouped by tile** — points on one tile share context; a point-level
  split would leak tile context across sides (same isolation argument as
  the NFI head's grouped split).
- **The pre-existing index split is honoured**: the L1 index carries a
  'test' side (71 points) frozen by an earlier experiment; its tiles are
  FORCED into our holdout so no prior freeze leaks into distill-train.
- **Holdout must cover all 11 crop classes** (it is the future validator;
  a class absent there is a class we can never score). Seeded retry search, like the NFI grouped_split.
- Same physical pinning as the NFI set: tiles must carry s1_vv_vh (the
  SAR-cohort intersection) and points must sit inside every column's
  crop window (row/col in [off, off+min_img)).

Outputs (PVC):
- ``lucas_crop_distill_index.parquet`` — the 70% side, extract-ready
  (tile_name/tile_path/row/col/unified_class/point_id).
- ``lucas_crop_split.json`` — provenance + the pinned plot list with
  ``key_cols`` so consumers verify the exact-match guard, + holdout tile
  list (points NOT enumerated here; the validator reads the index).

    python3 scripts/build_lucas_crop_split.py \
        --lucas-index /cephfs/lucas/lucas_tile_index.parquet \
        --data-dir /cephfs/unified_v2_512 \
        --out-dir /cephfs/distill
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_pinned_plot_set import (npz_key_names, npz_version_ok,
                                   REQUIRED_KEYS, _crop_offset)

CROP_CLASSES = tuple(range(11, 22))  # vete..majs, unified schema v5
SEED = 42
HOLDOUT_FRAC = 0.30
MIN_HOLDOUT_PER_CLASS = 5

INDEX_NAME = "lucas_crop_distill_index.parquet"
SPLIT_NAME = "lucas_crop_split.json"
MANIFEST_NAME = "lucas_crop_split.MANIFEST.json"
LOCK_NAME = ".lucas_crop_split.lock"
# What extract_plot_features actually reads from the index — the frozen
# parquet must carry every one of these to be consumable.
EXTRACT_COLUMNS = ("tile_name", "tile_path", "row", "col",
                   "unified_class", "point_id")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _publish(tmp_payload_writer, dest: Path) -> str:
    """Write via a pid-suffixed temp file + atomic rename; return sha256.

    No reader can ever observe a half-written artifact, and two writers
    cannot interleave into one file.
    """
    tmp = dest.with_name(f"{dest.name}.tmp{os.getpid()}")
    tmp_payload_writer(tmp)
    digest = _sha256(tmp)
    os.replace(tmp, dest)
    return digest


def required_npz_keys() -> tuple[str, ...]:
    """The TRUE six-column input intersection, derived — never hand-listed.

    REQUIRED_KEYS is the SAR base, but a column may filter on more:
    tessera's dataset drops embedding-less tiles at init. A split frozen
    on a subset strands the stricter column — its extract drops the
    unqualified tiles' points and the pinned-OOF merge aborts the Job.
    """
    from gen_ladder_manifests import DISTILL

    extra = {k for cfg in DISTILL.values() for k in cfg.get("require_keys", ())}
    return tuple(sorted(set(REQUIRED_KEYS) | extra))


def tile_qualifies(path: Path, keys: tuple[str, ...]) -> bool | None:
    """True/False qualification; None = unreadable (caller must abort)."""
    names = npz_key_names(path)
    if names is None:
        return None
    return all(k in names for k in keys) and npz_version_ok(path, keys)


def freeze_state(out_dir: Path) -> tuple[str, str]:
    """('frozen'|'corrupt'|'partial'|'absent', detail). Split is AUTHORITATIVE.

    The k8s Job is deletable state (TTL 48 h) but the split must be frozen
    ONCE: a re-run after the PVC changed would move previously trained
    tiles into holdout and silently burn the holdout's never-trained-on
    property.

    The MANIFEST is the integrity-bearing commit marker, published LAST:
    its presence claims a completed freeze and it binds both artifacts by
    content hash. Existence checks alone accepted a mixed pair from two
    racing builders — and a truncated parquet — as 'frozen'.
    """
    manifest_p = out_dir / MANIFEST_NAME
    if manifest_p.exists():
        try:
            m = json.loads(manifest_p.read_text())
            artifacts = m["artifacts"]
        except (ValueError, KeyError, OSError) as exc:
            return "corrupt", f"unreadable {MANIFEST_NAME}: {exc}"
        for name in (INDEX_NAME, SPLIT_NAME):
            p = out_dir / name
            if name not in artifacts:
                return "corrupt", f"{MANIFEST_NAME} lacks a hash for {name}"
            if not p.exists():
                return "corrupt", f"{name} missing but {MANIFEST_NAME} claims it"
            got = _sha256(p)
            if got != artifacts[name]:
                return "corrupt", (f"{name} sha256 {got[:12]}… does not match "
                                   f"{MANIFEST_NAME} ({artifacts[name][:12]}…)")
        problem = _validate_frozen_semantics(out_dir, m)
        if problem:
            return "corrupt", problem
        return "frozen", f"validated against {MANIFEST_NAME}"
    if (out_dir / INDEX_NAME).exists() or (out_dir / SPLIT_NAME).exists():
        return "partial", "artifact(s) present without a manifest — interrupted freeze"
    return "absent", ""


def _validate_frozen_semantics(out_dir: Path, manifest: dict) -> str | None:
    """Cross-artifact consistency — hashes alone only bind BYTES to the
    marker. An attacker-free but corrupted freeze (tampered parquet with a
    refreshed marker hash, a schema drift, a duplicate key) passes the
    hash check while the JSON and parquet disagree on which plots exist.
    Returns a problem description, or None if consistent.
    """
    split = json.loads((out_dir / SPLIT_NAME).read_text())
    for field in ("key_cols", "plots", "n_distill", "n_holdout",
                  "holdout_tiles", "required_keys", "truth_col"):
        if field not in split:
            return f"{SPLIT_NAME} lacks required field '{field}'"
    if split["key_cols"] != ["tile_name", "point_id"]:
        return f"unexpected key_cols {split['key_cols']}"
    plots = split["plots"]
    if not plots:
        return f"{SPLIT_NAME} has an empty plot list"
    json_keys = {(str(p["tile_name"]), int(p["point_id"])) for p in plots}
    if len(json_keys) != len(plots):
        return f"duplicate plot keys in {SPLIT_NAME}"
    if split["n_distill"] != len(plots):
        return (f"{SPLIT_NAME} n_distill={split['n_distill']} "
                f"!= len(plots)={len(plots)}")
    if manifest.get("n_distill") != len(plots):
        return (f"{MANIFEST_NAME} n_distill={manifest.get('n_distill')} "
                f"!= {SPLIT_NAME} plots={len(plots)}")
    if manifest.get("n_holdout") != split["n_holdout"]:
        return (f"{MANIFEST_NAME} n_holdout={manifest.get('n_holdout')} "
                f"!= {SPLIT_NAME} n_holdout={split['n_holdout']}")
    df = pd.read_parquet(out_dir / INDEX_NAME)
    # The index must be EXTRACT-READY, not merely keyed: a hash-consistent
    # parquet reduced to the key columns would freeze fine and then kill
    # every crop-distill job at extraction (Codex round 4).
    missing_cols = [c for c in EXTRACT_COLUMNS if c not in df.columns]
    if missing_cols:
        return f"{INDEX_NAME} lacks extract columns {missing_cols}"
    pq_keys = {(str(t), int(p))
               for t, p in df[["tile_name", "point_id"]].itertuples(index=False)}
    if len(pq_keys) != len(df):
        return f"duplicate keys in {INDEX_NAME}"
    if pq_keys != json_keys:
        return (f"key sets disagree: {SPLIT_NAME} has {len(json_keys)}, "
                f"{INDEX_NAME} has {len(pq_keys)} "
                f"({len(json_keys ^ pq_keys)} differing)")
    return None


def acquire_lock(out_dir: Path) -> Path:
    """Exclusive cross-process lock (O_EXCL) — two racing builders must
    never both reach the publish step; the loser dies here, loudly."""
    lock = out_dir / LOCK_NAME
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"freeze lock {lock} exists — another builder is running, or a "
            f"crashed one left it behind. Recovery: verify no "
            f"ladder-lucas-crop-split pod is active, then delete the lock "
            f"and re-run.")
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"pid": os.getpid(),
                             "host": socket.gethostname(),
                             "started": datetime.now(timezone.utc).isoformat()}))
    return lock


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lucas-index", help="required to build")
    ap.add_argument("--data-dir", help="required to build")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--verify", action="store_true",
                    help="validate an existing freeze (hashes + cross-"
                         "artifact consistency) and exit — 0 iff frozen-"
                         "valid. Consumers run this BEFORE extraction so "
                         "they can never read a publish window or a "
                         "crash-left pair.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--git-sha", default=None,
                    help="full commit SHA of the producing checkout; "
                         "recorded in the MANIFEST (the k8s job passes it)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.verify:
        state, detail = freeze_state(out_dir)
        if state == "frozen":
            print(f"freeze VALID: {detail}")
            return
        raise SystemExit(f"freeze INVALID ({state}): {detail}")
    if not args.lucas_index or not args.data_dir:
        ap.error("--lucas-index and --data-dir are required to build")

    out_dir.mkdir(parents=True, exist_ok=True)
    _guard(out_dir)
    lock = acquire_lock(out_dir)
    try:
        # Authoritative re-check UNDER the lock: the pre-lock guard is
        # advisory only — a racing builder may have published between the
        # check and the lock acquisition.
        _guard(out_dir)
        _build(args, out_dir)
    finally:
        lock.unlink()


def _guard(out_dir: Path) -> None:
    state, detail = freeze_state(out_dir)
    if state == "frozen":
        # Idempotent no-op, exit 0: a re-applied Job must not fail, and it
        # must NOT re-freeze. Re-freezing is a deliberate manual act.
        print(f"split already frozen in {out_dir} ({detail}) — refusing to "
              f"overwrite. To re-freeze deliberately: verify nothing has "
              f"trained on the current split, then delete {INDEX_NAME}, "
              f"{SPLIT_NAME} and {MANIFEST_NAME}, and re-run.")
        raise SystemExit(0)
    if state == "corrupt":
        raise SystemExit(
            f"CORRUPT freeze in {out_dir}: {detail}. The artifacts do not "
            f"match their commit marker — do NOT train on them. Recovery: "
            f"confirm nothing has trained on this split, delete "
            f"{INDEX_NAME}, {SPLIT_NAME} and {MANIFEST_NAME}, re-run.")
    if state == "partial":
        raise SystemExit(
            f"PARTIAL freeze in {out_dir}: {detail}; neither side is "
            f"trustworthy. Recovery: confirm nothing has trained on it, "
            f"delete the surviving file(s), re-run.")


def _build(args, out_dir: Path) -> None:
    from gen_ladder_manifests import DISTILL

    df = pd.read_parquet(args.lucas_index)
    crops = df[df["unified_class"].isin(CROP_CLASSES)].copy()
    print(f"crop points: {len(crops)}/{len(df)} on "
          f"{crops['tile_name'].nunique()} tiles")

    # Crop-window intersection — identical arithmetic to the NFI pinned set.
    min_img = min(cfg["img_size"] for cfg in DISTILL.values())
    off = _crop_offset(512, min_img)
    in_win = ((crops["row"] >= off) & (crops["row"] < off + min_img)
              & (crops["col"] >= off) & (crops["col"] < off + min_img))
    print(f"crop window [{off}, {off + min_img}): "
          f"{int((~in_win).sum())} border points excluded")
    crops = crops[in_win]

    # Tile qualification on the FULL six-column intersection (unreadable
    # aborts, as in the NFI set). REQUIRED_KEYS alone strands tessera.
    req_keys = required_npz_keys()
    print(f"qualifying on npz keys: {req_keys}")
    data_dir = Path(args.data_dir)
    qual: dict[str, bool] = {}
    unreadable: list[str] = []
    for name in crops["tile_name"].unique():
        p = data_dir / f"{name}.npz"
        if not p.exists():
            qual[name] = False
            continue
        ok = tile_qualifies(p, req_keys)
        if ok is None:
            unreadable.append(name)
        else:
            qual[name] = ok
    if unreadable:
        raise SystemExit(
            f"{len(unreadable)} unreadable tiles (first {unreadable[:5]}) — "
            f"no split is frozen on a degraded PVC.")
    crops = crops[crops["tile_name"].map(qual).fillna(False)]
    print(f"after key/window/existence pinning: {len(crops)} points on "
          f"{crops['tile_name'].nunique()} tiles")

    # The L1 index's own 'test' side is an earlier freeze — its tiles go
    # to OUR holdout unconditionally.
    forced_holdout = set(df.loc[df.get("split", "") == "test", "tile_name"])

    forced_present = forced_holdout & set(crops["tile_name"])
    tiles = np.array(sorted(set(crops["tile_name"]) - forced_holdout))
    # The holdout fraction is of ALL qualified tiles — forced ones included.
    # Computing it on the already-forced-reduced pool undershot the target
    # by HOLDOUT_FRAC × n_forced tiles (Codex round 4).
    n_total = len(tiles) + len(forced_present)
    n_hold = max(0, int(round(n_total * HOLDOUT_FRAC)) - len(forced_present))
    rng_base = args.seed
    best = None
    for trial in range(50):
        rng = np.random.default_rng(rng_base + trial)
        hold_tiles = set(rng.choice(tiles, size=n_hold, replace=False))
        hold_tiles |= forced_present
        hold = crops[crops["tile_name"].isin(hold_tiles)]
        support = hold["unified_class"].value_counts()
        cover = sum(1 for c in CROP_CLASSES
                    if support.get(c, 0) >= MIN_HOLDOUT_PER_CLASS)
        if best is None or cover > best[0]:
            best = (cover, trial, hold_tiles)
        if cover == len(CROP_CLASSES):
            break
    cover, trial, hold_tiles = best
    if cover < len(CROP_CLASSES):
        raise SystemExit(
            f"no seed in 50 trials gave every crop class >= "
            f"{MIN_HOLDOUT_PER_CLASS} holdout points (best {cover}/11) — "
            f"loosen MIN_HOLDOUT_PER_CLASS deliberately, do not ship a "
            f"validator that cannot score a class.")
    if trial:
        print(f"  note: seed+{trial} used for full holdout class coverage")

    hold = crops[crops["tile_name"].isin(hold_tiles)]
    dist = crops[~crops["tile_name"].isin(hold_tiles)]
    assert not (set(dist.tile_name) & set(hold.tile_name)), "tile leak"
    print(f"distill: {len(dist)} points / {dist.tile_name.nunique()} tiles; "
          f"holdout: {len(hold)} points / {hold.tile_name.nunique()} tiles")
    print("holdout class support:",
          hold["unified_class"].value_counts().sort_index().to_dict())

    dist = dist.sort_values(["tile_name", "point_id"]).reset_index(drop=True)
    split_doc = {
        "seed": args.seed, "trial_offset": trial,
        "holdout_frac": HOLDOUT_FRAC,
        "min_holdout_per_class": MIN_HOLDOUT_PER_CLASS,
        "required_keys": list(req_keys),
        "crop_window": [off, off + min_img],
        "key_cols": ["tile_name", "point_id"],
        "truth_col": "unified_class",
        "n_distill": int(len(dist)),
        "n_holdout": int(len(hold)),
        "holdout_tiles": sorted(hold_tiles),
        "forced_holdout_tiles_from_prior_split": sorted(
            forced_holdout & set(crops["tile_name"])),
        "plots": [
            {"tile_name": str(t), "point_id": int(p)}
            for t, p in dist[["tile_name", "point_id"]].itertuples(index=False)
        ],
    }

    # Publish order is load-bearing: artifacts first (each atomically),
    # the hash-bearing MANIFEST last. A crash at any point leaves either
    # 'absent' or 'partial' — never a state that validates as frozen.
    hashes = {
        INDEX_NAME: _publish(lambda p: dist.to_parquet(p), out_dir / INDEX_NAME),
        SPLIT_NAME: _publish(
            lambda p: p.write_text(json.dumps(split_doc, indent=1)),
            out_dir / SPLIT_NAME),
    }
    _publish(lambda p: p.write_text(json.dumps({
        "git_sha": args.git_sha,
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "run_args": {"lucas_index": args.lucas_index,
                     "data_dir": args.data_dir, "seed": args.seed},
        "artifacts": hashes,
        "n_distill": int(len(dist)),
        "n_holdout": int(len(hold)),
    }, indent=1)), out_dir / MANIFEST_NAME)
    print(f"wrote {out_dir}/{INDEX_NAME} + {SPLIT_NAME} + {MANIFEST_NAME}")


if __name__ == "__main__":
    main()
