#!/usr/bin/env python3
"""Write the five Skogsstyrelsen/SLU aux channels from the open-data mirror into existing tiles.

The NFI top-up tiles arrived without ``height``, ``volume``, ``basal_area``,
``diameter`` and ``markfukt``: the proxies behind them were retired, so the
fetch skipped them and ``k8s/nfi-topup-enrich-job.yaml`` marked every tile
``aux_complete=0`` (Tobias 2026-10-12, option a: never zero-fill). The open
mirror (``imint.training.skg_open``, IM-022) now holds all five. This pass
reads them with the repo's own fetchers (``scripts/prefetch_aux``) on the
tile's own grid (``resolve_fetch_bbox``) and sets the flags in the same
rewrite, per tile:

* a channel that is absent, or present but all zero/NaN, is read from the
  mirror; a channel already present with data is kept as it is, and is also
  re-read so the record says whether it equals the mirror's current version
  (``kept_equal_to_mirror``);
* a read that the mirror cannot serve (``SkgMirrorMissingError``: county not
  staged, no county covers the window) or that returns no valid pixel is a
  genuine gap: the key stays absent, never a zero array;
* ``aux_provenance`` (JSON list, one entry per pass that changed the tile)
  records what was written, the gaps, the git sha, and which county rasters
  the window reads, with their sha256 and source zip date
  (``skg_open.window_provenance``): the open products are versions of their
  own, documented per tile;
* all five present: ``aux_complete=1`` and ``aux_missing`` /
  ``aux_gap_reason`` removed; otherwise ``aux_complete=0`` with the missing
  channels and a dated reason naming each gap.

The rewrite is atomic (a hidden tmp file that is not ``*.npz``, then rename). Any other error leaves the tile
untouched and makes the run exit 1. The per-tile outcome goes to
``--report`` (on the PVC); the log carries counts only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from imint.training import skg_open  # noqa: E402
from imint.training.tile_bbox import resolve_fetch_bbox  # noqa: E402
from scripts.prefetch_aux import _CHANNEL_FETCHERS  # noqa: E402

AUX = ("height", "volume", "basal_area", "diameter", "markfukt")


def _git_sha() -> str:
    env = os.environ.get("IMINT_GIT_SHA")
    if env:
        return env
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _empty(arr) -> bool:
    """The enrich Job's rule: an image channel with no non-zero, non-NaN pixel carries nothing."""
    return getattr(arr, "ndim", 0) >= 2 and not np.any(np.nan_to_num(arr))


def _gap(exc: skg_open.SkgMirrorMissingError) -> str:
    """The reason without the window's coordinates."""
    msg = str(exc)
    if "is not staged" in msg:
        counties = re.search(r"\[[^]]*\]", msg).group(0)
        return f"county {counties} not staged in the mirror"
    if "no county" in msg:
        return "no county covers the window"
    return "mirror product missing"


def process(path: Path, *, stamp: str, git_sha: str, dry_run: bool) -> dict:
    with np.load(path, allow_pickle=True) as z:
        d = dict(z)
    need = [k for k in AUX if k not in d or _empty(d[k])]
    if not need and int(d.get("aux_complete", 1)) == 1 and "aux_missing" not in d:
        return {"status": "complete"}
    bbox, size = resolve_fetch_bbox(name=path.stem, npz_data=d)
    if bbox is None:
        raise RuntimeError("tile bbox cannot be resolved")
    west, south, east, north = bbox["west"], bbox["south"], bbox["east"], bbox["north"]

    written, gaps = [], {}
    for ch in need:
        try:
            arr = _CHANNEL_FETCHERS[ch](west, south, east, north, size_px=(size, size), cache_dir=None)
        except skg_open.SkgMirrorMissingError as exc:
            gaps[ch] = _gap(exc)
            continue
        if _empty(arr):
            gaps[ch] = "no valid pixel in the mirror window"
            continue
        d[ch] = arr
        written.append(ch)
    missing = [k for k in need if k not in written]
    for k in missing:
        d.pop(k, None)                       # never leave an empty array behind a present key
    kept = {}
    for ch in (k for k in AUX if k not in need):
        try:
            again = _CHANNEL_FETCHERS[ch](west, south, east, north, size_px=(size, size), cache_dir=None)
        except skg_open.SkgMirrorMissingError as exc:
            kept[ch] = _gap(exc)
            continue
        kept[ch] = bool(np.array_equal(np.asarray(d[ch], dtype=np.float32), again, equal_nan=True))

    entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_sha": git_sha,
             "written": written, "gaps": gaps, "kept_equal_to_mirror": kept,
             **skg_open.window_provenance(west, south, east, north, size_px=(size, size))}
    history = json.loads(str(d["aux_provenance"])) if "aux_provenance" in d else []
    d["aux_provenance"] = np.array(json.dumps([*history, entry]))
    if missing:
        d["aux_complete"] = np.array(0)
        d["aux_missing"] = np.array(missing)
        d["aux_gap_reason"] = np.array(f"skg_open_mirror_gap__verified_{stamp}: "
                                       + "; ".join(f"{k}: {gaps[k]}" for k in missing))
    else:
        d["aux_complete"] = np.array(1)
        d.pop("aux_missing", None)
        d.pop("aux_gap_reason", None)
    if not dry_run:
        tmp = path.with_name(f".{path.stem}.auxpass.tmp")   # never *.npz: a killed run leaves no fake tile
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **d)
        os.replace(tmp, path)
    return {"status": "incomplete" if missing else "completed", "written": written, "missing": missing,
            "gaps": gaps, "kept_equal_to_mirror": kept, "counties": entry["counties"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True, help="tile root (flat *.npz)")
    p.add_argument("--report", type=Path, required=True, help="per-tile outcome JSON")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true", help="read and decide, write no tile and no report")
    args = p.parse_args(argv)

    tiles = sorted(args.data_dir.glob("*.npz"))
    if not tiles:
        sys.exit(f"REFUSING: no tiles in {args.data_dir}")
    stamp, git_sha = time.strftime("%Y-%m-%d", time.gmtime()), _git_sha()
    print(f"{args.data_dir}: {len(tiles)} tiles, git {git_sha}, dry_run={args.dry_run}", flush=True)

    def run(path: Path) -> tuple[str, dict]:
        try:
            return path.stem, process(path, stamp=stamp, git_sha=git_sha, dry_run=args.dry_run)
        except Exception as exc:                      # the tile is untouched; reported, run exits 1
            return path.stem, {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = dict(pool.map(run, tiles))

    counts = Counter(r["status"] for r in results.values())
    written = Counter(ch for r in results.values() for ch in r.get("written", []))
    gaps = Counter(f"{ch}: {why}" for r in results.values() for ch, why in r.get("gaps", {}).items())
    kept = Counter(f"{ch}: {eq}" for r in results.values() for ch, eq in r.get("kept_equal_to_mirror", {}).items())
    errors = Counter(r["error"].split(":")[0] for r in results.values() if r["status"] == "error")
    print(f"tiles by outcome: {dict(sorted(counts.items()))}", flush=True)
    print(f"channels written: {dict(sorted(written.items()))}", flush=True)
    for g, n in sorted(gaps.items()):
        print(f"  gap {g}: {n} tiles", flush=True)
    for k, n in sorted(kept.items()):
        print(f"  kept, equal to the mirror {k}: {n} tiles", flush=True)
    for e, n in sorted(errors.items()):
        print(f"  ERROR {e}: {n} tiles (untouched)", flush=True)
    if not args.dry_run:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.report.with_name(args.report.name + ".tmp")
        tmp.write_text(json.dumps({"data_dir": str(args.data_dir), "verified": stamp, "git_sha": git_sha,
                                   "counts": dict(counts), "tiles": results}, indent=1, sort_keys=True))
        os.replace(tmp, args.report)
        print(f"report: {args.report}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
