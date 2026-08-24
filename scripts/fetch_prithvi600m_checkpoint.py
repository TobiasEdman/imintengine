#!/usr/bin/env python3
"""Fetch and atomically publish the exact IBM/NASA Prithvi 600M-TL file."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

if __package__:
    from scripts.era5_smoke_provenance import verify_foundation_checkpoint
else:
    from era5_smoke_provenance import verify_foundation_checkpoint


REPOSITORY = "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL"
FILENAME = "Prithvi_EO_V2_600M_TL.pt"
REVISION = "f4c19741895193f6eb6ec16748550fb730860aff"


def _publish_from_cache(source: Path, target: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".publish",
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as read_handle, os.fdopen(
            descriptor, "wb"
        ) as write_handle:
            shutil.copyfileobj(read_handle, write_handle, 8 * 1024 * 1024)
            write_handle.flush()
            os.fsync(write_handle.fileno())
        verify_foundation_checkpoint(temporary)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        if temporary.exists():
            temporary.unlink()
        raise


def fetch_checkpoint(target: Path, cache_dir: Path) -> dict:
    """Return verified identity, downloading once under an interprocess lock."""
    target.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f".{target.name}.lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if target.exists():
            identity = verify_foundation_checkpoint(target)
            return {**identity, "status": "existing", "path": str(target)}
        downloaded = Path(hf_hub_download(
            repo_id=REPOSITORY,
            filename=FILENAME,
            revision=REVISION,
            cache_dir=cache_dir,
        ))
        verify_foundation_checkpoint(downloaded)
        _publish_from_cache(downloaded, target)
        identity = verify_foundation_checkpoint(target)
        return {**identity, "status": "downloaded", "path": str(target)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(fetch_checkpoint(args.target, args.cache_dir), sort_keys=True))


if __name__ == "__main__":
    main()
