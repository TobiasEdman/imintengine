#!/usr/bin/env python3
"""Copy generated artifacts to the separate showcase, recording their hashes.

Only docs/data and docs/showcase are exported. Website source stays owned by
imint-showcase. Use --replace explicitly when updating existing artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ALLOWED_SUFFIXES = {".json", ".geojson", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


def _validate_destination(path: Path, site_root: Path) -> None:
    """Reject symlinks in every destination component before any writes."""
    for component in (path, *path.parents):
        if component == site_root:
            break
        if component.is_symlink():
            raise ValueError(f"Symlink destination escapes showcase ownership: {component}")
    if site_root not in path.resolve().parents:
        raise ValueError(f"Artifact escapes showcase: {path}")


def export_artifacts(source_root: Path, site_root: Path, *, replace: bool = False) -> dict:
    source_root = source_root.resolve()
    site_root = site_root.resolve()
    if source_root == site_root or source_root in site_root.parents or site_root in source_root.parents:
        raise ValueError("Source and showcase must be separate directories")
    if not (site_root / "docs/index.html").is_file():
        raise ValueError("Showcase must contain docs/index.html")
    receipt_path = site_root / "ARTIFACT_IMPORT.json"
    _validate_destination(receipt_path, site_root)
    plan = []
    for folder in ("docs/data", "docs/showcase"):
        for src in sorted((source_root / folder).rglob("*")):
            if not src.is_file() or src.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            if src.is_symlink() or source_root not in src.resolve().parents:
                raise ValueError(f"Artifact escapes source: {src}")
            rel = src.relative_to(source_root)
            dst = site_root / rel
            _validate_destination(dst, site_root)
            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() != digest and not replace:
                raise FileExistsError(f"Use --replace to update {rel}")
            plan.append((src, dst, rel, digest))
    if not plan:
        raise ValueError("No generated artifacts found")
    records = []
    for src, dst, rel, digest in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
            raise OSError(f"Artifact changed while copying: {rel}")
        records.append({"path": str(rel), "sha256": digest})
    receipt = {"source_root": str(source_root), "files": records}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--site-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    receipt = export_artifacts(args.source_root, args.site_root, replace=args.replace)
    print(f"Verified {len(receipt['files'])} exported artifacts")


if __name__ == "__main__":
    main()
