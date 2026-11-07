#!/usr/bin/env python3
"""Prepare the IM-023 amendment to the 2026-10-27 NFI freeze, on CPU.

Writes one new manifest; never touches the freeze. The amendment references
the frozen holdout by the original manifest's pinned digest and verifies that
manifest, its parquets and every frozen tile here, before anything is
written. A changed or missing frozen tile refuses the run and every such tile
is listed (Tobias 2026-11-07), as the freeze's own consumers refuse. It pins
the three cells' checkpoints (r4 must be the frozen ``tessera_r4``
checkpoint), their input contracts, the protocol constants, the amendment's
source files and the sealed runtime. No inference, no NMD sampling and no
scoring. The manifest ends in ``awaiting_user_go_no_go``.

Runs under the image's scoring interpreter, as prepare_nfi_holdout does; the
checkpoints are read by nfi_amendment_checkpoints.py under the model one.

    /opt/venvs/scoring/bin/python /opt/imintengine/scripts/prepare_nfi_amendment.py \
        --base-manifest /cephfs/nfi/nfi_holdout_freeze_2026-10-27/manifest.json \
        --checkpoint-main /cephfs/checkpoints/im023/tessera_b28/best_model.pt \
        --checkpoint-control /cephfs/checkpoints/im023/tessera_control/best_model.pt \
        --checkpoint-r4 /cephfs/checkpoints/ladder/tessera_r4/best_model.pt \
        --runtime-image ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:... \
        --runtime-manifest /opt/provenance/runtime.json --source-git-sha SHA \
        --out-dir /cephfs/nfi/im023_amendment_2026-11-06/freeze
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imint.eval.fieldtruth import NFI_KEY, load_frozen_holdout, sha256_file, verify_file_identity
from imint.eval.nfi_amendment import (
    AMENDMENT_CELLS, AMENDMENT_PROTOCOL, AMENDMENT_PROTOCOL_ID, AMENDMENT_SCHEMA,
    AMENDMENT_SOURCE_FILES, BASE_MANIFEST_SHA256, BASE_R4_CELL, CELL_HEAD, NUM_FRACTIONS,
    sealed_runtime, validate_amendment_structure,
)
from prepare_nfi_holdout import file_identity


def checkpoint_metadata(cells: dict, model_python: str) -> dict:
    """Heads and input contracts, read by the sealed model interpreter."""
    child = Path(__file__).resolve().parent / "nfi_amendment_checkpoints.py"
    request = {name: {"checkpoint": cell["checkpoint"]} for name, cell in cells.items()}
    try:
        done = subprocess.run([model_python, str(child)], input=json.dumps(request),
                              capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError("checkpoint metadata preparation failed: " + exc.stderr[-4000:]) from exc
    metadata = json.loads(done.stdout)
    if set(metadata) != set(cells):
        raise ValueError("checkpoint metadata roster mismatch")
    return metadata


def refuse_changed_tiles(tiles: dict) -> None:
    """Refuse if any frozen tile changed or is gone, naming every one, not the first."""
    changed = []
    for name, record in sorted(tiles.items()):
        try:
            verify_file_identity(record)
        except (OSError, ValueError) as exc:
            changed.append(f"{name}: {exc}")
    if changed:
        raise ValueError(f"{len(changed)} of {len(tiles)} frozen tiles changed since the freeze; "
                         "refusing:\n" + "\n".join(changed))


def amendment_cells(checkpoints: dict[str, Path], base: dict, model_python: str) -> dict:
    """Pin each cell; refuse a roster the freeze's tile audit does not cover."""
    if set(checkpoints) != set(AMENDMENT_CELLS):
        raise ValueError("amendment roster differs from the fixed cells")
    reference = base["cells"][BASE_R4_CELL]
    cells = {name: {"checkpoint": file_identity(path), "backbone": reference["backbone"],
                    "img_size": reference["img_size"], "num_classes": AMENDMENT_CELLS[name],
                    "head": CELL_HEAD}
             for name, path in checkpoints.items()}
    if cells["r4"]["checkpoint"]["sha256"] != reference["checkpoint"]["sha256"]:
        raise ValueError(f"r4 checkpoint is not the frozen {BASE_R4_CELL} checkpoint")
    # The freeze audited each tile's stored aux for the union its cells declared.
    audited = {name for cell in base["cells"].values()
               for name in cell["input_contract"]["enabled_aux_names"]}
    for name, metadata in checkpoint_metadata(cells, model_python).items():
        if metadata["num_classes"] != cells[name]["num_classes"] or not metadata["fraction_head"]:
            raise ValueError(f"{name}: checkpoint lacks a {cells[name]['num_classes']}-class head "
                             f"with a {NUM_FRACTIONS}-share fraction head")
        if not set(metadata["input_contract"]["enabled_aux_names"]) <= audited:
            raise ValueError(f"{name}: aux channels outside the freeze's tile audit")
        cells[name]["input_contract"] = metadata["input_contract"]
    if cells["r4"]["input_contract"] != reference["input_contract"]:
        raise ValueError(f"r4 input contract differs from the frozen {BASE_R4_CELL}")
    return cells


def build_manifest(base_path: Path, checkpoints: dict[str, Path], *, runtime_manifest: Path,
                   runtime_image: str, source_git_sha: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    runtime = sealed_runtime(root, runtime_manifest, source_git_sha=source_git_sha,
                             image_ref=runtime_image, environment="scoring")
    holdout, base = load_frozen_holdout(base_path, BASE_MANIFEST_SHA256)
    refuse_changed_tiles(base["tiles"])
    manifest = {
        "schema": AMENDMENT_SCHEMA, "protocol_id": AMENDMENT_PROTOCOL_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base": {"manifest_path": str(base_path.resolve()),
                 "manifest_sha256": base["_manifest_sha256"],
                 "holdout_sha256": base["holdout"]["sha256"],
                 "holdout_observations": len(holdout),
                 "git_sha": base["git_sha"], "runtime_image": base["runtime_image"]},
        "git_sha": source_git_sha, "runtime_image": runtime_image,
        "preparation_runtime": runtime,
        "source_sha256": {name: sha256_file(root / name) for name in sorted(AMENDMENT_SOURCE_FILES)},
        "identity": NFI_KEY,
        "cells": amendment_cells(checkpoints, base,
                                 runtime["environments"]["model"]["python"]["path"]),
        "protocol": AMENDMENT_PROTOCOL,
        "execution": "awaiting_user_go_no_go",
    }
    validate_amendment_structure(manifest)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-manifest", type=Path, required=True,
                    help="the 2026-10-27 freeze manifest; its digest is pinned in code")
    ap.add_argument("--checkpoint-main", type=Path, required=True, help="im023_main (29 classes)")
    ap.add_argument("--checkpoint-control", type=Path, required=True, help="im023_control (28 classes)")
    ap.add_argument("--checkpoint-r4", type=Path, required=True, help="the frozen tessera_r4 checkpoint")
    ap.add_argument("--runtime-image", required=True, help="the sealed image@sha256 the Jobs run")
    ap.add_argument("--runtime-manifest", type=Path, required=True,
                    help="the image's /opt/provenance/runtime.json")
    ap.add_argument("--source-git-sha", required=True, help="the amendment commit baked into the image")
    ap.add_argument("--out-dir", type=Path, required=True, help="new directory; never overwritten")
    args = ap.parse_args()
    if args.out_dir.exists():
        raise ValueError("amendment output already exists; never overwrite it")
    checkpoints = {"im023_main": args.checkpoint_main, "im023_control": args.checkpoint_control,
                   "r4": args.checkpoint_r4}
    manifest = build_manifest(args.base_manifest, checkpoints,
                              runtime_manifest=args.runtime_manifest,
                              runtime_image=args.runtime_image, source_git_sha=args.source_git_sha)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    dest = args.out_dir / "manifest.json"
    with dest.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(dest), "manifest_sha256": sha256_file(dest),
                      "base_manifest_sha256": manifest["base"]["manifest_sha256"],
                      "cells": {n: c["checkpoint"]["sha256"] for n, c in manifest["cells"].items()},
                      "state": manifest["execution"]}, indent=2))


if __name__ == "__main__":
    main()
