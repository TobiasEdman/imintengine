#!/usr/bin/env python3
"""Relabel a further model's distilled sidecars onto the existing class 28 NMD labels (IM-023).

``relabel_barrblandskog.py run`` writes the relabelled NMD sidecars and one
distilled cohort together. The NMD half does not depend on the model and is
already on disk for every tile (``/cephfs/nmd2023_labels_b28``, written with
the TESSERA cohort). ``run`` for a second model would rewrite those sidecars
and overwrite their manifest with that run's totals. This script reads the NMD
directory and never writes to it:

* the NMD directory's manifest must carry the rule and name the same input;
* for every tile of the cohort, the stored NMD label must equal
  ``relabel_nmd`` of its source, and the raw NMD of the distilled sidecar must
  equal the source's;
* each distilled sidecar is relabelled with ``relabel_distilled`` against that
  label and written atomically, stamped as ``run`` stamps it; current outputs
  are skipped;
* ``relabel_manifest.json`` is written into the distilled directory only, last,
  with the NMD totals of the reused manifest. It is the done signal.

The outputs are the files ``run`` would write for this cohort. The
verification gate (``relabel_barrblandskog.py verify``) is unchanged and runs
before this on the model's own canonical distillation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel_barrblandskog import (MANIFEST, RULE, digest, is_current, load, names,  # noqa: E402
                                   relabel_distilled, relabel_nmd, save_atomic)

#: Totals of the NMD half, model independent, carried over from the reused manifest.
NMD_TOTALS = ("nmd_tiles", "nmd_changed", "class_28_pixels")


def reused_manifest(nmd_in: Path, nmd_b28: Path) -> dict:
    """The NMD directory's manifest, checked against the rule and the input."""
    path = nmd_b28 / MANIFEST
    if not path.is_file():
        raise ValueError(f"{path} missing: the NMD labels are not relabelled")
    manifest = json.loads(path.read_text())
    if manifest["rule"] != RULE:
        raise ValueError(f"{path}: rule {manifest['rule']}, expected {RULE}")
    named = (manifest["inputs"]["nmd"], manifest["outputs"]["nmd"])
    if named != (str(nmd_in), str(nmd_b28)):
        raise ValueError(f"{path} relabelled {named[0]} into {named[1]}, not {nmd_in} into {nmd_b28}")
    return manifest


def stored_nmd_label(nmd_source: dict, nmd_b28_path: Path) -> np.ndarray | None:
    """The relabelled NMD label, if the stored sidecar holds exactly it; else None."""
    if not nmd_b28_path.is_file():
        return None
    expected = relabel_nmd(nmd_source["label"], nmd_source["nmd_label_raw"])
    stored = load(nmd_b28_path)["label"]
    return expected if stored.dtype == expected.dtype and np.array_equal(stored, expected) else None


def run(nmd_in: Path, nmd_b28: Path, distill_in: Path, distill_out: Path) -> dict:
    if distill_out.resolve() in {p.resolve() for p in (nmd_in, nmd_b28, distill_in)}:
        raise ValueError("the output must be a new directory, never an input directory")
    reused = reused_manifest(nmd_in, nmd_b28)
    # The reused manifest names the first model's distillation and its relabelled
    # copy; writing to either would replace that model's labels with this one's.
    if distill_out.resolve() in {Path(reused[side]["distilled"]).resolve() for side in ("inputs", "outputs")}:
        raise ValueError(f"{distill_out} holds the cohort the reused manifest was written for")
    previous = distill_out / MANIFEST
    if previous.is_file() and json.loads(previous.read_text())["inputs"]["distilled"] != str(distill_in):
        raise ValueError(f"{previous} relabelled another distillation than {distill_in}")
    cohort = names(distill_in)
    if not cohort:
        raise ValueError(f"no distilled sidecars in {distill_in}")
    orphans = sorted(cohort - names(nmd_in))
    if orphans:
        raise ValueError(f"{len(orphans)} distilled sidecars without an NMD sidecar, e.g. {orphans[:5]}")
    distill_out.mkdir(parents=True, exist_ok=True)

    totals = dict.fromkeys(("written", "skipped_current", "distilled_tiles", "distilled_changed"), 0)
    stale = []
    for name in sorted(cohort):
        f = f"{name}.npz"
        source, out = distill_in / f, distill_out / f
        nmd = load(nmd_in / f)
        new_nmd = stored_nmd_label(nmd, nmd_b28 / f)
        if new_nmd is None:
            stale.append(name)
            continue
        if is_current(out, source):
            # The stamp covers the distilled source only; the NMD label may have
            # been rebuilt since, so the stored output must still be the rule's.
            old, new = load(source)["label"], load(out)["label"]
            if not np.array_equal(new, relabel_distilled(old, new_nmd)):
                raise ValueError(f"{out.name}: stored output is not the rule applied to the current NMD label")
            totals["skipped_current"] += 1
        else:
            distilled = load(source)
            if not np.array_equal(distilled["nmd_label_raw"], nmd["nmd_label_raw"]):
                raise ValueError(f"{source.name}: raw NMD differs between NMD and distilled sidecar")
            old, new = distilled["label"], relabel_distilled(distilled["label"], new_nmd)
            save_atomic(out, {**distilled, "label": new, "relabel_rule": np.str_(RULE),
                              "relabel_source_sha256": np.str_(digest(source))})
            totals["written"] += 1
        totals["distilled_tiles"] += 1
        totals["distilled_changed"] += int((old != new).sum())
    if stale:
        raise ValueError(f"{len(stale)} NMD sidecars in {nmd_b28} are missing or not the rule applied "
                         f"to {nmd_in}, e.g. {stale[:5]}; no manifest written")

    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True).stdout.strip() or None
    reused_path = nmd_b28 / MANIFEST
    manifest = {"rule": RULE, "git_sha": git_sha,
                "inputs": {"nmd": str(nmd_in), "distilled": str(distill_in)},
                "outputs": {"nmd": str(nmd_b28), "distilled": str(distill_out)},
                "nmd_reused": {"manifest": str(reused_path), "sha256": digest(reused_path),
                               "git_sha": reused["git_sha"]},
                **totals, **{key: reused[key] for key in NMD_TOTALS}}
    # The manifest is the done signal: write it whole or not at all.
    tmp = distill_out / f".{MANIFEST}.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, distill_out / MANIFEST)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nmd-in", type=Path, required=True, help="the NMD sidecars R4 was distilled from")
    ap.add_argument("--nmd-b28", type=Path, required=True,
                    help="their relabelled copy, from relabel_barrblandskog.py run; read only")
    ap.add_argument("--distill-in", type=Path, required=True)
    ap.add_argument("--distill-out", type=Path, required=True)
    args = ap.parse_args(argv)
    print(json.dumps(run(args.nmd_in, args.nmd_b28, args.distill_in, args.distill_out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
