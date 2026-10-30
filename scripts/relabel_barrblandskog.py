#!/usr/bin/env python3
"""Give NMD 113 barrblandskog its unified class 28 in existing label sidecars (IM-023).

An exact shortcut for re-running the canonical chain (build_labels.py
--nmd-version 2023, then distill_forest_labels.py) after NMD2023 113 was
mapped to class 28:

* NMD sidecar: ``label = where(nmd_label_raw == 113 & label == 3, 28, label)``.
  LPIS only overrides agricultural codes and SKS harvest is applied on top of
  the base class, so no other pixel can change.
* Distilled sidecar: ``label = where(new NMD label == 28, 28, distilled label)``.
  The head overrides only where both the original and the head are forest
  types 1-4 (28 is not one) and its prediction does not depend on the label.

Subcommands:

``sample``  deterministic verification tiles (sha256 order of the name).
``verify``  compares the shortcut with the canonical chain run on the sample,
            bit for bit, and exits non-zero on any difference. Two gates:
            the relabelled NMD sidecar equals a canonical rebuild, and
            relabelling a fresh distillation of the old labels equals a fresh
            distillation of the canonical labels (both runs in the same job,
            so the gate tests the shortcut, not GPU reproducibility). Whether
            the fresh distillation reproduces the stored sidecars is reported
            separately, with how many pixels a head rerun changed.
``run``     writes every NMD sidecar and every distilled sidecar to two new
            directories, atomically, stamped with rule and source digest;
            tiles whose outputs are current are skipped. A manifest with the
            totals of the output state is written last, as the done signal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RULE = "im023-nmd113-to-28-v1"
NMD_BARRBLAND, OLD_CLASS, NEW_CLASS = 113, 3, 28
# Sidecar keys the trainer reads (unified_dataset._LABEL_SIDECAR_KEYS).
TRAINER_KEYS = ("label", "parcel_area_ha", "nmd_area_ha")
MANIFEST = "relabel_manifest.json"


def relabel_nmd(label: np.ndarray, raw: np.ndarray) -> np.ndarray:
    if label.shape != raw.shape:
        raise ValueError(f"label {label.shape} and nmd_label_raw {raw.shape} differ")
    return np.where((raw == NMD_BARRBLAND) & (label == OLD_CLASS), NEW_CLASS, label).astype(label.dtype)


def relabel_distilled(distilled: np.ndarray, new_nmd_label: np.ndarray) -> np.ndarray:
    if distilled.shape != new_nmd_label.shape:
        raise ValueError(f"distilled {distilled.shape} and NMD label {new_nmd_label.shape} differ")
    return np.where(new_nmd_label == NEW_CLASS, NEW_CLASS, distilled).astype(distilled.dtype)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def save_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp = path.with_name(path.stem + ".tmp")
    np.savez_compressed(tmp, **payload)  # numpy appends .npz
    os.replace(tmp.with_suffix(".tmp.npz"), path)


def names(directory: Path) -> set[str]:
    return {p.stem for p in directory.glob("*.npz")}


def is_current(out: Path, source: Path) -> bool:
    if not out.is_file():
        return False
    try:
        with np.load(out, allow_pickle=False) as z:
            return (str(z["relabel_rule"]) == RULE
                    and str(z["relabel_source_sha256"]) == digest(source))
    except (KeyError, OSError, ValueError):
        return False


def sample(nmd_dir: Path, distill_dir: Path, with_113: int, without_113: int) -> list[str]:
    """The first distilled tiles in sha256(name) order with and without NMD 113."""
    want = {True: with_113, False: without_113}
    picked: dict[bool, list[str]] = {True: [], False: []}
    for name in sorted(names(distill_dir), key=lambda n: hashlib.sha256(n.encode()).hexdigest()):
        with np.load(nmd_dir / f"{name}.npz", allow_pickle=False) as z:
            has = bool((z["nmd_label_raw"] == NMD_BARRBLAND).any())
        if len(picked[has]) < want[has]:
            picked[has].append(name)
        if all(len(picked[k]) == want[k] for k in want):
            return sorted(picked[True] + picked[False])
    raise ValueError(f"only {len(picked[True])} tiles with and {len(picked[False])} "
                     f"without NMD 113 among the distilled tiles")


def compare(expected: dict, actual: dict, name: str) -> tuple[list[str], list[str]]:
    """(problems, notes) of the shortcut against a canonical sidecar.

    Any value, dtype or shape difference on a shared key, and a trainer key on
    one side only, is a problem. Other keys on one side only are notes.
    """
    problems, notes = [], []
    for key in sorted(expected.keys() | actual.keys()):
        if key in expected and key in actual:
            a, b = np.asarray(expected[key]), np.asarray(actual[key])
            if a.dtype != b.dtype or a.shape != b.shape:
                problems.append(f"{name}: {key} is {b.dtype}{b.shape}, canonical {a.dtype}{a.shape}")
            elif not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
                problems.append(f"{name}: {key} differs in {int((a != b).sum())} cells")
        else:
            side = "canonical" if key in expected else "shortcut"
            (problems if key in TRAINER_KEYS else notes).append(f"{name}: {key} only in {side}")
    return problems, notes


def verify(nmd_in: Path, distill_in: Path, canon_nmd: Path,
           canon_distill_old: Path, canon_distill_new: Path) -> dict:
    """Compare the shortcut with the canonical chain on the tiles in ``canon_distill_new``.

    ``canon_nmd`` holds build_labels.py output for the sample;
    ``canon_distill_old``/``canon_distill_new`` hold distill_forest_labels.py
    output for the sample from ``nmd_in`` and from ``canon_nmd``.
    """
    tiles = sorted(names(canon_distill_new))
    if not tiles:
        raise ValueError(f"no canonical distilled sidecars in {canon_distill_new}")
    nmd_problems, distill_problems, notes = [], [], []
    head_rerun_px = stored_differs_px = 0
    stored_differs_tiles = []
    for name in tiles:
        f = f"{name}.npz"
        old_nmd, canonical_nmd = load(nmd_in / f), load(canon_nmd / f)
        shortcut_nmd = {**old_nmd, "label": relabel_nmd(old_nmd["label"], old_nmd["nmd_label_raw"])}
        problems, extra = compare(canonical_nmd, shortcut_nmd, f"{name} nmd")
        nmd_problems += problems
        notes += extra

        fresh_old, fresh_new = load(canon_distill_old / f), load(canon_distill_new / f)
        shortcut_distilled = {**fresh_old,
                              "label": relabel_distilled(fresh_old["label"], canonical_nmd["label"])}
        problems, extra = compare(fresh_new, shortcut_distilled, f"{name} distilled")
        distill_problems += problems
        notes += extra
        # Where both runs saw the same input label, a difference is the head itself.
        same_input = old_nmd["label"] == canonical_nmd["label"]
        head_rerun_px += int(((fresh_old["label"] != fresh_new["label"]) & same_input).sum())

        stored = load(distill_in / f)["label"]
        if stored.shape != fresh_old["label"].shape:
            raise ValueError(f"{name}: stored distilled label {stored.shape} vs fresh {fresh_old['label'].shape}")
        differs = int((stored != fresh_old["label"]).sum())
        if differs:
            stored_differs_tiles.append(name)
            stored_differs_px += differs
    return {"rule": RULE, "n_tiles": len(tiles), "tiles": tiles,
            "identical": not nmd_problems and not distill_problems,
            "nmd_problems": nmd_problems, "distill_problems": distill_problems,
            "head_rerun_differs_px": head_rerun_px,
            "stored_reproduced": not stored_differs_tiles,
            "stored_differs_px": stored_differs_px,
            "stored_differs_tiles": stored_differs_tiles,
            "notes": notes}


def relabel_tile(pairs: list[tuple[Path, Path]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Write the relabelled sidecars for one tile; (old, new) label per pair.

    ``pairs`` is the NMD (source, output) pair, then optionally the distilled one.
    """
    nmd = load(pairs[0][0])
    new_nmd = relabel_nmd(nmd["label"], nmd["nmd_label_raw"])
    outputs = [(pairs[0], nmd, new_nmd)]
    if len(pairs) == 2:
        distilled = load(pairs[1][0])
        if not np.array_equal(distilled["nmd_label_raw"], nmd["nmd_label_raw"]):
            raise ValueError(f"{pairs[1][0].name}: raw NMD differs between NMD and distilled sidecar")
        outputs.append((pairs[1], distilled, relabel_distilled(distilled["label"], new_nmd)))
    for (source, out), payload, label in outputs:
        save_atomic(out, {**payload, "label": label, "relabel_rule": np.str_(RULE),
                          "relabel_source_sha256": np.str_(digest(source))})
    return [(payload["label"], label) for _, payload, label in outputs]


def run(nmd_in: Path, distill_in: Path, nmd_out: Path, distill_out: Path,
        tile_ids: list[str] | None = None) -> dict:
    sources, targets = {nmd_in.resolve(), distill_in.resolve()}, {nmd_out.resolve(), distill_out.resolve()}
    if len(targets) != 2 or targets & sources:
        raise ValueError("outputs must be two new directories, never an input directory")
    nmd_names, distill_names = names(nmd_in), names(distill_in)
    orphans = sorted(distill_names - nmd_names)
    if orphans:
        raise ValueError(f"{len(orphans)} distilled sidecars without an NMD sidecar, e.g. {orphans[:5]}")
    if tile_ids:
        missing = sorted(set(tile_ids) - nmd_names)
        if missing:
            raise ValueError(f"requested tiles without an NMD sidecar: {missing}")
        nmd_names = set(tile_ids)
    if not nmd_names:
        raise ValueError(f"no NMD sidecars in {nmd_in}")
    for directory in targets:
        directory.mkdir(parents=True, exist_ok=True)

    totals = dict.fromkeys(("written", "skipped_current", "nmd_tiles", "distilled_tiles",
                            "nmd_changed", "distilled_changed", "class_28_pixels"), 0)
    for name in sorted(nmd_names):
        pairs = [(nmd_in / f"{name}.npz", nmd_out / f"{name}.npz")]
        if name in distill_names:
            pairs.append((distill_in / f"{name}.npz", distill_out / f"{name}.npz"))
        if all(is_current(out, source) for source, out in pairs):
            totals["skipped_current"] += 1
            labels = [(load(source)["label"], load(out)["label"]) for source, out in pairs]
        else:
            totals["written"] += 1
            labels = relabel_tile(pairs)
        (old_nmd, new_nmd), *distilled = labels
        totals["nmd_tiles"] += 1
        totals["nmd_changed"] += int((old_nmd != new_nmd).sum())
        totals["class_28_pixels"] += int((new_nmd == NEW_CLASS).sum())
        for old, new in distilled:
            totals["distilled_tiles"] += 1
            totals["distilled_changed"] += int((old != new).sum())

    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True).stdout.strip() or None
    manifest = {"rule": RULE, "git_sha": git_sha,
                "inputs": {"nmd": str(nmd_in), "distilled": str(distill_in)},
                "outputs": {"nmd": str(nmd_out), "distilled": str(distill_out)}, **totals}
    text = json.dumps(manifest, indent=2) + "\n"
    for directory in (nmd_out, distill_out):
        (directory / MANIFEST).write_text(text)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("sample", help="deterministic verification tiles")
    sp.add_argument("--nmd-in", type=Path, required=True)
    sp.add_argument("--distill-in", type=Path, required=True)
    sp.add_argument("--with-113", type=int, default=12)
    sp.add_argument("--without-113", type=int, default=3)

    vp = sub.add_parser("verify", help="shortcut against the canonical chain, bit for bit")
    vp.add_argument("--nmd-in", type=Path, required=True)
    vp.add_argument("--distill-in", type=Path, required=True)
    vp.add_argument("--canon-nmd", type=Path, required=True, help="build_labels.py output")
    vp.add_argument("--canon-distill-old", type=Path, required=True,
                    help="distill_forest_labels.py output from --nmd-in")
    vp.add_argument("--canon-distill-new", type=Path, required=True,
                    help="distill_forest_labels.py output from --canon-nmd")
    vp.add_argument("--report", type=Path, required=True)

    rp = sub.add_parser("run", help="write the relabelled sidecars")
    rp.add_argument("--nmd-in", type=Path, required=True)
    rp.add_argument("--distill-in", type=Path, required=True)
    rp.add_argument("--nmd-out", type=Path, required=True)
    rp.add_argument("--distill-out", type=Path, required=True)
    rp.add_argument("--tile-ids", nargs="+")
    args = ap.parse_args(argv)

    if args.command == "sample":
        print(" ".join(sample(args.nmd_in, args.distill_in, args.with_113, args.without_113)))
        return 0
    if args.command == "verify":
        report = verify(args.nmd_in, args.distill_in, args.canon_nmd,
                        args.canon_distill_old, args.canon_distill_new)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        summary = {k: report[k] for k in ("n_tiles", "identical", "head_rerun_differs_px",
                                          "stored_reproduced", "stored_differs_px")}
        print(json.dumps(summary | {"problems": (report["nmd_problems"] + report["distill_problems"])[:20],
                                    "notes": report["notes"][:10]}))
        return 0 if report["identical"] else 1
    manifest = run(args.nmd_in, args.distill_in, args.nmd_out, args.distill_out, args.tile_ids)
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
