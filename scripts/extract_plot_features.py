#!/usr/bin/env python3
"""scripts/extract_plot_features.py — dump pre-classifier features at NFI plots.

Hybrid-experiment step 1. Forward a NMD2023-pretrained UPerNet-on-Prithvi seg
model over each tile that carries NFI plots, hook the 256-dim feature map that
FEEDS the final 1×1 classifier (``head.head.2`` in ``imint/fm/upernet.py``), and
sample that 256-vector at every in-crop plot pixel. Also derive the NFI
field-truth forest class per plot (``derive_nfi_forest_class``). The result is a
per-plot feature parquet that ``nfi_head_cv.py`` k-fold-cross-validates a small
head on — testing whether a head trained DIRECTLY on field-truth can beat the
NMD label ceiling for forest-type classification.

Why hook the INPUT of ``head.head.2``? The seg head is
``Sequential(Identity, Dropout2d, Conv2d(256, num_classes, 1))``. The 256-ch
tensor entering that Conv2d is the decoder's fused feature — the representation
the network actually classifies from, before the (NMD-supervised) linear layer.
Training a fresh head on it against NFI truth isolates "is the information there"
from "did NMD supervision surface it".

Crop remap: ``run_inference`` centre-crops each tile to ``img_size`` (504 for the
600M patch-14 backbone on 512 tiles). Plot ``(row, col)`` are in full-tile
coords, so we subtract the crop offset ``off = (tile_h - crop_sz) // 2`` and drop
plots that fall in the discarded border — mirroring ``validate_against_nfi.main``
exactly, so the sampled pixel matches what the model scored.

    python scripts/extract_plot_features.py \
        --checkpoint /data/checkpoints/v8b_nmd2023_long/best_model.pt \
        --plot-index /data/nfi/nfi_index_unified_v2_512.parquet \
        --img-size 504 \
        --out /data/nfi_eval/nfi_plot_features_nmd2023.parquet
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root for imint.*, own dir for sibling scripts. The latter is implicit
# during direct execution but required when tests/tools load this file via an
# importlib spec.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_pinned_plot_set import npz_key_names, npz_version_ok
from validate_against_nfi import crop_offset, derive_nfi_forest_class

# Feature width is the classifier's in_channels — a BACKBONE property
# (256 for the UPerNet families, 128 for tessera), read from the model at
# runtime and carried by the parquet's column count. Never a constant.

_MAX_FAILED_TILE_DETAILS = 20
_MAX_ERROR_CHARS = 500
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _load_inference_comparison():
    """Import ``inference_comparison`` as a module (it lives beside this file).

    Uses spec-from-file-location — the same pattern ``validate_against_nfi``
    uses — so this works whether or not ``scripts/`` is an importable package.
    """
    spec = importlib.util.spec_from_file_location(
        "_infcmp", str(Path(__file__).resolve().parent / "inference_comparison.py"),
    )
    infcmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infcmp)
    return infcmp


def register_preclassifier_hook(model) -> dict:
    """Hook ``head.head.2`` to capture its INPUT (the 256-dim feature map).

    ``head.head`` is ``Sequential(Identity, Dropout2d, Conv2d(256, C, 1))``, so
    ``head.head[2]`` is the final classifier. A forward pre-hook style capture
    via the module's ``forward`` input grabs the (1, 256, H, W) tensor before
    the class projection. Returns a mutable dict whose ``["feat"]`` holds the
    most recent capture (CPU numpy, batch-squeezed → (256, H, W)).

    The hook stores the feature at classifier resolution (pre-upsample). The
    logits are bilinearly upsampled to input resolution AFTER the head, so to
    keep plot (row,col) — which are input-resolution pixels — aligned with the
    feature grid we upsample the captured map to (crop_sz, crop_sz) at sample
    time (see ``_sample_feature``).
    """
    store: dict = {"feat": None}
    classifier = find_classifier(model)

    def hook(_module, inputs, _output):
        # inputs is a tuple; inputs[0] is the (B, 256, H, W) feature map.
        feat = inputs[0].detach()
        store["feat"] = feat  # keep on device; squeeze/np at sample time

    handle = classifier.register_forward_hook(hook)
    store["handle"] = handle
    store["n_features"] = int(classifier.in_channels)
    return store


def find_classifier(model):
    """The final per-pixel class-projection Conv2d, across all six families.

    The hook predates the multi-backbone ladder and hardcoded Prithvi's
    ``head.head[2]``. Known per-family paths are tried first so a family
    refactor fails loudly HERE; then a structural fallback.

    NOTE the in_channels is NOT always 256: tessera's is 128
    (final_in = hidden // 2). The feature dim is a property of the
    backbone's head and travels with the export — consumers read the
    width from the data, never from a constant.
    """
    from torch import nn

    # Tessera: its own small head ends in self.classifier.
    mod = getattr(model, "classifier", None)
    if isinstance(mod, nn.Conv2d):
        return mod
    # Prithvi: model.head.head[2]; croma/terramind/clay wrap the shared
    # UPerNet as model.decoder_head -> .head.head[2].
    for holder_name in ("head", "decoder_head"):
        holder = getattr(model, holder_name, None)
        for seq in (getattr(holder, "head", None),
                    getattr(getattr(holder, "head", None), "head", None)):
            try:
                m = seq[2]
            except (TypeError, IndexError, KeyError):
                continue
            if isinstance(m, nn.Conv2d):
                return m
    # Structural fallback: among 1x1 convs the class projection has the
    # UNIQUE maximal out_channels (frac/binary heads carry 4/1). Guessing
    # between ambiguous candidates would silently hook the wrong feature.
    convs = [m for m in model.modules()
             if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1)]
    if convs:
        top = max(c.out_channels for c in convs)
        winners = [c for c in convs if c.out_channels == top]
        if len(winners) == 1:
            return winners[0]
    raise TypeError(
        f"cannot locate the classifier Conv2d on {type(model).__name__}; "
        f"1x1 convs seen: {[(c.in_channels, c.out_channels) for c in convs]}")

def _sample_feature(feat_map, rows, cols, crop_sz: int) -> np.ndarray:
    """Sample (N, 256) vectors at input-resolution (row,col) from (1,256,h,w).

    The captured feature grid is at classifier resolution (h,w) which may be
    smaller than ``crop_sz`` (the head's logits are upsampled AFTER the
    classifier). Bilinearly resize the feature map to (crop_sz, crop_sz) so the
    plot pixels — expressed in crop/input coordinates — index the right cell.
    """
    import torch.nn.functional as F

    if feat_map.shape[-1] != crop_sz or feat_map.shape[-2] != crop_sz:
        feat_map = F.interpolate(
            feat_map, size=(crop_sz, crop_sz),
            mode="bilinear", align_corners=True,
        )
    fm = feat_map.squeeze(0).cpu().numpy()  # (256, crop_sz, crop_sz)
    r = np.asarray(rows, dtype=np.int64)
    c = np.asarray(cols, dtype=np.int64)
    return fm[:, r, c].T.astype(np.float32)  # (N, 256)


def output_columns(truth_col: str | None, feat_cols: list[str]) -> list[str]:
    """Column order for the features parquet — the truth col follows the mode.

    The record dicts carry point_id and (in generic-truth mode) the truth
    column itself; a hard-coded NFI column list handed to ``from_records``
    silently DROPS both, leaving a LUCAS crop parquet with neither its pin
    key (tile_name, point_id) nor its labels.
    """
    return (["TractID", "PlotID", "point_id", "Easting", "Northing",
             "tile_name", truth_col or "nfi_forest"] + feat_cols)


def truth_summary(out_df: pd.DataFrame, truth_col: str | None) -> tuple[str, dict]:
    """Class distribution of the ACTIVE truth column.

    Must follow the mode exactly like ``output_columns`` does — a
    hard-coded ``nfi_forest`` here crashed the whole crop-distill Job
    (backoffLimit 0) after the parquet was already on disk.
    """
    name = truth_col or "nfi_forest"
    return name, out_df[name].value_counts().sort_index().to_dict()


def filter_index_by_npz_requirements(
    index_df: pd.DataFrame,
    required_keys: tuple[str, ...],
    *,
    log=print,
) -> pd.DataFrame:
    """Drop plot rows whose tile cannot satisfy ``required_keys``.

    Qualification deliberately shares the canonical probes used by
    ``build_pinned_plot_set`` and ``distill_forest_labels``. In particular,
    key presence alone is insufficient for SAR: ``s1_vv_vh`` also requires
    the current ``s1_enrich_v`` stamp. Filtering before model construction and
    forwarding keeps stale, non-pinned tiles out of the inference loop while
    the downstream ``nfi_head_cv --pinned-plots`` equality gate still rejects
    any missing pinned plot.
    """
    keys = tuple(dict.fromkeys(required_keys))
    if not keys:
        return index_df

    qualified_paths: set[str] = set()
    missing_key_paths: list[str] = []
    stale_version_paths: list[str] = []
    unreadable_paths: list[str] = []

    for raw_path in index_df["tile_path"].drop_duplicates():
        path = Path(raw_path)
        names = npz_key_names(path)
        if names is None:
            unreadable_paths.append(str(path))
        elif not all(key in names for key in keys):
            missing_key_paths.append(str(path))
        elif not npz_version_ok(path, keys):
            stale_version_paths.append(str(path))
        else:
            qualified_paths.add(str(path))

    path_strings = index_df["tile_path"].astype(str)
    keep = path_strings.isin(qualified_paths)
    dropped_rows = int((~keep).sum())
    dropped_tiles = int(path_strings[~keep].nunique())
    log(
        f"require {list(keys)}: kept {int(keep.sum())}/{len(index_df)} plots "
        f"on {len(qualified_paths)} tiles; dropped {dropped_rows} plots on "
        f"{dropped_tiles} unqualified tiles "
        f"(missing-key={len(missing_key_paths)}, "
        f"stale-version={len(stale_version_paths)}, "
        f"unreadable={len(unreadable_paths)})"
    )
    details = (
        [("missing key", p) for p in missing_key_paths]
        + [("stale version", p) for p in stale_version_paths]
        + [("unreadable", p) for p in unreadable_paths]
    )
    for reason, path in details[:_MAX_FAILED_TILE_DETAILS]:
        log(f"  DROP {Path(path).name}: {reason}")
    if len(details) > _MAX_FAILED_TILE_DETAILS:
        log(
            f"  ... {len(details) - _MAX_FAILED_TILE_DETAILS} additional "
            "unqualified tiles omitted"
        )
    return index_df.loc[keep].copy()


def load_tile_inventory(
    path: Path,
    *,
    partition: str,
    index_df: pd.DataFrame,
) -> dict[str, tuple[int, str]]:
    """Load the frozen byte identity for every tile in ``index_df``.

    The split builder hashes the complete cohort once.  During extraction we
    authenticate each tile lazily, in the same raw-byte read used by
    ``numpy.load``.  Requiring exact inventory/index equality prevents a
    caller from omitting the identity for just one tile.
    """
    if partition != "distill":
        raise ValueError("feature extraction may consume only distill inventory")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read frozen tile inventory {path}: {exc}") from exc
    field = f"{partition}_tile_inventory"
    records = document.get(field)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} has no non-empty {field}")

    identities: dict[str, tuple[int, str]] = {}
    names: set[str] = set()
    expected_fields = {"tile_name", "file_name", "tile_path", "size", "sha256"}
    for position, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError(f"{field}[{position}] has malformed fields")
        tile_name = record["tile_name"]
        file_name = record["file_name"]
        tile_path = record["tile_path"]
        size = record["size"]
        sha256 = record["sha256"]
        if (
            not isinstance(tile_name, str)
            or not tile_name
            or Path(tile_name).name != tile_name
            or not isinstance(file_name, str)
            or file_name != f"{tile_name}.npz"
            or not isinstance(tile_path, str)
            or not Path(tile_path).is_absolute()
            or str(Path(tile_path)) != tile_path
            or Path(tile_path).name != file_name
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(sha256, str)
            or _SHA256_RE.fullmatch(sha256) is None
        ):
            raise ValueError(f"{field}[{position}] has malformed identity")
        if tile_name in names or tile_path in identities:
            raise ValueError(f"{field} contains duplicate tile identity")
        names.add(tile_name)
        identities[tile_path] = (size, sha256)

    index_pairs = set(zip(
        index_df["tile_name"].astype(str),
        index_df["tile_path"].astype(str),
        strict=True,
    ))
    inventory_pairs = {
        (Path(tile_path).stem, tile_path) for tile_path in identities
    }
    if index_pairs != inventory_pairs:
        missing = sorted(index_pairs - inventory_pairs)[:3]
        extra = sorted(inventory_pairs - index_pairs)[:3]
        raise ValueError(
            "frozen distill tile inventory does not exactly match plot index; "
            f"missing={missing}, extra={extra}"
        )
    return identities


def _authenticated_tile_height(
    infcmp,
    tile_path: str,
    identity: tuple[int, str],
) -> int:
    """Read the sample geometry only after authenticating its frozen bytes."""
    expected_size, expected_sha256 = identity
    data, buffer = infcmp._load_npz_for_inference(
        tile_path,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    )
    try:
        if "spectral" not in data:
            raise ValueError(f"authenticated sample tile lacks spectral: {tile_path}")
        spectral = data["spectral"]
        if spectral.ndim < 2 or spectral.shape[-1] <= 0:
            raise ValueError(
                f"authenticated sample tile has invalid spectral shape: "
                f"{tile_path}: {spectral.shape}"
            )
        return int(spectral.shape[-1])
    finally:
        try:
            data.close()
        finally:
            buffer.close()


def prepare_plot_index(
    index_df: pd.DataFrame,
    *,
    tile_inventory: Path | None,
    tile_inventory_partition: str,
    required_keys: tuple[str, ...],
    img_size: int,
    infcmp,
    log=print,
) -> tuple[pd.DataFrame, dict[str, tuple[int, str]] | None, int]:
    """Validate input routing and apply the extraction crop window.

    A sealed crop run first validates the inventory against index metadata,
    before any tile path is probed.  Its sample geometry is then read through
    the same authenticated descriptor path used by inference.  The frozen
    bytes already encode qualification, so the legacy unauthenticated
    key/version filter is deliberately not run in this mode.
    """
    if index_df.empty:
        raise RuntimeError("feature extraction plot index is empty")

    tile_identities = None
    if tile_inventory is not None:
        tile_identities = load_tile_inventory(
            tile_inventory,
            partition=tile_inventory_partition,
            index_df=index_df,
        )
        sample_path = str(index_df["tile_path"].iloc[0])
        tile_h = _authenticated_tile_height(
            infcmp, sample_path, tile_identities[sample_path]
        )
        if required_keys:
            log(
                f"sealed inventory fixes required keys {list(required_keys)}; "
                "tiles will be parsed only through authenticated inference"
            )
    else:
        # Legacy, mutable indexes retain their historical shrink-on-missing
        # behavior.  Crop runs never enter this branch.
        exists = index_df["tile_path"].map(os.path.exists)
        if not exists.any():
            raise SystemExit(
                f"0 of {len(index_df)} indexed tile paths exist — mount "
                f"mismatch? First expected: {index_df['tile_path'].iloc[0]}"
            )
        if not exists.all():
            gone = int((~exists).sum())
            log(
                f"dropping {gone} plots on tiles no longer in the dataset "
                f"({index_df.loc[~exists, 'tile_name'].nunique()} tiles)"
            )
            index_df = index_df[exists].copy()

        index_df = filter_index_by_npz_requirements(index_df, required_keys, log=log)
        if index_df.empty:
            raise RuntimeError(
                "feature extraction has zero qualified plot rows after tile "
                "existence/key/version filtering"
            )
        sample_path = str(index_df["tile_path"].iloc[0])
        with np.load(sample_path, allow_pickle=False) as sample:
            tile_h = int(sample["spectral"].shape[-1])

    off = crop_offset(tile_h, img_size)
    crop_size = min(img_size, tile_h)
    before = len(index_df)
    index_df = index_df[
        (index_df["row"] >= off) & (index_df["row"] < off + crop_size)
        & (index_df["col"] >= off) & (index_df["col"] < off + crop_size)
    ].copy()
    index_df["row"] -= off
    index_df["col"] -= off
    kept = len(index_df)
    log(
        f"crop offset={off} (tile {tile_h}→{crop_size}); kept "
        f"{kept}/{before} plots in-crop ({before - kept} border-dropped)"
    )
    return index_df, tile_identities, crop_size


def write_parquet_create_only(frame: pd.DataFrame, path: Path) -> None:
    """Write a crop artifact through one exclusive, no-follow descriptor.

    A failed write intentionally leaves its private per-Pod pathname occupied:
    deleting it safely after a hostile path swap is not possible with Python's
    path APIs.  The caller must fail rather than reuse that run directory.
    """
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("this platform cannot enforce O_NOFOLLOW for output")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError(f"refusing to create crop output {path}: {exc}") from exc

    fd_needs_close = True
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise RuntimeError(f"crop output is not a single-link regular file: {path}")
        destination = os.fdopen(fd, "wb", closefd=True)
        fd_needs_close = False
        with destination:
            frame.to_parquet(destination, index=False)
            destination.flush()
            os.fsync(destination.fileno())
            completed = os.fstat(destination.fileno())
            if (
                not stat.S_ISREG(completed.st_mode)
                or completed.st_nlink != 1
                or completed.st_size <= 0
                or (completed.st_dev, completed.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise RuntimeError(f"crop output changed while writing: {path}")
            current = os.stat(path, follow_symlinks=False)
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or (current.st_dev, current.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise RuntimeError(f"crop output path changed while writing: {path}")
    except BaseException:
        if fd_needs_close:
            try:
                os.close(fd)
            except OSError:
                pass
        raise


def extract_feature_records(
    index_df: pd.DataFrame,
    *,
    model,
    device,
    img_size: int,
    crop_size: int,
    infcmp,
    store: dict,
    feat_cols: list[str],
    aux_names: list[str] | None,
    truth_col: str | None,
    dominant_frac: float,
    tile_identities: dict[str, tuple[int, str]] | None = None,
    log=print,
) -> tuple[list[dict], int, list[str]]:
    """Forward qualified tiles independently and return plot records.

    A tile failure is contained so already-computed and later valid records
    survive. The hook store is cleared before *every* forward: a failed or
    hook-less call can therefore never reuse the previous tile's features.
    Only ``Exception`` is caught; interrupts and other ``BaseException``
    subclasses retain their normal process-level semantics.

    Returns ``(records, failure_count, failure_details)``. Details are bounded
    in count and length so a damaged volume cannot produce an unbounded
    in-memory error ledger. Zero records is always fatal.
    """
    records: list[dict] = []
    failure_count = 0
    failure_details: list[str] = []

    for tile_name, grp in index_df.groupby("tile_name", sort=False):
        tile_path = str(grp["tile_path"].iloc[0])
        store["feat"] = None
        authenticated: dict[str, object] = {}
        if tile_identities is not None:
            if tile_path not in tile_identities:
                raise RuntimeError(
                    f"tile {tile_name} is absent from frozen inventory"
                )
            expected_size, expected_sha256 = tile_identities[tile_path]
            authenticated = {
                "expected_tile_size": expected_size,
                "expected_tile_sha256": expected_sha256,
            }
        try:
            infcmp.run_inference(
                model,
                tile_path,
                device,
                img_size=img_size,
                return_probs=True,
                aux_channel_names=aux_names,
                **authenticated,
            )
            feat_map = store["feat"]
            if feat_map is None:
                raise RuntimeError(
                    f"hook captured no feature for tile {tile_name}"
                )

            rows = grp["row"].to_numpy()
            cols = grp["col"].to_numpy()
            vecs = _sample_feature(feat_map, rows, cols, crop_size)
            tile_records: list[dict] = []
            for (_, row), vec in zip(grp.iterrows(), vecs):
                record = {
                    "TractID": row.get("TractID"),
                    "PlotID": row.get("PlotID"),
                    "point_id": row.get("point_id"),
                    "Easting": row.get("Easting"),
                    "Northing": row.get("Northing"),
                    "tile_name": str(tile_name),
                }
                if truth_col:
                    record[truth_col] = int(row[truth_col])
                else:
                    nfi_class = derive_nfi_forest_class(
                        row, dominant_frac=dominant_frac
                    )
                    record["nfi_forest"] = (
                        int(nfi_class) if nfi_class is not None else -1
                    )
                record.update(dict(zip(feat_cols, vec.tolist())))
                tile_records.append(record)
            records.extend(tile_records)
        except Exception as exc:
            identity_error = getattr(infcmp, "TileIdentityError", ())
            if tile_identities is not None and isinstance(exc, identity_error):
                raise
            failure_count += 1
            error_text = " ".join(str(exc).splitlines())
            message = (
                f"{tile_name}: {type(exc).__name__}: {error_text}"
            )[:_MAX_ERROR_CHARS]
            if len(failure_details) < _MAX_FAILED_TILE_DETAILS:
                failure_details.append(message)
                log(f"  ERROR {message}")
            elif failure_count == _MAX_FAILED_TILE_DETAILS + 1:
                log("  ERROR additional tile failures omitted from detail log")

    if failure_count:
        log(
            f"tile failures: {failure_count}; retained "
            f"{len(failure_details)} bounded details"
        )
    if not records:
        raise RuntimeError(
            "feature extraction produced zero records; refusing to write an "
            "empty parquet"
        )
    return records, failure_count, failure_details


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--checkpoint-size",
        type=int,
        help="exact checkpoint byte size; must accompany --checkpoint-sha256",
    )
    ap.add_argument(
        "--checkpoint-sha256",
        help="exact checkpoint SHA-256; must accompany --checkpoint-size",
    )
    ap.add_argument("--plot-index", required=True,
                    help="parquet from nfi_tile_coverage.py (tile_name, tile_path, "
                         "row, col, Easting, Northing, TractID, PlotID + NFI cols)")
    ap.add_argument("--img-size", type=int, default=504,
                    help="inference crop (504 = 600M patch-14 on 512 tiles)")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--dominant-frac", type=float, default=0.7,
                    help="conifer/deciduous dominance threshold for NFI truth")
    ap.add_argument("--truth-col", default=None,
                    help="take truth directly from this index column "
                         "(e.g. unified_class for the LUCAS crop index) "
                         "instead of deriving NFI forest class from volume "
                         "columns")
    ap.add_argument("--backbone-name", default=None,
                    help="registry backbone for load_model; without it a "
                         "checkpoint lacking backbone_name silently "
                         "resolves to prithvi_300m")
    ap.add_argument("--require-npz-key", action="append", default=[],
                    help="drop plot rows whose tile lacks this key or its "
                         "canonical version stamp (repeatable; SAR columns "
                         "pass s1_vv_vh)")
    ap.add_argument(
        "--tile-inventory",
        type=Path,
        help="frozen split JSON containing the exact distill tile byte "
             "inventory; crop jobs must supply it",
    )
    ap.add_argument(
        "--tile-inventory-partition",
        choices=("distill",),
        default="distill",
        help="only the distill partition is permitted to feature extraction",
    )
    ap.add_argument("--enable-markfukt", action="store_true",
                    help="feed markfukt as the 11th aux (wetness-aux checkpoint)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    checkpoint_identity_supplied = (
        args.checkpoint_size is not None,
        args.checkpoint_sha256 is not None,
    )
    if checkpoint_identity_supplied[0] != checkpoint_identity_supplied[1]:
        ap.error("--checkpoint-size and --checkpoint-sha256 are one unit")
    if args.tile_inventory is not None and not all(checkpoint_identity_supplied):
        ap.error("sealed crop extraction requires exact checkpoint identity")

    import torch

    infcmp = _load_inference_comparison()

    index_df = pd.read_parquet(args.plot_index)
    print(f"plot index: {len(index_df):,} co-located plots on "
          f"{index_df['tile_name'].nunique()} tiles")

    index_df, tile_identities, cs = prepare_plot_index(
        index_df,
        tile_inventory=args.tile_inventory,
        tile_inventory_partition=args.tile_inventory_partition,
        required_keys=tuple(args.require_npz_key),
        img_size=args.img_size,
        infcmp=infcmp,
    )

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # BOTH kwargs must reach load_model — the full call shape of
    # infer_tiles.py:243. img_size: clay/croma carry no pos_embed, so the
    # backbone is otherwise BUILT at a 224 grid while run_inference feeds
    # it 504px tiles — the 256-dim features this script exists to capture
    # would come off a wrongly-shaped head. backbone_name: checkpoint-only
    # resolution defaults to prithvi_300m when the saved config lacks the
    # field (pre-2026-08-24 trainer) — the wrong backbone entirely.
    # The config is returned from this one safe load; reading the checkpoint
    # a second time for aux names would bypass its authenticated descriptor.
    model, epoch, miou, model_img_size, checkpoint_config = infcmp.load_model(
        args.checkpoint,
        device,
        backbone_name=args.backbone_name,
        img_size=args.img_size,
        expected_checkpoint_size=args.checkpoint_size,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        return_checkpoint_config=True,
    )
    print(
        f"  [load_model] epoch={epoch} ckpt_mIoU={miou} "
        f"native_img={model_img_size}"
    )

    # The checkpoint KNOWS its aux set — reconstructing it from flags is
    # how terramind died: its r2 trained with 13 aux (the usual 11 plus
    # delta_vv/delta_vh ΔSAR), while --enable-markfukt rebuilt 11 and the
    # lidar_branch conv rejected the tensor. The saved config is the
    # single source of truth; the flag survives only as a fallback for
    # pre-config-era checkpoints.
    aux_names = checkpoint_config.get("enabled_aux_names")
    if aux_names:
        aux_names = list(aux_names)
        print(f"  aux from checkpoint config: {len(aux_names)} channels "
              f"({aux_names[-3:]}...)")
    elif args.enable_markfukt:
        from imint.training.unified_dataset import AUX_CHANNEL_NAMES
        aux_names = list(AUX_CHANNEL_NAMES) + ["markfukt"]
        print(f"  aux fallback (no config in ckpt): {len(aux_names)}")
    else:
        aux_names = None

    store = register_preclassifier_hook(model)

    n_features = store["n_features"]
    print(f"  classifier in_channels = {n_features} (native feature width)")
    feat_cols = [f"f{i:03d}" for i in range(n_features)]
    try:
        records, failure_count, _failure_details = extract_feature_records(
            index_df,
            model=model,
            device=device,
            img_size=args.img_size,
            crop_size=cs,
            infcmp=infcmp,
            store=store,
            feat_cols=feat_cols,
            aux_names=aux_names,
            truth_col=args.truth_col,
            dominant_frac=args.dominant_frac,
            tile_identities=tile_identities,
        )
    finally:
        store["handle"].remove()

    out_df = pd.DataFrame.from_records(
        records, columns=output_columns(args.truth_col, feat_cols))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.tile_inventory is not None:
        write_parquet_create_only(out_df, out)
    else:
        out_df.to_parquet(out, index=False)

    truth_name, n_by_class = truth_summary(out_df, args.truth_col)
    note = "" if args.truth_col else " (−1=treeless)"
    print(f"\nwrote {out} — {len(out_df)} plots × {n_features} features")
    print(f"  {truth_name} distribution{note}: {n_by_class}")
    if failure_count:
        print(
            f"  WARNING: {failure_count} tile(s) failed; downstream "
            "--pinned-plots equality remains the hard completeness gate"
        )


if __name__ == "__main__":
    main()
