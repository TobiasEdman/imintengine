"""Read frozen crop identity metadata and audit candidate identity overlap.

No validator parquet, image tile, feature vector or outcome is opened. The
expected manifest/source pins come from the reviewed crop run authority.
This check does not establish spatial separation or full evaluation eligibility.
The stdlib-only core can run on an existing read-only metadata mount.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import os
import re
import stat
from pathlib import Path

MANIFEST = "lucas_crop_split.MANIFEST.json"
SPLIT = "lucas_crop_split.json"
VALIDATOR = "lucas_crop_validator_holdout_index.parquet"
DISTILL = "lucas_crop_distill_index.parquet"
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_SHA = re.compile(r"[0-9a-f]{40}")


class HoldoutError(ValueError):
    """Invalid identity or protected observations in a candidate population."""

    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise HoldoutError("duplicate metadata key")
        result[key] = value
    return result


def read_pinned(path, expected_sha256, *, limit=32 << 20):
    """Return the same stable file bytes that matched the supplied full digest."""
    if not isinstance(expected_sha256, str) or not SHA256.fullmatch(expected_sha256):
        raise HoldoutError("expected a full SHA-256")
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or not 0 < before.st_size <= limit):
            raise HoldoutError("unsafe or oversized metadata file")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
        current = path.lstat()
        attrs = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        if (any(getattr(before, k) != getattr(after, k) for k in attrs)
                or any(getattr(before, k) != getattr(current, k) for k in attrs)):
            raise HoldoutError("metadata changed during read")
    if len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise HoldoutError("metadata digest mismatch")
    return raw


def _object(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HoldoutError("invalid metadata JSON") from exc
    if not isinstance(value, dict):
        raise HoldoutError("metadata must be an object")
    return value


def _tile(value):
    if (not isinstance(value, str) or not value or value.isspace()
            or "\x00" in value or "\\" in value or value in (".", "..")
            or Path(value).name != value):
        raise HoldoutError("invalid tile identity")
    return value


def point_id(value):
    """Use the frozen protocol's integer/canonical-decimal identity semantics."""
    if isinstance(value, bool):
        raise HoldoutError("invalid point identity")
    if isinstance(value, numbers.Integral) and value >= 0:
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"(?:0|[1-9][0-9]{0,31})", value):
        return int(value)
    raise HoldoutError("invalid point identity")


def _keys(records):
    if not isinstance(records, list) or not records:
        raise HoldoutError("empty or invalid frozen key list")
    result = []
    for row in records:
        if not isinstance(row, dict) or set(row) != {"tile_name", "point_id"}:
            raise HoldoutError("frozen keys have an unexpected schema")
        if type(row["point_id"]) is not int:
            raise HoldoutError("frozen point identities must be integers")
        result.append((_tile(row["tile_name"]), point_id(row["point_id"])))
    if len(set(result)) != len(result):
        raise HoldoutError("duplicate frozen observation identity")
    return result


def key_digest(records):
    """sha256-jsonl-v1, exactly the frozen split producer's canonical format."""
    digest = hashlib.sha256()
    for row in sorted(records):
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_crop_holdout(directory, *, manifest_sha256, source_git_sha):
    if not isinstance(source_git_sha, str) or not GIT_SHA.fullmatch(source_git_sha):
        raise HoldoutError("expected a full split-source Git SHA")
    directory = Path(directory)
    manifest_raw = read_pinned(directory / MANIFEST, manifest_sha256)
    manifest = _object(manifest_raw)
    try:
        artifacts = manifest["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != {SPLIT, VALIDATOR, DISTILL}:
            raise HoldoutError("unexpected frozen artifact inventory")
        if any(not isinstance(v, str) or not SHA256.fullmatch(v) for v in artifacts.values()):
            raise HoldoutError("invalid frozen artifact digest")
        split_sha = artifacts[SPLIT]
        split_raw = read_pinned(directory / SPLIT, split_sha)
        split = _object(split_raw)
        if manifest["git_sha"] != source_git_sha or split["git_sha"] != source_git_sha:
            raise HoldoutError("split source identity mismatch")
        if split["key_cols"] != ["tile_name", "point_id"]:
            raise HoldoutError("unexpected frozen key schema")
        if split["key_digest_format"] != "sha256-jsonl-v1":
            raise HoldoutError("unsupported frozen key digest format")
        train, holdout = _keys(split["plots"]), _keys(split["holdout_plots"])
        prior = _keys(split["prior_test_plots"])
        train_tiles, holdout_tiles = {t for t, p in train}, {t for t, p in holdout}
        prior_tiles_list = split["prior_test_tiles"]
        if (not isinstance(prior_tiles_list, list) or not prior_tiles_list
                or prior_tiles_list != sorted(set(map(_tile, prior_tiles_list)))):
            raise HoldoutError("invalid prior-test tile list")
        prior_tiles = set(prior_tiles_list)
        if not {t for t, p in prior} <= prior_tiles:
            raise HoldoutError("prior-test keys are outside their frozen tiles")
        if split["holdout_tiles"] != sorted(holdout_tiles):
            raise HoldoutError("holdout tiles disagree with frozen keys")
        protected_keys = set(holdout) | set(prior)
        protected_points = {p for t, p in protected_keys}
        protected_tiles = holdout_tiles | prior_tiles
        tile_overlap = len(train_tiles & protected_tiles)
        train_points = {p for t, p in train}
        point_overlap = len(train_points & protected_points)
        overlap_details = {
            "distill_crop_holdout_point_ids": len(train_points & {p for t, p in holdout}),
            "distill_prior_test_point_ids": len(train_points & {p for t, p in prior}),
            "distill_crop_holdout_exact_keys": len(set(train) & set(holdout)),
            "distill_prior_test_exact_keys": len(set(train) & set(prior)),
            "distill_unique_point_ids": len(train_points),
            "crop_holdout_unique_point_ids": len({p for t, p in holdout}),
            "prior_test_unique_point_ids": len({p for t, p in prior}),
        }
        counts = {"n_distill": len(train), "n_holdout": len(holdout),
                  "n_qualified": len(train) + len(holdout),
                  "prior_test_point_count": len(prior),
                  "prior_test_tile_count": len(prior_tiles)}
        for name, actual in counts.items():
            for obj in (manifest, split):
                if type(obj[name]) is not int or obj[name] != actual:
                    raise HoldoutError(f"frozen count mismatch: {name}")
        digests = {
            "distill_keys_sha256": key_digest(train),
            "holdout_keys_sha256": key_digest(holdout),
            "qualified_keys_sha256": key_digest(train + holdout),
            "partition_sha256": key_digest(
                [(t, p, "distill") for t, p in train]
                + [(t, p, "holdout") for t, p in holdout]),
            "prior_test_keys_sha256": key_digest(prior),
            "prior_test_tiles_sha256": hashlib.sha256(
                "\n".join(sorted(prior_tiles)).encode()).hexdigest(),
        }
        for name, actual in digests.items():
            if manifest[name] != actual or split[name] != actual:
                raise HoldoutError(f"frozen key digest mismatch: {name}")
    except (KeyError, TypeError) as exc:
        raise HoldoutError("incomplete frozen identity metadata") from exc
    result = {
        "validator_sha256": artifacts[VALIDATOR],
        "protected_tiles": frozenset(protected_tiles),
        "protected_point_ids": frozenset(protected_points),
        "summary": {
            "schema": "imint-crop-holdout-identity-audit-v1",
            "status": "frozen_identity_metadata_verified",
            "manifest_sha256": manifest_sha256, "split_sha256": split_sha,
            "source_git_sha": source_git_sha, "counts": counts,
            "key_digests": digests,
            "protected_point_ids": len(protected_points),
            "protected_tiles": len(protected_tiles),
            "identity_metadata_verified": True,
            "distill_protected_point_overlap": point_overlap,
            "distill_protected_tile_overlap": tile_overlap,
            "overlap_details": overlap_details,
            "files_read": [MANIFEST, SPLIT],
            "validator_parquet_read": False, "features_read": False,
            "spatial_alias_check": "not_done", "full_role_audit": "not_done",
            "cleared_for_scoring": False,
        },
    }

    if tile_overlap or point_overlap:
        result["summary"]["status"] = "protected_role_overlap_detected"
        raise HoldoutError("LUCAS identity crosses distill and protected roles",
                           diagnostics=result["summary"])
    return result


def audit_candidates(guard, records, *, source):
    """Counts only: protect point identity even after tile renaming/year changes.

    Tile names are checked for both truth sources. Point IDs are LUCAS-specific;
    NFI's different identifier namespace is not treated as LUCAS point IDs.
    Spatially overlapping tiles with different names still need geometry audit.
    """
    if source not in ("nfi", "lucas"):
        raise HoldoutError("unknown observation source")
    total = tile_hits = point_hits = blocked = 0
    for row in records:
        try:
            tile_hit = _tile(row["tile_name"]) in guard["protected_tiles"]
            point_hit = (point_id(row["point_id"]) in guard["protected_point_ids"]
                         if source == "lucas" else False)
        except (KeyError, TypeError) as exc:
            raise HoldoutError("candidate index lacks identity columns") from exc
        total += 1
        tile_hits += int(tile_hit)
        point_hits += int(point_hit)
        blocked += int(tile_hit or point_hit)
    if not total:
        raise HoldoutError("empty candidate population")
    return {"source": source, "rows": total, "protected_tile_hits": tile_hits,
            "protected_point_hits": point_hits, "blocked_rows": blocked,
            "status": "blocked" if blocked else "no_known_holdout_identity_overlap",
            "spatial_alias_check": "not_done", "cleared_for_scoring": False}



def read_candidate_identities(guard, path, expected_sha256, *, source):
    """Never open the declared validator, including a hash-identical renamed copy."""
    path = Path(path)
    if path.name == VALIDATOR or expected_sha256 == guard["validator_sha256"]:
        raise HoldoutError("validator index cannot be a candidate index")
    if source not in ("nfi", "lucas"):
        raise HoldoutError("unknown observation source")
    import io
    import pyarrow.parquet as pq
    raw = read_pinned(path, expected_sha256)
    columns = ["tile_name"] + (["point_id"] if source == "lucas" else [])
    rows = pq.read_table(io.BytesIO(raw), columns=columns, use_threads=False).to_pylist()
    return rows, columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--candidate-index")
    parser.add_argument("--candidate-index-sha256")
    parser.add_argument("--candidate-source", choices=("nfi", "lucas"))
    args = parser.parse_args()
    candidate_options = (args.candidate_index, args.candidate_index_sha256, args.candidate_source)
    if any(candidate_options) and not all(candidate_options):
        parser.error("candidate index, SHA-256 and source must be supplied together")
    try:
        guard = load_crop_holdout(args.split_dir, manifest_sha256=args.manifest_sha256,
                                  source_git_sha=args.source_git_sha)
        report = guard["summary"]
        if args.candidate_index:
            rows, columns = read_candidate_identities(
                guard, args.candidate_index, args.candidate_index_sha256,
                source=args.candidate_source)
            report["files_read"].append(str(args.candidate_index))
            report["candidates"] = audit_candidates(guard, rows, source=args.candidate_source)
            report["candidate_index_sha256"] = args.candidate_index_sha256
            report["candidate_columns_read"] = columns
        print(json.dumps(report, sort_keys=True))
        return 2 if report.get("candidates", {}).get("blocked_rows", 0) else 0
    except (HoldoutError, OSError) as exc:
        failure = {"status": "not_verified", "error": str(exc),
                   "cleared_for_scoring": False}
        if isinstance(exc, HoldoutError) and exc.diagnostics is not None:
            failure["diagnostics"] = exc.diagnostics
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
