"""Apply a reviewed, content-pinned fraction-supervision decision.

This is an enforcement layer, not a scientific provenance auditor. The caller
must obtain the expected policy digest from the reviewed campaign specification,
not compute it from an unreviewed policy at scoring time. Prediction-cache head
capability alone never authorizes fraction metrics.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from scripts import prediction_cache as cache

SCHEMA = "imint-fraction-scoring-policy-v1"


def _sha(value):
    if not isinstance(value, str) or cache.SHA256.fullmatch(value) is None:
        raise cache.CacheError("fraction policy requires a full SHA-256")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise cache.CacheError(f"duplicate fraction policy key: {key}")
        result[key] = value
    return result


def load_policy(path, expected_sha256):
    """Check the pinned digest before parsing the same stable regular bytes."""
    _sha(expected_sha256)
    path = Path(path)
    fd, opened = cache.inference._open_regular_nofollow(
        path, error_type=cache.CacheError, label="fraction policy")
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read()
        cache.inference._assert_open_file_unchanged(
            stream.fileno(), path, opened, error_type=cache.CacheError,
            label="fraction policy")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise cache.CacheError("fraction policy digest mismatch")
    try:
        policy = json.loads(raw, object_pairs_hook=_unique_object)
        if policy["schema"] != SCHEMA:
            raise cache.CacheError("unsupported fraction policy schema")
        entries = policy["entries"]
        if not isinstance(entries, list) or not entries:
            raise cache.CacheError("empty fraction policy")
        seen = set()
        for entry in entries:
            cache._check_identity(entry["checkpoint"])
            sha = entry["checkpoint"]["sha256"]
            if sha in seen:
                raise cache.CacheError("duplicate checkpoint in fraction policy")
            seen.add(sha)
            if not isinstance(entry["cell"], str) or not entry["cell"].strip():
                raise cache.CacheError("fraction policy needs a cell name")
            if entry["decision"] not in ("eligible", "ineligible", "unverified"):
                raise cache.CacheError("unknown fraction eligibility decision")
            if not isinstance(entry["reason"], str) or not entry["reason"].strip():
                raise cache.CacheError("fraction policy needs a reason")
            _sha(entry["training_log_sha256"])
            _sha(entry["basis_report_sha256"])
            if type(entry["head_enabled"]) is not bool:
                raise cache.CacheError("fraction head decision must be a boolean")
            if entry["fraction_loss_signal"] not in (
                    "recorded_nonzero", "recorded_zero", "not_recorded"):
                raise cache.CacheError("unknown fraction loss evidence")
            lineage = entry["target_lineage_review_sha256"]
            if lineage is not None:
                _sha(lineage)
            if entry["decision"] == "eligible" and not (
                    entry["head_enabled"]
                    and entry["fraction_loss_signal"] == "recorded_nonzero"
                    and lineage is not None):
                raise cache.CacheError("eligible fractions lack supervision evidence")
    except (ValueError, UnicodeError, KeyError, TypeError) as exc:
        raise cache.CacheError(f"invalid fraction policy: {exc}") from exc
    return policy


def select_fraction_mode(contract, *, requested=False, automatic=False,
                         policy_path=None, policy_sha256=None):
    """Return an auditable decision before truth data or predictions are read.

    Explicit requests fail unless eligible. Automatic mode requires a pinned
    policy for every checkpoint, and uses class-only scoring for known
    ineligible/unverified entries. Unknown checkpoints and malformed evidence
    always fail; they never silently become a negative eligibility decision.
    """
    if requested and automatic:
        raise cache.CacheError("choose explicit or automatic fraction scoring")
    if not requested and not automatic:
        if policy_path is not None or policy_sha256 is not None:
            raise cache.CacheError("fraction policy supplied without a fraction mode")
        return {"enabled": False, "mode": "class_only", "policy_sha256": None,
                "decision": "not_requested", "reason": "Class-head scoring requested"}
    if policy_path is None or policy_sha256 is None:
        raise cache.CacheError("fraction scoring requires a reviewed policy and its pinned SHA-256")
    policy = load_policy(policy_path, policy_sha256)
    checkpoint = contract["checkpoint"]
    entry = next((e for e in policy["entries"]
                  if e["checkpoint"]["sha256"] == checkpoint["sha256"]), None)
    if entry is None:
        raise cache.CacheError("checkpoint has no fraction eligibility decision")
    if entry["checkpoint"] != checkpoint:
        raise cache.CacheError("fraction policy checkpoint size mismatch")
    if entry["head_enabled"] is not contract["model"]["fraction_head"]:
        raise cache.CacheError("fraction policy and cache disagree about the head")
    enabled = entry["decision"] == "eligible"
    if requested and not enabled:
        raise cache.CacheError(
            f"fraction scoring not eligible for {entry['cell']}: {entry['reason']}")
    return {
        "enabled": enabled,
        "mode": "explicit" if requested else "automatic",
        "policy_sha256": policy_sha256,
        "cell": entry["cell"],
        "checkpoint": checkpoint,
        "decision": entry["decision"],
        "reason": entry["reason"],
        "training_log_sha256": entry["training_log_sha256"],
        "basis_report_sha256": entry["basis_report_sha256"],
        "target_lineage_review_sha256": entry["target_lineage_review_sha256"],
    }
