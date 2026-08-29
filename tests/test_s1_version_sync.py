"""The S1 enrichment version must be one constant, imported by both sides.

The v3→v4 bump (2026-08-24) updated only the enricher; the dataset's
hard-coded ``== 3`` gate then silently dropped every SAR tile, and
train-croma-v3 died with "No tiles found for split='val'" — twice. These
tests make a one-sided bump impossible: producer and consumer must reference
the same object in imint/training/s1_enrichment.py, and no local re-pin of
the version may reappear in either file.
"""
from __future__ import annotations

import re
from pathlib import Path

from imint.training.s1_enrichment import S1_ENRICH_VERSION

REPO = Path(__file__).resolve().parents[1]


def test_enricher_imports_shared_version():
    import scripts.enrich_tiles_s1 as enricher
    assert enricher.S1_ENRICH_VERSION is S1_ENRICH_VERSION


def test_dataset_imports_shared_version():
    import imint.training.unified_dataset as ds
    assert ds.S1_ENRICH_VERSION is S1_ENRICH_VERSION


def test_no_local_version_pins():
    """No integer literal re-pin of the version outside the shared module."""
    pattern = re.compile(r"S1_ENRICH_VERSION\s*=\s*\d")
    for rel in ("scripts/enrich_tiles_s1.py", "imint/training/unified_dataset.py"):
        text = (REPO / rel).read_text()
        assert not pattern.search(text), (
            f"{rel} re-pins S1_ENRICH_VERSION locally — import it from "
            f"imint/training/s1_enrichment.py instead"
        )


def test_no_hardcoded_version_gates_in_dataset():
    """The gates must compare against the constant, not a magic number."""
    text = (REPO / "imint/training/unified_dataset.py").read_text()
    assert not re.search(r"s1_enrich_v.{0,40}[!=]= ?[0-9]", text, re.DOTALL), (
        "unified_dataset.py compares s1_enrich_v against a literal — "
        "use S1_ENRICH_VERSION"
    )
