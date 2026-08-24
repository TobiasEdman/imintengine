"""Regression tests for the immutable Prithvi-600M checkpoint fetcher."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import fetch_prithvi600m_checkpoint as fetcher


def _install_small_identity(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    from scripts import era5_smoke_provenance as provenance

    monkeypatch.setattr(
        provenance, "FOUNDATION_CHECKPOINT_SIZE_BYTES", len(payload),
    )
    monkeypatch.setattr(
        provenance,
        "FOUNDATION_CHECKPOINT_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )


def test_fetches_verifies_and_reuses_persisted_checkpoint(tmp_path, monkeypatch):
    payload = b"official-prithvi-fixture"
    _install_small_identity(monkeypatch, payload)
    remote = tmp_path / "remote.pt"
    remote.write_bytes(payload)
    calls: list[dict] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(remote)

    monkeypatch.setattr(fetcher, "hf_hub_download", fake_download)
    target = tmp_path / "models" / fetcher.FILENAME
    result = fetcher.fetch_checkpoint(target, tmp_path / "cache")
    assert result["status"] == "downloaded"
    assert target.read_bytes() == payload
    assert calls == [{
        "repo_id": fetcher.REPOSITORY,
        "filename": fetcher.FILENAME,
        "revision": fetcher.REVISION,
        "cache_dir": tmp_path / "cache",
    }]

    assert fetcher.fetch_checkpoint(target, tmp_path / "cache")["status"] == "existing"
    assert len(calls) == 1


def test_never_publishes_wrong_checkpoint(tmp_path, monkeypatch):
    expected = b"expected"
    _install_small_identity(monkeypatch, expected)
    remote = tmp_path / "wrong.pt"
    remote.write_bytes(b"wrong---")
    monkeypatch.setattr(fetcher, "hf_hub_download", lambda **_: str(remote))
    target = tmp_path / "models" / fetcher.FILENAME

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        fetcher.fetch_checkpoint(target, tmp_path / "cache")
    assert not target.exists()


def test_source_revision_is_immutable_official_weights_commit():
    assert fetcher.REPOSITORY == "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL"
    assert fetcher.REVISION == "f4c19741895193f6eb6ec16748550fb730860aff"
