"""Tester för imint.exporters.manifest."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from imint.exporters.manifest import (
    _docker_image_digest,
    _file_sha256,
    _git_sha,
    _hash_files,
    read_manifest,
    write_manifest,
)


def test_write_manifest_minimal(tmp_path: Path):
    """Manifest med bara output_dir ska ha produced_at + git_sha."""
    p = write_manifest(tmp_path)
    assert p.is_file()
    data = json.loads(p.read_text())
    assert "produced_at" in data
    assert "git_sha" in data
    assert "git_dirty" in data


def test_write_manifest_full(tmp_path: Path):
    """Manifest med alla fält serialiseras stabilt."""
    proc_file = tmp_path / "graph.xml"
    proc_file.write_text("<graph/>")
    input_file = tmp_path / "in.dat"
    input_file.write_bytes(b"hello")

    p = write_manifest(
        tmp_path,
        image="imint-test:1.0",
        process_files=[str(proc_file)],
        run_args={"aoi": "POLYGON((0 0,1 1))", "netset": "C2X-Nets"},
        input_data=[input_file],
        outputs=["chl.png", "tsm.png"],
        extra={"note": "smoketest"},
    )
    data = json.loads(p.read_text())
    assert data["image"] == "imint-test:1.0"
    assert data["run_args"]["netset"] == "C2X-Nets"
    assert data["input_data_count"] == 1
    assert data["input_data_hash"].startswith("sha256:")
    assert data["outputs"] == ["chl.png", "tsm.png"]
    assert data["note"] == "smoketest"
    assert len(data["process_files"]) == 1
    assert data["process_files"][0]["sha256"].startswith("sha256:")


def test_read_manifest_roundtrip(tmp_path: Path):
    write_manifest(tmp_path, image="x:1.0")
    data = read_manifest(tmp_path)
    assert data is not None
    assert data["image"] == "x:1.0"


def test_read_manifest_missing(tmp_path: Path):
    assert read_manifest(tmp_path) is None


def test_file_sha256_deterministic(tmp_path: Path):
    f = tmp_path / "x.dat"
    f.write_bytes(b"hello")
    h1 = _file_sha256(f)
    h2 = _file_sha256(f)
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_hash_files_order_independent(tmp_path: Path):
    """Hash ska vara stabil oavsett ordning i input-listan."""
    a = tmp_path / "a.dat"; a.write_bytes(b"A")
    b = tmp_path / "b.dat"; b.write_bytes(b"B")
    h1 = _hash_files([a, b])
    h2 = _hash_files([b, a])
    assert h1 == h2


def test_hash_files_content_sensitive(tmp_path: Path):
    a = tmp_path / "a.dat"
    a.write_bytes(b"A")
    h1 = _hash_files([a])
    a.write_bytes(b"B")
    h2 = _hash_files([a])
    assert h1 != h2


def test_git_sha_in_repo(tmp_path: Path):
    """Default repo_root → upptäcker git-state. Testar bara att nåt returneras."""
    repo = Path(__file__).resolve().parent.parent
    sha, dirty = _git_sha(repo)
    assert isinstance(sha, str)
    assert isinstance(dirty, bool)


def test_docker_image_digest_missing():
    """Image som inte finns ska ge None, inte krasch."""
    assert _docker_image_digest("nonexistent-image-xyz-987:0.1") is None
