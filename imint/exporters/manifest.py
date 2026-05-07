"""MANIFEST.json sidecar — bind output-artefakter till exakt build + commit.

Bakgrund: i commit 52d19ae committades 9 PNG från en C2RCC-pipeline
utan att Dockerfile, SNAP graph eller run-args var versionerade. Sex
månader senare gick det inte att spåra vilken image-version, vilket
git-SHA, eller vilka run-args som producerade artefakterna.

Den här modulen är ``imint``-sidan av regeln *"output-artefakter får
inte committas innan processen som producerade dem är committad"*
(CLAUDE.md "Docker- och processversionering"). Varje pipeline-skript
som producerar artefakter i ``outputs/`` eller ``docs/showcase/`` ska
anropa ``write_manifest()`` bredvid utdata.

MANIFEST.json-strukturen
------------------------

::

    {
      "produced_at": "2026-05-06T14:32:11+02:00",   # ISO-8601 lokal tid
      "git_sha": "d7eaa40",                          # repo-state vid run
      "git_dirty": false,                            # true om uncommitted
      "image": "imint-snap-c2rcc:latest",            # docker tag
      "image_digest": "sha256:80a519fe8023...",      # image content-hash
      "graph": "docker/c2rcc-snap/c2rcc_msi_graph.xml",  # config-fil
      "graph_sha256": "abc123...",                   # config-hash
      "run_args": {...},                             # CLI-args eller dict
      "input_data_hash": "sha256:def456...",          # hash på input
      "outputs": ["chl.png", "tsm.png"]              # filer skapade
    }

Använd så här
-------------

::

    from imint.exporters.manifest import write_manifest

    write_manifest(
        output_dir="outputs/c2rcc_runs_lilla_karlso/2025-05-12",
        image="imint-snap-c2rcc:latest",
        process_files=["docker/c2rcc-snap/c2rcc_msi_graph.xml"],
        run_args={"aoi_wkt": "POLYGON((...))", "netset": "C2X-Nets"},
        input_data=[Path("demos/lilla_karlso_birds/cache_l1c/...")],
        outputs=["c2rcc_2025-05-12.dim", "chl.png"],
    )
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _git_sha(repo_root: Path) -> tuple[str, bool]:
    """Returnera (short SHA, dirty?) för repot. Faller tillbaka på (?, ?)."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        return sha, bool(status)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        return "unknown", True


def _docker_image_digest(image: str) -> str | None:
    """Returnera image content-hash (sha256:...) från lokal Docker-daemon."""
    try:
        result = subprocess.run(
            ["docker", "inspect", image, "--format", "{{.Id}}"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        digest = result.stdout.strip()
        return digest if digest.startswith("sha256:") else None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        return None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _hash_files(paths: Iterable[Path]) -> str:
    """Stable hash över en lista filer (för input-data-fingerprint)."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        h.update(str(p).encode())
        h.update(b"\0")
        if p.is_file():
            h.update(_file_sha256(p).encode())
        else:
            h.update(b"missing")
        h.update(b"\n")
    return f"sha256:{h.hexdigest()}"


def write_manifest(
    output_dir: str | Path,
    *,
    image: str | None = None,
    process_files: list[str | Path] | None = None,
    run_args: dict[str, Any] | None = None,
    input_data: list[str | Path] | None = None,
    outputs: list[str] | None = None,
    repo_root: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Skriv MANIFEST.json till ``output_dir`` med pipelinens fingerprint.

    Args:
        output_dir: Katalog där MANIFEST.json sparas. Skapas om saknas.
        image: Docker-tag som producerade artefakterna (``imint-snap-c2rcc:latest``).
        process_files: Config/script-filer som styr processen (Dockerfile,
            graph-XML, etc.). SHA-256-hash beräknas för varje.
        run_args: CLI-args eller config-dict som passerades till pipelinen.
        input_data: Input-filer (SAFE-arkiv, ingång-tiles). Hash beräknas
            över hela listan.
        outputs: Filnamn på artefakter producerade. Bara namn, inte path.
        repo_root: Override för repo-root (default: gå uppåt från output_dir).
        extra: Fält som läggs till manifestet utan validering.

    Returns:
        Path till skriven MANIFEST.json.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if repo_root is None:
        # Försök hitta git-roten genom att gå uppåt från output_dir
        cur = out_path.resolve()
        while cur != cur.parent and not (cur / ".git").exists():
            cur = cur.parent
        repo_root = cur if (cur / ".git").exists() else Path.cwd()

    git_sha, git_dirty = _git_sha(repo_root)

    manifest: dict[str, Any] = {
        "produced_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
    }

    if image:
        manifest["image"] = image
        digest = _docker_image_digest(image)
        if digest:
            manifest["image_digest"] = digest

    if process_files:
        manifest["process_files"] = []
        for pf in process_files:
            pf_path = Path(pf)
            if not pf_path.is_absolute():
                pf_path = repo_root / pf_path
            entry: dict[str, Any] = {"path": str(pf)}
            if pf_path.is_file():
                entry["sha256"] = _file_sha256(pf_path)
            else:
                entry["missing"] = True
            manifest["process_files"].append(entry)

    if run_args is not None:
        manifest["run_args"] = run_args

    if input_data:
        input_paths = [Path(p) for p in input_data]
        manifest["input_data_hash"] = _hash_files(input_paths)
        manifest["input_data_count"] = len(input_paths)

    if outputs:
        manifest["outputs"] = list(outputs)

    if extra:
        manifest.update(extra)

    manifest_path = out_path / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def read_manifest(output_dir: str | Path) -> dict[str, Any] | None:
    """Läs ``MANIFEST.json`` från ``output_dir``, returnera None om saknas."""
    p = Path(output_dir) / "MANIFEST.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
