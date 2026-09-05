#!/usr/bin/env python3
"""Build-time extraction and sealing for the crop-distill runtime image."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any


CROMA_GIT_SHA = "59505a6bcadbf36ba20767270154bf9f3067c5e7"
CROMA_ARCHIVE_SHA256 = (
    "939d0918991ad7604bbb0a782df2674b8e30ade6edc061bcad6ab486e6f94001"
)
CLAY_GIT_SHA = "f14e698f3c237cabf8d28dec669a362d66625381"
CLAY_ARCHIVE_SHA256 = (
    "0b908ea11d5348736c26512f695221a304883ec88bac68f66822ca07bf435d64"
)
BASE_IMAGE = (
    "python:3.11-slim@sha256:"
    "d1e9ca7c4e78d1e8ecadb5d44bfc8e956e7a65b659a9950f569f243d72b326d0"
)
SOURCE_ARCHIVE_SELECTION = (
    "config",
    "data/distill/distill_split.json",
    "imint",
    "scripts",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_member(member: tarfile.TarInfo, expected_top: str) -> None:
    pure = PurePosixPath(member.name)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe archive member: {member.name!r}")
    if pure.parts[0].lower() != expected_top.lower():
        raise ValueError(
            f"archive member {member.name!r} is outside {expected_top!r}"
        )
    if member.issym() or member.islnk():
        target = PurePosixPath(member.linkname)
        if target.is_absolute() or ".." in target.parts:
            raise ValueError(
                f"unsafe archive link {member.name!r} -> {member.linkname!r}"
            )


def extract_archive(
    archive: Path,
    destination: Path,
    *,
    repository: str,
    git_sha: str,
    expected_archive_sha256: str | None,
    include_prefixes: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Verify and extract one exact GitHub codeload archive."""
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    actual_archive_sha256 = sha256_file(archive)
    if (
        expected_archive_sha256 is not None
        and actual_archive_sha256 != expected_archive_sha256
    ):
        raise ValueError(
            f"{repository} archive SHA256 mismatch: expected "
            f"{expected_archive_sha256}, got {actual_archive_sha256}"
        )

    expected_top = f"{repository}-{git_sha}"
    temporary = destination.parent / f".{destination.name}.extract"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            if not members:
                raise ValueError(f"empty archive: {archive}")
            for member in members:
                _safe_member(member, expected_top)
            selected = members
            if include_prefixes is not None:
                prefixes = tuple(PurePosixPath(value) for value in include_prefixes)

                def included(member: tarfile.TarInfo) -> bool:
                    parts = PurePosixPath(member.name).parts
                    if len(parts) == 1:
                        return True
                    relative = PurePosixPath(*parts[1:])
                    return any(
                        relative == prefix or prefix in relative.parents
                        for prefix in prefixes
                    )

                selected = [member for member in members if included(member)]
            bundle.extractall(temporary, members=selected)
        children = list(temporary.iterdir())
        if len(children) != 1 or children[0].name.lower() != expected_top.lower():
            raise ValueError(
                f"{repository} archive root does not identify git SHA {git_sha}"
            )
        os.replace(children[0], destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    identity: dict[str, Any] = {
        "git_sha": git_sha,
        "archive_sha256": actual_archive_sha256,
    }
    if include_prefixes is not None:
        identity["selection"] = list(include_prefixes)
    return identity


def _load_provenance_module(source_root: Path) -> ModuleType:
    protocol_path = source_root / "scripts" / "crop_distill_protocol.py"
    module_path = source_root / "scripts" / "crop_distill_provenance.py"
    previous_protocol = sys.modules.pop("crop_distill_protocol", None)
    previous_provenance = sys.modules.pop("crop_distill_provenance", None)
    try:
        protocol_spec = importlib.util.spec_from_file_location(
            "crop_distill_protocol", protocol_path
        )
        if protocol_spec is None or protocol_spec.loader is None:
            raise ValueError(f"cannot import protocol helper: {protocol_path}")
        protocol = importlib.util.module_from_spec(protocol_spec)
        sys.modules["crop_distill_protocol"] = protocol
        protocol_spec.loader.exec_module(protocol)

        spec = importlib.util.spec_from_file_location(
            "crop_distill_provenance", module_path
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import provenance helper: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["crop_distill_provenance"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_protocol is None:
            sys.modules.pop("crop_distill_protocol", None)
        else:
            sys.modules["crop_distill_protocol"] = previous_protocol
        if previous_provenance is None:
            sys.modules.pop("crop_distill_provenance", None)
        else:
            sys.modules["crop_distill_provenance"] = previous_provenance


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"required build artifact is missing or empty: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _python_identity(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Python interpreter is missing or not executable: {path}")
    probe = (
        "import json, platform, sys; "
        "print(json.dumps({'implementation': platform.python_implementation(), "
        "'path': sys.executable, 'version': platform.python_version(), "
        "'version_info': list(sys.version_info[:3])}, sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [str(path), "-I", "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )
        identity = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot identify Python interpreter {path}: {exc}") from exc
    if identity.get("path") != str(path):
        raise ValueError(
            f"Python interpreter resolved to {identity.get('path')!r}, expected {path}"
        )
    return identity


def _seal_tree(
    module: ModuleType,
    root: Path,
    manifest_path: Path,
    archive_identity: dict[str, str],
) -> dict[str, Any]:
    entries = module.snapshot_tree(root)
    payload_sha256 = module.tree_payload_sha256(entries)
    _write_json(
        manifest_path,
        {
            "schema": module.TREE_SCHEMA,
            "entries": entries,
            "payload_sha256": payload_sha256,
        },
    )
    return {
        **archive_identity,
        "root": str(root),
        "files_manifest": str(manifest_path),
        "files_manifest_sha256": sha256_file(manifest_path),
        "payload_sha256": payload_sha256,
    }


def extract_command(args: argparse.Namespace) -> None:
    source_identity = extract_archive(
        args.source_archive,
        args.source_root,
        repository="imintengine",
        git_sha=args.source_git_sha,
        expected_archive_sha256=None,
        include_prefixes=SOURCE_ARCHIVE_SELECTION,
    )
    croma_identity = extract_archive(
        args.croma_archive,
        args.croma_root,
        repository="CROMA",
        git_sha=CROMA_GIT_SHA,
        expected_archive_sha256=CROMA_ARCHIVE_SHA256,
    )
    clay_identity = extract_archive(
        args.clay_archive,
        args.clay_root,
        repository="model",
        git_sha=CLAY_GIT_SHA,
        expected_archive_sha256=CLAY_ARCHIVE_SHA256,
    )
    _write_json(
        args.archive_identities,
        {
            "source": source_identity,
            "croma": croma_identity,
            "clay": clay_identity,
        },
    )


def seal_command(args: argparse.Namespace) -> None:
    identities = json.loads(args.archive_identities.read_text(encoding="utf-8"))
    module = _load_provenance_module(args.source_root)
    provenance_dir = args.runtime_manifest.parent
    provenance_dir.mkdir(parents=True, exist_ok=True)

    source = _seal_tree(
        module,
        args.source_root,
        provenance_dir / "imintengine-files.json",
        identities["source"],
    )
    croma = _seal_tree(
        module,
        args.croma_root,
        provenance_dir / "croma-files.json",
        identities["croma"],
    )
    clay = _seal_tree(
        module,
        args.clay_root,
        provenance_dir / "clay-files.json",
        identities["clay"],
    )
    environments = {
        "model": {
            "python": _python_identity(args.model_python),
            "requirements_lock": _file_identity(args.model_requirements_lock),
            "pip_freeze": _file_identity(args.model_pip_freeze),
        },
        "scoring": {
            "python": _python_identity(args.scoring_python),
            "requirements_lock": _file_identity(args.scoring_requirements_lock),
            "pip_freeze": _file_identity(args.scoring_pip_freeze),
        },
    }
    _write_json(
        args.runtime_manifest,
        {
            "schema": module.RUNTIME_SCHEMA,
            "base_image": BASE_IMAGE,
            "model_resolution": module.MODEL_RESOLUTION,
            "base_python": _python_identity(args.base_python),
            "source": source,
            "environments": environments,
            "external_sources": {"croma": croma, "clay": clay},
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--source-git-sha", required=True)
    extract.add_argument("--source-archive", type=Path, required=True)
    extract.add_argument("--source-root", type=Path, required=True)
    extract.add_argument("--croma-archive", type=Path, required=True)
    extract.add_argument("--croma-root", type=Path, required=True)
    extract.add_argument("--clay-archive", type=Path, required=True)
    extract.add_argument("--clay-root", type=Path, required=True)
    extract.add_argument("--archive-identities", type=Path, required=True)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--source-root", type=Path, required=True)
    seal.add_argument("--croma-root", type=Path, required=True)
    seal.add_argument("--clay-root", type=Path, required=True)
    seal.add_argument("--archive-identities", type=Path, required=True)
    seal.add_argument("--base-python", type=Path, required=True)
    seal.add_argument("--model-python", type=Path, required=True)
    seal.add_argument("--model-requirements-lock", type=Path, required=True)
    seal.add_argument("--model-pip-freeze", type=Path, required=True)
    seal.add_argument("--scoring-python", type=Path, required=True)
    seal.add_argument("--scoring-requirements-lock", type=Path, required=True)
    seal.add_argument("--scoring-pip-freeze", type=Path, required=True)
    seal.add_argument("--runtime-manifest", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extract":
        extract_command(args)
    else:
        seal_command(args)


if __name__ == "__main__":
    main()
