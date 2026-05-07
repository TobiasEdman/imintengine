"""Repo-hygien-tester — håller Docker-pipelines reproducerbara.

Bakgrund: commit 52d19ae committade 9 PNG-resultat utan att Dockerfile
eller run-skript versionerades. Detta blockerade replikering 6 månader
senare. CLAUDE.md "Docker- och processversionering" specificerar reglerna;
det här testet håller dem self-enforcing.

Två tester:
  1. ``test_every_docker_run_has_a_dockerfile`` — varje image som
     refereras via ``docker run X`` i Python/shell/Makefile måste ha
     Dockerfile under ``docker/<namn>/`` eller på annan känd plats.
  2. ``test_no_latest_tag_in_dockerfiles`` — Dockerfiles får inte
     använda ``FROM x:latest`` (driver mot reproducerbarhet).

Båda körs lokalt via pytest och i CI på varje PR.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Filer som scannas för ``docker run`` references.
SCAN_GLOBS = ("*.py", "*.sh", "Makefile", "*.yaml", "*.yml")

# Kataloger som hoppas över (genererad kod, externa beroenden, .git, etc.)
# Inkluderar notebooks-checkpoints och anteckningsboksformat med stora cellsträngar
# som regex-matchen blir kvadratisk på.
SKIP_PARTS = (
    ".venv", ".venv-sr", "node_modules", ".git", "site-packages",
    ".ipynb_checkpoints", "outputs/", "docs/showcase/",
    ".pytest_cache", "__pycache__",
)

# Imagi som har Dockerfile i repot — håll synkad med faktiska Dockerfiles.
# Ny pipeline-image → lägg till här samtidigt som du committar Dockerfile.
KNOWN_LOCAL_IMAGES = {
    "imint-engine",        # Dockerfile (root)
    "imint-engine-cuda",   # Dockerfile.cuda
    "imint-engine-api",    # Dockerfile.api
    "imint/cloud-models",  # docker/cloud-models/Dockerfile
    "imint-snap-c2rcc",    # docker/c2rcc-snap/Dockerfile
}

# Allowlist för upstream/registry-images vi inte bygger själva. Gränsa till
# images vi medvetet använder utan att versionera build-receptet (typiskt
# stable upstream som postgres). Om en image är i pipeline och vi vill ha
# kontroll — flytta den till KNOWN_LOCAL_IMAGES och skriv en Dockerfile.
ALLOWED_UPSTREAM = {
    "postgres", "redis", "minio/minio",
    "python", "debian", "ubuntu", "alpine",  # base-images i CI-skript
    "busybox",
}


# Image-tag-regel: lowercase, digit/dash/dot/slash, måste innehålla
# minst en av {/, -, .} eller ett digit för att accepteras (annars
# är det troligen ett vanligt engelskt/svenskt ord från en kommentar
# där "docker run" nämns i prosa).
_IMAGE_RE = re.compile(r"^[a-z0-9][a-z0-9._/\-]*(:[a-zA-Z0-9._\-]+)?$")


def _looks_like_docker_image(tok: str) -> bool:
    if not tok or len(tok) < 3:
        return False
    if tok.startswith(("-", "/", "$", ".", "&&", ";", "\\")):
        return False
    if not _IMAGE_RE.match(tok):
        return False
    # Reject pure lowercase words without digit/slash/dash — sannolikt
    # ett vanligt ord från en kommentar.
    if all(c.isalpha() or c == "_" for c in tok.split(":")[0]):
        return False
    if tok in {"true", "false", "echo", "bash", "sh",
               "python", "python3"}:
        return False
    return True


def _iter_repo_files() -> list[Path]:
    """Iterera över versionerade filer av relevant typ.

    Använder ``git ls-files`` istället för ``Path.rglob`` så vi inte
    spenderar 100+ sekunder på att vandra `.venv-sr/`-trädet (~500k
    Python-stdlib-filer). Faller tillbaka på rglob om vi inte är i en
    git-repo.
    """
    import subprocess
    suffixes = {".py", ".sh", ".yaml", ".yml"}
    basenames = {"Makefile"}
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO), "ls-files"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        paths = [REPO / line for line in result.stdout.splitlines() if line]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        # Fallback — ej git eller git saknas. Lite långsammare.
        paths = []
        for ext in SCAN_GLOBS:
            for f in REPO.rglob(ext):
                if any(part in str(f) for part in SKIP_PARTS):
                    continue
                paths.append(f)
        return paths

    files: list[Path] = []
    for p in paths:
        if any(part in str(p) for part in SKIP_PARTS):
            continue
        if p.suffix in suffixes or p.name in basenames:
            files.append(p)
    return files


def _extract_docker_run_images() -> set[str]:
    """Find image names referenced via ``docker run`` in repo scripts.

    Skannar bara filer som troligen innehåller ``docker run``-anrop
    (Dockerfile är inte intressant — där letar vi FROM separat).
    Använder enkel rad-för-rad-tokenisering istället för regex som
    kan bli kvadratisk på stora notebook-cellsträngar.

    Returns image names with optional registry prefix (without tag).
    """
    images: set[str] = set()
    for f in _iter_repo_files():
        try:
            content = f.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        if "docker run" not in content:
            continue
        # Sammanslå backslash-line-continuations (`docker run \\\n  X` blir
        # `docker run   X`) så multi-line shell-anrop tokeniseras korrekt.
        content = re.sub(r"\\\s*\n\s*", " ", content)
        for line in content.splitlines():
            # Hitta token "docker run" och plocka första non-flag-arg som
            # följer. Tokenisering är medvetet enkel — täcker shell- och
            # subprocess-anrop men kan missa rader med komplex quoting.
            line_low = line.replace('"', " ").replace("'", " ").replace(",", " ")
            tokens = line_low.split()
            if "docker" not in tokens:
                continue
            try:
                idx = tokens.index("docker")
            except ValueError:
                continue
            if idx + 1 >= len(tokens) or tokens[idx + 1] != "run":
                continue
            # Skip flags och deras värden tills vi hittar image
            i = idx + 2
            while i < len(tokens):
                tok = tokens[i]
                if tok.startswith("-"):
                    # Flagga; om = i token är värdet inkluderat, annars hoppa över nästa
                    if "=" not in tok and tok in {
                        "-v", "--volume", "-e", "--env", "-p", "--publish",
                        "-w", "--workdir", "--name", "--user", "-u",
                        "--platform", "--network", "--mount",
                    }:
                        i += 2
                        continue
                    i += 1
                    continue
                # Första non-flag-token är image — validera att det
                # ser ut som en Docker-image-tag och inte ett vanligt
                # ord från en kommentar/dokumentation.
                tag = tok
                if not _looks_like_docker_image(tag):
                    break
                images.add(tag)
                break
    return images


def test_every_docker_run_has_a_dockerfile() -> None:
    """Every image used in a ``docker run`` must be buildable from this repo.

    If a CI/dev script invokes a Docker image, the recipe to build that
    image must live in this repo (typically under ``docker/<name>/``) or
    be an explicit upstream allowed in :data:`ALLOWED_UPSTREAM`.

    To resolve a failure:
      1. Add ``docker/<name>/Dockerfile`` for the image, OR
      2. Append the image basename to :data:`KNOWN_LOCAL_IMAGES` if a
         Dockerfile already exists elsewhere in the repo, OR
      3. Append the image basename to :data:`ALLOWED_UPSTREAM` if it is a
         well-known upstream you intentionally don't fork.
    """
    referenced = _extract_docker_run_images()
    # Image basenames (strip path/tag): "imint/cloud-models:1.0" → "imint/cloud-models"
    bases = {img.split(":")[0] for img in referenced}
    missing = bases - KNOWN_LOCAL_IMAGES - ALLOWED_UPSTREAM

    assert not missing, (
        f"docker run-anrop refererar till {sorted(missing)}, men ingen "
        f"Dockerfile hittades i repot. Antingen lägg till Dockerfile under "
        f"docker/<namn>/, uppdatera KNOWN_LOCAL_IMAGES, eller lägg till i "
        f"ALLOWED_UPSTREAM om det är en avsiktlig upstream-image."
    )


def test_no_latest_tag_in_dockerfiles() -> None:
    """Dockerfiles must not use ``FROM x:latest`` — pin to explicit version.

    ``:latest`` makes builds non-reproducible: the same Dockerfile produces
    different images on different days. Pin to ``FROM mundialis/esa-snap:13.0.0``,
    ``FROM python:3.11.9-slim-bookworm``, etc.
    """
    # Använd git-tracked Dockerfiles, inte rglob (.venv-sr/ vandring → 15+ s).
    import subprocess
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "*Dockerfile*", "**/Dockerfile*"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        dockerfiles = [REPO / line for line in result.stdout.splitlines() if line]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        dockerfiles = list(REPO.rglob("Dockerfile*"))
    dockerfiles = [
        f for f in dockerfiles
        if not any(part in str(f) for part in SKIP_PARTS)
    ]

    violations: list[tuple[Path, int, str]] = []
    for f in dockerfiles:
        try:
            for lineno, line in enumerate(f.read_text().splitlines(), start=1):
                if line.strip().upper().startswith("FROM ") and ":latest" in line:
                    violations.append((f.relative_to(REPO), lineno, line.strip()))
        except (OSError, UnicodeDecodeError):
            continue

    assert not violations, (
        "Dockerfile FROM-rader använder :latest — ej reproducerbart:\n"
        + "\n".join(f"  {f}:{ln}  {src}" for f, ln, src in violations)
    )


def test_pipeline_images_have_run_script() -> None:
    """Each pipeline image directory should ship a runner (run.sh / Makefile).

    The cloud-models reference uses an ENTRYPOINT in the Dockerfile +
    a host-side Python driver. Either pattern is acceptable, but the
    docker/<name>/ directory must contain at least one of:
      - run.sh
      - Makefile
      - <name>.py with a __main__ block
      - README.md that documents docker run invocation
    """
    docker_root = REPO / "docker"
    if not docker_root.is_dir():
        pytest.skip("no docker/ directory")

    missing: list[str] = []
    for sub in sorted(docker_root.iterdir()):
        if not sub.is_dir() or not (sub / "Dockerfile").is_file():
            continue
        candidates = [
            sub / "run.sh",
            sub / "Makefile",
            sub / "README.md",
        ]
        candidates += list(sub.glob("*.py"))
        if not any(c.exists() for c in candidates):
            missing.append(str(sub.relative_to(REPO)))

    assert not missing, (
        f"Pipeline-image-kataloger saknar run-skript/README/Python-driver: "
        f"{missing}. Lägg till minst en av: run.sh, Makefile, README.md, "
        f"eller en *.py-fil med entry-point."
    )
