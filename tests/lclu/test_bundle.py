"""The demo bundle carries every script the LCLU service runs or hashes.

docker/lclu/build.py packs an explicit file list. A script the service reaches
through ``ROOT / 'scripts/<name>.py'`` but the list omits is present in tests
and on a developer's machine and missing only in the deployed bundle
(scripts/prefetch_lclu.py, found 2026-11-02 before the campaign redeploy).
"""
import re
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVICE = ("imint/inference/lclu.py", "imint/inference/lclu_sources.py",
           "imint/_cache_identity.py")


def test_bundle_lists_every_script_the_service_uses():
    listed = set(re.findall(r"'([a-z_]+\.py)'", (ROOT / "docker/lclu/build.py").read_text()))
    used = {name for path in SERVICE for name in re.findall(
        r"ROOT ?/ ?['\"]scripts/([a-z_]+\.py)['\"]", (ROOT / path).read_text())}
    assert used, "no script references found; the pattern no longer matches the service"
    assert used <= listed, f"missing from docker/lclu/build.py: {sorted(used - listed)}"


def test_dockerfile_python_payload_imports_launcher_outside_checkout(tmp_path):
    # Import the Python payload actually COPYed into the image. Importing from
    # the checkout hides runtime modules omitted from the Dockerfile.
    for line in (ROOT / 'docker/lclu/Dockerfile').read_text().splitlines():
        if not line.startswith('COPY '):
            continue
        _, source, destination = shlex.split(line)
        source = ROOT / source
        files = source.rglob('*.py') if source.is_dir() else [source]
        for path in files:
            if path.suffix != '.py':
                continue
            target = tmp_path / destination
            if source.is_dir():
                target /= path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    subprocess.run(
        [sys.executable, '-c', 'from docker.lclu.run import main; assert callable(main)'],
        cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(tmp_path)}, check=True,
    )
