"""The demo bundle carries every script the LCLU service runs or hashes.

docker/lclu/build.py packs an explicit file list. A script the service reaches
through ``ROOT / 'scripts/<name>.py'`` but the list omits is present in tests
and on a developer's machine and missing only in the deployed bundle
(scripts/prefetch_lclu.py, found 2026-11-02 before the campaign redeploy).
"""
import re
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
