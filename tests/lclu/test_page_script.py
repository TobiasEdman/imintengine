"""The annual-map page script under controlled timing (tests/lclu/page_harness.mjs).

The harness runs docs/js/lclu.js against a stub DOM, map and API whose answers
and timers the scenarios release in a chosen order, which is how the asynchronous
races on PR #128 were reproduced.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which('node')


@pytest.fixture(scope='module')
def results():
    if NODE is None:
        pytest.skip('Node.js runs the page harness')
    done = subprocess.run([NODE, str(Path(__file__).with_name('page_harness.mjs'))],
                          capture_output=True, text=True, timeout=60, check=True)
    return json.loads(done.stdout)


def test_release_creates_the_released_year_at_the_released_place(results):
    assert results['release']['creations'] == [{'place': 'A', 'year': 2023}]


def test_choosing_another_place_cancels_a_pending_release(results):
    assert results['release_then_place']['creations'] == []


def test_an_older_click_never_inspects_over_a_newer_selection(results):
    clicked = results['stale_click']
    assert clicked['pixel_jobs'] == [clicked['job_a']]
    assert clicked['title'].startswith('59.5000°') and clicked['year'] == '2024'


def test_restore_accepts_only_this_services_artifacts(results):
    assert results['restore']['overlays'] == ['https://api.test/v1/lclu/jobs/own/files/overlay.png']


def test_an_outdated_cached_page_asks_for_a_reload(results):
    outdated = results['outdated_page']
    assert outdated['error'] and outdated['requests'] == 0
    assert outdated['status'].startswith('Sidan har uppdaterats')


def test_the_page_versions_its_script():
    # A changed DOM contract must reach browsers that cached the old script.
    assert re.search(r'<script defer src="js/lclu\.js\?v=\d+"></script>', (ROOT/'docs/lclu.html').read_text())
