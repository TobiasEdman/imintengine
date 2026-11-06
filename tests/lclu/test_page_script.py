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
    page = (ROOT/'docs/lclu.html').read_text()
    assert re.search(r'<script defer src="js/lclu\.js\?v=\d+"></script>', page)
    assert re.search(r'<link rel="stylesheet" href="css/lclu\.css\?v=\d+">', page)


LABEL = '2025 · växtfenologi från 2024'
NOTE = 'Uppskattningsvis ~7 % av pixlarna kan avvika jämfört med en karta med årets växtfenologi.'


def test_a_substituted_year_carries_the_apis_label_wherever_it_is_named(results):
    shown = results['substituted']
    assert shown['error'] is None
    assert shown['creations'] == [{'place': 'A', 'year': 2025}, {'place': 'A', 'year': 2024}]
    chosen, made = shown['chosen'], shown['made']
    assert chosen['readout'] == chosen['valuetext'] == LABEL
    assert chosen['substituted_ticks'] == [['2025', LABEL]]
    assert made['status'] == f'Årskarta {LABEL} är klar.'
    assert made['result_shown'] and made['title'] == f'Årskarta {LABEL}' and made['note'] == NOTE
    assert made['readout'] == f'{LABEL} · skapad'
    assert shown['span'] == '2023⁠–⁠2025'
    assert results['single_year']['span'] == '2024'


def test_other_years_and_an_api_without_the_list_get_no_label(results):
    other, plain = results['substituted']['other'], results['substituted']['plain']
    assert other['status'] == '2024: ingen karta än. Släpp tidsaxeln för att skapa den.'
    assert not other['result_shown'] and other['readout'] == '2024'
    assert plain['title'] == 'Årskarta 2024' and plain['note'] is None
    assert plain['status'] == 'Årskarta 2024 är klar.'
    without = results['without_list']
    assert without['error'] is None and without['substituted_ticks'] == []
    assert without['title'] == 'Årskarta 2024' and without['note'] is None


def test_the_substitution_wording_lives_only_in_the_api():
    for path in ('docs/js/lclu.js', 'docs/lclu.html'):
        text = (ROOT/path).read_text()
        assert 'växtfenologi från' not in text and '~7 %' not in text, path
