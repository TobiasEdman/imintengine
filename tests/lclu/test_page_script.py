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
    # The released year first; the place's other year (2024) follows in the background.
    assert results['release']['creations'] == [{'place': 'A', 'year': 2023}, {'place': 'A', 'year': 2024}]


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


def test_the_stylesheet_reaches_only_the_page():
    # The showcase loads lclu.css into index.html for its Marktäckekartering
    # sub-tab, so every rule must be scoped to .lclu (lclu.html's body, the tab).
    css = re.sub(r'/\*.*?\*/', '', (ROOT/'docs/css/lclu.css').read_text(), flags=re.S)
    preludes = re.findall(r'([^{}]+)\{', re.sub(r'@media[^{]*\{', '', css))
    selectors = [s.strip() for prelude in preludes for s in prelude.split(',')]
    assert selectors and [s for s in selectors if s != '.lclu' and not s.startswith('.lclu ')] == []
    assert '<body class="lclu">' in (ROOT/'docs/lclu.html').read_text()


def test_the_showcase_tab_shows_this_page_first():
    index = (ROOT/'docs/index.html').read_text()
    nav = re.search(r'id="tab-lulc">\s*<div class="sub-tab-nav">(.*?)</div>', index, re.S)[1]
    assert re.findall(r'data-subtab="(\w+)"', nav)[0] == 'lclu'
    assert re.findall(r'class="sub-tab active" data-subtab="(\w+)"', nav) == ['lclu']
    assert re.search(r'<div class="sub-tab-content active lclu" id="tab-lclu" data-src="lclu\.html"></div>', index)
    assert "loadAnnualMap();" in (ROOT/'docs/js/app.js').read_text()


LABEL = '2025 · växtfenologi från 2024'
NOTE = 'Uppskattningsvis ~7 % av pixlarna kan avvika jämfört med en karta med årets växtfenologi.'


def test_a_substituted_year_carries_the_apis_label_wherever_it_is_named(results):
    shown = results['substituted']
    assert shown['error'] is None
    # 2024 is the nearest background year; the user's create waits for it, then 2023 follows.
    assert shown['creations'] == [{'place': 'A', 'year': 2025}, {'place': 'A', 'year': 2024}, {'place': 'A', 'year': 2023}]
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


# Tobias 2026-11-07/08: once a place and a year are chosen, make every year,
# nearest first, forward and backward, so the timeline is ready sooner.
def test_the_other_years_follow_nearest_first_later_first_at_a_tie(results):
    fill = results['fill_order']
    assert fill['years'] == [2020, 2021, 2019, 2022, 2018, 2023, 2017, 2024, 2025]
    assert fill['fill'] == 'Alla år för platsen är klara.'
    assert fill['status'] == 'Årskarta 2020 är klar.'  # the status line stays the user's


def test_one_job_at_a_time(results):
    step = results['one_at_a_time']
    assert step['during'] == [2020, 2021] and step['fill'] == 'Övriga år: 0 av 8 klara …'
    assert step['after'] == 9


def test_a_full_queue_is_retried_later_not_at_once(results):
    retry = results['retry']
    assert [(post['year'], post['at'], post['refused']) for post in retry['posts']] == [
        (2024, 0, False), (2023, 0, True), (2023, 15000, False)]
    assert retry['before'] == 2 and retry['years'] == [2024, 2023]


def test_another_place_stops_the_old_places_years(results):
    change = results['place_change']
    assert change['creations'] == [{'place': 'A', 'year': 2020}, {'place': 'A', 'year': 2021}]
    assert change['fill'] is None


def test_maps_already_in_the_session_are_not_made_again(results):
    assert results['existing']['years'] == [2020, 2019, 2022]


def test_a_restored_session_starts_no_background_years(results):
    assert results['restore_no_fill'] == {'posts': 0, 'fill': None}


def test_a_year_the_user_asks_for_goes_before_the_background_years(results):
    first = results['user_first']
    assert first['years'] == [2020, 2021, 2025, 2019, 2022, 2018, 2023, 2017, 2024]
    assert first['waiting'].endswith('står på tur …')
    assert first['status'].startswith('Årskarta 2025') and first['status'].endswith('är klar.')


def test_removing_a_place_stops_its_years_and_drops_late_results(results):
    removed = results['remove_place']
    assert removed == {'creations': [2020, 2021], 'listed': 0, 'fill': None}


def test_choosing_the_same_place_again_resumes_the_years(results):
    again = results['reselect']
    assert again['years'] == [2020, 2021, 2019, 2022, 2018, 2023, 2017, 2024, 2025]
    assert again['fill'] == 'Alla år för platsen är klara.'


def test_removing_the_place_during_the_users_year_drops_it_and_starts_nothing(results):
    assert results['user_remove'] == {'years': [2020, 2021, 2025], 'listed': 0, 'fill': None}


def test_choosing_the_same_place_again_keeps_a_retry_wait(results):
    again = results['retry_reselect']
    assert again['during'] == 2
    assert again['posts'] == [[2024, 0, False], [2023, 0, True], [2023, 15000, False]]


def test_a_year_queued_before_the_removal_is_cancelled(results):
    assert results['pending_remove'] == {'years': [2020, 2021], 'listed': 0, 'fill': None}
