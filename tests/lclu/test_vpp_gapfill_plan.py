import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import derive_missing_vpp_mgrs  # noqa: E402
from scripts import plan_lclu_vpp_gapfill as gapfill  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_mgrs(monkeypatch):
    """The mgrs wheel is a pod dependency, not a test one: one cell per 100 km of easting."""
    monkeypatch.setattr(derive_missing_vpp_mgrs, 'mgrs_tiles_for_bbox',
                        lambda b: {f"C{int(x // 100000)}" for x in (b['west'], b['east'])})


def write_inputs(tmp_path):
    areas = [{'id': 'a', 'bounds_3006': [450000, 6500000, 470000, 6510000]},
             {'id': 'b', 'bounds_3006': [590000, 6600000, 620000, 6610000]}]
    cells = [{'area_id': 'a', 'name': 'Alfa', 'year': 2019, 'status': 'source_ready',
              'vpp_missing_metrics': []},
             {'area_id': 'a', 'name': 'Alfa', 'year': 2020, 'status': 'missing_vpp',
              'vpp_missing_metrics': ['SOSD']},
             {'area_id': 'b', 'name': 'Beta', 'year': 2019, 'status': 'missing_vpp',
              'vpp_missing_metrics': ['SOSD', 'EOSD']},
             {'area_id': 'b', 'name': 'Beta', 'year': 2020, 'status': 'missing_vpp',
              'vpp_missing_metrics': []}]
    campaign, preflight = tmp_path / 'campaign.json', tmp_path / 'preflight.json'
    campaign.write_text(json.dumps({'areas': areas}))
    preflight.write_text(json.dumps({'years': [2019, 2020], 'cells': cells,
                                     'vpp_index': {'sha256': 'f' * 64, 'size': 1},
                                     'existing_vpp_files': 4, 'existing_vpp_bytes': 400}))
    return campaign, preflight


def test_area_is_split_so_interior_cells_are_not_missed(fake_mgrs):
    # Corners alone would give C4 and C7; the interior C5 and C6 only appear when split.
    assert gapfill.mgrs_tiles_for_area([450000, 0, 720000, 5000]) == {'C4', 'C5', 'C6', 'C7'}
    with pytest.raises(ValueError, match='non-empty'):
        gapfill.mgrs_tiles_for_area([1, 0, 1, 5])


def test_plan_lists_only_missing_vpp_pairs_and_writes_to_disk(fake_mgrs, tmp_path, monkeypatch):
    campaign, preflight = write_inputs(tmp_path)
    out, matrix = tmp_path / 'plan.json', tmp_path / 'matrix.md'
    monkeypatch.setattr(sys, 'argv', ['plan', '--campaign', str(campaign), '--preflight',
                                      str(preflight), '--output', str(out), '--matrix', str(matrix)])
    gapfill.main()
    plan = json.loads(out.read_text())
    assert plan['status_counts'] == {'missing_vpp': 3, 'source_ready': 1}
    assert plan['pairs_by_year'] == {'2019': ['C5', 'C6'], '2020': ['C4', 'C5', 'C6']}
    assert plan['n_pairs'] == 5
    assert plan['storage_estimate']['cogs'] == 25
    assert plan['storage_estimate']['bytes_at_cached_mean'] == 2500
    assert plan['prefetch_commands'][0] == (
        'python scripts/prefetch_vpp_wekeo.py --mgrs-tiles C5,C6 --years 2019 '
        '--dest-dir /data/vpp_wekeo')
    assert len(plan['preflight_sha256']) == 64
    table = matrix.read_text()
    assert '| Alfa | ✓ | VPP |' in table and '| Beta | VPP | VPP |' in table


def test_preflight_area_outside_campaign_fails_closed(fake_mgrs, tmp_path):
    campaign, preflight = write_inputs(tmp_path)
    campaign.write_text(json.dumps({'areas': [{'id': 'a', 'bounds_3006': [0, 0, 1, 1]}]}))
    with pytest.raises(ValueError, match='missing from the campaign'):
        gapfill.plan(campaign, preflight)


def test_real_campaign_areas_straddling_18e_get_both_utm_zones():
    pytest.importorskip('mgrs')
    areas = {a['id']: a for a in json.loads(
        (ROOT / 'config/lclu/municipalities.json').read_text())['areas']}
    stockholm = gapfill.mgrs_tiles_for_area(areas['0180']['bounds_3006'])
    assert {t[:2] for t in stockholm} == {'33', '34'}
    assert gapfill.mgrs_tiles_for_area(areas['kiruna_tatort']['bounds_3006']) == {'34WDA'}
