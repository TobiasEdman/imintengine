"""Offline regressions for PR68 review: synthetic identities and rasters only."""
import json
import socket
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import prepare_nfi_holdout as prep
from scripts.score_nfi_holdout import paired_report
from scripts.nfi_nmd_coverage import coverage_masks
from scripts.compare_nmd2023_nfi import sample_nmd_unified
from tests.test_nfi_preparation_e2e import prepared_inputs


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a, **kw):
        raise AssertionError('network prohibited in synthetic PR68 regression')
    for name in ('connect', 'connect_ex', 'bind'):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, 'getaddrinfo', denied)


def rows():
    return pd.DataFrame(dict(TractID=[1, 2, 3, 4], PlotID=[1]*4,
        Year=[2022, 2022, 2021, 2022], tile_name=['cohort', 'overlap', 'other_year', 'clear'],
        tile_role=['cohort', 'holdout_copy', 'holdout_copy', 'nfi2022'],
        row=[2]*4, col=[2]*4, Easting=[10., 10., 10., 100.], Northing=[10.]*4))


def metadata(frame):
    return {r.tile_name: dict(height=8, width=8, year=r.Year) for r in frame.itertuples()}


def test_primary_drops_cohort_plot_year_even_without_teacher_identity():
    frame = rows()
    held, counts = prep.select_holdout(frame, frame.iloc[:0][prep.NFI_KEY], metadata(frame))
    assert 'cohort' not in set(held.tile_role)
    assert counts['cohort_overlap_rows'] == 1


def test_same_year_footprint_excludes_other_root_and_duplicate_key(tmp_path):
    path = tmp_path/'unindexed-training.npz'
    np.savez(path, year=2022, bbox_3006=[0., 0., 20., 20.])
    footprints = prep.training_footprints({path.stem: path})
    frame = rows()
    # A second tile/coordinate for the same identity cannot bypass exclusion.
    duplicate = frame.iloc[[1]].assign(tile_name='duplicate', Easting=100.)
    frame = pd.concat([frame, duplicate], ignore_index=True)
    held, counts = prep.select_holdout(frame, frame.iloc[:0][prep.NFI_KEY], metadata(frame), footprints=footprints)
    assert held.TractID.tolist() == [3, 4]
    assert counts['training_footprint_overlap_rows'] == 2


@pytest.mark.parametrize('bbox', [[0, 0, 0, 20], [20, 0, 0, 20], [0, 0, float('nan'), 20], [0, 0, 20]])
def test_invalid_training_bbox_fails_closed(tmp_path, bbox):
    path = tmp_path/'train.npz'; np.savez(path, year=2022, bbox_3006=bbox)
    with pytest.raises(ValueError, match='footprint bbox'):
        prep.training_footprints({'train': path})


def test_legacy_footprint_uses_actual_shape_and_requires_year(tmp_path):
    path = tmp_path/'train.npz'
    data = dict(spectral=np.ones((24, 8, 12)), easting=100., northing=200.)
    np.savez(path, year=2022, **data)
    assert prep.training_footprints({'train': path})[0]['bbox'] == [40.,160.,160.,240.]
    np.savez(path, **data)
    with pytest.raises(ValueError, match='footprint year'):
        prep.training_footprints({'train': path})


def test_root_year_rejects_wrong_year_in_real_tile(tmp_path):
    path = tmp_path/'wrong.npz'; np.savez(path, year=2019)
    with pytest.raises(ValueError, match='root year'):
        prep.validate_root_year(path, 2021)
    prep.validate_root_year(path, None)


def test_all_roots_ignore_temporary_npz(tmp_path):
    (tmp_path/'tile.npz').write_bytes(b'fixture')
    (tmp_path/'tile.tmp.npz').write_bytes(b'incomplete')
    assert set(prep.tile_files(tmp_path)) == {'tile'}


def test_main_declares_role_year_and_filters_campaign_temp(prepared_inputs, monkeypatch):
    extra = prepared_inputs['out-dir'].parent/'nfi2019'; extra.mkdir()
    with np.load(prepared_inputs['staging-dir']/'campaign000.npz') as data:
        tile = dict(data)
    tile['dates'] = np.array(['2018-10-07','2019-05-29','2019-06-28','2019-07-28'])
    np.savez(extra/'extra2019.npz', **tile)
    (extra/'ignored.tmp.npz').write_bytes(b'incomplete')
    (prepared_inputs['staging-dir']/'ignored.tmp.npz').write_bytes(b'incomplete')
    path = prepared_inputs['plot-index']; frame = pd.read_parquet(path)
    pd.concat([frame, frame.iloc[[2]].assign(TractID=7, Year=2019, tile_name='extra2019')]).to_parquet(path)
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--evaluation-root', f'nfi2019:2019:{extra}'])
    prep.main()
    manifest = json.loads((prepared_inputs['out-dir']/'manifest.json').read_text())
    held = pd.read_parquet(prepared_inputs['out-dir']/'holdout.parquet')
    assert manifest['campaign']['tiles'] == 475
    assert manifest['evaluation_roots'][-1]['role'] == 'nfi2019'
    assert manifest['evaluation_roots'][-1]['year'] == 2019
    assert held.loc[held.Year == 2019, 'tile_role'].tolist() == ['nfi2019']
    assert len(manifest['training_footprint_inputs']) == 2


def test_main_rejects_wrong_root_year_before_freeze(prepared_inputs, monkeypatch):
    extra = prepared_inputs['out-dir'].parent/'nfi2021'; extra.mkdir()
    np.savez(extra/'wrong2019.npz', year=2019)
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--evaluation-root', f'nfi2021:2021:{extra}'])
    with pytest.raises(ValueError, match='root year'):
        prep.main()
    assert not prepared_inputs['out-dir'].exists()


def score_inputs():
    held = rows().assign(nfi_forest=[1,2,3,4])
    held['Year'] = [2021,2021,2022,2022]
    held['tile_role'] = ['holdout_copy','nfi2021','holdout_copy','nfi2022']
    held['Easting'] = [100000.,200000.,300000.,400000.]
    protocol = dict(primary_population='all', bootstrap_seed=1, bootstrap_samples=20, block_km=50, sesoi=.02)
    return held, {'model': np.array([1,1,3,3])}, {'NMD2023': (np.array([1,2,4,4]), np.array([111,112,0,114]))}, protocol


def test_year_and_role_year_scores_use_common_support():
    report = paired_report(*score_inputs())
    assert report['year_scores']['2021']['model']['n'] == 2
    assert report['year_scores']['2021']['model']['overall_exact'] == .5
    assert report['year_scores']['2022']['model']['n'] == 1
    assert report['year_scores']['2022']['model']['overall_exact'] == 0
    assert '2022' not in report['role_year_scores']['holdout_copy']
    for year, sources in report['year_scores'].items():
        for source, score in sources.items():
            assert np.asarray(score['confusion_matrix']).sum() == score['n']
            assert score['n'] == sum(roles.get(year, {}).get(source, {}).get('n', 0)
                                      for roles in report['role_year_scores'].values())


def test_score_requires_explicit_primary_population():
    held, predictions, baselines, protocol = score_inputs()
    del protocol['primary_population']
    with pytest.raises(ValueError, match='primary_population'):
        paired_report(held, predictions, baselines, protocol)


def test_nmd_masked_nonzero_nodata_matches_prepare_and_score(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    path = tmp_path/'nmd.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=3, height=1, count=1,
                       dtype='uint16', crs='EPSG:3006', nodata=999,
                       transform=from_origin(0, 10, 10, 10)) as dst:
        dst.write(np.array([[[111,999,0]]], dtype=np.uint16))
    xy = [[5.,5.],[15.,5.],[25.,5.],[100.,5.]]
    coverage = coverage_masks({'coordinates':xy,'baselines':{'NMD2023':str(path)}})['NMD2023']
    _, raw = sample_nmd_unified(str(path), *np.array(xy).T)
    assert coverage == [True,False,False,False]
    assert (raw != 0).tolist() == coverage


@pytest.mark.parametrize('year', [2021.5, float('nan'), [2021], '2021'])
def test_root_year_never_truncates_or_coerces_invalid_metadata(tmp_path, year):
    path = tmp_path/'invalid.npz'; np.savez(path, year=year)
    with pytest.raises(ValueError, match='tile year'):
        prep.validate_root_year(path, 2021)
