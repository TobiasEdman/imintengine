"""Acceptance must reject incomplete or internally inconsistent releases."""
import copy
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from shapely.geometry import box, mapping

from imint.inference.lclu import EMBEDDING, Grid, public_identity, write_json
from imint.data.wekeo_vpp import _VPP_PRODUCT_TYPES
from scripts.cache_predict import file_identity
from scripts import verify_lclu_campaign as audit
from tests.lclu.test_service import bundle


@pytest.fixture
def campaign(bundle, tmp_path, monkeypatch):
    grid, arrays, entry, input_path = bundle
    plan = {'version': 1, 'years': [2024, 2025], 'blocks': [
        {'year': y, 'lon': 17.65, 'lat': 59.85, 'grid_size': 256, 'scales_size': 256,
         'version': '1.0', 'variant': 'vultr'} for y in (2024, 2025)],
        'areas': [{'id': 'a', 'block_keys': ['2024_17.65_59.85', '2025_17.65_59.85'],
                   'bounds_3006': [grid.east-3700, grid.north-3700, grid.east+3700, grid.north+3700],
                   'municipality': {'geometry': mapping(box(17.63, 59.85, 17.65, 59.87)),
                                    'properties': {'name': 'Synthetic area'}}}]}
    plan_path = tmp_path/'campaign.json'
    write_json(plan_path, plan)
    # Only the private checkpoint boundary is replaced; inputs, hashes, geometry,
    # years, persisted entries and class arrays exercise the real audit.
    checkpoint = tmp_path/'synthetic-checkpoint'
    checkpoint.write_bytes(b'explicit synthetic fixture; no weights loaded')
    monkeypatch.setattr(audit, 'R4_SHA256', file_identity(checkpoint)['sha256'])
    ready = {'area_id': 'a', 'year': 2024, 'status': 'source_ready'}
    gap = {'area_id': 'a', 'year': 2025, 'status': 'missing_vpp',
           'vpp_missing_metrics': ['SOSD'], 'vpp_missing_files': []}
    write_json(tmp_path/'preflight.json', {'years': [2024, 2025], 'cells': [ready, gap]})
    root = tmp_path/'regions/a/2024'
    qualification = root/'qualification-2024'
    qualification.mkdir(parents=True)
    def source(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return file_identity(path)

    index_path = tmp_path/'vpp/index.json'
    vpp_files = {metric+'.tif': public_identity(source(tmp_path/'vpp'/f'{metric}.tif', metric.encode()))
                 for metric in _VPP_PRODUCT_TYPES}
    write_json(index_path, {name: {'year': 2024, 'season': 1, 'metric': name[:-4],
                                  'bounds_4326': [17, 59, 18, 61]} for name in vpp_files})
    skg_path = tmp_path/'skg/MANIFEST.json'
    raster = source(skg_path.parent/'derived.tif', b'synthetic forestry')
    forest = {'derived': {'derived.tif': {'sha256': raster['sha256'], 'from': 'source.zip',
                                        'bounds': plan['areas'][0]['bounds_3006']}},
              'files': {'source.zip': {'sha256': 'b'*64}}}
    write_json(skg_path, forest)
    vrts = ['markfukt/markfukt.vrt', 'tradhojd/tradhojd_10m.vrt',
            *[f'grunddata/{name}_omdrev1.vrt' for name in ('volym', 'grundyta', 'medeldiameter')]]
    skg_vrts = [{**source(skg_path.parent/name, b'synthetic VRT inventory'), 'path': name} for name in vrts]
    dem = source(tmp_path/'dem.tif', b'synthetic DEM')
    vectors = source(tmp_path/'vectors.tif', b'synthetic vectors')
    scales = source(tmp_path/'scales.tif', b'synthetic scales')
    cog_path = tmp_path/'cog-index.json'
    write_json(cog_path, {'version': 1, 'embedding': EMBEDDING, 'blocks': [
        {'year': 2024, 'lon': 17.65, 'lat': 59.85, 'vectors': vectors, 'scales': scales,
         'original': [{'size': 256}, {'size': 256}]}]})
    config = {'version': 1, 'embedding': EMBEDDING, 'vpp_index': file_identity(index_path),
              'identity_cache': str(tmp_path/'receipts'), 'vpp_files': vpp_files,
              'cog_index': file_identity(cog_path), 'skg_manifest': file_identity(skg_path),
              'skg_vrts': skg_vrts, 'dem_sources': [dem],
              'qualified_regions': [{'years': [2024], 'country': 'SE', 'name': 'Synthetic area',
                                     'bounds_3006': plan['areas'][0]['bounds_3006'],
                                     'geometry': plan['areas'][0]['municipality']['geometry']}]}
    entry['auxiliary_provenance'].update({
        'vpp': {'year': 2024, 'season': 1, 'index': public_identity(config['vpp_index']),
                'files': [{'file': name, **record} for name, record in vpp_files.items()]},
        'forestry': {'mirror_manifest_sha256': config['skg_manifest']['sha256'],
                    'counties': ['03'], 'grunddata_omdrev_pixels': {},
                    'rasters': {'derived.tif': {'sha256': raster['sha256'], 'source_sha256': 'b'*64}}},
        'dem': {'files': [public_identity(dem)]}})
    entry['embedding_provenance'] = {'cog_index': public_identity(config['cog_index']),
        'storage': 'compact-cog', 'blocks': 1, 'files': [public_identity(vectors), public_identity(scales)]}
    config_path = root/'sources-2024.json'
    entry_path = qualification/'entry.json'
    prediction_path = qualification/'prediction.npy'
    prediction = np.full((504, 504), 3, dtype=np.uint8)
    prediction[252, 252] = 4
    np.save(prediction_path, prediction)
    cell = {**ready, 'status': 'qualified', 'prediction': public_identity(file_identity(prediction_path)),
            'class_ids': [3, 4]}
    manifest = {'version': 1, 'device': 'cpu',
                'campaign': public_identity(file_identity(plan_path)), 'checkpoint': file_identity(checkpoint),
                'coverage': [cell, copy.deepcopy(gap)]}

    def save():
        checking = root/'sources-2024-checking.json'
        write_json(checking, {**config, 'qualified_regions': []})
        entry['source_config'] = public_identity(file_identity(checking))
        write_json(entry_path, entry)
        cell['qualification'] = public_identity(file_identity(entry_path))
        config['qualified_regions'][0]['verification'] = cell['qualification']['sha256']
        write_json(config_path, config)
        manifest['preparations'] = [{'config': file_identity(config_path), 'python': sys.executable,
                                    'timeout_seconds': 300}]
        write_json(tmp_path/'serving.json', manifest)
    save()
    return plan_path, tmp_path, manifest, config, entry, arrays, input_path, save


def test_acceptance_reads_real_arrays_and_reports_explicit_gap(campaign):
    plan, work, *_ = campaign
    report = audit.verify_campaign(plan, work)
    assert report['passed'], report['errors']
    assert report['counts'] == {'qualified': 1, 'missing_vpp': 1}
    assert report['cells'][0]['embedding_valid_pixels'] == 504**2 - 100
    assert report['cells'][0]['center_class'] == 4


@pytest.mark.parametrize('fault', ['missing_cell', 'duplicate_cell', 'prepared', 'wrong_classes', 'invented_year'])
def test_acceptance_rejects_incomplete_or_invented_coverage(campaign, fault):
    plan, work, manifest, config, entry, arrays, input_path, save = campaign
    if fault == 'missing_cell': manifest['coverage'].pop()
    if fault == 'duplicate_cell': manifest['coverage'].append(copy.deepcopy(manifest['coverage'][0]))
    if fault == 'prepared': manifest['coverage'][0]['status'] = 'prepared'
    if fault == 'wrong_classes': manifest['coverage'][0]['class_ids'] = [3]
    if fault == 'invented_year': manifest['coverage'][1]['status'] = 'qualified'
    save()
    assert not audit.verify_campaign(plan, work)['passed']


def test_acceptance_checks_npz_year_even_when_all_hashes_match(campaign):
    plan, work, manifest, config, entry, arrays, input_path, save = campaign
    arrays['vpp_year'] = np.int32(2023)
    np.savez_compressed(input_path, **arrays)
    entry['input'] = file_identity(input_path)
    save()
    report = audit.verify_campaign(plan, work)
    assert not report['passed']
    assert any('Input years must exactly match' in error for error in report['errors'])


def test_acceptance_detects_source_index_changed_since_staging(campaign):
    plan, work, manifest, config, entry, arrays, input_path, save = campaign
    write_json(Path(config['vpp_index']['path']), {'replacement': True})
    report = audit.verify_campaign(plan, work)
    assert not report['passed']
    assert any('Pinned file identity mismatch' in error for error in report['errors'])


def test_live_catalogue_cannot_enable_missing_year(campaign, monkeypatch):
    plan, work, manifest, config, entry, arrays, input_path, save = campaign
    area = json.loads(plan.read_text())['areas'][0]['municipality']
    area['properties']['years'] = [2024]
    def get(url, **kwargs):
        data = ({'ready': True, 'candidate_years': [2024, 2025], 'areas': [area]}
                if url.endswith('/v1/lclu') else {'years': [2024, 2025], 'unavailable_years': []})
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(audit, 'urlopen', get)
    report = audit.verify_campaign(plan, work, 'https://example.invalid')
    assert not report['passed']
    assert any('Live year availability differs' in error for error in report['errors'])


@pytest.mark.parametrize('fault', ['bounds', 'country', 'name', 'version', 'vpp_identity',
                                  'cog_identity', 'forest_identity', 'dem_identity', 'wrong_block'])
def test_acceptance_binds_evidence_to_serving_contract(campaign, fault):
    plan, work, manifest, config, entry, arrays, input_path, save = campaign
    if fault == 'bounds': config['qualified_regions'][0]['bounds_3006'] = [0, 0, 1, 1]
    if fault == 'country': config['qualified_regions'][0]['country'] = 'NO'
    if fault == 'name': config['qualified_regions'][0]['name'] = 'Wrong area'
    if fault == 'version': config['version'] = 99
    if fault == 'vpp_identity': entry['auxiliary_provenance']['vpp']['index']['sha256'] = 'f'*64
    if fault == 'cog_identity': entry['embedding_provenance']['cog_index']['sha256'] = 'f'*64
    if fault == 'forest_identity': entry['auxiliary_provenance']['forestry']['mirror_manifest_sha256'] = 'f'*64
    if fault == 'dem_identity': entry['auxiliary_provenance']['dem']['files'][0]['sha256'] = 'f'*64
    if fault == 'wrong_block':
        path = Path(config['cog_index']['path'])
        index = json.loads(path.read_text())
        index['blocks'][0].update(lon=100, lat=1)
        write_json(path, index)
        config['cog_index'] = file_identity(path)
        entry['embedding_provenance']['cog_index'] = public_identity(config['cog_index'])
    save()
    assert not audit.verify_campaign(plan, work)['passed']


@pytest.mark.parametrize('source', ['dem.tif', 'vectors.tif', 'scales.tif', 'skg/derived.tif', 'vpp/SOSD.tif'])
@pytest.mark.parametrize('fault', ['deleted', 'same_size_corruption'])
def test_acceptance_checks_actual_source_files(campaign, source, fault):
    plan, work, *_ = campaign
    path = work/source
    if fault == 'deleted':
        path.unlink()
    else:
        path.write_bytes(b'X' * path.stat().st_size)
    assert not audit.verify_campaign(plan, work)['passed']


def test_acceptance_rejects_changed_serving_config_after_qualification(campaign):
    plan, work, manifest, config, *_ = campaign
    config['identity_cache'] = str(work/'another-receipt-dir')
    path = Path(manifest['preparations'][0]['config']['path'])
    write_json(path, config)
    manifest['preparations'][0]['config'] = file_identity(path)
    write_json(work/'serving.json', manifest)
    report = audit.verify_campaign(plan, work)
    assert not report['passed']
    assert any('differ from the qualified preparation configuration' in error for error in report['errors'])


@pytest.mark.parametrize('settings', [{'timeout_seconds': 901}, {'python': '/nonexistent/python'}])
def test_acceptance_rejects_unusable_preparation_runtime(campaign, settings):
    plan, work, manifest, *_ = campaign
    manifest['preparations'][0].update(settings)
    write_json(work/'serving.json', manifest)
    assert not audit.verify_campaign(plan, work)['passed']


@pytest.mark.parametrize('wrong_grid', [False, True])
def test_live_catalogue_checks_native_grid(campaign, monkeypatch, wrong_grid):
    plan, work, *_ = campaign
    area = json.loads(plan.read_text())['areas'][0]['municipality']
    area['properties']['years'] = [2024]
    def get(url, **kwargs):
        data = ({'ready': True, 'candidate_years': [2024, 2025], 'areas': [area]}
                if url.endswith('/v1/lclu') else {'years': [2024], 'unavailable_years': [2025],
                    'geometry': {'crs': 'EPSG:4326', 'width': 1, 'height': 1} if wrong_grid
                    else Grid.from_lonlat(17.64, 59.86).describe()})
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(audit, 'urlopen', get)
    report = audit.verify_campaign(plan, work, 'https://example.invalid')
    assert report['passed'] is (not wrong_grid), report['errors']


def test_source_receipts_are_read_only_and_rechecked_after_change(tmp_path, monkeypatch):
    from scripts.prepare_lclu import verify_source_file
    path = tmp_path/'source'
    path.write_bytes(b'first')
    record = file_identity(path)
    receipts = tmp_path/'receipts'
    verify_source_file(record, tmp_path, receipts)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in receipts.iterdir()}
    verifier = audit.SourceVerifier()
    verifier.verify(record, tmp_path, receipts)
    verifier.verify(record, tmp_path, receipts)
    assert verifier.counts == {'existing_sha256_receipts': 1}
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in receipts.iterdir()}
    path.write_bytes(b'other')
    with pytest.raises(ValueError, match='identity mismatch'):
        verifier.verify(record, tmp_path, receipts)
