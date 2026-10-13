"""PR76 data follow-ups: synthetic files and actual production validators."""
from datetime import date
import inspect
import socket
import numpy as np
import pytest
import rasterio

from tests.test_build_labels_rebuild import case, builder, assert_preserved
from tests.test_repair_script_io import tile, call, prepare
from imint.training.tile_time import validate_smoke_temporal_metadata


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('network forbidden in data repair regression')
    for key in ('connect', 'connect_ex', 'bind'):
        monkeypatch.setattr(socket.socket, key, denied)
    monkeypatch.setattr(socket, 'getaddrinfo', denied)


def noncrop(case, year=2020):
    payload = case['payload'].copy()
    payload.update(dates=np.array([f'{year-1}-09-10', f'{year}-05-10', f'{year}-06-10', f'{year}-08-10']),
                   label=np.ones((8, 8), np.uint8), label_mask=np.zeros((8, 8), np.uint16),
                   parcel_area_ha=np.zeros((8, 8), np.float32), n_parcels=np.int32(0))
    np.savez(case['tile'], **payload)
    return payload


def build(case, years=(), version=2018):
    kwargs = ({'allow_nmd_only_years': years}
              if 'allow_nmd_only_years' in inspect.signature(builder.build_tile_label).parameters else {})
    return builder.build_tile_label(str(case['tile']), str(case['root']/'nmd.tif'),
        str(case['root']/'refs'), str(case['root']/'refs'), nmd_version=version,
        label_out_dir=str(case['out']) if case['sidecar'] else None,
        **kwargs)


@pytest.mark.parametrize('version', [2018, 2023])
@pytest.mark.parametrize('raw', [111, 61])
def test_explicit_year_allows_nmd_noncrop_without_lpis(case, version, raw):
    noncrop(case)
    with rasterio.open(case['root']/'nmd.tif', 'r+') as dst:
        dst.write(np.full((32,32), raw, dtype=np.uint16), 1)
    source = case['tile'].read_bytes()
    result = build(case, (2020,), version)
    assert result['status'] == 'ok', result
    assert result['lpis_policy'] == 'nmd_only_missing_lpis'
    with np.load(case['target'], allow_pickle=False) as data:
        assert data['label_year'] == 2020
        assert data['label_lpis_policy'] == 'nmd_only_missing_lpis'
        assert np.all(data['label'] == (1 if raw == 111 else 10))
        assert np.all(data['label_mask'] == 0) and data['n_parcels'] == 0
    if case['sidecar']:
        assert case['tile'].read_bytes() == source


@pytest.mark.parametrize('years', [(), (2019,), (2022,)])
def test_missing_lpis_requires_exact_allowance(case, years):
    noncrop(case)
    source, target = case['tile'].read_bytes(), case['target'].read_bytes()
    result = build(case, years)
    assert result['status'] == 'failed' and 'missing_lpis_reference:2020' in result['reason']
    assert_preserved(case, source, target)


@pytest.mark.parametrize('crop_evidence', ['parcels', 'mask', 'name', 'nmd', 'area', 'label'])
def test_allowance_cannot_bypass_crop_evidence(case, crop_evidence):
    payload = noncrop(case)
    if crop_evidence == 'parcels': payload['n_parcels'] = np.int32(1)
    if crop_evidence == 'mask': payload['label_mask'][0,0] = 4
    if crop_evidence == 'area': payload['parcel_area_ha'][0,0] = 0.01
    if crop_evidence == 'label': payload['label'][0,0] = 11
    if crop_evidence == 'name':
        renamed = case['tile'].with_name('crop_tile.npz')
        case['tile'].rename(renamed)
        case['tile'] = renamed
        case['target'] = case['out']/renamed.name if case['sidecar'] else renamed
        if case['sidecar']: np.savez(case['target'], label=np.zeros((8,8)))
    if crop_evidence == 'nmd':
        with rasterio.open(case['root']/'nmd.tif', 'r+') as dst:
            dst.write(np.full((32,32), 3, dtype=np.uint16),1)
    np.savez(case['tile'], **payload)
    source, target = case['tile'].read_bytes(), case['target'].read_bytes()
    result = build(case, (2020,))
    assert result['status'] == 'failed', result
    assert_preserved(case, source, target)


def test_allowance_keeps_available_same_year_lpis(case):
    noncrop(case, 2022)
    result = build(case, (2022,))
    assert result['status'] == 'ok', result
    with np.load(case['target']) as data:
        assert data['label_lpis_policy'] == 'same_year_lpis'
        assert data['label_year'] == 2022


@pytest.mark.parametrize('operation', ['patch','blank'])
def test_alias_conflict_fails_without_write(tmp_path, monkeypatch, operation):
    module=prepare(operation, monkeypatch)
    path, arrays=tile(tmp_path,'both')
    arrays['image'] = arrays['spectral'] + 1
    np.savez_compressed(path, **arrays)
    before=path.read_bytes()
    result=call(module,path,operation)
    assert result['status'] not in ('ok','blanked','already_patched','skip'), result
    assert 'alias' in result.get('msg','').lower(), result
    assert path.read_bytes()==before
    assert not path.with_suffix('.tmp.npz').exists()


@pytest.mark.parametrize('layout', ['spectral','image','both'])
def test_autumn_metadata_passes_production_validator(tmp_path, monkeypatch, layout):
    module=prepare('patch',monkeypatch)
    path, arrays=tile(tmp_path,layout)
    if layout=='both': arrays['image']=arrays['spectral'].copy()
    arrays['temporal_mask'][0]=0
    np.savez_compressed(path,**arrays)
    result=call(module,path,'patch')
    assert result['status']=='ok',result
    with np.load(path,allow_pickle=False) as data:
        report=validate_smoke_temporal_metadata(data,2018)
        assert report['valid_frames']==4
        assert data['doy'][0]==date(2017,9,20).timetuple().tm_yday
        assert data['temporal_mask'][0]==1
        if layout=='both':np.testing.assert_array_equal(data['image'],data['spectral'])


@pytest.mark.parametrize('bad_date', ['2018-09-20','2017-07-20','not-a-date'])
def test_bad_acquisition_preserves_original(tmp_path,monkeypatch,bad_date):
    module=prepare('patch',monkeypatch)
    module._fetch_single_scene=lambda *a,**kw:(np.ones((6,2,2),np.float32),bad_date)
    path,_=tile(tmp_path,'spectral');before=path.read_bytes()
    result=call(module,path,'patch')
    assert result['status'] not in ('ok','already_patched'),result
    assert path.read_bytes()==before
    assert not path.with_suffix('.tmp.npz').exists()


def test_blank_is_explicitly_ineligible_for_sealed_smoke(tmp_path,monkeypatch):
    module=prepare('blank',monkeypatch)
    path,arrays=tile(tmp_path,'both');arrays['image']=arrays['spectral'].copy()
    np.savez_compressed(path,**arrays)
    assert call(module,path,'blank')['status']=='blanked'
    with np.load(path,allow_pickle=False) as data:
        assert data['dates'][0]=='' and data['doy'][0]==0 and data['temporal_mask'][0]==0
        np.testing.assert_array_equal(data['image'],data['spectral'])
        assert np.all(data['spectral'][:6]==0)
        with pytest.raises(ValueError,match='valid autumn'):
            validate_smoke_temporal_metadata(data,2018)


@pytest.mark.parametrize('kind', ['utforda', 'anmalda'])
def test_lpis_allowance_does_not_relax_sks_requirement(case, kind):
    noncrop(case)
    case['files'][kind].unlink()
    source,target=case['tile'].read_bytes(),case['target'].read_bytes()
    result=build(case,(2020,))
    assert result['status']=='failed' and kind in result['reason']
    assert_preserved(case,source,target)


def test_allowance_does_not_hide_corrupt_same_year_lpis(case):
    noncrop(case,2022)
    case['files']['lpis'].write_bytes(b'not-parquet')
    source,target=case['tile'].read_bytes(),case['target'].read_bytes()
    assert build(case,(2022,))['status']=='failed'
    assert_preserved(case,source,target)


def test_explicit_allowance_cli_disk_roundtrip(case,monkeypatch):
    import sys
    noncrop(case)
    args=['build_labels.py','--data-dir',str(case['root']),
          '--nmd-raster',str(case['root']/'nmd.tif'),
          '--lpis-dir',str(case['root']/'refs'),'--sks-dir',str(case['root']/'refs'),
          '--workers','1','--allow-nmd-only-year','2020']
    if case['sidecar']:args += ['--label-out-dir',str(case['out'])]
    monkeypatch.setattr(sys,'argv',args)
    before=case['tile'].read_bytes()
    builder.main()
    with np.load(case['target'],allow_pickle=False) as data:
        assert data['label_year']==2020
        assert data['label_lpis_policy']=='nmd_only_missing_lpis'
    if case['sidecar']:assert case['tile'].read_bytes()==before


def test_previous_partial_autumn_repair_is_not_silently_skipped(tmp_path,monkeypatch):
    module=prepare('patch',monkeypatch)
    path,arrays=tile(tmp_path,'spectral')
    arrays['dates'][0]='2017-09-20'  # Old script left original DOY behind.
    np.savez_compressed(path,**arrays)
    assert call(module,path,'patch')['status']=='ok'
    with np.load(path,allow_pickle=False) as data:
        validate_smoke_temporal_metadata(data,2018)
        assert data['doy'][0]==263


def test_postwrite_validation_failure_preserves_original(tmp_path,monkeypatch):
    module=prepare('patch',monkeypatch)
    path,_=tile(tmp_path,'spectral');before=path.read_bytes()
    original=validate_smoke_temporal_metadata
    calls=0
    def validate(*args,**kwargs):
        nonlocal calls
        calls+=1
        result=original(*args,**kwargs)
        if calls==2:raise ValueError('synthetic post-write validation failure')
        return result
    monkeypatch.setattr(module,'validate_smoke_temporal_metadata',validate,raising=False)
    assert call(module,path,'patch')['status']=='write_error'
    assert calls==2 and path.read_bytes()==before
    assert not path.with_suffix('.tmp.npz').exists()
