"""Multi-year routing, shared storage budgets and fail-closed source coverage."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
from shapely.geometry import box, mapping
from shapely.ops import transform
from pyproj import Transformer

from imint.inference.lclu import Grid, EMBEDDING, write_json
from imint.inference.lclu_sources import RegionalSources
from scripts.cache_predict import file_identity
from scripts.stage_lclu_campaign import load_campaign, preflight, stage_campaign


def fixture(tmp_path):
    data = tmp_path/'data'
    index = {}
    for metric in ('SOSD', 'EOSD', 'LENGTH', 'MAXV', 'MINV'):
        name = f'{metric}_2024.tif'
        p = data/'vpp_wekeo'/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'file-presence-only')
        index[name] = {'year':2024, 'season':1, 'metric':metric, 'bounds_4326':[17,58,19,61]}
    write_json(data/'vpp_wekeo/index.json', index)
    for name in ('MANIFEST.json', 'markfukt/markfukt.vrt', 'tradhojd/tradhojd_10m.vrt',
                 'grunddata/volym_omdrev1.vrt', 'grunddata/grundyta_omdrev1.vrt',
                 'grunddata/medeldiameter_omdrev1.vrt'):
        p = data/'skg_open'/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')
    blocks = [{'year':y,'lon':18.05,'lat':59.35,'version':'1.0','variant':'vultr',
               'grid_size':128128,'scales_size':4128} for y in (2017,2024,2025)]
    area = {'id':'a', 'municipality':{'properties':{'name':'A'}},
            'bounds_3006':[665000,6576000,674000,6585000],
            'block_keys':[f'{y}_18.05_59.35' for y in (2017,2024,2025)]}
    plan = {'version':1,'years':[2017,2024,2025], 'registry':{},
            'areas':[area,{**copy.deepcopy(area),'id':'b'}], 'blocks':blocks}
    path = tmp_path/'campaign.json'; write_json(path,plan)
    return data, path, plan


def test_preflight_deduplicates_shared_blocks_and_never_substitutes_vpp_year(tmp_path):
    data,path,plan = fixture(tmp_path)
    report = preflight(load_campaign(path), [2017,2024,2025], data)
    assert len(report['cells']) == 6
    assert [c['status'] for c in report['cells']] == ['missing_vpp','source_ready','missing_vpp']*2
    assert report['requested_unique_blocks']==3
    assert report['source_ready_unique_blocks']==1
    assert report['source_ready_original_bytes']==132256
    assert report['existing_vpp_files']==5
    assert report['existing_vpp_bytes']==5*len(b'file-presence-only')
    (data/'vpp_wekeo/MINV_2024.tif').unlink()
    missing = preflight(plan,[2024],data)
    assert missing['source_ready_unique_blocks']==0
    assert missing['cells'][0]['vpp_missing_metrics']==['MINV']


def test_missing_shared_aux_blocks_all_years_and_duplicate_requests_rejected(tmp_path):
    data,path,plan=fixture(tmp_path)
    (data/'skg_open/tradhojd/tradhojd_10m.vrt').unlink()
    report=preflight(plan,[2017,2024],data)
    assert all(c['status']=='missing_static_aux' for c in report['cells'])
    assert report['shared_aux']['missing_files']==['skg_open/tradhojd/tradhojd_10m.vrt']
    for years in ([2024,2024],[2016],[]):
        with pytest.raises(ValueError): preflight(plan,years,data)


def test_stage_budget_fails_before_network_or_model_calls(tmp_path, monkeypatch):
    data,path,_=fixture(tmp_path)
    monkeypatch.setattr('scripts.stage_lclu_campaign.stage',lambda *a,**k:pytest.fail('No stage allowed'))
    with pytest.raises(ValueError,match='explicit budget'):
        stage_campaign(path,[2024],data,tmp_path/'work',132255)
    saved=json.loads((tmp_path/'work/preflight.json').read_text())
    assert saved['source_ready_original_bytes']==132256
    assert not (tmp_path/'work/serving.json').exists()


def test_invalid_campaign_blocks_ids_and_duplicate_inventory_rejected(tmp_path):
    data,path,plan=fixture(tmp_path)
    for change in ('duplicate','id','unknown'):
        broken=copy.deepcopy(plan)
        if change=='duplicate': broken['blocks'].append(broken['blocks'][0])
        if change=='id': broken['areas'][0]['id']='../../escape'
        if change=='unknown': broken['areas'][0]['block_keys']=['unknown']
        write_json(path,broken)
        with pytest.raises(ValueError): load_campaign(path)


def test_same_area_multiple_years_has_one_selector_entry_and_exact_routing(tmp_path):
    grid=Grid.from_lonlat(18.07,59.33)
    project=Transformer.from_crs(3006,4326,always_xy=True).transform
    geom=mapping(transform(project,box(grid.east-1000,grid.north-1000,grid.east+1000,grid.north+1000)))
    settings=[]
    for year in (2017,2024):
        path=tmp_path/f'{year}.json'
        write_json(path,{'version':1,'embedding':EMBEDDING,'qualified_regions':[
            {'country':'SE','name':'Stockholm','verification':'test','years':[year],
             'geometry':geom,'bounds_3006':[grid.east-4000,grid.north-4000,grid.east+4000,grid.north+4000]}]})
        settings.append({'config':file_identity(path),'python':'unused','timeout_seconds':10})
    sources=RegionalSources(settings,tmp_path,tmp_path/'cache')
    assert len(sources.areas())==len(sources.locations())==1
    assert sources.areas()[0]['properties']['years']==[2017,2024]
    assert sources.years(grid)==[2017,2024]
    for i,year in enumerate((2017,2024)):
        assert sources.identity(grid,year)==sources.sources[i].identity(grid,year)
    assert sources.years(Grid.from_lonlat(19,63))==[]


def test_real_inventory_contains_ten_municipalities_plus_kiruna_nine_years():
    path=Path(__file__).resolve().parents[2]/'config/lclu/municipalities.json'
    plan=load_campaign(path)
    assert [a['id'] for a in plan['areas']]==[
        '0180','1480','1280','0380','0580','1980','1880','1283','0680','0581','kiruna_tatort']
    assert plan['years']==list(range(2017,2026))
    assert len(plan['blocks'])==3842
    rows={f"{r['year']}_{r['lon']:.2f}_{r['lat']:.2f}":r for r in plan['blocks']}
    assert all({rows[k]['year'] for k in a['block_keys']}==set(plan['years']) for a in plan['areas'])
    assert sum(r['grid_size']+r['scales_size'] for r in plan['blocks'] if r['year']==2024)==38499588196


def test_large_source_bundle_is_split_and_reconstructed_exactly(tmp_path):
    import base64
    import hashlib
    import io
    import os
    import tarfile
    from docker.lclu.render_job import resources
    bootstrap=b'print("test archive")\n'
    archive=tmp_path/'source.tar.gz'
    contents={'docker/lclu/bootstrap.py':bootstrap,
              'SOURCE_MANIFEST.json':json.dumps({'files':{'docker/lclu/bootstrap.py':hashlib.sha256(bootstrap).hexdigest()}}).encode(),
              'fixture.bin':os.urandom(950000)}
    with tarfile.open(archive,'w:gz') as tar:
        for name,content in contents.items():
            member=tarfile.TarInfo(name);member.size=len(content);tar.addfile(member,io.BytesIO(content))
    image='python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f'
    rendered=resources(image,2024,archive,campaign_years=(2024,),storage_gib=100,
                       max_original_bytes=39_000_000_000)
    maps=[x for x in rendered['items'] if x['kind']=='ConfigMap']
    assert len(maps)==2 and all(len(json.dumps(x))<1_048_576 for x in maps)
    chunks={k:base64.b64decode(v) for cm in maps for k,v in cm['binaryData'].items()}
    assert b''.join(chunks[k] for k in sorted(chunks))==archive.read_bytes()
    job=rendered['items'][-1]
    env={v['name']:v['value'] for v in job['spec']['template']['spec']['containers'][0]['env']}
    assert env['LCLU_BUNDLE_SHA256']==hashlib.sha256(archive.read_bytes()).hexdigest()
    assert env['LCLU_YEARS']=='2024'
    assert rendered['items'][-2]['spec']['resources']['requests']['storage']=='100Gi'
    assert len(job['spec']['template']['spec']['volumes'][-1]['projected']['sources'])==2
    with pytest.raises(ValueError,match='insufficient'):
        resources(image,2024,campaign_years=(2024,),storage_gib=20,max_original_bytes=39_000_000_000)


def test_resume_requires_space_only_for_missing_blocks(tmp_path, monkeypatch):
    import types
    import numpy as np
    from scripts import stage_lclu_campaign as campaign
    data,path,plan=fixture(tmp_path)
    # A 2 GB source already present as verified COGs, with only 1 GiB free.
    for row in plan['blocks']:
        row['grid_size']=2_000_000_000
    plan['areas']=plan['areas'][:1]
    write_json(path,plan)
    monkeypatch.setattr(campaign,'cached_block',lambda *a: {'verified':'test boundary'})
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda p:types.SimpleNamespace(free=2**30))
    checkpoint={'path':'unused','size':1,'sha256':'c'*64}
    def stage(scope,year,data,root,**kwargs):
        folder=root/f'qualification-{year}';folder.mkdir(parents=True)
        input_path=folder/'input.npz';input_path.write_bytes(b'verified synthetic input')
        write_json(folder/'entry.json',{'input':file_identity(input_path)})
        write_json(root/'serving.json',{'checkpoint':checkpoint,'preparation':{'config':{'synthetic':True}}})
    monkeypatch.setattr(campaign,'stage',stage)
    class Model:
        def __init__(self,*args):pass
        def predict(self,*args):return np.ones((504,504),np.uint8)
    monkeypatch.setattr('imint.inference.lclu.TesseraR4',Model)
    result=stage_campaign(path,[2024],data,tmp_path/'work',3_000_000_000)
    assert result['coverage'][0]['status']=='qualified'
    report=json.loads((tmp_path/'work/preflight.json').read_text())
    assert report['remaining_original_bytes']==0
    assert report['required_free_bytes']==612*2**20
    prediction=tmp_path/'work/regions/a/2024/qualification-2024/prediction.npy'
    np.testing.assert_array_equal(np.load(prediction),np.ones((504,504),np.uint8))


def test_campaign_launch_keeps_running_demo_manifest_and_api_lock(tmp_path, monkeypatch):
    import os
    import scripts.stage_lclu_campaign as campaign_module
    spec = importlib.util.spec_from_file_location(
        'lclu_run', Path(__file__).resolve().parents[2] / 'docker/lclu/run.py')
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    source, work = tmp_path / 'source', tmp_path / 'work'
    source.mkdir()
    (source / 'SOURCE_MANIFEST.json').write_text('{"files": {}}')
    running = work / 'demo' / 'serving.json'
    running.parent.mkdir(parents=True)
    running.write_text('{"running": "r7"}')
    monkeypatch.setattr(campaign_module, 'stage_campaign',
                        lambda *a, **k: {'campaign': {'sha256': 'c' * 64}, 'preparations': []})
    monkeypatch.setenv('LCLU_CAMPAIGN', 'municipalities')
    monkeypatch.setenv('LCLU_YEARS', '2017,2024')
    monkeypatch.setenv('LCLU_MAX_ORIGINAL_BYTES', '1')
    monkeypatch.setenv('LCLU_IMAGE', 'python@sha256:' + '0' * 64)
    monkeypatch.setenv('IMINT_API_OUTPUT_DIR', str(work / 'results'))  # as bootstrap sets it
    # run.main writes IMINT_LCLU_MANIFEST into os.environ. setenv records the original
    # (usually absent) value so teardown restores it; delenv on an absent name records nothing.
    monkeypatch.setenv('IMINT_LCLU_MANIFEST', 'unset-by-test')

    class Served(Exception):
        pass

    def fake_execv(*_):
        raise Served
    monkeypatch.setattr(run.os, 'execv', fake_execv)
    with pytest.raises(Served):
        run.main(source, work, tmp_path / 'data')
    assert running.read_text() == '{"running": "r7"}'
    assert not (work / 'demo' / 'MANIFEST.json').exists()
    assert json.loads((work / 'campaign/demo/serving.json').read_text())['campaign']['sha256'] == 'c' * 64
    assert json.loads((work / 'campaign/demo/MANIFEST.json').read_text())['run_args']['years'] == [2017, 2024]
    assert os.environ['IMINT_API_OUTPUT_DIR'] == str(work / 'campaign/results')
    assert os.environ['IMINT_LCLU_MANIFEST'] == str(work / 'campaign/demo/serving.json')
    monkeypatch.delenv('LCLU_CAMPAIGN')
    assert run.deployment_root(work) == work
