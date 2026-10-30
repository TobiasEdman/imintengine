import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import audit_observation_indices as audit
from scripts import crop_holdout_guard as guard
from scripts.prepare_observation_index_audit import build_job, IMAGE
from tests.test_crop_holdout_guard import write_freeze


def fixture(root, *, overlap=True, held_year=2022):
    rows = [
        {'tile_name':'train','point_id':10,'Easting':500000.,'Northing':6600000.,'Year':2022,'row':256,'col':256},
        {'tile_name':'train','point_id':11,'Easting':500010.,'Northing':6600000.,'Year':2022,'row':256,'col':257},
        {'tile_name':'held','point_id':10 if overlap else 20,'Easting':500000.,'Northing':6600000.,'Year':held_year,'row':256,'col':255},
        {'tile_name':'held','point_id':21,'Easting':500010.,'Northing':6600000.,'Year':held_year,'row':256,'col':256},
        {'tile_name':'prior','point_id':30,'Easting':505120.,'Northing':6600000.,'Year':2021,'row':256,'col':256},
    ]
    for r in rows:
        r['unified_class'] = 'OUTCOME_MUST_NOT_BE_DECODED'
        r['f000'] = 'FEATURE_MUST_NOT_BE_DECODED'
    nfi = [{'tile_name':'train','TractID':1,'PlotID':1,'Easting':500010.,'Northing':6600000.,'Year':2022,'row':256,'col':257}]
    tiles = [{'tile_name':t,'easting':x,'northing':6600000.,'year':y}
             for t,x,y in [('train',500000.,2022),('held',500010.,held_year),('prior',505120.,2021)]]
    pins = {}
    for name, data in [('lucas',rows),('nfi',nfi),('tiles',tiles)]:
        p = root / audit.INPUTS[name][0]
        p.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist(data),p)
        pins[name] = {'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
    split_dir = root/'distill/crop_split'
    split_dir.mkdir(parents=True)
    def mutate(split):
        split['source_index_sha256'] = pins['lucas']['sha256']
        if overlap:
            split['holdout_plots'][0]['point_id'] = 10
            train = guard._keys(split['plots'])
            held = guard._keys(split['holdout_plots'])
            split['holdout_keys_sha256'] = guard.key_digest(held)
            split['qualified_keys_sha256'] = guard.key_digest(train+held)
            split['partition_sha256'] = guard.key_digest([(t,p,'distill') for t,p in train]+[(t,p,'holdout') for t,p in held])
    frozen = write_freeze(split_dir,mutate=mutate)
    return {'schema':audit.SCHEMA,'inputs':pins,'geometry':audit.GEOMETRY.copy(),
            'split_manifest_sha256':frozen['manifest_sha256'],'split_source_git_sha':frozen['source_git_sha']}


def test_real_parquet_audit_keeps_failed_independence_and_counts_rows(tmp_path):
    spec = fixture(tmp_path)
    result = audit.audit(tmp_path,spec)
    assert result['status'] == 'diagnostic_audit_complete'
    assert result['cleared_for_scoring'] is False
    assert result['frozen_identity']['status'] == 'protected_role_overlap_detected'
    assert result['identity_overlap']['crop_holdout'] == {
        'shared_ids':1,'left_rows_with_shared_id':1,'right_rows_with_shared_id':1,
        'shared_id_years':1,'shared_ids_same_year':1,'shared_ids_only_different_years':0}
    assert result['distill_protected_tile_overlap']['tile_pairs_positive_area'] == 1
    assert result['different_lucas_ids_near_protected']['pairs_at_identical_coordinates'] == 1
    assert result['nfi_near_protected_lucas']['distinct_id_pairs'] == 2
    assert result['geometry_consistency']['crop_holdout']['pixel_mismatch'] == 0
    raw = json.dumps(result)
    for hidden in ['OUTCOME_MUST_NOT_BE_DECODED','FEATURE_MUST_NOT_BE_DECODED','6600000','500010']:
        assert hidden not in raw
    assert result['outcome_columns_decoded'] == []


def test_projection_is_explicit_before_arrow_decode(tmp_path,monkeypatch):
    spec = fixture(tmp_path)
    original = pq.ParquetFile
    seen = []
    class Spy:
        def __init__(self,*a,**kw): self.inner = original(*a,**kw)
        def __getattr__(self,name): return getattr(self.inner,name)
        def read(self,*,columns,use_threads):
            assert columns and not {'unified_class','f000'} & set(columns)
            seen.append(columns)
            return self.inner.read(columns=columns,use_threads=use_threads)
    monkeypatch.setattr(pq,'ParquetFile',Spy)
    audit.audit(tmp_path,spec)
    assert seen == [list(cols) for _,cols in audit.INPUTS.values()]


def test_return_visits_separated_from_same_year_duplicates(tmp_path):
    result = audit.audit(tmp_path,fixture(tmp_path,held_year=2023))
    overlap = result['identity_overlap']['crop_holdout']
    assert overlap['shared_ids_same_year'] == 0
    assert overlap['shared_ids_only_different_years'] == 1
    assert result['distill_protected_tile_overlap']['pairs_different_years'] == 1


def test_clean_point_ids_still_does_not_clear_scoring(tmp_path):
    result = audit.audit(tmp_path,fixture(tmp_path,overlap=False))
    assert result['identity_overlap']['crop_holdout']['shared_ids'] == 0
    assert not result['cleared_for_scoring']


@pytest.mark.parametrize('name',audit.INPUTS)
def test_byte_corruption_stops_before_parse(tmp_path,name):
    spec = fixture(tmp_path)
    (tmp_path/audit.INPUTS[name][0]).write_bytes(b'corrupt')
    with pytest.raises(guard.HoldoutError,match='digest mismatch'):
        audit.audit(tmp_path,spec)


def test_wrong_source_authority_fails_before_index_open(tmp_path,monkeypatch):
    spec = fixture(tmp_path)
    spec['inputs']['lucas']['sha256'] = '1'*64
    monkeypatch.setattr(audit,'read_table',lambda *a,**kw: pytest.fail('index opened'))
    with pytest.raises(guard.HoldoutError,match='source no longer matches'):
        audit.audit(tmp_path,spec)


@pytest.mark.parametrize('name,pin',[(guard.VALIDATOR,'e'*64),('renamed.parquet','e'*64)])
def test_validator_bytes_refused_before_open(tmp_path,name,pin,monkeypatch):
    monkeypatch.setattr(guard,'read_pinned',lambda *a,**kw: pytest.fail('validator opened'))
    with pytest.raises(guard.HoldoutError,match='outside this audit'):
        audit.read_table(tmp_path/name,{'sha256':pin,'size_bytes':100},['tile_name'],'e'*64)


def test_missing_and_inconsistent_geometry_cannot_report_complete(tmp_path,monkeypatch):
    spec = fixture(tmp_path)
    original = audit.tile_geometries
    def incomplete(rows):
        tiles, unresolved = original(rows)
        del tiles['held']
        tiles['train']['year'] = 2019
        tiles['prior']['west'] += 10
        return tiles,unresolved
    monkeypatch.setattr(audit,'tile_geometries',incomplete)
    result = audit.audit(tmp_path,spec)
    assert result['status'] == 'incomplete_geometry'
    assert result['geometry_consistency']['crop_holdout']['missing_manifest_geometry'] == 2
    assert result['geometry_consistency']['distill']['year_mismatch'] == 2
    assert result['geometry_consistency']['prior_test']['pixel_mismatch'] == 1


def test_spatial_boundaries_and_years():
    tiles = {
        'a':dict(west=0,east=5120,south=0,north=5120,year=2022),
        'touch':dict(west=5120,east=10240,south=0,north=5120,year=2022),
        'overlap':dict(west=5110,east=10230,south=0,north=5120,year=2023),
    }
    result = audit.tile_overlap({'a'},{'touch','overlap'},tiles)
    assert result['tile_pairs_positive_area'] == result['pairs_different_years'] == 1
    assert result['pairs_same_year'] == 0
    a = dict(id=1,x=-1.,y=0.,year=2022)
    b = dict(id=2,x=9.,y=0.,year=2022)
    assert audit.near_points([a],[b],same_namespace=True)['distinct_id_pairs'] == 1
    b['x'] += 0.01
    assert audit.near_points([a],[b],same_namespace=True)['distinct_id_pairs'] == 0


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),True,'500000'])
def test_bad_geometry_is_rejected(bad):
    with pytest.raises(guard.HoldoutError): audit.coordinate(bad)


def test_duplicate_keys_fail():
    row = {'tile_name':'a','point_id':1,'Easting':0,'Northing':0,'Year':2022,'row':0,'col':0}
    with pytest.raises(guard.HoldoutError,match='duplicate observation'):
        audit.observations([row,row],'lucas')


def test_proximity_comparison_limit_fails(monkeypatch):
    monkeypatch.setattr(audit,'MAX_PAIRS',1)
    row = dict(id=1,x=0.,y=0.,year=2022)
    with pytest.raises(guard.HoldoutError,match='comparison budget'):
        audit.near_points([row,row],[dict(row,id=2)],same_namespace=True)


def test_tile_comparison_budget_includes_nonoverlapping_candidates(monkeypatch):
    monkeypatch.setattr(audit,'MAX_PAIRS',1)
    tiles = {'a':dict(west=0,east=5120,south=0,north=5120,year=2022),
             'b':dict(west=5120,east=10240,south=0,north=5120,year=2022),
             'c':dict(west=5120,east=10240,south=0,north=5120,year=2023)}
    with pytest.raises(guard.HoldoutError,match='comparison budget'):
        audit.tile_overlap({'a'},{'b','c'},tiles)


def test_proximity_pairs_are_explicitly_role_ordered():
    rows = [dict(id=1,x=0.,y=0.,year=2022),dict(id=2,x=0.,y=0.,year=2022)]
    result = audit.near_points(rows,rows,same_namespace=True)
    assert result['pair_semantics'] == 'ordered(left_role_id,right_role_id)'
    assert result['distinct_id_pairs'] == result['pairs_at_identical_coordinates'] == 2


def write_spec(root,spec):
    path = root/'inputs.json'
    path.write_text(json.dumps(spec))
    return path,hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_cli_persists_aggregate_json_without_private_rows(tmp_path):
    path, digest = write_spec(tmp_path,fixture(tmp_path))
    output = tmp_path/'report.json'
    command = [sys.executable,'-m','scripts.audit_observation_indices','--input-root',str(tmp_path),
               '--spec',str(path),'--spec-sha256',digest,'--output',str(output)]
    run = subprocess.run(command,text=True,capture_output=True,timeout=60)
    assert run.returncode == 0,run.stderr
    report = json.loads(output.read_text())
    assert report == json.loads(run.stdout)
    assert report['execution']['spec_sha256'] == digest
    assert report['execution']['elapsed_seconds'] >= 0
    assert report['execution']['max_rss_bytes'] > 0
    assert report['source_sha256']['holdout_guard'] == hashlib.sha256(Path(guard.__file__).read_bytes()).hexdigest()
    assert not report['cleared_for_scoring']


def test_rendered_job_is_bounded_and_only_mounts_five_metadata_files(tmp_path):
    path,digest = write_spec(tmp_path,fixture(tmp_path))
    config,job = build_job(path,digest)['items']
    assert config['immutable']
    assert job['spec']['activeDeadlineSeconds'] == 600
    assert job['spec']['backoffLimit'] == 0
    pod = job['spec']['template']['spec']
    assert not pod['automountServiceAccountToken']
    assert pod['restartPolicy'] == 'Never'
    assert pod['terminationGracePeriodSeconds'] == 0
    c = pod['containers'][0]
    assert c['command'][0] == '/opt/venvs/scoring/bin/python'
    assert c['image'] == IMAGE and '@sha256:' in IMAGE
    assert c['resources']['limits'] == c['resources']['requests'] == {'cpu':'2','memory':'4Gi','ephemeral-storage':'1Gi'}
    mounts = [m for m in c['volumeMounts'] if m['name'] == 'metadata']
    assert len(mounts) == 5 and all(m['readOnly'] and m['subPath'] for m in mounts)
    assert all('validator' not in m['subPath'] and not m['subPath'].endswith('.npz') for m in mounts)
    assert config['data']['inputs.json'] == path.read_text()
