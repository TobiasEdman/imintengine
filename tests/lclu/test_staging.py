"""Fail-closed exact-year staging and bounded private deployment recipe."""
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts.stage_lclu_demo import select_vpp, fetch


def test_vpp_requires_all_metrics_over_whole_buffer_for_exact_year():
    index = {metric: {'year': 2024, 'season': 1, 'metric': metric,
                      'bounds_4326': [17, 59, 19, 60]}
             for metric in ('SOSD', 'EOSD', 'LENGTH', 'MAXV', 'MINV')}
    bounds = (17.7, 59.2, 18.3, 59.5)
    assert len(select_vpp(index, 2024, bounds)) == 5
    with pytest.raises(ValueError, match='2025'):
        select_vpp(index, 2025, bounds)
    index['MINV']['bounds_4326'] = [17, 59, 18, 60]
    with pytest.raises(ValueError, match='MINV'):
        select_vpp(index, 2024, bounds)


def test_download_resume_verifies_bytes_and_rejects_partial_file(tmp_path, monkeypatch):
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def raise_for_status(self): pass
        def iter_content(self, *_): yield b'1234'
    calls = []
    def get(*args, **kwargs):
        calls.append(args)
        return Response()
    monkeypatch.setattr('scripts.stage_lclu_demo.requests.get', get)
    path = tmp_path/'source.npy'
    with pytest.raises(ValueError, match='Incomplete'):
        fetch('https://example.org/source', path, maximum=5, expected=5)
    assert not path.exists() and not path.with_suffix('.npy.partial').exists()
    fetch('https://example.org/source', path, maximum=4, expected=4)
    fetch('https://example.org/source', path, maximum=4, expected=4)
    assert len(calls) == 2 and path.read_bytes() == b'1234'
    path.write_bytes(b'abcd')
    with pytest.raises(ValueError, match='identity mismatch'):
        fetch('https://example.org/source', path, maximum=4, expected=4)
    assert len(calls) == 2


def test_private_recipe_caps_resources_and_keeps_reference_data_read_only(tmp_path):
    path = Path(__file__).resolve().parents[2]/'docker/lclu/render_job.py'
    spec = importlib.util.spec_from_file_location('render_lclu_job', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = 'python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f'
    archive = tmp_path/'source.tar.gz'
    bootstrap = b'print("archived bootstrap, independent of working-tree changes")\n'
    def make_archive(digest):
        manifest = json.dumps({'files': {'docker/lclu/bootstrap.py': digest}}).encode()
        with tarfile.open(archive, 'w:gz') as stream:
            for name, content in [('docker/lclu/bootstrap.py', bootstrap), ('SOURCE_MANIFEST.json', manifest)]:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                stream.addfile(member, io.BytesIO(content))
    make_archive(hashlib.sha256(bootstrap).hexdigest())
    items = module.resources(image, 2024, archive)['items']
    assert {v['kind'] for v in items} == {'ConfigMap', 'Job', 'PersistentVolumeClaim'}
    assert items[0]['immutable'] is True
    assert items[0]['data']['bootstrap.py'] == bootstrap.decode()
    job = items[-1]
    assert job['spec']['activeDeadlineSeconds'] == 86400
    pod = job['spec']['template']['spec']
    assert pod['automountServiceAccountToken'] is False
    assert pod['containers'][0]['resources']['limits'] == {'cpu': '4', 'memory': '8Gi'}
    assert next(v for v in pod['volumes'] if v['name'] == 'data')['persistentVolumeClaim']['readOnly'] is True
    with pytest.raises(ValueError, match='digest'):
        module.resources('python:3.12-slim', 2024)
    with pytest.raises(ValueError, match='years'):
        module.resources(image, 2023)
    resumed = module.resources(image, 2024, archive, job_name='lclu-stockholm-0124-r1',
                               reuse_volume=True, deadline_seconds=80000)
    assert [item['kind'] for item in resumed['items']] == ['ConfigMap', 'Job']
    assert resumed['items'][-1]['spec']['activeDeadlineSeconds'] == 80000
    assert resumed['items'][-1]['metadata']['name'] == 'lclu-stockholm-0124-r1'
    with pytest.raises(ValueError, match='24 hours'):
        module.resources(image, 2024, deadline_seconds=86401)
    make_archive('0'*64)
    with pytest.raises(ValueError, match='bootstrap'):
        module.resources(image, 2024, archive)


def test_system_runtime_rejects_corrupted_cached_package_before_extraction(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2]/'docker/lclu/system_runtime.py'
    spec = importlib.util.spec_from_file_location('lclu_system_runtime', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path/(module.PACKAGE['sha256']+'.deb')).write_bytes(b'corrupt')
    monkeypatch.setattr(module.subprocess, 'run', lambda *_a, **_k: pytest.fail('Must not extract unverified bytes'))
    with pytest.raises(ValueError, match='identity'):
        module.install(tmp_path)


def test_run_publishes_two_verified_regions_into_fresh_directory(tmp_path, monkeypatch):
    from docker.lclu import run
    from imint.inference.lclu import write_json
    source, work, data = (tmp_path/name for name in ('source','work','data'))
    source.mkdir()
    write_json(source/'SOURCE_MANIFEST.json', {'files':{}})
    calls=[]
    checkpoint={'path':'/data/model.pt','sha256':'a'*64,'size':1}
    def staged(scope, year, reference, root):
        assert year==2024 and reference==data
        root.mkdir(parents=True)
        preparation={'config':{'path':str(root/'sources.json'),'sha256':'b'*64,'size':2},
                     'python':'/runtime/bin/python','timeout_seconds':300}
        write_json(root/'serving.json', {'checkpoint':checkpoint,'preparation':preparation})
        calls.append(scope.stem)
    monkeypatch.setattr(run,'stage',staged)
    monkeypatch.setenv('LCLU_YEAR','2024')
    monkeypatch.setenv('LCLU_IMAGE','python@sha256:'+'c'*64)
    monkeypatch.setenv('IMINT_LCLU_MANIFEST','unused')
    launched=[]
    monkeypatch.setattr(run.os,'execv',lambda *args:launched.append(args))
    run.main(source,work,data)
    assert calls==['stockholm','kiruna_tatort']
    published=json.loads((work/'demo/serving.json').read_text())
    assert published['checkpoint']==checkpoint and len(published['preparations'])==2
    assert run.os.environ['IMINT_LCLU_MANIFEST']==str(work/'demo/serving.json')
    assert json.loads((work/'demo/MANIFEST.json').read_text())['run_args']['areas']==['Stockholm','Kiruna tätort']
    assert len(launched)==1
