"""Annual API integration with synthetic, on-disk canonical-shape inputs."""
import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from .helpers import drain

REQUEST = {'lon': 17.64, 'lat': 59.86, 'year': 2024}


def install(api, monkeypatch):
    # Import after the API fixture's isolated module setup.
    from imint.inference.lclu import Grid, LcluService, BundleCatalog, EMBEDDING, AUX_NAMES
    from scripts.cache_predict import file_identity
    grid = Grid.from_lonlat(17.64, 59.86)
    path = api.tmp / 'synthetic.npz'
    arrays = {'tessera': np.full((128,512,512), .25, np.float16),
              'valid_embedding': np.ones((512,512), bool), 'year': 2024,
              'tessera_year': 2024, 'vpp_year': 2024, 'crs': np.array('EPSG:3006'),
              'transform': np.array(list(grid.transform())[:6])}
    arrays.update({name: np.ones((512,512), np.float32) for name in AUX_NAMES})
    np.savez_compressed(path, **arrays)
    entries = []
    for year in [2023,2024]:
        year_path = api.tmp / f'synthetic-{year}.npz'
        np.savez_compressed(year_path, **{**arrays, 'year':year, 'tessera_year':year, 'vpp_year':year})
        entries.append({'center_3006':[grid.east,grid.north], 'year':year, 'public_area':True,
                        'input':file_identity(year_path), 'embedding':EMBEDDING,
                        'auxiliary_provenance':{'vpp': {'year': year, 'season': 1, 'index': {'sha256': 'a'*64, 'size': 1}, 'files': []}}})
    class Model:
        contract = {'synthetic': True}
        calls = 0
        def predict(self, path, record):
            self.calls += 1
            return np.zeros((504,504), np.uint8)
    service = LcluService(Model(), BundleCatalog(entries,api.tmp), api.tmp / 'artifacts')
    monkeypatch.setattr(api.server, '_lclu_service', service)
    return service


def test_no_config_and_geometry(api):
    with TestClient(api.server.app) as client:
        assert client.get('/v1/lclu').json()['ready'] is False
        assert client.post('/v1/lclu',json=REQUEST).status_code == 503
        geometry=client.get('/v1/lclu/availability',params={'lon':17.64,'lat':59.86}).json()
        assert geometry['years'] == [] and geometry['geometry']['shape'] == [504,504]
        assert client.get('/v1/lclu/availability?lon=nan&lat=59').status_code == 422
        assert client.get('/lclu').status_code == 200
        assert client.get('/lclu-assets/lclu.js').status_code == 200
        assert client.get('/lclu-assets/secret.json').status_code == 404


@pytest.mark.parametrize('changes', [{'year':2017},{'year':2026},{'year':2024.0},{'year':True},
                                     {'year':'2024'},{'date':'2024-07-01'},{'lon':'17.64'},
                                     {'lon':False},{'checkpoint':'private.pt'}])
def test_strict_annual_request(api, changes):
    with TestClient(api.server.app) as client:
        assert client.post('/v1/lclu',json={**REQUEST,**changes}).status_code == 422


def test_async_success_dedup_download_and_no_des(api, monkeypatch):
    async def run():
        async with api.server.lifespan(api.server.app):
            service=install(api,monkeypatch)
            req=api.server.LcluRequest(**REQUEST)
            submissions=await asyncio.gather(*(api.server.submit_lclu(req) for _ in range(12)))
            assert len({r.job_id for r in submissions}) == 1
            await asyncio.gather(*tuple(api.server._pending_jobs))
            job=await api.server.get_job(submissions[0].job_id)
            assert job.status == 'success' and job.year == 2024 and job.date is None
            assert job.phase == 'complete' and service.model.calls == 1
            assert not api.calls and not api.jobs  # exact-day engine never invoked
            response=await api.server.lclu_file(job.job_id,'classification.tif')
            assert Path(response.path).is_file()
            assert job.result['artifacts']['provenance.json'].endswith('/provenance.json')
            with pytest.raises(api.server.HTTPException):
                await api.server.lclu_file(job.job_id,'input.npz')
    asyncio.run(run())


def test_queue_bound_failure_retry_and_shutdown(api,monkeypatch):
    started,release=threading.Event(),threading.Event()
    async def run():
        life=api.server.lifespan(api.server.app)
        await life.__aenter__()
        service=install(api,monkeypatch)
        monkeypatch.setattr(api.server,'_max_jobs',1)
        original=service.run
        def blocked(*args):
            started.set()
            assert release.wait(10)
            return original(*args)
        monkeypatch.setattr(service,'run',blocked)
        first=await api.server.submit_lclu(api.server.LcluRequest(**REQUEST))
        try:
            assert await asyncio.to_thread(started.wait,2)
            with pytest.raises(api.server.HTTPException) as full:
                await api.server.submit_lclu(api.server.LcluRequest(**{**REQUEST,'year':2023}))
            assert full.value.status_code == 429
            same=await api.server.submit_lclu(api.server.LcluRequest(**REQUEST))
            assert same.job_id == first.job_id
            shutdown=asyncio.create_task(life.__aexit__(None,None,None))
            await asyncio.sleep(.02)
            assert not shutdown.done()
            release.set()
            await shutdown
            assert api.server._jobs[first.job_id]['status'] == 'success'
        finally:
            release.set()
        async with api.server.lifespan(api.server.app):
            service=install(api,monkeypatch)
            def fail(*args): raise RuntimeError('/private/checkpoint secret location')
            monkeypatch.setattr(service,'run',fail)
            failed=await api.server.submit_lclu(api.server.LcluRequest(**REQUEST))
            await asyncio.gather(*tuple(api.server._pending_jobs))
            job=await api.server.get_job(failed.job_id)
            assert job.status == 'failed' and '/private/' not in job.error
            retried=await api.server.submit_lclu(api.server.LcluRequest(**REQUEST))
            assert retried.job_id != failed.job_id
    asyncio.run(run())
