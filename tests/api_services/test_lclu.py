"""Annual API integration with synthetic, on-disk canonical-shape inputs."""
import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from .helpers import drain

REQUEST = {'lon': 17.64, 'lat': 59.86, 'year': 2024}


def install(api, monkeypatch, *, prediction=None, valid=None):
    # Import after the API fixture's isolated module setup.
    from imint.inference.lclu import Grid, LcluService, BundleCatalog, EMBEDDING, AUX_NAMES
    from scripts.cache_predict import file_identity
    grid = Grid.from_lonlat(17.64, 59.86)
    path = api.tmp / 'synthetic.npz'
    arrays = {'tessera': np.full((128,512,512), .25, np.float16),
              'valid_embedding': np.ones((512,512), bool) if valid is None else valid, 'year': 2024,
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
            return np.zeros((504,504), np.uint8) if prediction is None else prediction.copy()
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


@pytest.mark.parametrize('changes', [{'year':2016},{'year':2026},{'year':2024.0},{'year':True},
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


def test_pixel_inspection_reads_native_classes_nodata_and_edges_without_inference(api, monkeypatch):
    from pyproj import Transformer
    from imint.inference.lclu import Grid
    grid = Grid.from_lonlat(REQUEST['lon'], REQUEST['lat'])
    tx = Transformer.from_crs(3006, 4326, always_xy=True)
    def position(row, col):
        return tx.transform(*(grid.transform(504)*(col+.5, row+.5)))
    prediction=np.zeros((504,504),np.uint8)
    prediction[0,0]=9;prediction[1,2]=7;prediction[503,503]=27
    valid=np.ones((512,512),bool);valid[6,7]=False
    async def run():
        async with api.server.lifespan(api.server.app):
            service=install(api,monkeypatch,prediction=prediction,valid=valid)
            submitted=await api.server.submit_lclu(api.server.LcluRequest(**REQUEST))
            await asyncio.gather(*tuple(api.server._pending_jobs))
            for row,col,value,name in [(0,0,9,'Exploaterad mark'),(1,2,7,'Öppen våtmark'),
                                       (2,3,255,'Data saknas'),(3,4,0,'bakgrund'),
                                       (503,503,27,'öppen mark utan vegetation')]:
                lon,lat=position(row,col)
                pixel=await api.server.lclu_pixel(submitted.job_id,lon,lat)
                assert pixel['inside'] and (pixel['row'],pixel['col'])==(row,col)
                assert (pixel['class_id'],pixel['name'])==(value,name)
                assert pixel['nodata']==(value==255) and pixel['year']==2024
                assert (pixel['color'] is None)==(value==255)
            for row,col in [(-1,0),(504,0),(0,-1),(0,504)]:
                assert (await api.server.lclu_pixel(submitted.job_id,*position(row,col)))['inside'] is False
            assert (await api.server.lclu_pixel(submitted.job_id,-75,0))['inside'] is False
            for lon,lat in [(float('nan'),59),(18,float('inf')),(181,59)]:
                with pytest.raises(api.server.HTTPException) as invalid:
                    await api.server.lclu_pixel(submitted.job_id,lon,lat)
                assert invalid.value.status_code==422
            with pytest.raises(api.server.HTTPException) as absent:
                await api.server.lclu_pixel('unknown',REQUEST['lon'],REQUEST['lat'])
            assert absent.value.status_code==404
            assert service.model.calls==1 and not api.calls and not api.jobs
            result=api.server._jobs[submitted.job_id]['result']
            path=service.output_root/result['cache_key']/'classification.tif'
            path.write_bytes(b'corrupted map')
            with pytest.raises(api.server.HTTPException) as missing:
                await api.server.lclu_pixel(submitted.job_id,REQUEST['lon'],REQUEST['lat'])
            assert missing.value.status_code==503 and service.model.calls==1
    asyncio.run(run())


@pytest.mark.parametrize('origin', ['*', 'http://showcase.example', 'https://site.example/path',
                                    'https://user:pass@site.example', 'https://site.example?x=1'])
def test_showcase_rejects_ambiguous_origins(origin):
    from imint.api.v1.lclu_cors import validate_origins
    with pytest.raises(ValueError):
        validate_origins([origin])


def test_showcase_cross_origin_submission_poll_and_download(api, monkeypatch):
    from imint.api.v1.lclu_cors import LcluCORSMiddleware
    origin='https://showcase.example'
    app=LcluCORSMiddleware(api.server.app, origins=[origin])
    headers={'Origin':origin}
    with TestClient(app) as client:
        install(api,monkeypatch)
        preflight=client.options('/v1/lclu',headers={**headers,
            'Access-Control-Request-Method':'POST', 'Access-Control-Request-Headers':'Content-Type'})
        assert preflight.status_code==200
        assert preflight.headers['access-control-allow-origin']==origin
        submit=client.post('/v1/lclu',headers=headers,json=REQUEST)
        assert submit.status_code==202 and submit.headers['access-control-allow-origin']==origin
        poll=submit.json()['poll_url']
        assert poll.startswith('/v1/lclu/jobs/')
        # Synchronise with the actual executor; no arbitrary polling sleep.
        async def finish():
            await asyncio.gather(*tuple(api.server._pending_jobs))
        client.portal.call(finish)
        result=client.get(poll,headers=headers)
        assert result.headers['access-control-allow-origin']==origin
        assert result.json()['status']=='success'
        for path in result.json()['result']['artifacts'].values():
            artifact=client.get(path,headers=headers)
            assert artifact.status_code==200
            assert artifact.headers['access-control-allow-origin']==origin
        assert client.post('/v1/lclu',headers=headers,json=REQUEST).json()['poll_url']==poll
        assert client.get('/v1/lclu/jobs/missing',headers=headers).status_code==404
        api.server._jobs['ordinary']={'kind':'analysis'}
        assert client.get('/v1/lclu/jobs/ordinary',headers=headers).status_code==404
        for path in ('/v1/jobs/missing','/v1/lclux','/health'):
            assert 'access-control-allow-origin' not in client.get(path,headers=headers).headers
        assert 'access-control-allow-origin' not in client.get('/v1/lclu',headers={'Origin':'https://untrusted.example'}).headers


def test_showcase_assets_work_for_static_and_api_hosting(api):
    import re
    root=Path(__file__).resolve().parents[2]/'docs'
    static=(root/'lclu.html').read_text()
    assets=re.findall(r'(?:src|href)="((?:css|js)/[^"?]+)(?:\?[^" ]*)?"',static)
    assert assets and all((root/name).is_file() for name in assets)
    assert 'href="index.html"' in static
    with TestClient(api.server.app) as client:
        html=client.get('/lclu').text
        served=re.findall(r'(?:src|href)="(/lclu-assets/[^"?]+)(?:\?[^" ]*)?"',html)
        assert len(served)==len(assets)
        for path in served:
            assert client.get(path).status_code==200
        assert client.get('/lclu-assets/../config/analyzers.yaml').status_code==404
