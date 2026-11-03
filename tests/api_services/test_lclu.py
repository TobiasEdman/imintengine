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


def test_api_hosted_page_uses_its_origin_with_public_static_configuration(api, monkeypatch):
    root = api.tmp/'static-showcase'
    (root/'js').mkdir(parents=True)
    public = root/'js/lclu-config.js'
    configured = "window.IMINT_LCLU_CONFIG = {apiBase: 'https://lclu.icedc.se'};\n"
    public.write_text(configured)
    monkeypatch.setattr(api.server, '_WEB_ROOT', root)
    with TestClient(api.server.app, base_url='http://127.0.0.1:8788') as client:
        response = client.get('/lclu-assets/lclu-config.js')
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('application/javascript')
        assert response.headers['cache-control'] == 'no-store'
        assert response.text == "window.IMINT_LCLU_CONFIG = {apiBase: ''};\n"
        assert client.get('/v1/lclu').status_code == 200
    assert public.read_text() == configured


def test_prepare_warms_the_newest_request_outside_the_job_queue(api, monkeypatch):
    gates = {0: (threading.Event(), threading.Event()), 3: (threading.Event(), threading.Event())}
    async def run():
        req = lambda year: api.server.LcluRequest(**{**REQUEST, 'year': year})
        with pytest.raises(api.server.HTTPException) as unconfigured:
            await api.server.prepare_lclu(req(2024))
        assert unconfigured.value.status_code == 503
        life = api.server.lifespan(api.server.app)
        await life.__aenter__()
        service = install(api, monkeypatch)
        calls, original = [], service.sources.prepare
        def recorded(grid, year):
            index = len(calls)
            calls.append(year)
            if index in gates:
                gates[index][0].set()
                assert gates[index][1].wait(10)
            return original(grid, year)
        monkeypatch.setattr(service.sources, 'prepare', recorded)
        assert (await api.server.prepare_lclu(req(2023))) == {'state': 'queued'}
        assert await asyncio.to_thread(gates[0][0].wait, 2)
        await api.server.prepare_lclu(req(2024))     # waits behind the running one ...
        await api.server.prepare_lclu(req(2023))     # ... and is replaced by the newest
        with pytest.raises(api.server.HTTPException) as unqualified:
            await api.server.prepare_lclu(req(2025))
        assert unqualified.value.status_code == 422
        # A map request is not held back by the running warm-up.
        job = await api.server.submit_lclu(req(2024))
        for _ in range(2400):   # a hang guard, not a speed test
            if api.server._jobs[job.job_id]['status'] == 'success':
                break
            await asyncio.sleep(.05)
        assert api.server._jobs[job.job_id]['status'] == 'success'
        assert calls == [2023, 2024]                 # the warm-up, then the job's own
        gates[0][1].set()
        await api.server._lclu_warm_tasks["place"]
        assert calls == [2023, 2024, 2023]           # the replaced 2024 warm-up never ran
        # Shutdown waits for a running warm-up's writes and drops a waiting one.
        await api.server.prepare_lclu(req(2024))
        assert await asyncio.to_thread(gates[3][0].wait, 2)
        await api.server.prepare_lclu(req(2023))
        shutdown = asyncio.create_task(life.__aexit__(None, None, None))
        await asyncio.sleep(.05)
        assert not shutdown.done()
        gates[3][1].set()
        await shutdown
        assert calls == [2023, 2024, 2023, 2024] and not api.server._pending_jobs
    try:
        asyncio.run(run())
    finally:
        for _, release in gates.values():
            release.set()


def test_prepare_endpoint_validates_like_a_map_request(api):
    with TestClient(api.server.app) as client:
        for changes in ({'year': 2016}, {'year': 2026}, {'checkpoint': 'private.pt'}, {'lon': 9}):
            assert client.post('/v1/lclu/prepare', json={**REQUEST, **changes}).status_code == 422


def test_storage_limit_rejects_new_map_but_keeps_fixed_inputs_and_downloads(api, monkeypatch):
    from imint.inference.lclu_storage import MAP_BYTES, STORAGE_MESSAGE

    async def run():
        async with api.server.lifespan(api.server.app):
            service = install(api, monkeypatch)
            request = api.server.LcluRequest(**REQUEST)
            first = await api.server.submit_lclu(request)
            await asyncio.gather(*tuple(api.server._pending_jobs))
            assert (await api.server.get_job(first.job_id)).status == 'success'
            service.storage.max_bytes = service.storage.used_bytes() + MAP_BYTES - 1
            new = api.server.LcluRequest(**{**REQUEST, 'year':2023})
            # Fixed qualified bundles already exist and require no allocation.
            assert await api.server.prepare_lclu(new) == {'state': 'queued'}
            await asyncio.gather(*tuple(api.server._pending_jobs))
            rejected = await api.server.submit_lclu(new)
            await asyncio.gather(*tuple(api.server._pending_jobs))
            result = await api.server.get_job(rejected.job_id)
            assert result.status == 'failed' and result.error == STORAGE_MESSAGE
            assert service.model.calls == 1
            assert (await api.server.submit_lclu(request)).job_id == first.job_id
            download = await api.server.lclu_file(first.job_id, 'classification.tif')
            assert Path(download.path).is_file()

    asyncio.run(run())


def test_a_warmed_place_skips_preparation_in_the_map_request(api, monkeypatch):
    """End to end through RepositorySources: the warm-up's input is the job's input."""
    import time
    from imint.inference.lclu import AUX_NAMES, EMBEDDING, Grid, LcluService, public_identity, write_json
    from imint.inference.lclu_sources import RepositorySources
    from scripts.cache_predict import file_identity
    grid = Grid.from_lonlat(REQUEST['lon'], REQUEST['lat'])
    b = grid.bounds()
    config = api.tmp/'sources.json'
    write_json(config, {'version': 1, 'embedding': EMBEDDING, 'qualified_regions': [
        {'country': 'SE', 'verification': 'unit-test', 'years': [2023, 2024],
         'bounds_3006': [b[k] for k in ('west', 'south', 'east', 'north')]}]})
    runs = []
    def preparation(command, **kwargs):
        year = int(command[command.index('--year') + 1])
        runs.append(year)
        time.sleep(1.0)   # stands in for the measured 12.9 s Stockholm preparation
        out = Path(command[command.index('--output') + 1])
        arrays = {'tessera': np.full((128, 512, 512), .25, np.float16),
                  'valid_embedding': np.ones((512, 512), bool), 'year': np.int32(year),
                  'tessera_year': np.int32(year), 'vpp_year': np.int32(year),
                  'crs': np.array('EPSG:3006'), 'transform': np.array(list(grid.transform())[:6]),
                  **{name: np.ones((512, 512), np.float32) for name in AUX_NAMES}}
        np.savez_compressed(out/'input.npz', **arrays)
        write_json(out/'entry.json', {
            'year': year, 'center_3006': [grid.east, grid.north], 'public_area': True,
            'input': {**file_identity(out/'input.npz'), 'path': 'input.npz'}, 'embedding': EMBEDDING,
            'source_config': public_identity(file_identity(config)),
            'auxiliary_provenance': {'vpp': {'year': year, 'season': 1,
                                             'index': {'sha256': 'a'*64, 'size': 1}, 'files': []}}})
    monkeypatch.setattr('imint.inference.lclu_sources.run_preparation', preparation)
    class Model:
        contract = {'synthetic': True}
        def predict(self, path, record):
            return np.zeros((504, 504), np.uint8)
    async def finished(job_id):
        for _ in range(2400):   # a hang guard, not a speed test
            job = api.server._jobs[job_id]
            if job['status'] in ('success', 'failed'):
                return job
            await asyncio.sleep(.05)
        raise AssertionError('job did not finish')
    async def run():
        async with api.server.lifespan(api.server.app):
            sources = RepositorySources({'config': file_identity(config), 'python': 'python3.12',
                                         'timeout_seconds': 30}, api.tmp, api.tmp/'artifacts')
            monkeypatch.setattr(api.server, '_lclu_service',
                                LcluService(Model(), sources, api.tmp/'artifacts'))
            req = lambda year: api.server.LcluRequest(**{**REQUEST, 'year': year})
            from imint.inference.lclu_storage import STORAGE_MESSAGE
            maximum = sources.storage.max_bytes
            sources.storage.max_bytes = 1
            with pytest.raises(api.server.HTTPException) as blocked:
                await api.server.prepare_lclu(req(2023))
            assert blocked.value.status_code == 503 and blocked.value.detail == STORAGE_MESSAGE
            assert not api.server._lclu_warm_tasks
            sources.storage.max_bytes = maximum
            cold = await finished((await api.server.submit_lclu(req(2023))).job_id)
            await api.server.prepare_lclu(req(2024))
            await api.server._lclu_warm_tasks["place"]
            sources.storage.max_bytes = 1
            assert await api.server.prepare_lclu(req(2024)) == {'state': 'queued'}
            await api.server._lclu_warm_tasks["place"]
            sources.storage.max_bytes = maximum
            warm = await finished((await api.server.submit_lclu(req(2024))).job_id)
            return cold, warm
    cold, warm = asyncio.run(run())
    assert cold['status'] == warm['status'] == 'success'
    assert runs == [2023, 2024]          # the map request started no second preparation
    assert cold['result']['timing_ms']['preparation'] >= 1000
    assert warm['result']['timing_ms']['preparation'] < 500


def test_area_prefetch_runs_beside_a_blocked_place_warm_up(api, monkeypatch):
    started, release = threading.Event(), threading.Event()
    box = {'west': 17.55, 'south': 59.82, 'east': 17.72, 'north': 59.9, 'year': 2024}
    async def run():
        async with api.server.lifespan(api.server.app):
            service = install(api, monkeypatch)
            prefetched = []
            def blocked(grid, year):
                started.set()
                assert release.wait(10)
            monkeypatch.setattr(service.sources, 'prepare', blocked)
            monkeypatch.setattr(service.sources, 'prefetch',
                                lambda bounds, year: prefetched.append((bounds, year)))
            await api.server.prepare_lclu(api.server.LcluRequest(**REQUEST))
            assert await asyncio.to_thread(started.wait, 2)
            assert await api.server.prefetch_lclu(api.server.LcluPrefetchRequest(**box)) == {'state': 'queued'}
            await api.server._lclu_warm_tasks['area']        # finishes while 'place' is blocked
            assert len(prefetched) == 1 and prefetched[0][1] == 2024
            west, south, east, north = prefetched[0][0]
            assert west < east and south < north and 6_600_000 < south < 6_700_000  # EPSG:3006
            assert not api.server._lclu_warm_tasks['place'].done()
            release.set()
    try:
        asyncio.run(run())
    finally:
        release.set()


def test_prefetch_endpoint_bounds_the_box(api, monkeypatch):
    box = {'west': 17.55, 'south': 59.82, 'east': 17.72, 'north': 59.9, 'year': 2024}
    with TestClient(api.server.app) as client:
        assert client.post('/v1/lclu/prefetch', json=box).status_code == 503   # no service
        for changes in ({'year': 2026}, {'west': 9.0}, {'west': '17.5'}, {'path': '/x'}):
            assert client.post('/v1/lclu/prefetch', json={**box, **changes}).status_code == 422
    async def run():
        async with api.server.lifespan(api.server.app):
            install(api, monkeypatch)
            for changes in ({'east': 17.5}, {'east': 20.0, 'north': 61.0}):   # empty, too large
                with pytest.raises(api.server.HTTPException) as refused:
                    await api.server.prefetch_lclu(api.server.LcluPrefetchRequest(**{**box, **changes}))
                assert refused.value.status_code == 422
            assert (await api.server.prefetch_lclu(api.server.LcluPrefetchRequest(**box))) == {'state': 'queued'}
            await api.server._lclu_warm_tasks['area']      # the catalogue has nothing to fetch
    asyncio.run(run())
