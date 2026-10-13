import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import pytest
import numpy as np
from fastapi.testclient import TestClient
from .helpers import ROOT, REQ, drain, load


def test_A02_exact_production_fetch_and_unique_outputs(api):
    async def run():
        async with api.server.app.router.lifespan_context(api.server.app):
            ids = []
            for _ in range(2):
                r = await api.server.submit_analysis(api.server.AnalyzeRequest(**REQ)); ids.append(r.job_id)
            await drain(api.server)
            for jid in ids:
                result = await api.server.get_job(jid)
                assert result.status == 'success'
                assert json.loads(Path(result.summary_path).read_text())['job_id'] == jid
            assert len(api.calls) == 2
            assert len({j.output_dir for j in api.jobs}) == 2
            for called, job in zip(api.calls, api.jobs):
                assert called == {'source': 'des', 'date': REQ['date'], 'coords': REQ['bbox'], 'cloud_threshold': 0.1, 'token': None, 'include_scl': True, 'date_window': 0}
                assert job.rgb is api.fetched.rgb and job.bands is api.fetched.bands
                assert job.scl is api.fetched.scl and job.geo is api.fetched.geo
                assert job.config_path == str(api.tmp/'config/analyzers.yaml')
                assert Path(job.output_dir) == api.tmp/'artifacts'/job.job_id
                np.testing.assert_array_equal(np.load(Path(job.output_dir)/'rgb.npy'), api.fetched.rgb)
    asyncio.run(run())


@pytest.mark.parametrize('exc', [RuntimeError('synthetic fetch failure'), ImportError('synthetic missing dependency')])
def test_A02_fetch_failure_never_calls_engine(api, exc):
    def fail(**kw): api.calls.append(kw); raise exc
    api.scene_ns['_fetch_s2_via_openeo'] = fail
    async def run():
        async with api.server.app.router.lifespan_context(api.server.app):
            r = await api.server.submit_analysis(api.server.AnalyzeRequest(**REQ)); await drain(api.server)
            result = await api.server.get_job(r.job_id)
            assert result.status == 'failed' and str(exc) in result.error
            assert len(api.calls) == 1 and not api.jobs
            assert not (api.tmp/'artifacts'/r.job_id).exists()
    asyncio.run(run())


def test_A02_cloud_rejection_and_missing_config(api):
    def cloudy(**kw): raise api.scene.FetchError('synthetic cloud threshold exceeded')
    api.scene_ns['_fetch_s2_via_openeo'] = cloudy
    async def run():
        async with api.server.app.router.lifespan_context(api.server.app):
            r = await api.server.submit_analysis(api.server.AnalyzeRequest(**REQ)); await drain(api.server)
            assert 'cloud threshold' in (await api.server.get_job(r.job_id)).error
            (api.tmp/'config/analyzers.yaml').unlink()
            r = await api.server.submit_analysis(api.server.AnalyzeRequest(**REQ)); await drain(api.server)
            assert 'config/analyzers.yaml' in (await api.server.get_job(r.job_id)).error
            assert not api.jobs
    asyncio.run(run())


@pytest.mark.parametrize('change', [{'date':'../../marker'}, {'date':'2026-02-30'}, {'date':'20260425'}, {'bbox':{**REQ['bbox'], 'crs':'EPSG:3006'}}, {'bbox':{**REQ['bbox'], 'west':12}}, {'bbox':{**REQ['bbox'], 'south':59}}])
def test_A02_invalid_fetch_inputs_rejected(api, change):
    with TestClient(api.server.app) as c:
        assert c.post('/v1/analyze', json={**REQ, **change}).status_code == 422
        assert not api.server._jobs and not api.calls


def test_real_engine_entry_persists_band_cache(api, monkeypatch):
    """Use the real engine module, with science implementations isolated."""
    import importlib
    import types

    with monkeypatch.context() as local:
        for name in (
            "change_detection", "spectral", "object_detection", "prithvi",
            "nmd", "cot", "marine_vessels", "ai2_vessels_s1",
            "vegetation_edge", "samgeo", "insar", "water_quality",
        ):
            module = types.ModuleType("imint.analyzers." + name)
            module.__getattr__ = lambda name: type(name, (), {})
            local.setitem(sys.modules, module.__name__, module)
        for name in ("imint.exporters.export", "imint.exporters.html_report"):
            module = types.ModuleType(name)
            module.__getattr__ = lambda name: (lambda *args, **kwargs: None)
            local.setitem(sys.modules, name, module)
        local.delitem(sys.modules, "imint.engine")
        engine = importlib.import_module("imint.engine")
        engine.ANALYZER_REGISTRY = {}
        engine.save_rgb_png = lambda rgb, path: Path(path).write_bytes(b"synthetic exporter")
        engine._generate_html_report = lambda *args: None

        def summary(results, date, out):
            path = Path(out) / "summary.json"
            path.write_text(json.dumps({"date": date}))
            return str(path)

        engine.save_summary_report = summary

        async def run():
            async with api.server.app.router.lifespan_context(api.server.app):
                response = await api.server.submit_analysis(api.server.AnalyzeRequest(**REQ))
                await drain(api.server)
                result = await api.server.get_job(response.job_id)
                assert result.status == "success", result.error
                out = api.tmp / "artifacts" / response.job_id
                np.testing.assert_array_equal(
                    np.load(out / "bands/2026-04-25_B04.npy"), api.fetched.bands["B04"]
                )
                metadata = json.loads((out / "bands/2026-04-25_bands_meta.json").read_text())
                assert metadata["geo"]["crs"] == "EPSG:3006"
                assert metadata["coords"] == REQ["bbox"]
                assert len(api.calls) == 1

        asyncio.run(run())
