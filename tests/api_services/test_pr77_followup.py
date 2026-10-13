"""Offline exact-head follow-ups; never connect to a fetch service."""
import asyncio
from contextlib import contextmanager
import json
import sys
import types
import numpy as np
import pytest
from fastapi.testclient import TestClient
from .helpers import REQ, drain

SMALL = {**REQ, 'bbox': dict(west=11.5, south=58.0, east=11.51, north=58.01)}


def test_primitives_survive_two_contended_event_loops(api):
    async def run():
        async with api.server.lifespan(api.server.app):
            for primitive in (api.server._jobs_lock, api.server._analysis_slot):
                await primitive.acquire()
                waiter = asyncio.create_task(primitive.acquire())
                await asyncio.sleep(0)
                primitive.release()
                await waiter
                primitive.release()
            responses = [await api.server.submit_analysis(api.server.AnalyzeRequest(**SMALL)) for _ in range(2)]
            await drain(api.server)
            for r in responses:
                assert (await api.server.get_job(r.job_id)).status == 'success'
    asyncio.run(run())
    asyncio.run(run())


@pytest.mark.parametrize('bbox', [
    dict(west=-180, south=-90, east=180, north=90),
    dict(west=11.5, south=58, east=12, north=58.5),
    dict(west=11.5, south=58, east=11.500001, north=80),
])
def test_oversized_bbox_rejected_before_admission(api, bbox):
    with TestClient(api.server.app) as client:
        response = client.post('/v1/analyze', json={**REQ, 'bbox': bbox})
        assert response.status_code == 422
        assert not api.server._jobs and not api.calls


@pytest.mark.parametrize('change', [{'west': '11.5'}, {'west': True}, {'extra': 1}])
def test_bbox_strict_boundary(api, change):
    with TestClient(api.server.app) as client:
        response = client.post('/v1/analyze', json={**SMALL, 'bbox': {**SMALL['bbox'], **change}})
        assert response.status_code == 422
        assert not api.server._jobs


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_validation_is_422(api, value):
    with TestClient(api.server.app, raise_server_exceptions=False) as client:
        body = {**SMALL, 'bbox': {**SMALL['bbox'], 'west': value}}
        response = client.post('/v1/analyze', content=json.dumps(body), headers={'Content-Type': 'application/json'})
        assert response.status_code == 422
        assert response.json()['detail']
        assert not api.server._jobs


@pytest.mark.parametrize('when', ['before_start', 'waiting_slot', 'before_thread'])
def test_cancelled_job_always_terminal(api, monkeypatch, when):
    async def run():
        async with api.server.lifespan(api.server.app):
            entered = asyncio.Event()
            if when == 'before_thread':
                original = api.server._put_job
                async def pause(job_id, payload):
                    await original(job_id, payload)
                    if payload['status'] == 'running':
                        entered.set()
                        await asyncio.Event().wait()
                monkeypatch.setattr(api.server, '_put_job', pause)
            if when == 'waiting_slot':
                await api.server._analysis_slot.acquire()
            response = await api.server.submit_analysis(api.server.AnalyzeRequest(**SMALL))
            task, = api.server._pending_jobs
            if when == 'waiting_slot':
                await asyncio.sleep(0)
            if when == 'before_thread':
                await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)
            if when == 'waiting_slot':
                api.server._analysis_slot.release()
            result = await api.server.get_job(response.job_id)
            assert result.status == 'failed'
            assert result.success is False and result.completed_at is not None
            assert 'cancel' in result.error.lower()
            assert not api.calls
    asyncio.run(run())


def test_result_schema_reports_real_output_names(api):
    row = api.analysis_result('spectral', True, outputs={'ndvi': np.ones((2, 2)), 'report': 'synthetic.json'})
    api.engine.run_job = lambda job: api.job.IMINTResult(job.job_id, job.date, True, [row])
    async def run():
        async with api.server.lifespan(api.server.app):
            r = await api.server.submit_analysis(api.server.AnalyzeRequest(**SMALL))
            await drain(api.server)
            result = (await api.server.get_job(r.job_id)).model_dump(mode='json')['analyzers'][0]
            assert result['output_names'] == ['ndvi', 'report']
            assert result['artifacts'] == [] and result['duration_ms'] is None
    asyncio.run(run())


def test_span_receives_matched_template(api, monkeypatch):
    spans = []
    class Span:
        def set_attribute(self, name, value): self.attributes[name] = value
    @contextmanager
    def record(*args, **attributes):
        value = Span(); value.attributes = attributes; spans.append(value)
        yield  # Shared instrumentation.span does not yield its Span object.
    trace = types.SimpleNamespace(get_current_span=lambda: spans[-1])
    monkeypatch.setitem(sys.modules, 'opentelemetry', types.SimpleNamespace(trace=trace))
    monkeypatch.setattr(api.server, '_HAS_OTEL', True)
    monkeypatch.setattr(api.server, 'span', record)
    with TestClient(api.server.app) as client:
        assert client.get('/v1/jobs/unknown-id').status_code == 404
        assert client.get('/not-a-route').status_code == 404
    assert [s.attributes['route'] for s in spans] == ['/v1/jobs/{job_id}', 'unmatched']


@pytest.mark.parametrize('shape,accepted', [((1000, 1000), True), ((1000001, 1), False),
                                          ((1, 1000001), False), ((0, 100), False),
                                          ((1000000, 1), True), ((1000, 1001), False)])
def test_exact_projected_budget_boundary(api, monkeypatch, shape, accepted):
    width, height = shape
    monkeypatch.setattr(api.scene, '_to_nmd_grid', lambda coords: dict(
        west=0, south=0, east=width * 10, north=height * 10))
    with TestClient(api.server.app) as client:
        response = client.post('/v1/analyze', json=SMALL)
        assert response.status_code == (202 if accepted else 422)
        if not accepted:
            assert not api.server._jobs and not api.calls


def test_projection_failure_rejected_before_admission(api, monkeypatch):
    from rasterio.errors import TransformError
    def fail(coords):
        raise TransformError('synthetic unprojectable bbox')
    monkeypatch.setattr(api.scene, '_to_nmd_grid', fail)
    with TestClient(api.server.app) as client:
        assert client.post('/v1/analyze', json=SMALL).status_code == 422
        assert not api.server._jobs and not api.calls


@pytest.mark.parametrize('budget', [None, '', '0', '-1', '1.5', 'NaN', 'Infinity', '1e6'])
def test_missing_or_invalid_budget_fails_startup(api, monkeypatch, budget):
    if budget is None:
        monkeypatch.delenv('IMINT_API_MAX_PIXELS', raising=False)
    else:
        monkeypatch.setenv('IMINT_API_MAX_PIXELS', budget)
    with pytest.raises(RuntimeError, match='IMINT_API_MAX_PIXELS'):
        with TestClient(api.server.app):
            pass
    assert not api.server._accepting_jobs
    assert not api.server._jobs and not api.calls
    assert not (api.tmp / 'artifacts').exists()


@pytest.mark.parametrize('edge', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_projected_grid_rejected(api, monkeypatch, edge):
    monkeypatch.setattr(api.scene, '_to_nmd_grid', lambda coords: dict(
        west=0, south=0, east=edge, north=100))
    with TestClient(api.server.app) as client:
        assert client.post('/v1/analyze', json=SMALL).status_code == 422
        assert not api.server._jobs and not api.calls
