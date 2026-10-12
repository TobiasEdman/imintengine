"""Scoped offline guard. All monkeypatches are restored after each test."""
import socket
import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Service regression attempted network access")
    for name in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


import importlib
import json
import sys
import types
from pathlib import Path
import numpy as np
from .helpers import ROOT


@pytest.fixture
def api(monkeypatch, tmp_path):
    # Replace only this fixture's imports, restoring every module afterwards.
    saved = {name: mod for name, mod in sys.modules.items() if name == 'imint' or name.startswith('imint.')}
    for name in saved: del sys.modules[name]
    for name, path in [('imint', ROOT/'imint'), ('imint.api', ROOT/'imint/api'), ('imint.api.v1', ROOT/'imint/api/v1')]:
        mod = types.ModuleType(name); mod.__path__ = [str(path)]; sys.modules[name] = mod
    # A fresh metrics registry prevents fixture-to-fixture collector duplication.
    import prometheus_client as prom
    registry = prom.CollectorRegistry()
    fake_prom = types.ModuleType('prometheus_client')
    for name in ['Counter', 'Gauge', 'Histogram']:
        def ctor(*a, _cls=getattr(prom, name), **kw): return _cls(*a, registry=registry, **kw)
        setattr(fake_prom, name, ctor)
    fake_prom.CONTENT_TYPE_LATEST = prom.CONTENT_TYPE_LATEST
    fake_prom.generate_latest = lambda: prom.generate_latest(registry)
    monkeypatch.setitem(sys.modules, 'prometheus_client', fake_prom)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('IMINT_API_OUTPUT_DIR', str(tmp_path/'artifacts'))
    (tmp_path/'config').mkdir(); (tmp_path/'config/analyzers.yaml').write_text('{}')
    job = importlib.import_module('imint.job')
    analyzers = types.ModuleType('imint.analyzers')
    analyzers.__path__ = [str(ROOT / 'imint/analyzers')]
    sys.modules['imint.analyzers'] = analyzers
    analysis_result = importlib.import_module('imint.analyzers.base').AnalysisResult
    engine = types.ModuleType('imint.engine')
    engine.ANALYZER_REGISTRY = {}
    sys.modules['imint.engine'] = engine
    # Import the actual production fetch module. Only the outbound
    # _fetch_s2_via_openeo call is stubbed; no source extraction is used.
    scene = importlib.import_module('imint.fetch')
    scene_ns = scene.__dict__
    rgb = np.ones((2, 2, 3), np.float32)
    bands = {'B04': np.ones((2, 2), np.float32)}
    scl = np.zeros((2, 2), np.uint8)
    geo = job.GeoContext('EPSG:3006', (1, 0, 0, 0, -1, 0), {}, {}, (2, 2))
    fetched = scene.FetchResult(bands, scl, 0.0, rgb, geo)
    calls, jobs = [], []
    def fetch_boundary(**kw): calls.append(kw); return fetched
    scene_ns['_fetch_s2_via_openeo'] = fetch_boundary
    def run(job_in):
        jobs.append(job_in)
        if job_in.rgb is None:
            return job.IMINTResult(job_in.job_id, job_in.date, False, error='rgb missing')
        # Observable, harmless persistence at the engine boundary.
        out = Path(job_in.output_dir); out.mkdir(parents=True, exist_ok=True)
        np.save(out/'rgb.npy', job_in.rgb)
        (out/'summary.json').write_text(json.dumps({'job_id': job_in.job_id}))
        return job.IMINTResult(job_in.job_id, job_in.date, True, summary_path=str(out/'summary.json'))
    engine.run_job = run
    server = importlib.import_module('imint.api.v1.server')
    sys.modules['imint.api.v1'].app = server.app
    server.emit_access_log = lambda **kw: None
    if hasattr(server, '_notify_callback'):
        async def no_outbound_callback(*args): pass
        server._notify_callback = no_outbound_callback
    ctx = types.SimpleNamespace(analysis_result=analysis_result, server=server, calls=calls, jobs=jobs, scene=scene, scene_ns=scene_ns, fetched=fetched, engine=engine, job=job, tmp=tmp_path)
    try:
        yield ctx
    finally:
        for name in list(sys.modules):
            if name == 'imint' or name.startswith('imint.'):
                del sys.modules[name]
        sys.modules.update(saved)
