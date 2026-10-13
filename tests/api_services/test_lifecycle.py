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


def test_A06_lifespan_required(api):
    c = TestClient(api.server.app)  # intentionally no lifespan startup
    assert c.post('/v1/analyze', json=REQ).status_code == 503
    assert not api.server._jobs


def test_A06_second_local_worker_rejected_and_restart_releases(api):
    other = load('imint.api.v1.second_worker', ROOT/'imint/api/v1/server.py')
    async def run():
        async with api.server.app.router.lifespan_context(api.server.app):
            with pytest.raises(RuntimeError, match='one worker'):
                async with other.app.router.lifespan_context(other.app): pass
            # Separate process confirms this is a real OS lock, not only a
            # module flag. No app/server/credentials imported by child.
            script = '''import fcntl,sys
with open(sys.argv[1], "a") as f:
 try: fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError: sys.exit(23)
sys.exit(0)
'''
            r = subprocess.run([sys.executable, '-c', script, str(api.tmp/'artifacts/.api-worker.lock')], capture_output=True)
            assert r.returncode == 23
        async with other.app.router.lifespan_context(other.app):
            r = await other.submit_analysis(other.AnalyzeRequest(**REQ)); await drain(other)
            assert (await other.get_job(r.job_id)).status == 'success'
    asyncio.run(run())


def test_A06_serial_execution_and_shutdown_drains(api):
    started, release = threading.Event(), threading.Event()
    lock = threading.Lock(); active = 0; maximum = 0
    original = api.engine.run_job
    def block(job):
        nonlocal active, maximum
        with lock: active += 1; maximum = max(maximum, active)
        started.set()
        assert release.wait(5), 'test release missing'
        try: return original(job)
        finally:
            with lock: active -= 1
    api.engine.run_job = block
    async def run():
        life = api.server.app.router.lifespan_context(api.server.app)
        await life.__aenter__()
        try:
            first = await api.server.submit_analysis(api.server.AnalyzeRequest(**REQ))
            second = await api.server.submit_analysis(api.server.AnalyzeRequest(**REQ))
            assert await asyncio.to_thread(started.wait, 2)
            assert (await api.server.get_job(second.job_id)).status == 'queued'
            shutdown = asyncio.create_task(life.__aexit__(None, None, None))
            await asyncio.sleep(0.01)
            assert not shutdown.done()
            with pytest.raises(api.server.HTTPException) as error:
                await api.server.submit_analysis(api.server.AnalyzeRequest(**REQ))
            assert error.value.status_code == 503
            release.set(); await shutdown
            assert maximum == 1
            assert all(api.server._jobs[jid]['status'] == 'success' for jid in [first.job_id, second.job_id])
        finally:
            release.set()
    asyncio.run(run())


def test_A06_cancelled_shutdown_keeps_lock_until_thread_finishes(api):
    import fcntl
    started, release = threading.Event(), threading.Event()
    original = api.engine.run_job
    def block(job):
        started.set()
        assert release.wait(5), 'test release missing'
        return original(job)
    api.engine.run_job = block
    async def run():
        life = api.server.app.router.lifespan_context(api.server.app)
        await life.__aenter__()
        try:
            r = await api.server.submit_analysis(api.server.AnalyzeRequest(**REQ))
            assert await asyncio.to_thread(started.wait, 2)
            shutdown = asyncio.create_task(life.__aexit__(None, None, None))
            await asyncio.sleep(0.01); shutdown.cancel(); await asyncio.sleep(0.01)
            assert not shutdown.done()
            with (api.tmp/'artifacts/.api-worker.lock').open('a') as probe:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release.set()
            with pytest.raises(asyncio.CancelledError): await shutdown
            assert api.server._jobs[r.job_id]['status'] == 'success'
            with (api.tmp/'artifacts/.api-worker.lock').open('a') as probe:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally: release.set()
    asyncio.run(run())


def test_single_worker_deployment_defaults():
    """Validate shipped launch arguments and replica strategy without Docker/K8s."""
    import shlex
    import yaml

    dockerfile = (ROOT / "Dockerfile.api").read_text()
    cmd_line = next(line[4:] for line in dockerfile.splitlines() if line.startswith("CMD "))
    command = json.loads(cmd_line)
    argv = shlex.split(command[-1])
    assert argv[:3] == ["exec", "uvicorn", "imint.api.v1:app"]
    assert argv[argv.index("--workers") + 1] == "1"
    assert argv[argv.index("--lifespan") + 1] == "on"
    assert "GUNICORN_WORKERS" not in dockerfile
    deployment = next(doc for doc in yaml.safe_load_all(
        (ROOT / "k8s/api-deployment.yaml").read_text()
    ) if doc["kind"] == "Deployment")
    assert deployment["spec"]["replicas"] == 1
    assert deployment["spec"]["strategy"] == {"type": "Recreate"}
    variables = deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    assert not any(item["name"] == "GUNICORN_WORKERS" for item in variables)
    budget = next(item for item in variables if item["name"] == "IMINT_API_MAX_PIXELS")
    assert budget == {"name": "IMINT_API_MAX_PIXELS", "valueFrom": {"configMapKeyRef": {
        "name": "imint-api-resource-budget", "key": "max_pixels", "optional": False,
    }}}



@pytest.mark.parametrize("worker_fails", [False, True])
def test_A06_cancel_all_tasks_retains_running_thread_ownership(api, worker_fails):
    import fcntl
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = api.engine.run_job

    def block(job):
        started.set()
        try:
            assert release.wait(5), 'test release missing'
            if worker_fails:
                raise RuntimeError('synthetic worker failure')
            return original(job)
        finally:
            finished.set()

    api.engine.run_job = block

    async def run():
        life = api.server.lifespan(api.server.app)
        await life.__aenter__()
        shutdown = None
        try:
            response = await api.server.submit_analysis(api.server.AnalyzeRequest(**REQ))
            assert await asyncio.to_thread(started.wait, 2)
            jobs = tuple(api.server._pending_jobs)
            shutdown = asyncio.create_task(life.__aexit__(None, None, None))
            await asyncio.sleep(0)
            # Model runner teardown: cancel all tasks except this test driver.
            # Repeat cancellation while the blocking work is being joined.
            for _ in range(2):
                for task in asyncio.all_tasks() - {asyncio.current_task()}:
                    task.cancel()
                await asyncio.sleep(0.01)
                assert not finished.is_set()
                assert not shutdown.done()
                assert api.server._analysis_slot.locked()
                assert all(not task.done() for task in jobs)
                with (api.tmp/'artifacts/.api-worker.lock').open('a') as probe:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await shutdown
            assert finished.is_set()
            assert all(task.cancelled() for task in jobs)
            assert not api.server._pending_jobs
            assert not api.server._analysis_slot.locked()
            job = api.server._jobs[response.job_id]
            assert job['status'] == ('failed' if worker_fails else 'success')
            assert job['completed_at'] is not None
            if worker_fails:
                assert 'synthetic worker failure' in job['error']
            with (api.tmp/'artifacts/.api-worker.lock').open('a') as probe:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            release.set()
            if shutdown is None:
                await life.__aexit__(None, None, None)
            else:
                await asyncio.gather(shutdown, return_exceptions=True)
            assert await asyncio.to_thread(finished.wait, 2)

    asyncio.run(run())
