"""Small in-process service-test helpers; no listener is started."""
from pathlib import Path
import importlib.util
import io

ROOT = Path(__file__).resolve().parents[2]
REQ = {"date": "2026-04-25", "bbox": {"west": 11.5, "south": 58, "east": 11.51, "north": 58.01}}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def handler(cls, path):
    h=object.__new__(cls); h.path=path; h.wfile=io.BytesIO()
    h.send_response=lambda code,*a: setattr(h,'status',code)
    h.send_error=lambda code,*a: setattr(h,'status',code)
    h.send_header=lambda *a:None; h.end_headers=lambda:None
    return h


import asyncio


async def drain(server):
    # Poll only the synthetic worker, yielding the event loop, without sockets.
    for _ in range(1000):
        if server._jobs and all(j['status'] in ('success', 'failed') for j in server._jobs.values()): return
        await asyncio.sleep(0.001)
    raise AssertionError('Synthetic job did not terminate')
