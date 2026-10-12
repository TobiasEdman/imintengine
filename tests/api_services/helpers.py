"""Small in-process service-test helpers; no listener is started."""
from pathlib import Path
import importlib.util
import io

ROOT = Path(__file__).resolve().parents[2]
REQ = {"date": "2026-04-25", "bbox": {"west": 11.5, "south": 58, "east": 12, "north": 58.5}}


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
