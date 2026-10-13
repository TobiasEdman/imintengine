from email.message import Message
import io
import json
import types
import pytest
from .helpers import ROOT, load


@pytest.fixture
def picker(tmp_path):
    module = load('offline_picker', ROOT/'demos/aircraft_parallax/picker_server.py')
    module.REPO_ROOT = tmp_path; module.PICKS_FILE = tmp_path/'picks.json'
    module.PICKS_FILE.write_text('{"picks":{"B02":{"native":[1,2]}}}')
    return module


def request_picker(picker, changes=None, *, body=b'{"picks":{}}', duplicate=None):
    h = object.__new__(picker.PickerHandler)
    h.path = '/api/picks'; h.rfile = io.BytesIO(body); h.wfile = io.BytesIO()
    h.server = types.SimpleNamespace(server_address=('127.0.0.1', 8096))
    headers = {'Host':'localhost:8096', 'Origin':'http://localhost:8096', 'Content-Type':'application/json', 'Content-Length':str(len(body))}
    headers.update(changes or {})
    h.headers = Message()
    for key, val in headers.items():
        if val is not None: h.headers[key] = val
    if duplicate: h.headers[duplicate[0]] = duplicate[1]
    h.send_response = lambda code, *a: setattr(h, 'status', code)
    h.send_header = lambda *a: None; h.end_headers = lambda: None
    h.do_POST()
    return h


@pytest.mark.parametrize('changes', [
    {'Origin':'https://untrusted.invalid'}, {'Origin':'null'}, {'Origin':None},
    {'Origin':'http://localhost:9999'}, {'Origin':'https://localhost:8096'},
    {'Host':'untrusted.invalid:8096','Origin':'http://untrusted.invalid:8096'},
    {'Host':'localhost:9999','Origin':'http://localhost:9999'},
    {'Host':'localhost:bad'}, {'Host':None},
    {'Origin':'http://user@localhost:8096'}, {'Origin':'http://localhost:8096/path'},
])
def test_A09_cross_origin_or_host_rejected_without_write(picker, changes):
    before = picker.PICKS_FILE.read_bytes()
    assert request_picker(picker, changes).status == 403
    assert picker.PICKS_FILE.read_bytes() == before


@pytest.mark.parametrize('content_type', ['text/plain', 'application/x-www-form-urlencoded', None])
def test_A09_requires_json_media_type(picker, content_type):
    before = picker.PICKS_FILE.read_bytes()
    assert request_picker(picker, {'Content-Type':content_type}).status == 415
    assert picker.PICKS_FILE.read_bytes() == before


@pytest.mark.parametrize('host', ['localhost:8096','127.0.0.1:8096'])
def test_A09_actual_client_json_contract_persists(picker, host):
    source = (ROOT/'docs/aircraft_parallax_picker.html').read_text()
    assert "fetch('/api/picks', {" in source and "'Content-Type': 'application/json'" in source
    body = b'{"picks":{"B02":{"native":[3,4]}}}'
    h = request_picker(picker, {'Host':host, 'Origin':'http://'+host, 'Content-Type':'application/json; charset=utf-8'}, body=body)
    assert h.status == 200
    assert json.loads(picker.PICKS_FILE.read_text())['picks']['B02']['native'] == [3,4]


@pytest.mark.parametrize('duplicate', [('Host','localhost:8096'), ('Origin','http://localhost:8096'), ('Content-Type','application/json'), ('Content-Length','12')])
def test_A09_duplicate_boundary_headers_rejected(picker, duplicate):
    before = picker.PICKS_FILE.read_bytes()
    assert request_picker(picker, duplicate=duplicate).status in (400,403,415)
    assert picker.PICKS_FILE.read_bytes() == before


@pytest.mark.parametrize('changes,body', [({'Content-Length':'oops'},b'{}'), ({'Content-Length':'-1'},b'{}'), ({'Transfer-Encoding':'chunked'},b'{}'), ({},b'\xff'), ({},b'[]'), ({},b'{invalid')])
def test_A09_bad_body_does_not_mutate(picker, changes, body):
    before = picker.PICKS_FILE.read_bytes()
    assert request_picker(picker, changes, body=body).status == 400
    assert picker.PICKS_FILE.read_bytes() == before
