import ast
import io
import json
import pytest
import textwrap
from .helpers import ROOT, handler


def test_A04_dashboard_routes(tmp_path):
    text=(ROOT/'k8s/training-dashboard-server.yaml').read_text()
    source=textwrap.dedent(text.split("cat > /serve.py <<'PY'\n",1)[1].split('\n              PY',1)[0])
    www=tmp_path/'www';(www/'compare').mkdir(parents=True)
    outside=tmp_path/'outside';outside.mkdir()
    (outside/'marker.html').write_text('OUTSIDE')
    (outside/'marker.json').write_text('{"status":"OUTSIDE"}')
    (outside/'training_log.json').write_text('{"status":"published"}')
    (www/'compare/index.html').write_text('<head></head>')
    (www/'published').symlink_to(outside,target_is_directory=True)
    source=source.replace('"/www"',repr(str(www)))
    tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef)) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='COMPARE_SLIC_STRIP' for t in n.targets))]
    ns={'TILE_COUNTS':{}}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'dashboard-inline','exec'),ns)
    for path in ['//../outside/training_log.json', '/compare/../../outside/marker.html','/../outside/marker.json?training_log.json?','/compare/%2e%2e/%2e%2e/outside/marker.html']:
        h=handler(ns['Handler'],path);h.do_GET();assert h.status==400
    for path in ['/compare/','/published/training_log.json']:
        h=handler(ns['Handler'],path);h.do_GET();assert h.status==200


@pytest.mark.parametrize("target, marker", [
    ("/compare/review#1/../../../results-sibling/marker.html", b"OUTSIDE_HTML"),
    ("/compare/review#1/../../../results-sibling/marker.json?training_log.json?", b"OUTSIDE_JSON"),
])
def test_A04_raw_fragment_request_target(tmp_path, target, marker):
    """Exercise the real HTTP parser with the traversable # directory present."""
    text = (ROOT / 'k8s/training-dashboard-server.yaml').read_text()
    source = textwrap.dedent(text.split("cat > /serve.py <<'PY'\n", 1)[1].split('\n              PY', 1)[0])
    www = tmp_path / 'www'
    www.mkdir()
    published = tmp_path / 'pvc' / 'compare'
    (published / 'review#1').mkdir(parents=True)
    (www / 'compare').symlink_to(published, target_is_directory=True)
    outside = tmp_path / 'results-sibling'
    outside.mkdir()
    (outside / 'marker.html').write_text('OUTSIDE_HTML')
    (outside / 'marker.json').write_text('{"status":"OUTSIDE_JSON"}')
    (published / 'index.html').write_text('<head></head>PUBLISHED_HTML')
    (published / 'training_log.json').write_text('{"status":"PUBLISHED_JSON"}')
    source = source.replace('"/www"', repr(str(www)))
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)) or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'COMPARE_SLIC_STRIP' for t in n.targets))]
    ns = {'TILE_COUNTS': {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'dashboard-inline', 'exec'), ns)

    def request(path):
        h = object.__new__(ns['Handler'])
        h.rfile = io.BytesIO(f'GET {path} HTTP/1.1\r\nHost: offline.invalid\r\n\r\n'.encode('ascii'))
        h.wfile = io.BytesIO()
        h.client_address = ('127.0.0.1', 0)
        h.log_message = lambda *args: None
        h.handle_one_request()
        return h.wfile.getvalue()

    response = request(target)
    assert response.split(b'\r\n', 1)[0].split()[1] == b'400'
    assert marker not in response
    # Intentional external publication and harmless query contents still work.
    for path, expected in [('/compare/?next=../ok', b'PUBLISHED_HTML'),
                           ('/compare/training_log.json?next=../ok', b'PUBLISHED_JSON')]:
        response = request(path)
        assert response.split(b'\r\n', 1)[0].split()[1] == b'200'
        assert expected in response


# ── publication allowlist (2026-10-27) ──────────────────────────────────────
# The dashboard is reachable through a public ingress, and the PVC trees it
# links hold per-plot NFI files (TractID, PlotID, Easting, Northing) beside
# the files the dashboards read. These run the real HTTP parser.

SECRET = b"TractID,PlotID,Easting,Northing 6543210 543210"


def _dashboard(tmp_path):
    text = (ROOT / 'k8s/training-dashboard-server.yaml').read_text()
    source = textwrap.dedent(text.split("cat > /serve.py <<'PY'\n", 1)[1].split('\n              PY', 1)[0])
    www = tmp_path / 'www'
    source = source.replace('"/www"', repr(str(www)))
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)) or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'COMPARE_SLIC_STRIP' for t in n.targets))]
    ns = {'TILE_COUNTS': {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'dashboard-inline', 'exec'), ns)
    return www, ns


def _request(cls, www, path, method='GET'):
    h = object.__new__(cls)
    h.directory = str(www)
    h.rfile = io.BytesIO(f'{method} {path} HTTP/1.1\r\nHost: offline.invalid\r\n\r\n'.encode('ascii'))
    h.wfile = io.BytesIO()
    h.client_address = ('127.0.0.1', 0)
    h.log_message = lambda *args: None
    h.handle_one_request()
    response = h.wfile.getvalue()
    return int(response.split(b'\r\n', 1)[0].split()[1]), response


def _tree(tmp_path, www):
    pvc = tmp_path / 'pvc'
    files = {
        'distill/heads/clay_r2_plot_features.parquet': SECRET,
        'distill/heads/clay_r2_split.json': b'{"train_tiles": ["SECRET_TILE"]}',
        'distill/heads/clay_r2_distillability.json': b'{"oof": 0.5}',
        'distill/clay_r2/_GATE_OK': b'',
        'distill/clay_r2/tile_1.npz': SECRET,
        'ladder_eval/fieldtruth/nfi-per-plot-clay_r1.parquet': SECRET,
        'ladder_eval/fieldtruth-standings.json': b'{"nfi": []}',
        'ladder_eval/sameplots/sameplots-clay_r1.json': b'{"n_plots_compared": 1}',
        'checkpoints/clay_r1/best_model.pt': SECRET,
        'checkpoints/clay_r1/training_log.json': b'{"epochs": []}',
        'ladder_inference/matrix.json': b'{"tiles": []}',
        'ladder_inference/raw/t1.png': b'\x89PNG',
    }
    for rel, data in files.items():
        (pvc / rel).parent.mkdir(parents=True, exist_ok=True)
        (pvc / rel).write_bytes(data)
    (www / 'ladder').mkdir(parents=True)
    (www / 'ladder/index.html').write_text('<html>LADDER</html>')
    for link, target in [('data', 'checkpoints'), ('distill', 'distill'),
                         ('inference', 'ladder_inference'), ('fieldtruth', 'ladder_eval')]:
        (www / 'ladder' / link).symlink_to(pvc / target, target_is_directory=True)


def test_dashboard_serves_only_the_allowlist(tmp_path):
    www, ns = _dashboard(tmp_path)
    _tree(tmp_path, www)
    cls = ns['Handler']
    refused = [
        '/ladder/distill/heads/clay_r2_plot_features.parquet',
        '/ladder/distill/heads/clay_r2_split.json',
        '/ladder/distill/clay_r2/tile_1.npz',
        '/ladder/distill/heads/',
        '/ladder/distill/',
        '/ladder/fieldtruth/fieldtruth/nfi-per-plot-clay_r1.parquet',
        '/ladder/fieldtruth/fieldtruth/',
        '/ladder/fieldtruth/',
        '/ladder/data/clay_r1/best_model.pt',
        '/ladder/data/',
        '/ladder/inference/raw/',
    ]
    for path in refused:
        status, body = _request(cls, www, path)
        assert status == 404, (path, status)
        assert SECRET not in body and b'SECRET_TILE' not in body, path
        assert b'nfi-per-plot' not in body and b'plot_features' not in body, path
    served = [
        ('/ladder/', b'LADDER'),
        ('/ladder/distill/heads/clay_r2_distillability.json', b'oof'),
        ('/ladder/fieldtruth/fieldtruth-standings.json', b'nfi'),
        ('/ladder/fieldtruth/sameplots/sameplots-clay_r1.json', b'n_plots_compared'),
        ('/ladder/inference/matrix.json', b'tiles'),
        ('/ladder/inference/raw/t1.png', b'PNG'),
    ]
    for path, marker in served:
        status, body = _request(cls, www, path + '?t=1')
        assert status == 200 and marker in body, (path, status)


def test_dashboard_head_probes_send_no_body(tmp_path):
    www, ns = _dashboard(tmp_path)
    _tree(tmp_path, www)
    cls = ns['Handler']
    for path in ['/ladder/distill/heads/clay_r2_plot_features.parquet', '/ladder/distill/clay_r2/_GATE_OK']:
        status, response = _request(cls, www, path, 'HEAD')
        assert status == 200, path
        assert SECRET not in response
    status, _ = _request(cls, www, '/ladder/distill/heads/missing.parquet', 'HEAD')
    assert status == 404
    status, _ = _request(cls, www, '/ladder/../pvc/distill/heads/clay_r2_plot_features.parquet', 'HEAD')
    assert status == 400
