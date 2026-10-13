import ast
import io
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
