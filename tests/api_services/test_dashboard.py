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


def test_nfi_inference_status_reads_only_named_metadata_fields(tmp_path):
    www, ns = _dashboard(tmp_path)
    (www / 'ladder').mkdir(parents=True)
    out = tmp_path / 'nfi_inference'
    out.mkdir()
    (out / 'nfi-per-plot-clay_r1.parquet').write_bytes(SECRET)
    (out / 'nfi-per-plot-clay_r1.parquet.meta.json').write_text(json.dumps({
        'cell': 'clay_r1', 'status': 'success', 'plots_in_crop': 5194,
        'prediction_sha256': 'ab' * 32, 'checkpoint_sha256': 'cd' * 32,
        'holdout_manifest_sha256': 'ef' * 32,
        'Easting': 6543210, 'TractID': 'SECRET_TRACT', 'accuracy': 0.99}))
    (out / 'nfi-per-plot-clay_r2.parquet.meta.json').write_text('{broken')
    cls = ns['Handler']
    cls.NFI_INFERENCE_DIR = str(out)
    status, response = _request(cls, www, '/ladder/nfi_inference.json?t=1')
    assert status == 200
    body = json.loads(response.split(b'\r\n\r\n', 1)[1])
    assert body['published'] == 1 and body['expected_cells'] == 28
    assert body['manifest_sha256'] == 'ef' * 6
    assert set(body['cells']) == {'clay_r1'}
    assert set(body['cells']['clay_r1']) == {'status', 'plots', 'prediction_sha256',
                                             'checkpoint_sha256', 'published_at'}
    assert body['cells']['clay_r1']['prediction_sha256'] == 'ab' * 6
    for leak in (b'6543210', b'SECRET_TRACT', b'0.99', SECRET):
        assert leak not in response
    cls.NFI_INFERENCE_DIR = str(tmp_path / 'absent')
    status, response = _request(cls, www, '/ladder/nfi_inference.json')
    assert status == 200 and json.loads(response.split(b'\r\n\r\n', 1)[1])['cells'] == {}


def test_served_json_never_carries_identity_keys(tmp_path):
    """2026-10-28 private probe: the inference matrix's tiles carried
    easting/northing. Every JSON read from a PVC tree is re-serialized without
    identity keys, at any depth; a file that does not parse is refused."""
    www, ns = _dashboard(tmp_path)
    _tree(tmp_path, www)
    pvc = tmp_path / 'pvc'
    (pvc / 'ladder_inference/matrix.json').write_text(json.dumps({
        'git_sha': 'abc', 'tiles': [{'name': 'holdoutval_1', 'easting': 654321.5,
                                     'Northing': 6543210.5, 'year': 2022}],
        'cells': {'clay_r1': {'epoch': 3, 'geometry': {'coordinates': [[1, 2]]}}},
        'plots': [{'TractID': 'T1', 'PlotID': 'P1', 'lat': 59.1, 'lon': 18.1}]}))
    (pvc / 'ladder_eval/broken.json').write_text('{not json')
    cls = ns['Handler']
    status, response = _request(cls, www, '/ladder/inference/matrix.json?t=1')
    assert status == 200
    body = json.loads(response.split(b'\r\n\r\n', 1)[1])
    assert body['tiles'] == [{'name': 'holdoutval_1', 'year': 2022}]
    assert body['cells'] == {'clay_r1': {'epoch': 3}}
    assert body['plots'] == [{}]
    for leak in (b'654321', b'6543210', b'T1', b'P1', b'59.1', b'18.1'):
        assert leak not in response
    status, response = _request(cls, www, '/ladder/fieldtruth/broken.json')
    assert status == 404 and b'not json' not in response


@pytest.mark.parametrize("path", [
    "/ladder//distill/heads/clay_r2_split.json",            # Codex, #114
    "/ladder/./distill/heads/clay_r2_split.json",           # Codex, #114
    "/ladder/%252e/distill/heads/clay_r2_split.json",       # Codex, #114: double-encoded "."
    "//ladder/distill/heads/clay_r2_split.json",
    "/ladder/distill%2fheads/clay_r2_split.json",            # encoded slash
    "/ladder/distill/heads/./clay_r2_split.json",
    "/ladder/distill/heads//clay_r2_plot_features.parquet",
    "/ladder/alias/heads/clay_r2_split.json",               # symlink alias into the tree
    "/ladder/alias/heads/clay_r2_plot_features.parquet",
    "/ladder/distill/heads/%2e%2e/heads/clay_r2_split.json",
    "/ladder/distill/heads/clay_r2_split.json%00.json",
])
def test_no_url_spelling_reaches_a_refused_body(tmp_path, path):
    www, ns = _dashboard(tmp_path)
    _tree(tmp_path, www)
    (www / 'ladder' / 'alias').symlink_to(tmp_path / 'pvc' / 'distill', target_is_directory=True)
    status, body = _request(ns['Handler'], www, path)
    assert status in (400, 404), (path, status)
    assert SECRET not in body and b'SECRET_TILE' not in body, path


def test_equivalent_spellings_of_an_allowed_file_still_work(tmp_path):
    www, ns = _dashboard(tmp_path)
    _tree(tmp_path, www)
    for path in ['/ladder//distill/heads/clay_r2_distillability.json',
                 '/ladder/./distill/heads/clay_r2_distillability.json']:
        status, body = _request(ns['Handler'], www, path)
        assert status == 200 and b'oof' in body, path


def test_training_log_normalization_strips_identity_keys(tmp_path):
    """Codex, #114: the training_log.json branch serves normalize()'s output,
    which keeps nested per-class objects; it must be stripped like any JSON."""
    www, ns = _dashboard(tmp_path)
    _tree(tmp_path, www)
    log = tmp_path / 'pvc' / 'checkpoints' / 'clay_r1' / 'training_log.json'
    log.write_text(json.dumps({
        'config': {'epochs': 2}, 'status': 'running',
        'epochs': [{'epoch': 1, 'val_miou': 0.5, 'train_loss': 1.0,
                    'per_class_iou': {'1': {'TractID': 'T9', 'Easting': 654321, 'iou': 0.4}},
                    'confusion_matrix': [[1, 0], [0, 1]]}]}))
    status, response = _request(ns['Handler'], www, '/ladder/data/clay_r1/training_log.json?t=1')
    assert status == 200
    body = json.loads(response.split(b'\r\n\r\n', 1)[1])
    assert body['per_class_iou'] == {'1': {'iou': 0.4}}
    assert body['epochs'][0]['per_class_iou'] == {'1': {'iou': 0.4}}
    assert b'T9' not in response and b'654321' not in response


def test_request_target_is_canonicalized_once(tmp_path):
    """The canonical path is what the policy checks and what every later step
    reads; anything still encoded after one decode is refused."""
    www, ns = _dashboard(tmp_path)
    for raw, expected in [('//ladder//./data/x.json?t=1', '/ladder/data/x.json'),
                          ('/ladder/distill%2fheads/a.json', '/ladder/distill/heads/a.json'),
                          ('/a//b/./', '/a/b/'), ('/', '/')]:
        h = handler(ns['Handler'], raw)
        assert h._unsafe_target() == expected, raw
        assert h.path.split('?', 1)[0] == expected, raw
    for raw in ['/ladder/%252e/x.json', '/ladder/%2e%2e/x.json', '/a\\b.json', '/a%00b.json', 'relative.json']:
        h = handler(ns['Handler'], raw)
        assert h._unsafe_target() is None and h.status == 400, raw


@pytest.mark.parametrize("path", [
    "/ladder/inference/matrix.json%3f.png",          # Codex, #114 (eeccb4c)
    "/ladder/inference/matrix.json%3F.png",
    "/ladder/inference/matrix.json%23.png",          # Codex, #114 (eeccb4c)
    "/ladder/data/clay_r1/best_model.pt%3f.png",     # Codex, #114 (eeccb4c): binary body
    "/ladder/data/clay_r1/best_model.pt%23.json",
    "/ladder/inference/matrix.json%20.png",
    "/ladder/inference/matrix.json%0a.png",
    "/ladder/inference/matrix.json%3b.png",
])
def test_encoded_delimiters_never_end_the_path_late(tmp_path, path):
    """A decoded "?" or "#" ended the path only when the file was opened, so
    the policy saw ".png" while the raw JSON or checkpoint was served."""
    www, ns = _dashboard(tmp_path)
    _tree(tmp_path, www)
    (tmp_path / 'pvc' / 'ladder_inference' / 'matrix.json').write_text(
        json.dumps({'tiles': [{'name': 'holdoutval_1', 'easting': 654321.5}]}))
    status, body = _request(ns['Handler'], www, path)
    assert status == 400, (path, status)
    assert SECRET not in body and b'654321' not in body, path
