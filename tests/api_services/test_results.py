import json
from .helpers import ROOT, handler, load


def test_A01_startup_and_containment(tmp_path):
    source=ROOT/'scripts/results_server.py'
    compile(source.read_text(),str(source),'exec')
    m=load('offline_results',source)
    m.RESULTS_DIR=tmp_path/'results'; m.CHECKPOINTS_DIR=tmp_path/'checkpoints'
    for root in [m.RESULTS_DIR,m.CHECKPOINTS_DIR]: (root/'ok').mkdir(parents=True)
    (m.RESULTS_DIR/'ok/training_log.json').write_text('{"status":"valid"}')
    (m.RESULTS_DIR/'ok/checkpoint_meta.json').write_text('{}')
    (m.CHECKPOINTS_DIR/'ok/best_model.pt').write_bytes(b'synthetic')
    (tmp_path/'training_log.json').write_text('{"status":"outside"}')
    (m.RESULTS_DIR/'escape').symlink_to(tmp_path,target_is_directory=True)
    for path in ['/health','/api/runs','/api/runs/ok/log','/api/runs/ok/meta','/api/runs/ok/checkpoint']:
        h=handler(m.ResultsHandler,path);h.do_GET();assert h.status==200
    for path in ['/api/runs/../log','/api/runs/escape/log','/api/runs/ok/extra/log']:
        h=handler(m.ResultsHandler,path);h.do_GET();assert h.status==404
    h=handler(m.ResultsHandler,'/api/runs');h.do_GET()
    assert [r['id'] for r in json.loads(h.wfile.getvalue())['runs']]==['ok']
