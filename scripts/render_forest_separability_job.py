#!/usr/bin/env python3
"""Render (never submit) the bounded CPU class diagnostic and exact source bundle."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:921aed5400857ae48eec8214007add39e9f0543f518e5d042af893562370d86d'
SCRIPTS = ('extract_forest_separability.py', 'probe_forest_separability.py',
           'inference_comparison.py', 'cache_predict.py', 'extract_plot_features.py',
           'build_pinned_plot_set.py', 'validate_against_nfi.py', 'infer_tiles.py')

BOOTSTRAP = '''import hashlib,json,os,pathlib,subprocess,tarfile,time
root=pathlib.Path('/scratch/source'); root.mkdir()
bundle=pathlib.Path('/recipe/source.tar.gz')
assert hashlib.sha256(bundle.read_bytes()).hexdigest()==os.environ['SOURCE_SHA256']
with tarfile.open(bundle) as archive:
    archive.extractall(root, filter='data')
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
for name, sha in manifest['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha, name
os.environ['PYTHONPATH']=str(root); os.chdir(root)
python='/opt/venvs/model/bin/python'
status=0
try:
    subprocess.run([python,'-c','import torch,numpy,scipy,sklearn,pandas; print(torch.__version__,numpy.__version__,sklearn.__version__)'],check=True)
    with open('/scratch/extract.log','w') as stream:
        subprocess.run([python,'-u','scripts/extract_forest_separability.py',
            '--data-dir','/data/unified_v2_512','--label-dir','/data/nmd2023_labels',
            '--distill-dir','/data/distill/tessera_r2',
            '--checkpoint','/data/checkpoints/ladder/tessera_r4/best_model.pt',
            '--output','/scratch/results','--feature-year','2022','--max-tiles','24','--per-class','128'],
            stdout=stream,stderr=subprocess.STDOUT,check=True)
    with open('/scratch/probe.log','w') as stream:
        subprocess.run([python,'scripts/probe_forest_separability.py',
            '--features','/scratch/results/features.npz','--output','/scratch/results/probe.json'],
            stdout=stream,stderr=subprocess.STDOUT,check=True)
except Exception as exc:
    print(type(exc).__name__,str(exc),flush=True); status=1
pathlib.Path('/scratch/exit-status').write_text(str(status))
print('OUTPUT_READY',status,flush=True)
# The parent copies private artifacts, then acknowledges before this job exits.
# Job-level1800s hard expiry bounds failed collection as well as computation.
while not pathlib.Path('/scratch/ACK').exists():
    time.sleep(2)
raise SystemExit(status)
'''


def render(output: Path, deadline_seconds: int = 1800):
    if not 60 <= deadline_seconds <= 1800:
        raise ValueError("deadline must be60–1800seconds")
    output.mkdir(parents=True, exist_ok=False)
    files = sorted((ROOT / 'imint').rglob('*.py'))
    files += [ROOT / 'scripts' / name for name in SCRIPTS]
    files += sorted((ROOT / 'imint/config').glob('*.yaml'))
    identities = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest = {'git_sha': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'uncommitted_source': True, 'files': identities, 'image': IMAGE}
    manifest_bytes = (json.dumps(manifest,indent=2)+'\n').encode()
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w:gz') as tar:
        for p in files:
            tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        info = tarfile.TarInfo('SOURCE_MANIFEST.json'); info.size=len(manifest_bytes)
        tar.addfile(info,io.BytesIO(manifest_bytes))
    bundle=raw.getvalue()
    if len(bundle)>900000:
        raise ValueError('source bundle exceeds ConfigMap cap')
    sha=hashlib.sha256(bundle).hexdigest()
    name='forest-separability-0127-'+sha[:8]
    cm={'apiVersion':'v1','kind':'ConfigMap','metadata':{'name':name,'namespace':'prithvi-training-default'},
        'immutable':True,'binaryData':{'source.tar.gz':base64.b64encode(bundle).decode()},
        'data':{'bootstrap.py':BOOTSTRAP}}
    job={'apiVersion':'batch/v1','kind':'Job','metadata':{'name':name,'namespace':'prithvi-training-default'},
         'spec':{'activeDeadlineSeconds':deadline_seconds,'backoffLimit':0,'ttlSecondsAfterFinished':3600,
             'template':{'spec':{'restartPolicy':'Never','automountServiceAccountToken':False,
                 'imagePullSecrets':[{'name':'ghcr-push'}],
                 'securityContext':{'runAsNonRoot':True,'runAsUser':2000,'runAsGroup':2000,
                                    'fsGroup':2000,'seccompProfile':{'type':'RuntimeDefault'}},
                 'containers':[{'name':'probe','image':IMAGE,
                     'command':['/opt/venvs/model/bin/python','-u','/recipe/bootstrap.py'],
                     'securityContext':{'allowPrivilegeEscalation':False,'capabilities':{'drop':['ALL']}},
                     'env':[{'name':k,'value':v} for k,v in {
                         'SOURCE_SHA256':sha,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                         'OPENBLAS_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1',
                         'PYTHONUNBUFFERED':'1','XDG_CACHE_HOME':'/scratch/cache',
                         'MPLCONFIGDIR':'/scratch/mpl','TORCH_HOME':'/scratch/torch','TMPDIR':'/scratch',
                         'USER':'probe','LOGNAME':'probe'}.items()],
                     'resources':{'requests':{'cpu':'2','memory':'4Gi'},'limits':{'cpu':'2','memory':'4Gi'}},
                     'volumeMounts':[{'name':'data','mountPath':'/data','readOnly':True},
                                     {'name':'scratch','mountPath':'/scratch'},
                                     {'name':'recipe','mountPath':'/recipe','readOnly':True}]}],
                 'volumes':[{'name':'data','persistentVolumeClaim':{'claimName':'training-data-cephfs','readOnly':True}},
                            {'name':'scratch','emptyDir':{'sizeLimit':'2Gi'}},
                            {'name':'recipe','configMap':{'name':name}}]}}}}
    (output/'source.tar.gz').write_bytes(bundle)
    (output/'SOURCE_MANIFEST.json').write_bytes(manifest_bytes)
    (output/'job.json').write_text(json.dumps({'apiVersion':'v1','kind':'List','items':[cm,job]},indent=2)+'\n')
    (output/'identity.json').write_text(json.dumps({'job':name,'bundle_sha256':sha,'bundle_bytes':len(bundle)},indent=2)+'\n')
    print(json.dumps({'job':name,'bundle_sha256':sha,'bundle_bytes':len(bundle)}))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--deadline-seconds',default=1800,type=int)
    args=ap.parse_args()
    render(args.output,args.deadline_seconds)
