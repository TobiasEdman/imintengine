#!/usr/bin/env python3
"""Render (never submit) the bounded CPU class diagnostic and exact source bundle."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.forest_separability_schemes import (  # stdlib only, no numpy here
    CLASS_SCHEMES, FEATURE_YEARS, resolve_model, run_tag)
IMAGE = 'ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:921aed5400857ae48eec8214007add39e9f0543f518e5d042af893562370d86d'
SCRIPTS = ('extract_forest_separability.py', 'probe_forest_separability.py',
           'forest_separability_schemes.py',
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
            '--checkpoint',os.environ['PROBE_CHECKPOINT'],
            '--model-name',os.environ['PROBE_MODEL_NAME'],'--num-classes',os.environ['PROBE_NUM_CLASSES'],
            '--output','/scratch/results','--feature-year',os.environ['PROBE_FEATURE_YEAR'],
            '--scheme',os.environ['PROBE_SCHEME'],'--max-tiles','24','--per-class','128'],
            stdout=stream,stderr=subprocess.STDOUT,check=True)
    with open('/scratch/probe.log','w') as stream:
        subprocess.run([python,'scripts/probe_forest_separability.py',
            '--features','/scratch/results/features.npz','--output','/scratch/results/probe.json',
            '--scheme',os.environ['PROBE_SCHEME'],'--model-name',os.environ['PROBE_MODEL_NAME']],
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


def check_checkpoint(path: str) -> None:
    """The Job mounts the training PVC read-only at /data; a checkpoint must be on it."""
    p = PurePosixPath(path)
    if p.parts[:3] != ('/', 'data', 'checkpoints') or '..' in p.parts or p.suffix != '.pt':
        raise ValueError(f"checkpoint {path!r} must be a .pt file under /data/checkpoints/")


def render(output: Path, *, scheme: str, feature_year: int, deadline_seconds: int = 1800,
           model_name: str | None = None, checkpoint: str | None = None,
           num_classes: int | None = None):
    """The model arguments are all given or all None (R4)."""
    if not 60 <= deadline_seconds <= 1800:
        raise ValueError("deadline must be60–1800seconds")
    if scheme not in CLASS_SCHEMES or feature_year not in FEATURE_YEARS:
        raise ValueError("scheme must be pre-registered and feature year 2022 or 2023")
    model_name, checkpoint, num_classes = resolve_model(model_name, checkpoint, num_classes)
    check_checkpoint(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    files = sorted((ROOT / 'imint').rglob('*.py'))
    files += [ROOT / 'scripts' / name for name in SCRIPTS]
    files += sorted((ROOT / 'imint/config').glob('*.yaml'))
    identities = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest = {'git_sha': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'uncommitted_source': bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
                'files': identities, 'image': IMAGE, 'scheme': scheme, 'feature_year': feature_year,
                'model_name': model_name, 'checkpoint': checkpoint, 'num_classes': num_classes}
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
    tag=run_tag(model_name, scheme, feature_year)
    name=f'forest-separability-0127-{tag}-{sha[:8]}'
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
                         'SOURCE_SHA256':sha,'PROBE_SCHEME':scheme,'PROBE_FEATURE_YEAR':str(feature_year),
                         'PROBE_MODEL_NAME':model_name,'PROBE_CHECKPOINT':checkpoint,
                         'PROBE_NUM_CLASSES':str(num_classes),
                         'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
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
    identity={'job':name,'tag':tag,'bundle_sha256':sha,'bundle_bytes':len(bundle)}
    (output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(identity))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--scheme',required=True,choices=sorted(CLASS_SCHEMES))
    ap.add_argument('--feature-year',required=True,type=int)
    ap.add_argument('--deadline-seconds',default=1800,type=int)
    # All three model options, or none for R4.
    ap.add_argument('--model-name',help='keys features, report, evidence and Job name')
    ap.add_argument('--checkpoint',help='path inside the Job')
    ap.add_argument('--num-classes',type=int,help="the checkpoint's class count (28 or 29)")
    args=ap.parse_args()
    render(args.output,scheme=args.scheme,feature_year=args.feature_year,
           deadline_seconds=args.deadline_seconds,model_name=args.model_name,
           checkpoint=args.checkpoint,num_classes=args.num_classes)
