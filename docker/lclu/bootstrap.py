"""Reproduce the Dockerfile's locked environment on the isolated demo PVC.

For a temporary private test when local Docker cannot build. The published
Python base is digest-pinned; source bundle and all Python distributions are
hash-pinned. This process never alters the mounted reference-data volume.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

bundle = Path('/recipe/source.tar.gz')
bundle_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
if bundle_hash != os.environ['LCLU_BUNDLE_SHA256']:
    raise ValueError('Source archive identity mismatch')
source = Path('/work/source')/bundle_hash
if not source.exists():
    staging = source.with_name(bundle_hash+'.partial')
    staging.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle) as archive:
        archive.extractall(staging, filter='data')
    staging.rename(source)
manifest = json.loads((source/'SOURCE_MANIFEST.json').read_text())
for relative, digest in manifest['files'].items():
    if hashlib.sha256((source/relative).read_bytes()).hexdigest() != digest:
        raise ValueError(f'Source identity mismatch: {relative}')
lock = source/'docker/lclu/requirements.lock'
lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest()
environment = Path('/work/runtime')/lock_hash
python = environment/'bin/python'
if not python.exists():
    subprocess.run([sys.executable, '-m', 'venv', str(environment)], check=True)
if not (environment/'READY').exists():
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir', '--require-hashes',
                    '-r', str(lock)], check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
    with (environment/'installed.txt').open('w') as stream:
        subprocess.run([str(python), '-m', 'pip', 'freeze', '--all'], stdout=stream, check=True)
    (environment/'READY').write_text(lock_hash+'\n')
os.environ['PYTHONPATH'] = str(source)
os.environ['IMINT_LCLU_MANIFEST'] = '/work/demo/serving.json'
os.environ['IMINT_API_OUTPUT_DIR'] = '/work/results'
os.environ['LCLU_EXECUTION_MODE'] = 'source-recipe'
system_root = Path('/work/system/expat-2.5.0-1-deb12u5')
os.environ['LD_LIBRARY_PATH'] = subprocess.check_output(
    [sys.executable, str(source/'docker/lclu/system_runtime.py'), str(system_root)], text=True).strip()
os.chdir(source)
subprocess.run([str(python), str(source/'docker/lclu/smoke.py')], check=True)
os.execv(python, [str(python), str(source/'docker/lclu/run.py')])
