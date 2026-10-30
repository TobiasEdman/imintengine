#!/usr/bin/env python3
"""Render the bounded E01 metadata audit; never submit cluster resources."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_observation_indices import INPUTS, validate_spec
from scripts.crop_holdout_guard import _object, read_pinned

IMAGE = 'ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:25919e5cc0bc5ccba770b80f5906d0721cf47a779e3b4591a07cf5c5f3dc3599'
NAMESPACE = 'prithvi-training-default'


def build_job(spec_path, spec_sha256):
    raw = read_pinned(spec_path, spec_sha256)
    validate_spec(_object(raw))
    root = Path(__file__).resolve().parents[1]
    files = {'audit_observation_indices.py':'scripts/audit_observation_indices.py',
             'crop_holdout_guard.py':'scripts/crop_holdout_guard.py',
             'tile_config.py':'imint/training/tile_config.py'}
    data = {name:(root/relative).read_text() for name,relative in files.items()}
    data['inputs.json'] = raw.decode()
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    name = 'fm-e01-index-audit-' + digest[:12]
    labels = {'app':'fm-experiments', 'purpose':'observation-index-audit'}
    config = {'apiVersion':'v1', 'kind':'ConfigMap',
              'metadata':{'name':name, 'namespace':NAMESPACE, 'labels':labels},
              'immutable':True, 'data':data}
    mounts = [{'name':'audit-code', 'mountPath':'/opt/imintengine/'+relative,
               'subPath':name, 'readOnly':True} for name,relative in files.items()]
    mounts += [{'name':'audit-code','mountPath':'/audit/inputs.json','subPath':'inputs.json','readOnly':True},
               {'name':'work','mountPath':'/work'}]
    paths = [relative for relative, _ in INPUTS.values()] + [
        'distill/crop_split/lucas_crop_split.json',
        'distill/crop_split/lucas_crop_split.MANIFEST.json']
    mounts += [{'name':'metadata','mountPath':'/cephfs/'+relative,'subPath':relative,'readOnly':True}
               for relative in paths]
    resources = {'cpu':'2', 'memory':'4Gi', 'ephemeral-storage':'1Gi'}
    container = {
        'name':'audit','image':IMAGE,'imagePullPolicy':'IfNotPresent',
        'command':['/opt/venvs/scoring/bin/python','-m','scripts.audit_observation_indices'],
        'args':['--input-root','/cephfs','--spec','/audit/inputs.json','--spec-sha256',spec_sha256,
                '--output','/work/result.json'],
        'workingDir':'/opt/imintengine',
        'env':[{'name':'PYTHONPATH','value':'/opt/imintengine'},
               {'name':'PYTHONDONTWRITEBYTECODE','value':'1'},
               {'name':'TMPDIR','value':'/work'},
               {'name':'OMP_NUM_THREADS','value':'2'},
               {'name':'ARROW_NUM_THREADS','value':'2'},
               {'name':'OPENBLAS_NUM_THREADS','value':'2'},
               {'name':'AUDIT_IMAGE','value':IMAGE},
               {'name':'POD_UID','valueFrom':{'fieldRef':{'fieldPath':'metadata.uid'}}}],
        'securityContext':{'allowPrivilegeEscalation':False,'readOnlyRootFilesystem':True,
                           'capabilities':{'drop':['ALL']}},
        'resources':{'requests':resources.copy(),'limits':resources.copy()},'volumeMounts':mounts,
    }
    job = {'apiVersion':'batch/v1','kind':'Job',
           'metadata':{'name':name,'namespace':NAMESPACE,'labels':labels},
           'spec':{'backoffLimit':0,'activeDeadlineSeconds':600,'ttlSecondsAfterFinished':604800,
                   'template':{'metadata':{'labels':labels},'spec':{
                       'restartPolicy':'Never','automountServiceAccountToken':False,
                       'terminationGracePeriodSeconds':0,
                       'imagePullSecrets':[{'name':'ghcr-push'}],
                       'securityContext':{'runAsNonRoot':True,'runAsUser':2007,'runAsGroup':2000,
                                          'fsGroup':2000,'seccompProfile':{'type':'RuntimeDefault'}},
                       'containers':[container],
                       'volumes':[{'name':'audit-code','configMap':{'name':name}},
                                  {'name':'metadata','persistentVolumeClaim':{
                                      'claimName':'training-data-cephfs','readOnly':True}},
                                  {'name':'work','emptyDir':{'sizeLimit':'1Gi'}}],
                   }}}}
    return {'apiVersion':'v1','kind':'List','items':[config,job]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True,type=Path)
    parser.add_argument('--spec-sha256',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    manifest = build_job(args.spec,args.spec_sha256)
    with args.output.open('x') as stream:
        json.dump(manifest,stream,indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
