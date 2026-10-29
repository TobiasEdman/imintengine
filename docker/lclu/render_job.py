#!/usr/bin/env python3
"""Render the bounded private ICE demo. This command never applies resources."""
import argparse
import json
from pathlib import Path
import re
import base64
import hashlib
import io
import tarfile


def resources(image: str, year: int, recipe: Path | None = None, *,
              job_name: str = 'lclu-stockholm-0124', reuse_volume: bool = False,
              deadline_seconds: int = 86400) -> dict:
    if not re.fullmatch(r'[a-z0-9./_-]+@sha256:[0-9a-f]{64}', image):
        raise ValueError('A published image digest is required, not a tag')
    if year not in (2024, 2025):
        raise ValueError('Only inventoried demo years are allowed')
    if not re.fullmatch(r'lclu-stockholm-0124(?:-r[0-9]+)?', job_name):
        raise ValueError('Job must belong to the Stockholm demo')
    if not 1 <= deadline_seconds <= 86400:
        raise ValueError('Deadline must remain within the approved 24 hours')
    metadata = {'namespace': 'prithvi-training-default', 'labels': {'app': 'lclu-stockholm', 'task': '0124'}}
    pvc = {'apiVersion': 'v1', 'kind': 'PersistentVolumeClaim',
           'metadata': {**metadata, 'name': 'lclu-stockholm-0124'},
           'spec': {'storageClassName': 'rook-ceph-fs', 'accessModes': ['ReadWriteMany'],
                    'resources': {'requests': {'storage': '20Gi'}}}}
    job = {'apiVersion': 'batch/v1', 'kind': 'Job',
           'metadata': {**metadata, 'name': job_name},
           'spec': {'activeDeadlineSeconds': deadline_seconds, 'ttlSecondsAfterFinished': 86400,
                    'backoffLimit': 0, 'template': {
               'metadata': {'labels': metadata['labels']}, 'spec': {
                   'restartPolicy': 'Never', 'automountServiceAccountToken': False,
                   'enableServiceLinks': False, 'terminationGracePeriodSeconds': 360,
                   'containers': [{'name': 'demo', 'image': image,
                       'command': ['python', '/opt/imintengine/docker/lclu/run.py'],
                       'env': [{'name': 'LCLU_YEAR', 'value': str(year)},
                               {'name': 'LCLU_IMAGE', 'value': image}],
                       'ports': [{'name': 'http', 'containerPort': 8000}],
                       'resources': {'requests': {'cpu': '4', 'memory': '8Gi'},
                                     'limits': {'cpu': '4', 'memory': '8Gi'}},
                       'securityContext': {'allowPrivilegeEscalation': False,
                                           'capabilities': {'drop': ['ALL']}},
                       'readinessProbe': {'httpGet': {'path': '/v1/lclu', 'port': 8000},
                                          'periodSeconds': 10, 'timeoutSeconds': 3},
                       'volumeMounts': [{'name': 'data', 'mountPath': '/data', 'readOnly': True},
                                        {'name': 'work', 'mountPath': '/work'}]}],
                   'volumes': [{'name': 'data', 'persistentVolumeClaim': {
                                   'claimName': 'training-data-cephfs', 'readOnly': True}},
                               {'name': 'work', 'persistentVolumeClaim': {'claimName': 'lclu-stockholm-0124'}}]}}}}
    items = [job] if reuse_volume else [pvc, job]
    if recipe is not None:
        expected_base = 'python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f'
        if image != expected_base:
            raise ValueError('Recipe mode requires the Dockerfile\'s exact Python base')
        archive = recipe.read_bytes()
        if len(archive) > 900_000:
            raise ValueError('Source archive exceeds the bounded ConfigMap budget')
        digest = hashlib.sha256(archive).hexdigest()
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            bootstrap = source.extractfile('docker/lclu/bootstrap.py').read()
            manifest = json.load(source.extractfile('SOURCE_MANIFEST.json'))
        if hashlib.sha256(bootstrap).hexdigest() != manifest['files']['docker/lclu/bootstrap.py']:
            raise ValueError('Archived bootstrap does not match its source manifest')
        name = 'lclu-stockholm-source-'+digest[:12]
        configmap = {'apiVersion': 'v1', 'kind': 'ConfigMap',
            'metadata': {**metadata, 'name': name}, 'immutable': True,
            'data': {'bootstrap.py': bootstrap.decode('utf-8')},
            'binaryData': {'source.tar.gz': base64.b64encode(archive).decode('ascii')}}
        spec = job['spec']['template']['spec']
        container = spec['containers'][0]
        container['command'] = ['python', '-u', '/recipe/bootstrap.py']
        container['env'] += [{'name': k, 'value': v} for k, v in {
            'LCLU_BUNDLE_SHA256': digest, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
            'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'GDAL_CACHEMAX': '256',
            'VPP_SOURCE': 'wekeo', 'IMINT_API_MAX_PIXELS': '262144', 'IMINT_API_MAX_JOBS': '4'}.items()]
        container['volumeMounts'].append({'name': 'recipe', 'mountPath': '/recipe', 'readOnly': True})
        spec['volumes'].append({'name': 'recipe', 'configMap': {'name': name}})
        items.insert(0, configmap)
    return {'apiVersion': 'v1', 'kind': 'List', 'items': items}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, help='Source archive for a hash-locked runtime on the demo volume')
    parser.add_argument('--job-name', default='lclu-stockholm-0124')
    parser.add_argument('--reuse-volume', action='store_true')
    parser.add_argument('--deadline-seconds', type=int, default=86400)
    args = parser.parse_args()
    args.output.write_text(json.dumps(resources(args.image, args.year, args.recipe,
        job_name=args.job_name, reuse_volume=args.reuse_volume,
        deadline_seconds=args.deadline_seconds), indent=2)+'\n')


if __name__ == '__main__':
    main()
