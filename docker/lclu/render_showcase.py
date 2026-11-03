#!/usr/bin/env python3
"""Prepare public showcase routing and configuration; never apply resources.

Input is a fresh Job proposal from render_job.py --reuse-volume --isolate-campaign. The generated
Job preserves its source, resource and expiry settings, adding exact CORS
origins. Publish lclu-config.js with the static showcase after live acceptance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


def release(document: dict, api_host: str, origins: list[str]) -> dict:
    hostname = re.compile(r'(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+'
                          r'[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?')
    if not hostname.fullmatch(api_host):
        raise ValueError('API host must be an explicit DNS hostname, without scheme or path')
    if not origins or len(set(origins)) != len(origins):
        raise ValueError('Provide unique, explicit showcase origins')
    for origin in origins:
        value = urlsplit(origin)
        try:
            port = value.port
        except ValueError:
            raise ValueError('Showcase origins must use valid HTTPS ports') from None
        canonical = f'https://{value.hostname}' + (f':{port}' if port not in (None, 443) else '')
        if (value.scheme != 'https' or not value.hostname or value.username or value.password
                or not hostname.fullmatch(value.hostname) or port == 0 or origin != canonical
                or value.path or value.query or value.fragment):
            raise ValueError('Showcase origins must be canonical HTTPS origins without paths')
    job_document = deepcopy(document)
    if (job_document.get('kind') != 'List' or
            any(item['kind'] not in ('Job', 'ConfigMap') for item in job_document['items'])):
        raise ValueError('Use a Job proposal with --reuse-volume; no PVC or other resources')
    jobs = [item for item in job_document['items'] if item['kind'] == 'Job']
    if len(jobs) != 1:
        raise ValueError('Exactly one demo Job is required')
    job = jobs[0]
    name = job['metadata']['name']
    namespace = job['metadata']['namespace']
    if not re.fullmatch(r'lclu-stockholm-0124-r[0-9]+', name) or namespace != 'prithvi-training-default':
        raise ValueError('Expected a versioned LCLU demo Job in its existing namespace')
    containers = job['spec']['template']['spec']['containers']
    if len(containers) != 1 or containers[0]['name'] != 'demo':
        raise ValueError('Expected the single demo container')
    container = containers[0]
    environment = {entry['name']: entry.get('value') for entry in container['env']}
    if environment.get('LCLU_CAMPAIGN_RELEASE') != name or not environment.get('LCLU_CAMPAIGN'):
        raise ValueError('Use an isolated campaign proposal to preserve the running release')
    container['env'] = [e for e in container['env'] if e['name'] != 'IMINT_LCLU_ORIGINS']
    container['env'].append({'name': 'IMINT_LCLU_ORIGINS', 'value': ','.join(origins)})
    metadata = {'name': 'lclu-showcase', 'namespace': namespace}
    service = {'apiVersion': 'v1', 'kind': 'Service', 'metadata': metadata,
               'spec': {'type': 'ClusterIP',
                        'selector': {'batch.kubernetes.io/job-name': name},
                        'ports': [{'name': 'http', 'port': 80, 'targetPort': 8000}]}}
    paths = [{'path': path, 'pathType': kind,
              'backend': {'service': {'name': metadata['name'], 'port': {'name': 'http'}}}}
             for path, kind in (('/v1/lclu', 'Prefix'), ('/lclu', 'Exact'), ('/lclu-assets', 'Prefix'))]
    ingress = {'apiVersion': 'networking.k8s.io/v1', 'kind': 'Ingress',
               'metadata': {**metadata, 'annotations': {'cert-manager.io/issuer': 'letsencrypt-prod'}},
               'spec': {'ingressClassName': 'nginx',
                        'tls': [{'hosts': [api_host], 'secretName': 'lclu-showcase-tls'}],
                        'rules': [{'host': api_host, 'http': {'paths': paths}}]}}
    return {'job': job_document, 'routing': {'apiVersion': 'v1', 'kind': 'List', 'items': [service, ingress]},
            'config': 'window.IMINT_LCLU_CONFIG = ' + json.dumps({'apiBase': f'https://{api_host}'}) + ';\n'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-manifest', required=True, type=Path)
    parser.add_argument('--api-host', required=True)
    parser.add_argument('--showcase-origin', action='append', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    proposal = release(json.loads(args.job_manifest.read_text()), args.api_host, args.showcase_origin)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('job-with-origins.json', proposal['job']), ('routing.json', proposal['routing'])]:
        (args.output/name).write_text(json.dumps(value, indent=2)+'\n')
    (args.output/'lclu-config.js').write_text(proposal['config'])
    files = ('job-with-origins.json', 'routing.json', 'lclu-config.js')
    (args.output/'release.json').write_text(json.dumps({
        'api_host': args.api_host, 'showcase_origins': args.showcase_origin, 'applied': False,
        'campaign_work': '/work/releases/' + next(
            item['metadata']['name'] for item in proposal['job']['items'] if item['kind'] == 'Job'),
        'input_job_sha256': hashlib.sha256(args.job_manifest.read_bytes()).hexdigest(),
        'files': {name: hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in files},
        'gates': ['Confirm existing compute approval permits the proposed Job before starting it',
                  'Complete campaign acceptance and browser checks',
                  'Verify DNS/TLS and showcase-origin CORS before publishing the static config'],
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
