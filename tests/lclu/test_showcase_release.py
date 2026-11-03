"""Public routing must isolate one campaign and retain its approved allocation."""
from copy import deepcopy

import pytest

from docker.lclu.render_job import resources
from docker.lclu.render_showcase import release

BASE = 'python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f'


def job():
    return resources(BASE, 2024, job_name='lclu-stockholm-0124-r9', reuse_volume=True,
                     storage_gib=694, campaign_years=tuple(range(2017, 2026)),
                     max_original_bytes=305680252544, isolate_campaign=True)


def test_release_preserves_job_and_only_adds_exact_cors_origins():
    before = job()
    snapshot = deepcopy(before)
    result = release(before, 'lclu.icedc.se', ['https://digitalearth.se', 'https://tobiasedman.github.io'])
    assert before == snapshot
    changed = result['job']['items'][0]['spec']['template']['spec']['containers'][0]['env'].pop()
    assert changed == {'name': 'IMINT_LCLU_ORIGINS',
                       'value': 'https://digitalearth.se,https://tobiasedman.github.io'}
    assert result['job'] == before  # CPU/RAM, volumes, source pin, years and expiry are byte-identical.
    service, ingress = result['routing']['items']
    assert service['spec']['selector'] == {'batch.kubernetes.io/job-name': 'lclu-stockholm-0124-r9'}
    assert service['spec']['type'] == 'ClusterIP'
    paths = ingress['spec']['rules'][0]['http']['paths']
    assert [(p['path'], p['pathType']) for p in paths] == [
        ('/v1/lclu', 'Prefix'), ('/lclu', 'Exact'), ('/lclu-assets', 'Prefix')]
    assert ingress['spec']['tls'][0]['hosts'] == ['lclu.icedc.se']
    assert ingress['metadata']['annotations'] == {'cert-manager.io/issuer': 'letsencrypt-prod'}
    assert result['config'] == 'window.IMINT_LCLU_CONFIG = {"apiBase": "https://lclu.icedc.se"};\n'


@pytest.mark.parametrize('host', ['https://lclu.icedc.se', 'lclu.icedc.se/path', '*.icedc.se',
                                 'lclu.icedc.se:8000', '-bad.icedc.se'])
def test_rejects_ambiguous_api_hosts(host):
    with pytest.raises(ValueError, match='hostname'):
        release(job(), host, ['https://digitalearth.se'])


@pytest.mark.parametrize('origins', [[], ['*'], ['http://digitalearth.se'],
                                    ['https://digitalearth.se/case/'],
                                    ['https://digitalearth.se']*2,
                                    ['https://a.invalid,https://b.invalid'],
                                    ['https://digitalearth.se:bogus'], ['https://a b.invalid'],
                                    ['https://DigitalEarth.se'], ['https://digitalearth.se:443'],
                                    ['https://digitalearth.se:0'], ['https://digitalearth.se:99999']])
def test_rejects_ambiguous_cors_origins(origins):
    with pytest.raises(ValueError, match='origins'):
        release(job(), 'lclu.icedc.se', origins)


def test_rejects_storage_changes_and_multiple_jobs():
    proposal = job()
    proposal['items'].append({'kind': 'PersistentVolumeClaim'})
    with pytest.raises(ValueError, match='no PVC'):
        release(proposal, 'lclu.icedc.se', ['https://digitalearth.se'])
    proposal = job()
    proposal['items'] *= 2
    with pytest.raises(ValueError, match='Exactly one'):
        release(proposal, 'lclu.icedc.se', ['https://digitalearth.se'])


def test_rejects_a_release_that_would_overwrite_the_running_campaign():
    proposal = job()
    env = proposal['items'][0]['spec']['template']['spec']['containers'][0]['env']
    env[:] = [entry for entry in env if entry['name'] != 'LCLU_CAMPAIGN_RELEASE']
    with pytest.raises(ValueError, match='isolated campaign'):
        release(proposal, 'lclu.icedc.se', ['https://digitalearth.se'])
