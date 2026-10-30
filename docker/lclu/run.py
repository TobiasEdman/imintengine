#!/usr/bin/env python3
"""Stage the public demo once, persist its process manifest, then serve it."""
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

from scripts.stage_lclu_demo import stage
from imint.inference.lclu import write_json, public_identity
from scripts.cache_predict import file_identity


def deployment_root(work: Path) -> Path:
    """A campaign stages beside the running two-area demo on the same volume.

    Its manifest, process MANIFEST and API output directory (which holds the
    single-worker flock) must not be the running demo's, or it would overwrite
    that demo's provenance and fail to start while the demo holds the lock.
    """
    return work/'campaign' if os.environ.get('LCLU_CAMPAIGN') else work


def main(source: Path, work: Path, data: Path) -> None:
    base = deployment_root(work)
    root = base/'demo'
    root.mkdir(parents=True, exist_ok=True)
    os.environ['IMINT_API_OUTPUT_DIR'] = str(base/'results')
    if os.environ.get('LCLU_CAMPAIGN'):
        from scripts.stage_lclu_campaign import stage_campaign
        years = [int(value) for value in os.environ['LCLU_YEARS'].split(',')]
        manifest = stage_campaign(source/'config/lclu/municipalities.json', years, data,
                                  work/'campaign', int(os.environ['LCLU_MAX_ORIGINAL_BYTES']),
                                  reuse_roots=(work/'stockholm', work/'kiruna_tatort'))
        write_json(root/'serving.json', manifest)
        run_args = {'years': years, 'campaign': manifest['campaign']}
    else:
        year = int(os.environ['LCLU_YEAR'])
        manifests = []
        for name in ('stockholm', 'kiruna_tatort'):
            region = work/name
            stage(source/f'config/lclu/{name}.json', year, data, region)
            manifests.append(json.loads((region/'serving.json').read_text()))
        if any(m['checkpoint'] != manifests[0]['checkpoint'] for m in manifests):
            raise ValueError('Demo regions must use the same pinned R4 checkpoint')
        write_json(root/'serving.json', {'version': 1, 'device': 'cpu',
            'checkpoint': manifests[0]['checkpoint'],
            'preparations': [m['preparation'] for m in manifests]})
        run_args = {'year': year, 'areas': ['Stockholm', 'Kiruna tätort']}
    os.environ['IMINT_LCLU_MANIFEST'] = str(root/'serving.json')
    write_json(root/'MANIFEST.json', {
        'image': os.environ['LCLU_IMAGE'], 'image_digest': os.environ['LCLU_IMAGE'].split('@', 1)[1],
        'execution_mode': os.environ.get('LCLU_EXECUTION_MODE', 'built-image'),
        'source': json.loads((source/'SOURCE_MANIFEST.json').read_text()),
        'run_args': run_args,
        'input_data_hash': public_identity(file_identity(root/'serving.json')),
        'produced_at': datetime.now(timezone.utc).isoformat(),
    })
    os.execv(sys.executable, [sys.executable, '-m', 'uvicorn', 'imint.api.v1:app',
        '--host', '0.0.0.0', '--port', '8000', '--workers', '1', '--lifespan', 'on'])


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[2], Path('/work'), Path('/data'))
