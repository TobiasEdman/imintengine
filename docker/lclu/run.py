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


def main(source: Path, work: Path, data: Path) -> None:
    root = work/'demo'
    root.mkdir(parents=True, exist_ok=True)
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
    os.environ['IMINT_LCLU_MANIFEST'] = str(root/'serving.json')
    write_json(root/'MANIFEST.json', {
        'image': os.environ['LCLU_IMAGE'], 'image_digest': os.environ['LCLU_IMAGE'].split('@', 1)[1],
        'execution_mode': os.environ.get('LCLU_EXECUTION_MODE', 'built-image'),
        'source': json.loads((source/'SOURCE_MANIFEST.json').read_text()),
        'run_args': {'year': year, 'areas': ['Stockholm', 'Kiruna tätort']},
        'input_data_hash': public_identity(file_identity(root/'serving.json')),
        'produced_at': datetime.now(timezone.utc).isoformat(),
    })
    os.execv(sys.executable, [sys.executable, '-m', 'uvicorn', 'imint.api.v1:app',
        '--host', '0.0.0.0', '--port', '8000', '--workers', '1', '--lifespan', 'on'])


if __name__ == '__main__':
    main(Path(__file__).resolve().parents[2], Path('/work'), Path('/data'))
