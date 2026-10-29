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

root = Path('/work/stockholm')
source = Path(__file__).resolve().parents[2]
year = int(os.environ['LCLU_YEAR'])
stage(source/'config/lclu/stockholm.json', year, Path('/data'), root)
write_json(root/'MANIFEST.json', {
    'image': os.environ['LCLU_IMAGE'], 'image_digest': os.environ['LCLU_IMAGE'].split('@', 1)[1],
    'execution_mode': os.environ.get('LCLU_EXECUTION_MODE', 'built-image'),
    'source': json.loads((source/'SOURCE_MANIFEST.json').read_text()),
    'run_args': {'year': year, 'municipality': '0180'},
    'input_data_hash': public_identity(file_identity(root/f'sources-{year}.json')),
    'produced_at': datetime.now(timezone.utc).isoformat(),
})
os.execv(sys.executable, [sys.executable, '-m', 'uvicorn', 'imint.api.v1:app',
    '--host', '0.0.0.0', '--port', '8000', '--workers', '1', '--lifespan', 'on'])
