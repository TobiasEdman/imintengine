#!/usr/bin/env python3
"""Launch the local annual-map prototype with real inputs or an explicit demo.

    python scripts/serve_lclu.py --manifest /path/to/serving.json
    python scripts/serve_lclu.py --demo

The demo uses synthetic inputs and random weights, labelled in the page and
downloaded provenance. It never downloads or substitutes a trained checkpoint.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--manifest', type=Path, help='Identity-pinned real R4 serving manifest')
    mode.add_argument('--demo', action='store_true', help='Explicit synthetic interaction demo')
    parser.add_argument('--output', type=Path, help='Persistent local data/cache directory')
    parser.add_argument('--port', type=int, default=8784)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if args.manifest is not None and not args.manifest.is_file():
        parser.error('--manifest must identify an existing serving manifest')
    output = (args.output or ROOT / 'outputs' / ('lclu-demo' if args.demo else 'lclu-api')).resolve()
    print(f'Local annual-map prototype: http://127.0.0.1:{args.port}/lclu', flush=True)
    print('Stop with Ctrl+C. Nothing is publicly deployed.', flush=True)
    if args.demo:
        print('SYNTHETIC DEMO: random weights; no real R4 classifications.', flush=True)
        from tests.lclu.browser_fixture import main as run_demo
        run_demo(['--output', str(output), '--port', str(args.port)])
        return
    os.environ['IMINT_LCLU_MANIFEST'] = str(args.manifest.resolve())
    os.environ['IMINT_API_OUTPUT_DIR'] = str(output)
    os.environ.setdefault('IMINT_API_MAX_PIXELS', str(512 * 512))
    os.environ.setdefault('IMINT_API_MAX_JOBS', '4')
    # Existing API configuration is repository-relative; invocation from another
    # working directory must resolve the same checked-in files.
    os.chdir(ROOT)
    import uvicorn
    uvicorn.run('imint.api.v1:app', host='127.0.0.1', port=args.port,
                workers=1, lifespan='on')


if __name__ == '__main__':
    main()
