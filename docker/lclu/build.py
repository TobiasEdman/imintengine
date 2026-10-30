#!/usr/bin/env python3
"""Build the exact local prototype source, recording every included file hash.

No model, observation, credential or output files enter this context. Building
does not push a registry image or change ICE. Use a published digest in K8s.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle-only', action='store_true', help='Create a source bundle without Docker')
    args = parser.parse_args()
    if ':' not in args.tag or args.tag.endswith(':latest'):
        parser.error('Use an explicit version tag, never latest')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT/'imint').rglob('*.py')) + sorted((ROOT/'imint/config').glob('*.yaml'))
    files += [ROOT/'scripts'/name for name in (
        'inference_comparison.py', 'cache_predict.py', 'enrich_tiles_tessera.py',
        'prepare_lclu.py', 'stage_lclu_demo.py')]
    files += [ROOT/path for path in (
        'config/lclu/stockholm.json', 'config/lclu/kiruna_tatort.json', 'docs/lclu.html', 'docs/js/lclu.js',
        'docs/js/vendor/leaflet.min.js', 'docs/css/lclu.css', 'docs/css/styles.css',
        'docs/css/leaflet.css', 'docker/lclu/Dockerfile', 'docker/lclu/requirements.lock',
        'docker/lclu/run.py', 'docker/lclu/bootstrap.py', 'docker/lclu/smoke.py',
        'docker/lclu/system_runtime.py')]
    identities = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    source_hash = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    manifest = {'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'uncommitted_source': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
                'source_sha256': source_hash, 'files': identities, 'image_tag': args.tag,
                'base_digest': 'sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f'}
    with tempfile.TemporaryDirectory(prefix='lclu-build-') as folder:
        context = Path(folder)
        for p in files:
            target = context/p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        (context/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
        shutil.copyfile(context/'SOURCE_MANIFEST.json', output/'SOURCE_MANIFEST.json')
        if args.bundle_only:
            with tarfile.open(output/'source.tar.gz', 'w:gz') as archive:
                for p in sorted(context.rglob('*')):
                    if p.is_file():
                        archive.add(p, arcname=str(p.relative_to(context)), recursive=False)
            print(json.dumps({'source_sha256': source_hash, 'bundle_bytes': (output/'source.tar.gz').stat().st_size}))
            return
        subprocess.run(['docker', 'build', '--platform', 'linux/amd64', '--tag', args.tag,
                        '--iidfile', str(output/'image-id.txt'), '--file', str(context/'docker/lclu/Dockerfile'),
                        str(context)], check=True)
    print(json.dumps({'source_sha256': source_hash, 'image_id': (output/'image-id.txt').read_text().strip()}))


if __name__ == '__main__':
    main()
