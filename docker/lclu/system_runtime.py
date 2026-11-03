"""Install the hash-pinned Debian XML runtime into an isolated directory."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

PACKAGE = {
    'name': 'libexpat1', 'version': '2.5.0-1+deb12u5', 'architecture': 'amd64',
    'url': 'https://security.debian.org/debian-security/pool/updates/main/e/expat/libexpat1_2.5.0-1+deb12u5_amd64.deb',
    'sha256': 'cae5ded666d8573dd33060bc66fbac21eea8b101a6b56e2af8bd590fcfa65d79',
    'size': 107940,
    'metadata': 'https://packages.debian.org/bookworm/amd64/libexpat1/download',
}


def install(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    package = root / (PACKAGE['sha256'] + '.deb')
    if not package.exists():
        with urllib.request.urlopen(PACKAGE['url'], timeout=60) as response:
            data = response.read(PACKAGE['size'] + 1)
    else:
        data = package.read_bytes()
    if len(data) != PACKAGE['size'] or hashlib.sha256(data).hexdigest() != PACKAGE['sha256']:
        raise ValueError('System package identity mismatch')
    package.write_bytes(data)
    # Extract only this verified package; never alter the reference-data volume.
    subprocess.run(['dpkg-deb', '--extract', str(package), str(root)], check=True)
    (root / 'MANIFEST.json').write_text(json.dumps(PACKAGE, indent=2) + '\n')
    return ':'.join(str(root / p) for p in ('lib/x86_64-linux-gnu', 'usr/lib/x86_64-linux-gnu'))


if __name__ == '__main__':
    print(install(Path(sys.argv[1])))
