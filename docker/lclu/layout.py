"""Filesystem boundaries shared by bootstrap and the demo launcher."""
import os
from pathlib import Path
import re


def deployment_root(work: Path) -> Path:
    release = os.environ.get('LCLU_CAMPAIGN_RELEASE')
    if release is not None:
        if not os.environ.get('LCLU_CAMPAIGN') or not re.fullmatch(r'lclu-stockholm-0124-r[0-9]+', release):
            raise ValueError('An isolated release requires a versioned campaign Job name')
        return work/'releases'/release
    return work/'campaign' if os.environ.get('LCLU_CAMPAIGN') else work


def system_runtime_root(work: Path) -> Path:
    # Installing a new release must not replace libraries loaded by the old one.
    base = deployment_root(work) if 'LCLU_CAMPAIGN_RELEASE' in os.environ else work
    return base/'system/expat-2.5.0-1-deb12u5'
