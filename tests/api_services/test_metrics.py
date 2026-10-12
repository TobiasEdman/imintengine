import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import pytest
import numpy as np
from fastapi.testclient import TestClient
from .helpers import ROOT, REQ, drain, load


def test_A05_bounded_metric_route_labels(api):
    with TestClient(api.server.app) as c:
        for i in range(3):
            assert c.get(f'/v1/jobs/unbounded-probe-{i}').status_code==404
            assert c.get(f'/missing/unbounded-probe-{i}').status_code==404
        payload=c.get('/v1/metrics').text
    assert 'unbounded-probe' not in payload
    assert 'route="/v1/jobs/{job_id}"' in payload and 'route="unmatched"' in payload
