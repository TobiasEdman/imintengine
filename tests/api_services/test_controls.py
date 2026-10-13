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


@pytest.mark.parametrize('name,value', [('analyzers', ['spectral']), ('disable_analyzers', ['prithvi']), ('config_overrides', {'spectral': {'enabled': False}}), ('analyzers', []), ('config_overrides', None)])
def test_A08_controls_rejected_before_enqueue(api, name, value):
    with TestClient(api.server.app) as c:
        r = c.post('/v1/analyze', json={**REQ, name: value})
        assert r.status_code == 422
        assert any(e['loc'][-1] == name for e in r.json()['detail'])
        assert not api.server._jobs and not api.calls


def test_A08_controls_absent_from_openapi(api):
    props = api.server.app.openapi()['components']['schemas']['AnalyzeRequest']['properties']
    assert set(props) == {'date', 'bbox'}
