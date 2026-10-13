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


@pytest.mark.parametrize('value', ['http://127.0.0.1:9999/internal', 'https://example.invalid/notify', None])
def test_A03_callback_rejected_before_enqueue(api, value):
    with TestClient(api.server.app) as c:
        r = c.post('/v1/analyze', json={**REQ, 'callback_url': value})
        assert r.status_code == 422
        assert any(e['loc'][-1] == 'callback_url' for e in r.json()['detail'])
        assert not api.server._jobs and not api.calls


def test_A03_callback_not_advertised_or_executable(api):
    props = api.server.app.openapi()['components']['schemas']['AnalyzeRequest']['properties']
    assert 'callback_url' not in props
    assert not hasattr(api.server, '_notify_callback')
