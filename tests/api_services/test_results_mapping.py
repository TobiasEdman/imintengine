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


def test_A07_real_result_fields(api):
    rows=api.analysis_result
    outcomes=[rows('failed',False,error='synthetic'),rows('ok',True),rows('skip',False,skipped=True)]
    api.engine.run_job=lambda job:api.job.IMINTResult(job.job_id,job.date,False,outcomes)
    async def run():
        async with api.server.app.router.lifespan_context(api.server.app):
            r=await api.server.submit_analysis(api.server.AnalyzeRequest(**REQ));await drain(api.server)
            status=await api.server.get_job(r.job_id)
            assert [(s.name,s.status) for s in status.analyzers]==[('failed','failed'),('ok','success'),('skip','skipped')]
    asyncio.run(run())
