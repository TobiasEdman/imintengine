"""ImintEngine HTTP API.

Per agentic_workflow rollout plan W4.1. The HTTP boundary that
decouples ImintEngine from any specific executor (CLI, ColonyOS, K8s
job, Airflow) and lets external customers integrate without depending
on Python.

v1 lives in `imint.api.v1`. Future major versions live in sibling
packages (`imint.api.v2`, etc.) so consumers can pin to a stable API
version independent of ImintEngine's internal changes.
"""
