#!/usr/bin/env python3
"""Prepare, but never submit, the bounded E00 CPU audit Job and immutable inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_ladder_checkpoints import IMAGE, load_inventory

NAMESPACE = "prithvi-training-default"


def build_job(inventory: Path) -> dict:
    _, raw = load_inventory(inventory)
    script = Path(__file__).with_name("audit_ladder_checkpoints.py").read_text()
    suffix = hashlib.sha256(script.encode() + raw).hexdigest()[:12]
    name = f"fm-e00-checkpoint-audit-{suffix}"
    labels = {"app": "fm-experiments", "purpose": "checkpoint-integrity"}
    config = {
        "apiVersion": "v1", "kind": "ConfigMap",
        "metadata": {"name": name, "namespace": NAMESPACE, "labels": labels},
        "immutable": True,
        "data": {"audit_ladder_checkpoints.py": script, "inventory.jsonl": raw.decode()},
    }
    container = {
        "name": "audit", "image": IMAGE, "imagePullPolicy": "IfNotPresent",
        "command": ["/opt/venvs/model/bin/python", "-c"],
        "args": [
            "import os,runpy,sys; "
            "os.makedirs('/work/tmp',mode=0o700,exist_ok=True); "
            "sys.argv=['/audit/audit_ladder_checkpoints.py','--inventory','/audit/inventory.jsonl',"
            "'--output','/work/results.jsonl']; runpy.run_path(sys.argv[0],run_name='__main__')"
        ],
        "env": [
            {"name": "PYTHONPATH", "value": "/opt/imintengine"},
            {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
            {"name": "TMPDIR", "value": "/work/tmp"},
            {"name": "HOME", "value": "/work"},
            {"name": "USER", "value": "checkpoint-audit"},
            {"name": "LOGNAME", "value": "checkpoint-audit"},
            {"name": "TORCHINDUCTOR_CACHE_DIR", "value": "/work/torch-inductor"},
            {"name": "OMP_NUM_THREADS", "value": "4"},
            {"name": "AUDIT_IMAGE", "value": IMAGE},
            {"name": "POD_UID", "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}},
        ],
        "securityContext": {
            "allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True,
            "capabilities": {"drop": ["ALL"]},
        },
        "resources": {
            "requests": {"cpu": "4", "memory": "16Gi", "ephemeral-storage": "8Gi"},
            "limits": {"cpu": "4", "memory": "16Gi", "ephemeral-storage": "8Gi"},
        },
        "volumeMounts": [
            {"name": "inputs", "mountPath": "/audit/audit_ladder_checkpoints.py",
             "subPath": "audit_ladder_checkpoints.py", "readOnly": True},
            {"name": "inputs", "mountPath": "/audit/inventory.jsonl",
             "subPath": "inventory.jsonl", "readOnly": True},
            {"name": "checkpoints", "mountPath": "/cephfs/checkpoints/ladder",
             "subPath": "checkpoints/ladder", "readOnly": True},
            {"name": "work", "mountPath": "/work"},
        ],
    }
    job = {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "namespace": NAMESPACE, "labels": labels},
        "spec": {
            "backoffLimit": 0, "activeDeadlineSeconds": 1800, "ttlSecondsAfterFinished": 604800,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never", "automountServiceAccountToken": False,
                    "imagePullSecrets": [{"name": "ghcr-push"}],
                    "securityContext": {
                        "runAsNonRoot": True, "runAsUser": 2007, "runAsGroup": 2000,
                        "fsGroup": 2000, "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "containers": [container],
                    "volumes": [
                        {"name": "inputs", "configMap": {"name": name}},
                        {"name": "checkpoints", "persistentVolumeClaim": {
                            "claimName": "training-data-cephfs", "readOnly": True}},
                        {"name": "work", "emptyDir": {"sizeLimit": "8Gi"}},
                    ],
                },
            },
        },
    }
    return {"apiVersion": "v1", "kind": "List", "items": [config, job]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_job(args.inventory)
    with args.output.open("x") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
