#!/usr/bin/env python3
"""Create/apply the content-addressed Kaniko build for the smoke runtime."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CONTEXT_FILES = {
    "Dockerfile": REPO_ROOT / "docker/era5-smoke/Dockerfile",
    "requirements.lock": REPO_ROOT / "docker/era5-smoke/requirements.lock",
    "runtime_smoke.py": REPO_ROOT / "docker/era5-smoke/runtime_smoke.py",
}
KANIKO_IMAGE = (
    "gcr.io/kaniko-project/executor@"
    "sha256:c3109d5926a997b100c4343944e06c6b30a6804b2f9abe0994d3de6ef92b028e"
)


def context_sha256(contents: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(contents):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contents[name])
        digest.update(b"\0")
    return digest.hexdigest()


def build_manifests(namespace: str) -> tuple[dict, dict, dict]:
    contents = {name: path.read_bytes() for name, path in CONTEXT_FILES.items()}
    digest = context_sha256(contents)
    suffix = digest[:12]
    configmap_name = f"era5-smoke-runtime-build-{suffix}"
    image_tag = f"ghcr.io/tobiasedman/imint-era5-smoke:20260821-{suffix}"
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": configmap_name,
            "namespace": namespace,
            "labels": {"purpose": "era5-prithvi600m-smoke-runtime"},
            "annotations": {"imint.se/build-context-sha256": digest},
        },
        "immutable": True,
        "binaryData": {
            name: base64.b64encode(payload).decode("ascii")
            for name, payload in contents.items()
        },
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"build-era5-smoke-runtime-{suffix}",
            "namespace": namespace,
            "labels": {"purpose": "era5-prithvi600m-smoke-runtime"},
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 3600,
            "ttlSecondsAfterFinished": 172800,
            "template": {
                "metadata": {
                    "labels": {"purpose": "era5-prithvi600m-smoke-runtime"},
                },
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "kaniko",
                        "image": KANIKO_IMAGE,
                        "args": [
                            "--dockerfile=/workspace/Dockerfile",
                            "--context=dir:///workspace",
                            f"--destination={image_tag}",
                            "--cleanup",
                            "--snapshot-mode=redo",
                            "--digest-file=/dev/termination-log",
                        ],
                        "volumeMounts": [
                            {"name": "context", "mountPath": "/workspace", "readOnly": True},
                            {"name": "ghcr-config", "mountPath": "/kaniko/.docker", "readOnly": True},
                        ],
                        "resources": {
                            # Installing the CUDA-enabled PyTorch wheel expands
                            # several GiB of shared libraries.  Kaniko's first
                            # full-filesystem snapshot exceeded 8 GiB RSS in a
                            # real cluster build, so keep enough headroom for
                            # both the unpacked layer and snapshot metadata.
                            "requests": {"cpu": "4", "memory": "32Gi"},
                            "limits": {"cpu": "4", "memory": "32Gi"},
                        },
                    }],
                    "volumes": [
                        {"name": "context", "configMap": {"name": configmap_name}},
                        {"name": "ghcr-config", "secret": {
                            "secretName": "ghcr-push",
                            "items": [{"key": ".dockerconfigjson", "path": "config.json"}],
                        }},
                    ],
                },
            },
        },
    }
    identity = {
        "context_sha256": digest,
        "configmap": configmap_name,
        "job": job["metadata"]["name"],
        "image_tag": image_tag,
        "kaniko_image": KANIKO_IMAGE,
    }
    return configmap, job, identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="prithvi-training-default")
    parser.add_argument("--context", default="icekube")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args()
    configmap, job, identity = build_manifests(args.namespace)
    if args.apply or args.dry_run:
        for manifest in (configmap, job):
            command = ["kubectl", "--context", args.context, "apply"]
            if args.dry_run:
                command.append("--dry-run=server")
            command.extend(["-f", "-"])
            subprocess.run(
                command,
                input=json.dumps(manifest), text=True, check=True,
            )
    if args.manifest:
        print(json.dumps({"configmap": configmap, "job": job}, indent=2))
    else:
        print(json.dumps(identity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
