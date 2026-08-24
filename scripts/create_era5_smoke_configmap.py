#!/usr/bin/env python3
"""Create the content-addressed ConfigMap used by the ERA5 smoke jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

if __package__:
    from scripts.build_era5_smoke_runtime import CONTEXT_FILES, context_sha256
    from scripts.era5_smoke_provenance import CODE_BUNDLE_SCHEMA, code_bundle_sha256
else:
    from build_era5_smoke_runtime import CONTEXT_FILES, context_sha256
    from era5_smoke_provenance import CODE_BUNDLE_SCHEMA, code_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parent.parent
PATCH_FILES = {
    "config.py": REPO_ROOT / "imint/training/config.py",
    "era5_aux.py": REPO_ROOT / "imint/training/era5_aux.py",
    "tile_time.py": REPO_ROOT / "imint/training/tile_time.py",
    "trainer.py": REPO_ROOT / "imint/training/trainer.py",
    "unified_dataset.py": REPO_ROOT / "imint/training/unified_dataset.py",
    "upernet.py": REPO_ROOT / "imint/fm/upernet.py",
    "train_unified.py": REPO_ROOT / "scripts/train_unified.py",
    "build_cohort.py": REPO_ROOT / "scripts/build_era5_smoke_cohort.py",
    "fetch_era5.py": REPO_ROOT / "scripts/fetch_era5_aux.py",
    "analyze_smoke.py": REPO_ROOT / "scripts/analyze_era5_smoke.py",
    "era5_smoke_provenance.py": REPO_ROOT / "scripts/era5_smoke_provenance.py",
    "data_preflight.py": REPO_ROOT / "scripts/preflight_era5_smoke_data.py",
    "fetch_foundation.py": REPO_ROOT / "scripts/fetch_prithvi600m_checkpoint.py",
    "runtime_smoke.py": REPO_ROOT / "docker/era5-smoke/runtime_smoke.py",
    "run_arm.sh": REPO_ROOT / "scripts/run_era5_smoke_arm.sh",
}
JOB_MANIFESTS = {
    "control": REPO_ROOT / "k8s/era5-prithvi600m-smoke-control.yaml",
    "treatment": REPO_ROOT / "k8s/era5-prithvi600m-smoke-treatment.yaml",
}
RUNTIME_MANIFEST = REPO_ROOT / "docker/era5-smoke/MANIFEST.json"
BASE_GIT_SHA = "602c86dee8d4dbdc191ab2750c261cf86b56ac50"
RUNTIME_REPOSITORY = "ghcr.io/tobiasedman/imint-era5-smoke"
RUNTIME_IMAGE_RE = re.compile(
    rf"^{re.escape(RUNTIME_REPOSITORY)}@sha256:[0-9a-f]{{64}}$"
)
RUNTIME_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
RUN_COMMAND = ["/bin/bash", "/patches/run_arm.sh"]
IMAGE_PULL_SECRET = "ghcr-push"
REQUIRED_PVC_MOUNTS = {
    "cephfs": ("training-data-cephfs", "/cephfs", True),
    "checkpoints": ("training-checkpoints", "/checkpoints", False),
    "training-data": ("training-data", "/data", True),
}


def build_configmap(namespace: str) -> tuple[dict, dict]:
    contents = {name: path.read_bytes() for name, path in PATCH_FILES.items()}
    bundle_hash = code_bundle_sha256(contents)
    manifest = {
        "schema": CODE_BUNDLE_SCHEMA,
        "bundle_sha256": bundle_hash,
        "files": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(contents.items())
        },
    }
    name = f"era5-smoke-code-{bundle_hash[:12]}"
    data = {key: value.decode("utf-8") for key, value in contents.items()}
    data["bundle_manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {"purpose": "era5-prithvi600m-smoke"},
            "annotations": {"imint.se/code-sha256": bundle_hash},
        },
        "immutable": True,
        "data": data,
    }
    return configmap, manifest


def _environment(container: dict, source: Path | str) -> dict[str, str | None]:
    """Return a Job's literal environment while rejecting ambiguous duplicates."""
    values: dict[str, str | None] = {}
    for item in container.get("env", []):
        name = item.get("name")
        if not isinstance(name, str):
            raise ValueError(f"{source} contains an environment entry without a name")
        if name in values:
            raise ValueError(f"{source} declares duplicate environment variable {name}")
        values[name] = item.get("value")
    return values


def _verify_job_manifest(
    job: dict,
    *,
    arm: str,
    source: Path | str,
    configmap_name: str,
    bundle_sha256: str,
) -> None:
    """Validate the immutable launch contract for one ERA5 smoke-test arm."""
    suffix = bundle_sha256[:12]
    expected_run_id = f"era5-p600m-20260821-{suffix}"
    expected_job_name = f"era5-p600m-{arm}-{suffix}"
    if job.get("metadata", {}).get("name") != expected_job_name:
        raise ValueError(f"{source} does not use job name {expected_job_name}")

    pod_template = job.get("spec", {}).get("template", {})
    pod_spec = pod_template.get("spec", {})
    containers = pod_spec.get("containers", [])
    if len(containers) != 1:
        raise ValueError(f"{source} must declare exactly one container")
    container = containers[0]
    if container.get("command") != RUN_COMMAND:
        raise ValueError(f"{source} must use command {RUN_COMMAND!r}")

    image = container.get("image")
    if not isinstance(image, str) or RUNTIME_IMAGE_RE.fullmatch(image) is None:
        raise ValueError(
            f"{source} image must be a digest-pinned "
            "ghcr.io/tobiasedman/imint-era5-smoke image"
        )

    environment = _environment(container, source)
    if environment.get("CONTAINER_IMAGE") != image:
        raise ValueError(f"{source} CONTAINER_IMAGE must exactly equal image {image}")
    if environment.get("BASE_GIT_SHA") != BASE_GIT_SHA:
        raise ValueError(f"{source} does not use immutable BASE_GIT_SHA={BASE_GIT_SHA}")
    if environment.get("ARM") != arm:
        raise ValueError(f"{source} does not declare ARM={arm}")
    if environment.get("RUN_ID") != expected_run_id:
        raise ValueError(f"{source} does not use RUN_ID={expected_run_id}")

    pull_secrets = pod_spec.get("imagePullSecrets", [])
    if {item.get("name") for item in pull_secrets} != {IMAGE_PULL_SECRET}:
        raise ValueError(f"{source} must use imagePullSecret {IMAGE_PULL_SECRET}")

    labels = pod_template.get("metadata", {}).get("labels", {})
    if labels.get("arm") != arm:
        raise ValueError(f"{source} pod template does not use arm label {arm}")

    volumes_by_name = {
        volume.get("name"): volume for volume in pod_spec.get("volumes", [])
    }
    mounts_by_name = {
        mount.get("name"): mount for mount in container.get("volumeMounts", [])
    }
    for name, (claim, mount_path, read_only) in REQUIRED_PVC_MOUNTS.items():
        actual_claim = volumes_by_name.get(name, {}).get(
            "persistentVolumeClaim", {}
        ).get("claimName")
        mount = mounts_by_name.get(name, {})
        if actual_claim != claim:
            raise ValueError(f"{source} volume {name} must use PVC {claim}")
        if (
            mount.get("mountPath") != mount_path
            or bool(mount.get("readOnly", False)) is not read_only
        ):
            mode = "read-only" if read_only else "writable"
            raise ValueError(
                f"{source} volume {name} must mount {mount_path} {mode}"
            )

    patch_volumes = [
        volume for volume in pod_spec.get("volumes", [])
        if volume.get("name") == "patches"
    ]
    if len(patch_volumes) != 1:
        raise ValueError(f"{source} must declare exactly one patches volume")
    actual_configmap = patch_volumes[0].get("configMap", {}).get("name")
    if actual_configmap != configmap_name:
        raise ValueError(f"{source} does not use ConfigMap {configmap_name}")


def _current_runtime_context_sha256() -> str:
    contents = {name: path.read_bytes() for name, path in CONTEXT_FILES.items()}
    return context_sha256(contents)


def _runtime_reference_from_manifest(manifest: dict) -> str:
    """Return the sealed digest reference after validating build provenance."""
    if manifest.get("schema") != "imint-pipeline-image-v1":
        raise ValueError("ERA5 runtime MANIFEST.json has an unsupported schema")
    context_digest = _current_runtime_context_sha256()
    if manifest.get("build_context_sha256") != context_digest:
        raise ValueError("ERA5 runtime MANIFEST.json build context is stale")
    expected_tag = (
        f"{RUNTIME_REPOSITORY}:20260821-{context_digest[:12]}"
    )
    if manifest.get("image") != expected_tag:
        raise ValueError(
            f"ERA5 runtime MANIFEST.json image must be {expected_tag}"
        )
    digest = manifest.get("image_digest")
    if not isinstance(digest, str) or RUNTIME_DIGEST_RE.fullmatch(digest) is None:
        raise ValueError(
            "ERA5 runtime MANIFEST.json image_digest must be sha256:<64 hex>"
        )
    return f"{RUNTIME_REPOSITORY}@{digest}"


def _verify_runtime_image_parity(images: dict[str, str], manifest: dict) -> None:
    expected = _runtime_reference_from_manifest(manifest)
    if set(images) != {"control", "treatment"}:
        raise ValueError("Both control and treatment runtime images are required")
    mismatches = {
        arm: image for arm, image in images.items() if image != expected
    }
    if mismatches:
        raise ValueError(
            "ERA5 smoke Jobs must use the same MANIFEST-pinned runtime image "
            f"{expected}; got {mismatches}"
        )


def verify_job_manifests(configmap: dict, bundle: dict) -> None:
    """Fail if checked-in Jobs violate the immutable launch contract."""
    images: dict[str, str] = {}
    for arm, path in JOB_MANIFESTS.items():
        job = yaml.safe_load(path.read_text())
        _verify_job_manifest(
            job,
            arm=arm,
            source=path,
            configmap_name=configmap["metadata"]["name"],
            bundle_sha256=bundle["bundle_sha256"],
        )
        images[arm] = job["spec"]["template"]["spec"]["containers"][0]["image"]
    try:
        runtime_manifest = json.loads(RUNTIME_MANIFEST.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid ERA5 runtime manifest: {RUNTIME_MANIFEST}") from exc
    _verify_runtime_image_parity(images, runtime_manifest)


def create_immutable_configmap(configmap: dict, *, context: str) -> None:
    """Create a content-addressed ConfigMap without a last-applied annotation.

    ``kubectl apply`` serializes the complete object into the
    ``kubectl.kubernetes.io/last-applied-configuration`` annotation.  The
    embedded smoke bundle is larger than Kubernetes' 256 KiB annotation
    limit, so an immutable, content-addressed object must be created directly.
    A name collision is intentionally fatal: it signals an unexpected stale
    or tampered cluster object instead of mutating it in place.
    """
    subprocess.run(
        ["kubectl", "--context", context, "create", "-f", "-"],
        input=json.dumps(configmap), text=True, check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="prithvi-training-default")
    parser.add_argument("--context", default="icekube")
    parser.add_argument(
        "--apply", action="store_true",
        help="Create the immutable ConfigMap; default only prints identity.",
    )
    parser.add_argument(
        "--manifest", action="store_true",
        help="Print the full Kubernetes JSON manifest instead of identity.",
    )
    parser.add_argument(
        "--check-jobs", action="store_true",
        help="Require both checked-in Job manifests to reference this bundle.",
    )
    args = parser.parse_args()
    configmap, bundle = build_configmap(args.namespace)
    if args.check_jobs or args.apply:
        verify_job_manifests(configmap, bundle)
    if args.apply:
        create_immutable_configmap(configmap, context=args.context)
    if args.manifest:
        print(json.dumps(configmap, indent=2, sort_keys=True))
    else:
        print(json.dumps({
            "configmap": configmap["metadata"]["name"],
            "bundle_sha256": bundle["bundle_sha256"],
            "files": bundle["files"],
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
