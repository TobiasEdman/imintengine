"""Supply-chain pins for the crop-distill runtime and build workflow."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.crop_distill_protocol import runtime_identity

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "build-pipeline-images.yml"
DOCKERFILE = REPO / "docker" / "ladder-crop-distill" / "Dockerfile"


def _job_blocks(text: str) -> dict[str, str]:
    jobs_text = text.split("\njobs:\n", maxsplit=1)[1]
    headers = list(
        re.finditer(r"^  ([a-z][a-z0-9_-]*):\n", jobs_text, re.MULTILINE)
    )
    return {
        match.group(1): jobs_text[
            match.end() : headers[index + 1].start()
            if index + 1 < len(headers)
            else len(jobs_text)
        ]
        for index, match in enumerate(headers)
    }


def _permissions(job_block: str) -> dict[str, str]:
    match = re.search(
        r"^    permissions:\n(?P<body>(?:^      [a-z-]+:\s+\w+.*\n)+)",
        job_block,
        re.MULTILINE,
    )
    assert match is not None
    return dict(
        re.findall(
            r"^      ([a-z-]+):\s+(\w+)",
            match.group("body"),
            re.MULTILINE,
        )
    )


def _workflow_permissions(text: str) -> dict[str, str]:
    match = re.search(
        r"^permissions:\n(?P<body>(?:^  [a-z-]+:\s+\w+.*\n)+)",
        text,
        re.MULTILINE,
    )
    assert match is not None
    return dict(
        re.findall(
            r"^  ([a-z-]+):\s+(\w+)",
            match.group("body"),
            re.MULTILINE,
        )
    )


def test_privileged_workflow_actions_are_full_commit_pins():
    text = WORKFLOW.read_text(encoding="utf-8")
    uses = re.findall(r"^\s*uses:\s+([^\s#]+)", text, flags=re.MULTILINE)

    assert uses
    assert all(re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", value) for value in uses)
    assert not re.search(r"^\s*uses:\s+[^\s]+@v\d+", text, flags=re.MULTILINE)


def test_pull_request_jobs_cannot_receive_publish_permissions():
    text = WORKFLOW.read_text(encoding="utf-8")
    jobs = _job_blocks(text)

    assert _workflow_permissions(text) == {"contents": "read"}
    assert "if: github.event_name == 'pull_request'" in jobs["pr_build"]
    assert _permissions(jobs["pr_build"]) == {"contents": "read"}

    privileged = {
        name
        for name, block in jobs.items()
        if "permissions:" in block
        if any(value == "write" for value in _permissions(block).values())
    }
    assert privileged == {"publish"}
    assert "if: github.event_name != 'pull_request'" in jobs["publish"]
    assert _permissions(jobs["publish"]) == {
        "contents": "read",
        "packages": "write",
        "id-token": "write",
    }
    for permission in ("packages", "id-token"):
        assert f"{permission}: write" not in jobs["pr_build"]
    assert "attestations: write" not in text


def test_publish_smokes_exact_pushed_digest_before_signing():
    text = WORKFLOW.read_text(encoding="utf-8")
    publish = _job_blocks(text)["publish"]
    build_at = publish.index("- name: Build and push")
    smoke_at = publish.index("- name: Smoke-test pushed digest (amd64)")
    sign_at = publish.index("- name: Sign image manifest digests")
    smoke = publish[smoke_at:sign_at]

    assert build_at < smoke_at < sign_at
    assert 'docker pull --platform linux/amd64 "${IMAGE}@${DIGEST}"' in smoke
    assert "docker buildx build" not in smoke
    assert "smoke-${{ matrix.image }}:test" not in smoke
    assert smoke.count('"${IMAGE}@${DIGEST}"') >= 6
    assert "--pull never" in smoke

    pr_build = _job_blocks(text)["pr_build"]
    assert "docker buildx build" in pr_build
    assert "Smoke-test local PR candidate (amd64)" in pr_build
    assert "docker pull" not in pr_build


def test_storage_prep_smoke_uses_exact_linux_security_boundary():
    text = WORKFLOW.read_text(encoding="utf-8")
    pr_build = _job_blocks(text)["pr_build"]
    image_match = re.search(r"^          PR_IMAGE_REF: (\S+)$", pr_build, re.MULTILINE)
    assert image_match is not None
    identity = runtime_identity(
        {
            "CROP_DISTILL_SOURCE_GIT_SHA": "a" * 40,
            "CROP_DISTILL_IMAGE": image_match.group(1),
            "POD_UID": "storage-prep-smoke",
        }
    )
    assert identity.image_ref == image_match.group(1)

    for required in (
        "--user 0:2000",
        '--env CROP_DISTILL_SOURCE_GIT_SHA="$SOURCE_GIT_SHA"',
        '--env CROP_DISTILL_IMAGE="$PR_IMAGE_REF"',
        "--env POD_UID=storage-prep-smoke",
        "--cap-drop ALL",
        "--cap-add CHOWN",
        "--cap-add FOWNER",
        "--security-opt no-new-privileges",
        "--read-only",
        "--network none",
        "/opt/imintengine/scripts/prepare_crop_distill_storage.py",
    ):
        assert required in pr_build

    publish = _job_blocks(text)["publish"]
    assert '--env CROP_DISTILL_IMAGE="${IMAGE}@${DIGEST}"' in publish


def test_cosign_verification_is_exact_and_cannot_mask_failure():
    text = WORKFLOW.read_text(encoding="utf-8")
    verify_step = text.split("- name: Verify signature", maxsplit=1)[1].split(
        "\n      # Publish the digest hand-off", maxsplit=1
    )[0]
    assert "shell: bash" in verify_step
    assert "set -euo pipefail" in verify_step
    assert "cosign verify" in verify_step
    assert "| jq" in verify_step
    assert "continue-on-error" not in verify_step
    assert '--certificate-identity "https://github.com/${GITHUB_WORKFLOW_REF}"' in verify_step
    assert '--certificate-github-workflow-sha "$GITHUB_SHA"' in verify_step
    assert (
        '--certificate-github-workflow-repository "$GITHUB_REPOSITORY"'
        in verify_step
    )
    assert (
        "--certificate-oidc-issuer https://token.actions.githubusercontent.com"
        in verify_step
    )
    assert "--certificate-identity-regexp" not in verify_step


def test_crop_digest_evidence_is_published_only_after_verification():
    text = WORKFLOW.read_text(encoding="utf-8")
    verify_at = text.index("- name: Verify signature")
    record_at = text.index("- name: Record crop-distill digest")
    upload_at = text.index("- name: Upload crop-distill digest evidence")

    assert verify_at < record_at < upload_at
    record_step = text[record_at:upload_at]
    assert "source_git_sha=%s" in record_step
    assert "workflow_sha=%s" in record_step
    assert "workflow_repository=%s" in record_step
    assert "workflow_ref=%s" in record_step
    assert "$GITHUB_SHA" in record_step
    assert "$GITHUB_REPOSITORY" in record_step
    assert "run_url" not in record_step.lower()
    assert "GITHUB_RUN_ID" not in record_step


def test_summary_distinguishes_pr_validation_from_signed_publish():
    summary = _job_blocks(WORKFLOW.read_text(encoding="utf-8"))["summary"]

    assert "PR_BUILD_RESULT" in summary
    assert "PUBLISH_RESULT" in summary
    assert "inget publicerades" in summary
    assert "publicerades och signerades" in summary


def test_runtime_defaults_to_the_protocol_nonroot_identity():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"^USER 2000:2000$", text, flags=re.MULTILINE)
    assert "USER root" not in text
