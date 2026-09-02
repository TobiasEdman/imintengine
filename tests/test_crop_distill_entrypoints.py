"""Contract tests for the image-baked LUCAS crop-distill entrypoints."""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import stat
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS))

import scripts.crop_distill_protocol as protocol
import scripts.crop_distill_provenance as provenance
import scripts.gen_ladder_manifests as manifests
import scripts.run_crop_distill_job as crop_job
import scripts.run_lucas_crop_split_job as split_job

SOURCE_SHA = "a" * 40
IMAGE_REF = "ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:" + "b" * 64
SPLIT_SHA256 = "c" * 64
IDENTITY = protocol.RuntimeIdentity(SOURCE_SHA, IMAGE_REF, "pod-uid-123")

EXPECTED_MODELS = {
    "clay": (
        504,
        "clay_v1_5",
        (),
        2_601_012_332,
        "0a37ebdbbae8ac61145424350ae8f2990225d2cb15a3e1c178c9d42134c226e2",
    ),
    "croma": (
        504,
        "croma_base",
        ("s1_vv_vh",),
        834_654_805,
        "dbfc04cf9475ca6b604dd5133191854736e961deebeb992be855f211d152bd80",
    ),
    "prithvi300m": (
        496,
        "prithvi_300m",
        (),
        1_285_893_675,
        "a27dadd9caf1c9ccfba6ecbd76ac7815fcb7236978e9df807e1d1bf7a498cda0",
    ),
    "prithvi600m": (
        504,
        "prithvi_600m",
        (),
        2_741_619_081,
        "89d544c06fd353772722dec5600a4ba8696fd8971250f471b47f6b53828d1d46",
    ),
    "terramind": (
        496,
        "terramind_v1_base",
        ("s1_vv_vh",),
        401_358_843,
        "97316cf22612288072f0278f5c90e1a987a845a35acb1dcb431cc13432b4fc8f",
    ),
    "tessera": (
        504,
        "tessera_v1",
        ("tessera",),
        1_596_322,
        "9dd7cfcad09b26576d23c846c29c3fd540d463b97a72df0b7557f6558dbced04",
    ),
}


@pytest.fixture
def render_identity(monkeypatch):
    """Render B manifests in memory while their real A identities are pending."""
    monkeypatch.setattr(manifests, "CROP_DISTILL_SOURCE_GIT_SHA", SOURCE_SHA)
    monkeypatch.setattr(manifests, "CROP_DISTILL_IMAGE", IMAGE_REF)
    monkeypatch.setattr(
        manifests,
        "CROP_DISTILL_SPLIT_MANIFEST_SHA256",
        SPLIT_SHA256,
    )


def _container(text: str) -> tuple[dict, dict]:
    document = yaml.safe_load(text)
    pod = document["spec"]["template"]["spec"]
    return pod, pod["containers"][0]


def _option_value(command: list[str], option: str) -> str:
    if option in command:
        return command[command.index(option) + 1]
    prefix = option + "="
    return next(
        value.removeprefix(prefix) for value in command if value.startswith(prefix)
    )


def _crop_split_source(root: Path) -> tuple[Path, str]:
    root.mkdir()
    index = root / protocol.CROP_INDEX.name
    split = root / protocol.CROP_SPLIT.name
    validator = root / "lucas_crop_validator_holdout_index.parquet"
    index.write_bytes(b"parquet-bytes")
    split.write_bytes(b'{"plots": []}\n')
    validator.write_bytes(b"secret-holdout")
    artifacts = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (index, split, validator)
    }
    manifest = root / protocol.CROP_SPLIT_MANIFEST.name
    manifest.write_text(json.dumps({"artifacts": artifacts}))
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_protocol_pins_all_six_model_columns_exactly():
    assert protocol.MODEL_KEYS == tuple(sorted(EXPECTED_MODELS))
    assert set(protocol.CROP_MODELS) == set(EXPECTED_MODELS)
    for model, expected in EXPECTED_MODELS.items():
        cfg = protocol.model_protocol(model)
        actual = (
            cfg.img_size,
            cfg.backbone,
            cfg.required_npz_keys,
            cfg.checkpoint_size,
            cfg.checkpoint_sha256,
        )
        assert actual == expected
        assert cfg.checkpoint_name == "best_model.pt"
        assert cfg.checkpoint_path == Path(
            f"/cephfs/checkpoints/ladder/{model}_r2/best_model.pt"
        )


def test_protocol_is_the_generators_model_source():
    assert manifests.CROP_MODEL_UIDS is protocol.CROP_MODEL_UIDS
    for model, cfg in protocol.CROP_MODELS.items():
        assert manifests.DISTILL[model]["img_size"] == cfg.img_size
        assert manifests.DISTILL[model]["backbone"] == cfg.backbone
        assert manifests.DISTILL[model].get("require_keys", ()) == (
            cfg.required_npz_keys
        )
        assert manifests.CROP_CHECKPOINTS[model] == {
            "size": cfg.checkpoint_size,
            "sha256": cfg.checkpoint_sha256,
        }


@pytest.mark.parametrize(
    "script",
    (
        "crop_distill_protocol.py",
        "run_crop_distill_job.py",
        "run_lucas_crop_split_job.py",
    ),
)
def test_baked_entrypoint_scripts_py_compile(script, tmp_path):
    py_compile.compile(
        str(SCRIPTS / script),
        cfile=str(tmp_path / f"{script}.pyc"),
        doraise=True,
    )


def test_runtime_identity_requires_only_exact_manifest_values():
    assert (
        protocol.runtime_identity(
            {
                "CROP_DISTILL_SOURCE_GIT_SHA": SOURCE_SHA,
                "CROP_DISTILL_IMAGE": IMAGE_REF,
                "POD_UID": "pod-uid-123",
            }
        )
        == IDENTITY
    )
    with pytest.raises(ValueError, match="missing required"):
        protocol.runtime_identity({})
    with pytest.raises(ValueError, match="unsafe"):
        protocol.runtime_identity(
            {
                "CROP_DISTILL_SOURCE_GIT_SHA": SOURCE_SHA,
                "CROP_DISTILL_IMAGE": IMAGE_REF,
                "POD_UID": "../escape",
            }
        )


@pytest.mark.parametrize("model", protocol.MODEL_KEYS)
def test_crop_entrypoint_builds_the_entire_baked_protocol(monkeypatch, model):
    calls: list[tuple[str, list[str]]] = []
    job = crop_job.CropDistillJob(model, IDENTITY, SPLIT_SHA256)
    snapshot = crop_job.CropSplitSnapshot(
        root=job.snapshot_dir,
        index=job.snapshot_dir / protocol.CROP_INDEX.name,
        split=job.snapshot_dir / protocol.CROP_SPLIT.name,
        manifest=job.snapshot_dir / protocol.CROP_SPLIT_MANIFEST.name,
        manifest_sha256=SPLIT_SHA256,
    )
    monkeypatch.setattr(job, "_prepare_work_directories", lambda: None)
    monkeypatch.setattr(job, "_prepare_output_dir", lambda: None)
    monkeypatch.setattr(
        crop_job,
        "snapshot_crop_inputs",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        crop_job,
        "publish_create_only",
        lambda _source, destination: crop_job.PublishedArtifact(
            path=destination,
            size_bytes=123,
            sha256="d" * 64,
        ),
    )

    def record(stage, command):
        calls.append((stage, [str(value) for value in command]))

    monkeypatch.setattr(job, "_run", record)
    job.execute()

    assert [stage for stage, _ in calls] == [
        "verify-runtime",
        "verify-split",
        "extract-features",
        "score-oof",
        "publish-completion",
    ]
    extract = calls[2][1]
    cfg = protocol.CROP_MODELS[model]
    assert extract[:2] == [str(protocol.MODEL_PYTHON), str(protocol.EXTRACT_SCRIPT)]
    assert _option_value(extract, "--checkpoint") == str(cfg.checkpoint_path)
    assert _option_value(extract, "--checkpoint-size") == str(cfg.checkpoint_size)
    assert _option_value(extract, "--checkpoint-sha256") == (cfg.checkpoint_sha256)
    assert _option_value(extract, "--plot-index") == str(snapshot.index)
    assert _option_value(extract, "--img-size") == str(cfg.img_size)
    assert _option_value(extract, "--backbone-name") == cfg.backbone
    assert _option_value(extract, "--tile-inventory") == str(snapshot.split)
    assert _option_value(extract, "--tile-inventory-partition") == "distill"
    actual_keys = [
        extract[index + 1]
        for index, value in enumerate(extract)
        if value == "--require-npz-key"
    ]
    assert actual_keys == list(cfg.required_npz_keys)

    scoring = calls[3][1]
    assert scoring[:2] == [str(protocol.SCORING_PYTHON), str(protocol.SCORE_SCRIPT)]
    assert _option_value(scoring, "--folds") == "5"
    assert _option_value(scoring, "--heads") == "mlp"
    assert _option_value(scoring, "--truth-col") == "unified_class"
    assert _option_value(scoring, "--group-col") == "tile_name"
    assert _option_value(scoring, "--features") == str(job.features_work)
    assert _option_value(scoring, "--pinned-plots") == str(snapshot.split)

    completion = calls[4][1]
    assert _option_value(completion, "--status") == "completed"
    assert _option_value(completion, "--checkpoint-sha256") == (cfg.checkpoint_sha256)
    assert _option_value(completion, "--split-sha256") == SPLIT_SHA256
    assert _option_value(completion, "--split-manifest") == str(snapshot.manifest)
    assert completion.count("--artifact-size") == 2
    assert completion.count("--artifact-sha256") == 2
    assert "validator" not in " ".join(completion).lower()


def test_crop_entrypoint_preserves_the_failing_stage_and_exit(monkeypatch):
    published: list[int] = []
    monkeypatch.setattr(crop_job, "runtime_claims", lambda _env: IDENTITY)
    monkeypatch.setattr(crop_job, "runtime_identity", lambda _env: IDENTITY)
    monkeypatch.setattr(crop_job, "split_manifest_claim", lambda _env: SPLIT_SHA256)
    monkeypatch.setattr(crop_job, "require_process_identity", lambda *_args, **_kwargs: None)

    def fail(self):
        self.failure_stage = "extract-features"
        raise subprocess.CalledProcessError(23, ["extract"])

    monkeypatch.setattr(crop_job.CropDistillJob, "execute", fail)
    monkeypatch.setattr(
        crop_job.CropDistillJob,
        "publish_failure",
        lambda _self, exit_code: published.append(exit_code),
    )
    assert crop_job.main(["--model", "clay"], environ={}) == 23
    assert published == [23]


def test_runtime_process_identity_is_exact(monkeypatch):
    monkeypatch.setattr(protocol.os, "geteuid", lambda: 2001)
    monkeypatch.setattr(protocol.os, "getegid", lambda: protocol.STORAGE_GID)
    protocol.require_process_identity(2001, role="test crop")

    monkeypatch.setattr(protocol.os, "geteuid", lambda: 2002)
    with pytest.raises(RuntimeError, match="requires effective UID:GID 2001:2000"):
        protocol.require_process_identity(2001, role="test crop")

    monkeypatch.setattr(protocol.os, "geteuid", lambda: 2001)
    monkeypatch.setattr(protocol.os, "getegid", lambda: 3000)
    with pytest.raises(RuntimeError, match="running as 2001:3000"):
        protocol.require_process_identity(2001, role="test crop")


def test_crop_main_rejects_manifest_uid_drift_before_execute(monkeypatch):
    published: list[tuple[str, int]] = []
    monkeypatch.setattr(crop_job, "runtime_claims", lambda _env: IDENTITY)
    monkeypatch.setattr(crop_job, "runtime_identity", lambda _env: IDENTITY)
    monkeypatch.setattr(crop_job, "split_manifest_claim", lambda _env: SPLIT_SHA256)
    monkeypatch.setattr(
        crop_job,
        "require_process_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("running as wrong UID")
        ),
    )
    monkeypatch.setattr(
        crop_job.CropDistillJob,
        "execute",
        lambda _self: pytest.fail("identity drift reached crop execution"),
    )
    monkeypatch.setattr(
        crop_job.CropDistillJob,
        "publish_failure",
        lambda self, code: published.append((self.failure_stage, code)),
    )

    assert crop_job.main(["--model", "clay"], environ={}) == 1
    assert published == [("validate-process-identity", 1)]


def test_split_main_rejects_manifest_uid_drift_before_execute(monkeypatch):
    published: list[tuple[str, int]] = []
    monkeypatch.setattr(split_job, "runtime_claims", lambda _env: IDENTITY)
    monkeypatch.setattr(split_job, "runtime_identity", lambda _env: IDENTITY)
    monkeypatch.setattr(
        split_job,
        "require_process_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("running as wrong UID")
        ),
    )
    monkeypatch.setattr(
        split_job.LucasCropSplitJob,
        "execute",
        lambda _self: pytest.fail("identity drift reached split execution"),
    )
    monkeypatch.setattr(
        split_job.LucasCropSplitJob,
        "publish_failure",
        lambda self, code: published.append((self.failure_stage, code)),
    )

    assert split_job.main([], environ={}) == 1
    assert published == [("validate-process-identity", 1)]


def test_invalid_crop_argument_publishes_terminal_failure(monkeypatch):
    calls: list[list[str]] = []
    environment = {
        "POD_UID": "safe-pod-uid",
        "CROP_DISTILL_SOURCE_GIT_SHA": SOURCE_SHA,
        "CROP_DISTILL_IMAGE": IMAGE_REF,
        "CROP_DISTILL_SPLIT_MANIFEST_SHA256": SPLIT_SHA256,
    }

    def record(command, **_kwargs):
        calls.append([str(value) for value in command])

    monkeypatch.setattr(crop_job.subprocess, "run", record)
    assert crop_job.main(["--" + "x" * 5000], environ=environment) == 2
    assert len(calls) == 1
    command = calls[0]
    assert _option_value(command, "--status") == "failed"
    assert _option_value(command, "--exit-code") == "2"
    failure_stage = _option_value(command, "--failure-stage")
    assert failure_stage.startswith("parse-arguments:")
    assert len(failure_stage) <= 191


def test_crop_artifact_publication_is_create_only(tmp_path):
    temporary = tmp_path / ".artifact.tmp"
    destination = tmp_path / "artifact.json"
    temporary.write_text("first")
    identity = crop_job.publish_create_only(temporary, destination)
    assert identity == crop_job.PublishedArtifact(
        path=destination,
        size_bytes=5,
        sha256=hashlib.sha256(b"first").hexdigest(),
    )
    assert destination.read_text() == "first"
    assert temporary.read_text() == "first"
    assert destination.stat().st_mode & 0o777 == 0o444

    retry = tmp_path / ".artifact.retry"
    retry.write_text("second")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        crop_job.publish_create_only(retry, destination)
    assert destination.read_text() == "first"


def test_crop_artifact_publication_rejects_symlinked_parent(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("evidence")
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink or non-directory"):
        crop_job.publish_create_only(source, alias / "artifact.json")
    assert not (real / "artifact.json").exists()


def test_crop_outputs_leave_private_work_for_group_readable_model_root(
    tmp_path, monkeypatch
):
    heads = tmp_path / "heads"
    heads.mkdir(mode=0o750)
    monkeypatch.setattr(crop_job, "CROP_HEADS_DIR", heads)
    job = crop_job.CropDistillJob("clay", IDENTITY, SPLIT_SHA256)

    job._prepare_output_dir()

    assert job.output_parent.stat().st_mode & 0o777 == 0o750
    assert job.features.parent == job.output_parent
    assert job.oof.parent == job.output_parent
    assert IDENTITY.pod_uid in job.features.name
    assert IDENTITY.pod_uid in job.oof.name
    assert job.work_dir not in job.features.parents
    assert job.work_dir not in job.oof.parents


def test_crop_split_snapshot_uses_external_authority_and_excludes_validator(
    tmp_path,
):
    source = tmp_path / "source"
    manifest, manifest_sha256 = _crop_split_source(source)
    # Crop workers authenticate the validator declaration but never need the
    # validator bytes or metadata in their mounted/snapshotted view.
    (source / "lucas_crop_validator_holdout_index.parquet").unlink()
    destination = tmp_path / "private" / "split"
    destination.parent.mkdir()

    snapshot = crop_job.snapshot_crop_inputs(
        source,
        destination,
        expected_manifest_sha256=manifest_sha256,
    )
    assert snapshot.manifest_sha256 == manifest_sha256
    assert snapshot.manifest.read_bytes() == manifest.read_bytes()
    assert snapshot.index.read_bytes() == b"parquet-bytes"
    assert snapshot.split.read_bytes() == b'{"plots": []}\n'
    assert sorted(path.name for path in destination.iterdir()) == sorted(
        {
            protocol.CROP_INDEX.name,
            protocol.CROP_SPLIT.name,
            protocol.CROP_SPLIT_MANIFEST.name,
        }
    )
    assert not (destination / "lucas_crop_validator_holdout_index.parquet").exists()


def test_crop_split_snapshot_rejects_unreviewed_manifest(tmp_path):
    source = tmp_path / "source"
    _manifest, _manifest_sha256 = _crop_split_source(source)
    destination = tmp_path / "private" / "split"
    destination.parent.mkdir()

    with pytest.raises(RuntimeError, match="manifest SHA256 mismatch"):
        crop_job.snapshot_crop_inputs(
            source,
            destination,
            expected_manifest_sha256="f" * 64,
        )
    assert not (destination / protocol.CROP_INDEX.name).exists()


@pytest.mark.parametrize("attack", ["symlink", "hardlink"])
def test_crop_split_snapshot_rejects_aliased_consumer_file(tmp_path, attack):
    source = tmp_path / "source"
    manifest, _manifest_sha256 = _crop_split_source(source)
    index = source / protocol.CROP_INDEX.name
    target = tmp_path / "attacker-index"
    target.write_bytes(index.read_bytes())
    index.unlink()
    if attack == "symlink":
        index.symlink_to(target)
    else:
        os.link(target, index)

    manifest_doc = json.loads(manifest.read_text())
    manifest_doc["artifacts"][index.name] = hashlib.sha256(
        target.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(manifest_doc))
    reviewed_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    destination = tmp_path / "private" / "split"
    destination.parent.mkdir()

    match = "securely open" if attack == "symlink" else "exactly one hard link"
    with pytest.raises(RuntimeError, match=match):
        crop_job.snapshot_crop_inputs(
            source,
            destination,
            expected_manifest_sha256=reviewed_sha256,
        )


def test_missing_split_authority_is_failed_inside_provenance_boundary(
    monkeypatch,
):
    published: list[tuple[str, str]] = []
    environment = {
        "POD_UID": "safe-pod-uid",
        "CROP_DISTILL_SOURCE_GIT_SHA": SOURCE_SHA,
        "CROP_DISTILL_IMAGE": IMAGE_REF,
    }

    def record_failure(self, _exit_code):
        published.append((self.split_manifest_sha256, self.failure_stage))

    monkeypatch.setattr(crop_job.CropDistillJob, "publish_failure", record_failure)
    assert crop_job.main(["--model", "clay"], environ=environment) == 1
    assert published == [("<missing>", "validate-split-manifest-environment")]


def test_failure_provenance_bounds_raw_environment_claims():
    claims = protocol.RuntimeIdentity(
        "bad-source\n" + "x" * 5000,
        "bad-image\n" + "y" * 5000,
        "safe-pod",
    )
    job = crop_job.CropDistillJob("clay", claims, "bad-split\n" + "z" * 5000)
    command = job._provenance_base(status="failed", exit_code=1)

    for option in ("--source-git-sha", "--image-ref", "--split-sha256"):
        value = _option_value(command, option)
        assert "\n" not in value
        assert len(value) <= 1024


def test_failure_provenance_preserves_dash_prefixed_claims_as_values():
    claims = protocol.RuntimeIdentity("--help", "--status=completed", "safe-pod")
    job = crop_job.CropDistillJob("clay", claims, "--artifact=/attacker")

    command = job._provenance_base(status="failed", exit_code=1)

    assert "--source-git-sha=--help" in command
    assert "--image-ref=--status=completed" in command
    assert "--split-sha256=--artifact=/attacker" in command
    parsed = provenance.build_parser().parse_args(command[2:])
    assert parsed.source_git_sha == "--help"
    assert parsed.image_ref == "--status=completed"
    assert parsed.split_sha256 == "--artifact=/attacker"


def test_invalid_source_claim_is_published_when_pod_uid_is_safe(monkeypatch):
    published: list[tuple[str, str, int]] = []
    environment = {
        "POD_UID": "safe-pod-uid",
        "CROP_DISTILL_SOURCE_GIT_SHA": "not-a-sha",
        "CROP_DISTILL_IMAGE": IMAGE_REF,
    }

    def record_failure(self, exit_code):
        published.append((self.identity.source_git_sha, self.failure_stage, exit_code))

    monkeypatch.setattr(crop_job.CropDistillJob, "publish_failure", record_failure)
    assert crop_job.main(["--model", "clay"], environ=environment) == 1
    assert published == [("not-a-sha", "validate-runtime-environment", 1)]


def test_split_failure_bounds_and_quotes_untrusted_identity_claims():
    identity = protocol.RuntimeIdentity(
        "--help\n" + "x" * 5000,
        "--status=completed\n" + "y" * 5000,
        "safe-pod",
    )
    job = split_job.LucasCropSplitJob(identity)

    command = job._provenance_base(status="failed", exit_code=1)

    source = next(value for value in command if value.startswith("--source-git-sha="))
    image = next(value for value in command if value.startswith("--image-ref="))
    assert source.startswith("--source-git-sha=--help ")
    assert image.startswith("--image-ref=--status=completed ")
    assert len(source.removeprefix("--source-git-sha=")) <= 1024
    assert len(image.removeprefix("--image-ref=")) <= 1024
    parsed = provenance.build_parser().parse_args(command[2:])
    assert parsed.source_git_sha.startswith("--help ")
    assert parsed.image_ref.startswith("--status=completed ")


def test_split_entrypoint_has_one_fixed_build_and_full_verify(monkeypatch):
    calls: list[tuple[str, list[str]]] = []
    job = split_job.LucasCropSplitJob(IDENTITY)
    monkeypatch.setattr(job, "_prepare_directories", lambda: None)
    monkeypatch.setattr(split_job, "sha256_file", lambda _path: "c" * 64)
    monkeypatch.setattr(
        split_job, "_ensure_real_directory", lambda _path, *, create: None
    )
    monkeypatch.setattr(split_job, "_set_directory_mode", lambda *_args: None)

    def record(stage, command):
        calls.append((stage, [str(value) for value in command]))

    monkeypatch.setattr(job, "_run", record)
    job.execute()
    assert [stage for stage, _ in calls] == [
        "verify-runtime",
        "freeze-split",
        "verify-split",
        "publish-completion",
    ]
    freeze = calls[1][1]
    assert _option_value(freeze, "--lucas-index") == str(protocol.LUCAS_SOURCE_INDEX)
    assert _option_value(freeze, "--data-dir") == str(protocol.DATA_DIR)
    assert _option_value(freeze, "--out-dir") == str(protocol.DISTILL_DIR)
    verify = calls[2][1]
    assert "--verify" in verify
    assert "--verify-consumer" not in verify
    completion = calls[3][1]
    assert completion.count("--artifact") == 4
    assert _option_value(completion, "--status") == "completed"


def test_split_entrypoint_accepts_no_behavioural_arguments():
    with pytest.raises(split_job.JobArgumentError):
        split_job.build_parser().parse_args(["--model", "clay"])


def test_invalid_split_argument_publishes_terminal_failure(monkeypatch):
    published: list[tuple[str, int]] = []
    monkeypatch.setattr(split_job, "runtime_claims", lambda _env: IDENTITY)
    monkeypatch.setattr(
        split_job.LucasCropSplitJob,
        "publish_failure",
        lambda self, code: published.append((self.failure_stage, code)),
    )

    assert split_job.main(["--model", "clay"], environ={}) == 2
    assert published == [("parse-arguments", 2)]


def test_split_owner_can_explicitly_unlock_frozen_root_for_refreeze(tmp_path):
    root = tmp_path / "crop_split"
    root.mkdir(mode=0o750)
    root.chmod(0o550)

    split_job._set_directory_mode(root, 0o770)

    assert stat.S_IMODE(root.stat().st_mode) == 0o770


def test_split_directory_preflight_rejects_symlink_component(
    monkeypatch,
    tmp_path,
):
    pvc = tmp_path / "cephfs"
    pvc.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    (pvc / "distill").symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(split_job, "PVC_ROOT", pvc)

    with pytest.raises(RuntimeError, match="not a real directory"):
        split_job._ensure_real_directory(pvc / "distill" / "heads", create=True)


@pytest.mark.parametrize("model", protocol.MODEL_KEYS)
def test_rendered_crop_job_is_shell_free_and_declarative(render_identity, model):
    text = manifests.render_crop_distill(model)
    pod, container = _container(text)
    assert container["command"] == ["/usr/local/bin/python"]
    assert container["args"] == [
        "/opt/imintengine/scripts/run_crop_distill_job.py",
        "--model",
        model,
    ]
    assert pod["imagePullSecrets"] == [{"name": "ghcr-push"}]
    assert container["image"] == IMAGE_REF
    env = {item["name"]: item for item in container["env"]}
    assert set(env) == {
        "CROP_DISTILL_SOURCE_GIT_SHA",
        "CROP_DISTILL_IMAGE",
        "CROP_DISTILL_SPLIT_MANIFEST_SHA256",
        "HOME",
        "TMPDIR",
        "POD_UID",
    }
    assert env["CROP_DISTILL_SOURCE_GIT_SHA"]["value"] == SOURCE_SHA
    assert env["CROP_DISTILL_IMAGE"]["value"] == IMAGE_REF
    assert env["CROP_DISTILL_SPLIT_MANIFEST_SHA256"]["value"] == (SPLIT_SHA256)
    assert env["HOME"]["value"] == "/work/home"
    assert env["TMPDIR"]["value"] == "/work/tmp"
    assert env["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == ("metadata.uid")

    forbidden = (
        "bash",
        "sh -c",
        "apt-get",
        "pip install",
        "git clone",
        "git fetch",
        "curl ",
        "wget ",
        "--img-size",
        "--backbone-name",
        "--checkpoint",
        "--require-npz-key",
        "--truth-col",
        "--folds",
        "--heads",
        "validator",
    )
    lowered = text.lower()
    assert not [token for token in forbidden if token in lowered]


@pytest.mark.parametrize("model", protocol.MODEL_KEYS)
def test_crop_mounts_are_minimal_subpath_projections(render_identity, model):
    _pod, container = _container(manifests.render_crop_distill(model))
    mounts = container["volumeMounts"]
    assert mounts == [
        {
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/unified_v2_512",
            "subPath": "unified_v2_512",
            "readOnly": True,
        },
        {
            "name": "training-data-cephfs",
            "mountPath": f"/cephfs/checkpoints/ladder/{model}_r2",
            "subPath": f"checkpoints/ladder/{model}_r2",
            "readOnly": True,
        },
        {
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/distill/crop_split",
            "subPath": "distill/crop_split/crop_consumer",
            "readOnly": True,
        },
        {
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/crop-heads",
            "subPath": f"distill/crop_heads/{model}_r2_crop_runs",
        },
        {
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/crop-records",
            "subPath": f"ops/crop-distill/{model}",
        },
        {"name": "work", "mountPath": "/work"},
    ]
    assert all(item["mountPath"] != "/cephfs" for item in mounts)


def test_rendered_split_job_only_invokes_the_baked_entrypoint(render_identity):
    text = manifests.render_lucas_crop_split()
    pod, container = _container(text)
    assert container["command"] == ["/usr/local/bin/python"]
    assert container["args"] == ["/opt/imintengine/scripts/run_lucas_crop_split_job.py"]
    assert container["volumeMounts"] == [
        {
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/unified_v2_512",
            "subPath": "unified_v2_512",
            "readOnly": True,
        },
        {
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/lucas",
            "subPath": "lucas",
            "readOnly": True,
        },
        {
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/distill/crop_split",
            "subPath": "distill/crop_split",
        },
        {
            "name": "training-data-cephfs",
            "mountPath": "/cephfs/ops/crop-distill",
            "subPath": "ops/crop-distill/split",
        },
        {"name": "work", "mountPath": "/work"},
    ]
    assert pod["volumes"] == [
        {
            "name": "training-data-cephfs",
            "persistentVolumeClaim": {"claimName": "training-data-cephfs"},
        },
        {"name": "work", "emptyDir": {}},
    ]
    lowered = text.lower()
    for forbidden in (
        "bash",
        "sh -c",
        "apt-get",
        "pip install",
        "git clone",
        "git fetch",
        "curl ",
        "wget ",
        "--lucas-index",
        "--data-dir",
        "--out-dir",
        "--verify",
    ):
        assert forbidden not in lowered


def test_rendering_still_refuses_bootstrap_sentinels(monkeypatch):
    monkeypatch.setattr(manifests, "CROP_DISTILL_SOURCE_GIT_SHA", "0" * 40)
    monkeypatch.setattr(manifests, "CROP_DISTILL_IMAGE", IMAGE_REF)
    with pytest.raises(ValueError, match="nonzero"):
        manifests.render_crop_distill("clay")
