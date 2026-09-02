"""The LUCAS crop-distill stage must be evidence-only and protocol-pinned.

The stage produces the numbers the R5 decision is made on [user-stated
2026-08-31: distillability before any retraining]. Two properties carry
that: (1) the OOF protocol is identical across columns — same split file,
same folds, same head, same truth column — so the numbers compare; (2) the
stage opens NO gate, so the ladder queue cannot auto-train a rung 5 the
decision has not approved. These tests pin both, plus the two generalization
holes found on the way in (from_records dropping the LUCAS columns;
accuracy_suite collapsing crop ids to non-forest).
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import scripts.crop_distill_protocol as crop_protocol  # noqa: E402
import scripts.gen_ladder_manifests as glm  # noqa: E402
import scripts.run_crop_distill_job as crop_entrypoint  # noqa: E402
import scripts.run_lucas_crop_split_job as split_entrypoint  # noqa: E402
from scripts.gen_ladder_manifests import (  # noqa: E402
    CROP_CHECKPOINTS,
    CROP_INDEX,
    CROP_SPLIT,
    CROP_SPLIT_MANIFEST,
    CROP_TRUTH_COL,
    DISTILL,
    OUT_DIR,
    PYTHON_IMAGE,
    render_crop_deny_egress,
    render_crop_distill,
    render_crop_storage_prep,
    render_distill,
    render_lucas_crop_split,
)

MODELS = sorted(DISTILL)
FIXTURE_GIT_SHA = "a" * 40
FIXTURE_CROP_IMAGE = (
    "ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:" + "b" * 64
)
FIXTURE_SPLIT_MANIFEST_SHA256 = "c" * 64
CANONICAL_PRIOR_CROP_TILE = "45024074"
RUNTIME_IDENTITY = crop_protocol.RuntimeIdentity(
    FIXTURE_GIT_SHA, FIXTURE_CROP_IMAGE, "pod-uid-123"
)


@pytest.fixture(autouse=True)
def _use_complete_crop_runtime_identity(monkeypatch):
    """Unit tests render in memory before Commit A supplies real identities."""
    monkeypatch.setattr(glm, "CROP_DISTILL_SOURCE_GIT_SHA", FIXTURE_GIT_SHA)
    monkeypatch.setattr(glm, "CROP_DISTILL_IMAGE", FIXTURE_CROP_IMAGE)
    monkeypatch.setattr(
        glm,
        "CROP_DISTILL_SPLIT_MANIFEST_SHA256",
        FIXTURE_SPLIT_MANIFEST_SHA256,
    )


def _manifest_text(path: Path) -> str:
    """Render bootstrap manifests in memory until Commit B has identities."""
    match = path.name
    if match == "lucas-crop-split-job.yaml":
        return render_lucas_crop_split()
    if match.startswith("crop-distill-") and match.endswith("-job.yaml"):
        model = match.removeprefix("crop-distill-").removesuffix("-job.yaml")
        return render_crop_distill(model)
    return path.read_text()


def _job_doc(path: Path) -> dict:
    return yaml.safe_load(_manifest_text(path))


def _job_text(path: Path) -> str:
    doc = _job_doc(path)
    c = doc["spec"]["template"]["spec"]["containers"][0]
    return "\n".join([*(c.get("command") or []), *(c.get("args") or [])])


def _container(path: Path) -> dict:
    return _job_doc(path)["spec"]["template"]["spec"]["containers"][0]


def _crop_path(model: str) -> Path:
    return OUT_DIR / f"crop-distill-{model}-job.yaml"


def _crop_job(model: str) -> crop_entrypoint.CropDistillJob:
    return crop_entrypoint.CropDistillJob(
        model,
        RUNTIME_IDENTITY,
        FIXTURE_SPLIT_MANIFEST_SHA256,
    )


def test_crop_renderers_cover_every_column_without_committed_manifests():
    """Commit A carries renderers, not unsafe pre-bootstrap YAML files."""
    rendered = {
        model: yaml.safe_load(render_crop_distill(model)) for model in MODELS
    }

    assert set(rendered) == set(MODELS)
    assert all(doc["kind"] == "Job" for doc in rendered.values())
    assert yaml.safe_load(render_lucas_crop_split())["kind"] == "Job"
    assert yaml.safe_load(render_crop_storage_prep())["kind"] == "Job"
    assert yaml.safe_load(render_crop_deny_egress())["kind"] == "NetworkPolicy"


@pytest.mark.parametrize("model", MODELS)
def test_crop_protocol_is_pinned_and_uniform(model):
    """Folds/head/truth/split identical across columns — or the numbers
    do not compare and the R5 decision rests on noise."""
    assert crop_protocol.OOF_FOLDS == 5
    assert crop_protocol.OOF_HEADS == "mlp"
    assert crop_protocol.TRUTH_COLUMN == CROP_TRUTH_COL
    assert str(crop_protocol.CROP_SPLIT) == CROP_SPLIT
    assert str(crop_protocol.CROP_INDEX) == CROP_INDEX
    assert model in crop_protocol.CROP_MODELS


@pytest.mark.parametrize("model", MODELS)
def test_crop_stage_follows_the_column_regime(model):
    """img-size and backbone-name are the column's own — clay/croma build
    their backbone at the img-size grid, so a wrong value is not a detail."""
    cfg = DISTILL[model]
    baked = crop_protocol.CROP_MODELS[model]
    assert baked.img_size == cfg["img_size"]
    assert baked.backbone == cfg["backbone"]
    assert baked.required_npz_keys == cfg.get("require_keys", ())
    assert baked.checkpoint_path == Path(
        f"/cephfs/checkpoints/ladder/{model}_r2/best_model.pt"
    )


@pytest.mark.parametrize("model", MODELS)
def test_crop_outputs_are_model_scoped(model):
    job = _crop_job(model)
    text = "\n".join((str(job.features), str(job.oof)))
    assert f"{model}_r2_crop_features.parquet" in str(job.features)
    assert f"{model}_r2_crop_distillability.json" in str(job.oof)
    for other in MODELS:
        if other != model:
            assert f"{other}_r2" not in text


@pytest.mark.parametrize("model", MODELS)
def test_crop_stage_opens_no_gate(model):
    """THE no-front-run guard: the stage must never WRITE a gate marker —
    it would let the ladder queue auto-submit a rung 5 before the human
    decision. (The manifest may MENTION _GATE_OK in its warning comment.)"""
    manifest = _job_text(_crop_path(model))
    entrypoint = inspect.getsource(crop_entrypoint.CropDistillJob)
    assert "_GATE_OK" not in manifest + entrypoint


@pytest.mark.parametrize("model", MODELS)
def test_crop_stage_never_touches_h100_quota(model):
    doc = _job_doc(_crop_path(model))
    spec = doc["spec"]["template"]["spec"]
    assert spec["nodeSelector"] == {"accelerator": "nvidia-gtx-2080ti"}
    c = spec["containers"][0]
    assert c["resources"]["requests"]["memory"] == "24Gi"
    assert c["resources"]["requests"]["ephemeral-storage"] == "8Gi"
    assert c["resources"]["limits"]["ephemeral-storage"] == "8Gi"


@pytest.mark.parametrize("model", MODELS)
def test_crop_stage_mounts_only_protocol_subpaths(model):
    """Crop pods cannot see the PVC root or the held-back partition."""
    mounts = {
        item["name"]: item for item in _container(_crop_path(model))["volumeMounts"]
    }
    assert mounts == {
        "tiles": {
            "name": "tiles", "mountPath": "/cephfs/unified_v2_512",
            "subPath": "unified_v2_512", "readOnly": True,
        },
        "checkpoint": {
            "name": "checkpoint",
            "mountPath": f"/cephfs/checkpoints/ladder/{model}_r2",
            "subPath": f"checkpoints/ladder/{model}_r2", "readOnly": True,
        },
        "split": {
            "name": "split", "mountPath": "/cephfs/distill/crop_split",
            "subPath": "distill/crop_split/crop_consumer", "readOnly": True,
        },
        "heads": {
            "name": "heads", "mountPath": "/cephfs/crop-heads",
            "subPath": f"distill/crop_heads/{model}_r2_crop_runs",
        },
        "records": {
            "name": "records", "mountPath": "/cephfs/crop-records",
            "subPath": f"ops/crop-distill/{model}",
        },
        "work": {"name": "work", "mountPath": "/work"},
    }
    assert all(mount["mountPath"] not in {"/cephfs", "/data"}
               for mount in mounts.values())


@pytest.mark.parametrize("model", MODELS)
def test_crop_stage_uses_only_baked_dependencies(model):
    """All six columns run the same immutable, offline dependency stack."""
    text = _manifest_text(_crop_path(model)) + inspect.getsource(crop_entrypoint)
    for forbidden in (
        "apt-get", "pip install", "git clone", "git fetch", "curl ", "wget ",
        "urllib.request", "requests.get", "http://", "https://",
    ):
        assert forbidden not in text


def test_split_job_freezes_the_canonical_split():
    path = OUT_DIR / "lucas-crop-split-job.yaml"
    container = _container(path)
    assert container["command"] == ["/usr/local/bin/python"]
    assert container["args"] == [
        "/opt/imintengine/scripts/run_lucas_crop_split_job.py"
    ]
    assert crop_protocol.LUCAS_SOURCE_INDEX == Path(
        "/cephfs/lucas/lucas_tile_index.parquet"
    )
    assert crop_protocol.DATA_DIR == Path("/cephfs/unified_v2_512")
    assert crop_protocol.DISTILL_DIR == Path("/cephfs/distill/crop_split")
    doc = _job_doc(OUT_DIR / "lucas-crop-split-job.yaml")
    c = doc["spec"]["template"]["spec"]["containers"][0]
    assert "nvidia.com/gpu" not in c["resources"]["requests"], \
        "the split is CPU work; a GPU request wastes a 2080ti slot"
    mounts = {item["name"]: item for item in container["volumeMounts"]}
    assert mounts == {
        "tiles": {
            "name": "tiles", "mountPath": "/cephfs/unified_v2_512",
            "subPath": "unified_v2_512", "readOnly": True,
        },
        "lucas": {
            "name": "lucas", "mountPath": "/cephfs/lucas",
            "subPath": "lucas", "readOnly": True,
        },
        "distill": {
            "name": "distill", "mountPath": "/cephfs/distill/crop_split",
            "subPath": "distill/crop_split",
        },
        "ops": {
            "name": "ops", "mountPath": "/cephfs/ops/crop-distill",
            "subPath": "ops/crop-distill/split",
        },
        "work": {"name": "work", "mountPath": "/work"},
    }
    assert all(mount["mountPath"] != "/cephfs" for mount in mounts.values())


@pytest.mark.parametrize("model", MODELS)
def test_nfi_extract_and_dense_pass_share_required_tile_filter(model):
    """A feature row and its later dense sidecar must come from the same
    trainable tile cohort, including the minimum SAR enrichment version."""
    doc = yaml.safe_load(render_distill(model))
    script = "\n".join(
        doc["spec"]["template"]["spec"]["containers"][0]["command"]
    )
    expected = set(DISTILL[model].get("require_keys", ()))
    for key in {"s1_vv_vh", "tessera"}:
        count = script.count(f"--require-npz-key {key}")
        assert count == (2 if key in expected else 0)


@pytest.mark.parametrize(
    "path",
    [_crop_path(model) for model in MODELS]
    + [OUT_DIR / "lucas-crop-split-job.yaml"],
    ids=lambda path: path.name,
)
def test_crop_jobs_use_one_pinned_offline_runtime(path):
    doc = _job_doc(path)
    pod_spec = doc["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    env = {item["name"]: item for item in container["env"]}
    is_crop = path.name.startswith("crop-distill-")
    model = (
        path.name.removeprefix("crop-distill-").removesuffix("-job.yaml")
        if is_crop
        else None
    )

    assert container["image"] == glm.CROP_DISTILL_IMAGE
    assert pod_spec["imagePullSecrets"] == [{"name": "ghcr-push"}]
    assert pod_spec["automountServiceAccountToken"] is False
    assert pod_spec["securityContext"]["runAsNonRoot"] is True
    run_uid = (
        glm.CROP_MODEL_UIDS[model]
        if is_crop
        else crop_protocol.STORAGE_UID
    )
    assert pod_spec["securityContext"]["runAsUser"] == run_uid
    assert pod_spec["securityContext"]["runAsGroup"] == crop_protocol.STORAGE_GID
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["capabilities"] == {"drop": ["ALL"]}
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["securityContext"]["runAsNonRoot"] is True
    assert container["securityContext"]["runAsUser"] == run_uid
    assert container["securityContext"]["runAsGroup"] == crop_protocol.STORAGE_GID
    volumes = {item["name"]: item for item in pod_spec["volumes"]}
    expected_work = (
        {"name": "work", "emptyDir": {"sizeLimit": "8Gi"}}
        if is_crop
        else {"name": "work", "emptyDir": {}}
    )
    assert volumes["work"] == expected_work
    assert container["command"] == ["/usr/local/bin/python"]
    if is_crop:
        assert model is not None
        assert container["args"] == [
            "/opt/imintengine/scripts/run_crop_distill_job.py", "--model", model,
        ]
    else:
        assert container["args"] == [
            "/opt/imintengine/scripts/run_lucas_crop_split_job.py"
        ]
    expected_env = {
        "CROP_DISTILL_IMAGE", "CROP_DISTILL_SOURCE_GIT_SHA", "HOME",
        "TMPDIR", "POD_UID",
    }
    if is_crop:
        expected_env.add("CROP_DISTILL_SPLIT_MANIFEST_SHA256")
    assert set(env) == expected_env
    assert env["CROP_DISTILL_IMAGE"]["value"] == glm.CROP_DISTILL_IMAGE
    assert env["CROP_DISTILL_SOURCE_GIT_SHA"]["value"] == (
        glm.CROP_DISTILL_SOURCE_GIT_SHA
    )
    assert env["HOME"]["value"] == "/work/home"
    assert env["TMPDIR"]["value"] == "/work/tmp"
    if is_crop:
        assert env["CROP_DISTILL_SPLIT_MANIFEST_SHA256"]["value"] == (
            glm.CROP_DISTILL_SPLIT_MANIFEST_SHA256
        )
    assert env["POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == (
        "metadata.uid"
    )
    text = _manifest_text(path)
    for forbidden in (
        "bash", "sh -c", "apt-get", "pip install", "git clone", "git fetch",
        "curl ", "wget ",
    ):
        assert forbidden not in text


@pytest.mark.parametrize(
    "source_git_sha,image",
    [
        ("0" * 40, FIXTURE_CROP_IMAGE),
        ("a" * 12, FIXTURE_CROP_IMAGE),
        ("A" * 40, FIXTURE_CROP_IMAGE),
        (FIXTURE_GIT_SHA,
         "ghcr.io/tobiasedman/imint-ladder-crop-distill:latest"),
        (FIXTURE_GIT_SHA,
         "ghcr.io/tobiasedman/wrong-repository@sha256:" + "b" * 64),
        (FIXTURE_GIT_SHA,
         "ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:" + "0" * 64),
    ],
    ids=(
        "zero-source-sha", "abbreviated-source-sha", "uppercase-source-sha",
        "mutable-image-tag", "wrong-image-repository", "zero-image-digest",
    ),
)
def test_generator_refuses_incomplete_crop_runtime_identity(
    monkeypatch, tmp_path, capsys, source_git_sha, image,
):
    monkeypatch.setattr(glm, "CROP_DISTILL_SOURCE_GIT_SHA", source_git_sha)
    monkeypatch.setattr(glm, "CROP_DISTILL_IMAGE", image)
    monkeypatch.setattr(glm, "OUT_DIR", tmp_path / "must-not-exist")

    with pytest.raises(ValueError):
        glm.render_crop_distill("clay")
    with pytest.raises(ValueError):
        glm.render_lucas_crop_split()

    monkeypatch.setattr(sys, "argv", ["gen_ladder_manifests.py", "--check"])
    assert glm.main() == 2
    first_error = capsys.readouterr().err
    assert first_error.startswith("REFUSING crop-distill manifest generation:")
    assert glm.main() == 2
    assert capsys.readouterr().err == first_error
    monkeypatch.setattr(sys, "argv", ["gen_ladder_manifests.py"])
    assert glm.main() == 2
    assert capsys.readouterr().err == first_error
    assert not glm.OUT_DIR.exists()


def test_only_crop_consumers_require_the_frozen_split_digest(monkeypatch):
    monkeypatch.setattr(glm, "CROP_DISTILL_SPLIT_MANIFEST_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="SPLIT_MANIFEST_SHA256"):
        glm.render_crop_distill("clay")
    assert "ladder-lucas-crop-split" in glm.render_lucas_crop_split()


def test_crop_bootstrap_cli_emits_only_split_producers(
    monkeypatch, tmp_path, capsys,
):
    """Commit B may generate producer YAML before the split digest exists."""
    output_dir = tmp_path / "generated"
    monkeypatch.setattr(glm, "REPO", tmp_path)
    monkeypatch.setattr(glm, "OUT_DIR", output_dir)
    monkeypatch.setattr(glm, "CROP_DISTILL_SPLIT_MANIFEST_SHA256", "0" * 64)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gen_ladder_manifests.py", "--crop-bootstrap-only"],
    )

    assert glm.main() == 0
    assert {path.name for path in output_dir.iterdir()} == {
        "crop-distill-deny-egress.yaml",
        "crop-distill-storage-prep-job.yaml",
        "lucas-crop-split-job.yaml",
    }
    assert (output_dir / "crop-distill-deny-egress.yaml").read_text() == (
        render_crop_deny_egress()
    )
    assert (output_dir / "crop-distill-storage-prep-job.yaml").read_text() == (
        render_crop_storage_prep()
    )
    assert (output_dir / "lucas-crop-split-job.yaml").read_text() == (
        render_lucas_crop_split()
    )
    assert all(
        not (output_dir / f"crop-distill-{model}-job.yaml").exists()
        for model in MODELS
    )
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gen_ladder_manifests.py",
            "--check",
            "--crop-bootstrap-only",
        ],
    )
    assert glm.main() == 0
    assert "all 3 ladder manifests up to date" in capsys.readouterr().out

    stale_consumer = output_dir / f"crop-distill-{MODELS[0]}-job.yaml"
    stale_consumer.write_text("unsafe pre-bootstrap consumer")
    monkeypatch.setattr(
        sys,
        "argv",
        ["gen_ladder_manifests.py", "--crop-bootstrap-only"],
    )
    assert glm.main() == 2
    assert "stale consumer manifests exist" in capsys.readouterr().err


def test_full_generator_refuses_zero_split_digest(
    monkeypatch, tmp_path, capsys,
):
    """Commit C cannot publish consumers until it pins the frozen digest."""
    monkeypatch.setattr(glm, "OUT_DIR", tmp_path / "must-not-exist")
    monkeypatch.setattr(glm, "CROP_DISTILL_SPLIT_MANIFEST_SHA256", "0" * 64)
    monkeypatch.setattr(sys, "argv", ["gen_ladder_manifests.py"])

    assert glm.main() == 2
    assert "CROP_DISTILL_SPLIT_MANIFEST_SHA256" in capsys.readouterr().err
    assert not glm.OUT_DIR.exists()


@pytest.mark.parametrize(
    "path",
    [_crop_path(model) for model in MODELS]
    + [OUT_DIR / "lucas-crop-split-job.yaml"],
    ids=lambda path: path.name,
)
def test_generated_crop_job_has_no_inline_shell(path):
    container = _container(path)
    assert container["command"] == ["/usr/local/bin/python"]
    assert len(container["args"]) in {1, 3}
    text = _manifest_text(path).lower()
    for token in ("bash", "sh -c", "set -e", "trap ", "|", "&&"):
        assert token not in text


@pytest.mark.parametrize("model", MODELS)
def test_crop_checkpoint_identity_is_exact(model):
    expected = CROP_CHECKPOINTS[model]
    protocol = crop_protocol.CROP_MODELS[model]
    assert protocol.checkpoint_size == expected["size"]
    assert protocol.checkpoint_sha256 == expected["sha256"]
    assert protocol.checkpoint_name == "best_model.pt"
    execute_source = inspect.getsource(crop_entrypoint.CropDistillJob.execute)
    assert '"--checkpoint-size"' in execute_source
    assert "self.protocol.checkpoint_size" in execute_source
    assert '"--checkpoint-sha256"' in execute_source
    assert "self.protocol.checkpoint_sha256" in execute_source


@pytest.mark.parametrize("model", MODELS)
def test_crop_outputs_are_pod_scoped_readable_and_create_only(model):
    import os

    job = _crop_job(model)
    assert job.output_parent == Path("/cephfs/crop-heads")
    assert job.features == job.output_parent / (
        f"{RUNTIME_IDENTITY.pod_uid}--{model}_r2_crop_features.parquet"
    )
    assert job.oof == job.output_parent / (
        f"{RUNTIME_IDENTITY.pod_uid}--{model}_r2_crop_distillability.json"
    )
    prepare_source = inspect.getsource(job._prepare_output_dir)
    directory_source = inspect.getsource(crop_entrypoint._ensure_publish_directory)
    publish_source = inspect.getsource(crop_entrypoint.publish_create_only)
    assert "_ensure_publish_directory(self.output_parent)" in prepare_source
    assert "os.mkdir(" in directory_source
    assert "0o750" in directory_source
    assert "identity.st_uid != os.geteuid()" in directory_source
    assert crop_entrypoint._CREATE_FLAGS & os.O_EXCL
    assert crop_entrypoint._CREATE_FLAGS & os.O_NOFOLLOW
    assert "os.fchmod(temporary_fd, 0o444)" in publish_source
    assert "os.link(" in publish_source
    assert "exist_ok" not in prepare_source + directory_source + publish_source


@pytest.mark.parametrize(
    "path,kind",
    [(_crop_path("tessera"), "crop"),
     (OUT_DIR / "lucas-crop-split-job.yaml", "split")],
    ids=lambda item: item.name if isinstance(item, Path) else item,
)
def test_entrypoint_failure_is_fail_closed_and_record_is_minimal(path, kind):
    if kind == "crop":
        job = _crop_job("tessera")
        module = crop_entrypoint
    else:
        job = split_entrypoint.LucasCropSplitJob(RUNTIME_IDENTITY)
        module = split_entrypoint
    failure_call = job._provenance_base(status="failed", exit_code=1)
    assert failure_call[failure_call.index("--kind") + 1] == kind
    assert failure_call[failure_call.index("--status") + 1] == "failed"
    for optional in ("--split-manifest", "--checkpoint", "--artifact"):
        assert optional not in failure_call
    if kind == "crop":
        split_claim = next(
            value.removeprefix("--split-sha256=")
            for value in failure_call
            if value.startswith("--split-sha256=")
        )
        assert split_claim == FIXTURE_SPLIT_MANIFEST_SHA256
    else:
        assert "--split-sha256" not in failure_call
    main_source = inspect.getsource(module.main)
    assert "runtime_claims(environment)" in main_source
    assert main_source.index("try:") < main_source.index(
        "runtime_identity(environment)"
    )
    assert "return 97" in main_source


@pytest.mark.parametrize("model", MODELS)
def test_crop_completion_binds_all_inputs_and_outputs(model):
    assert str(crop_protocol.CROP_SPLIT_MANIFEST) == CROP_SPLIT_MANIFEST
    source = inspect.getsource(crop_entrypoint.CropDistillJob.execute)
    source += inspect.getsource(crop_entrypoint.CropDistillJob._provenance_base)
    assert "--expected-git-sha" in source
    assert "--verify-consumer" in source
    for option in (
        "--split-manifest", "--split-sha256", "--checkpoint",
        "--checkpoint-sha256", "--checkpoint-size", "--artifact-size",
        "--artifact-sha256",
        '"features": publish_create_only(self.features_work, self.features)',
        '"oof": publish_create_only(self.oof_work, self.oof)',
        'f"{name}={identity.path}"',
        'f"{name}={identity.size_bytes}"',
        'f"{name}={identity.sha256}"',
    ):
        assert option in source


@pytest.mark.parametrize(
    "path", [_crop_path(model) for model in MODELS]
    + [OUT_DIR / "lucas-crop-split-job.yaml"],
    ids=lambda path: path.name,
)
def test_crop_jobs_use_the_baked_interpreters(path):
    assert _container(path)["command"] == [str(crop_protocol.BASE_PYTHON)]
    assert crop_protocol.BASE_PYTHON == Path("/usr/local/bin/python")
    assert crop_protocol.SCORING_PYTHON == Path("/opt/venvs/scoring/bin/python")
    if path.name.startswith("crop-distill-"):
        source = inspect.getsource(crop_entrypoint.CropDistillJob.execute)
        assert crop_protocol.MODEL_PYTHON == Path("/opt/venvs/model/bin/python")
        assert "MODEL_PYTHON" in source and "EXTRACT_SCRIPT" in source
        assert "SCORING_PYTHON" in source and "SCORE_SCRIPT" in source
    else:
        source = inspect.getsource(split_entrypoint.LucasCropSplitJob.execute)
        assert "SCORING_PYTHON" in source and "SPLIT_SCRIPT" in source


def test_split_completion_binds_exactly_four_split_artifacts():
    source = inspect.getsource(split_entrypoint.LucasCropSplitJob.execute)
    assert "--verify-consumer" not in source
    assert '"--verify"' in source
    expected = {
        "index": "lucas_crop_distill_index.parquet",
        "validator_holdout": "lucas_crop_validator_holdout_index.parquet",
        "split": "lucas_crop_split.json",
        "manifest": "lucas_crop_split.MANIFEST.json",
    }
    assert source.count('"--artifact"') == 4
    for name, filename in expected.items():
        assert f'f"{name}=' in source
        assert getattr(split_entrypoint, {
            "index": "DISTILL_INDEX",
            "validator_holdout": "VALIDATOR_INDEX",
            "split": "SPLIT",
            "manifest": "SPLIT_MANIFEST",
        }[name]).name == filename


def test_output_columns_keep_the_lucas_key_and_truth():
    """from_records(columns=…) DROPS record keys missing from the list —
    the hard-coded NFI list lost point_id and unified_class entirely."""
    from extract_plot_features import output_columns

    cols = output_columns("unified_class", ["f000", "f001"])
    assert "point_id" in cols
    assert "unified_class" in cols
    assert "nfi_forest" not in cols
    assert cols[-2:] == ["f000", "f001"]

    nfi = output_columns(None, ["f000"])
    assert "nfi_forest" in nfi
    assert "unified_class" not in nfi


def test_truth_summary_follows_the_mode():
    """The post-write summary crashed on KeyError('nfi_forest') in crop
    mode — after the parquet was written, failing the whole Job
    (backoffLimit 0). Drive the real writer-schema + summary end to end."""
    import pandas as pd
    from extract_plot_features import output_columns, truth_summary

    feat_cols = ["f000", "f001"]
    records = [
        {"TractID": None, "PlotID": None, "point_id": 7, "Easting": None,
         "Northing": None, "tile_name": "t1", "unified_class": 11,
         "f000": 0.1, "f001": 0.2},
        {"TractID": None, "PlotID": None, "point_id": 8, "Easting": None,
         "Northing": None, "tile_name": "t1", "unified_class": 12,
         "f000": 0.3, "f001": 0.4},
    ]
    crop_df = pd.DataFrame.from_records(
        records, columns=output_columns("unified_class", feat_cols))
    name, dist = truth_summary(crop_df, "unified_class")
    assert name == "unified_class"
    assert dist == {11: 1, 12: 1}

    nfi_df = pd.DataFrame.from_records(
        [{**r, "nfi_forest": 1} for r in records],
        columns=output_columns(None, feat_cols))
    name, dist = truth_summary(nfi_df, None)
    assert name == "nfi_forest"
    assert dist == {1: 2}


def test_generic_suite_scores_in_the_truths_own_space():
    """accuracy_suite collapses ids outside {1..4} to 0 — on crop truth
    every plot lands in one class and overall reads a meaningless 1.0.
    The generic suite must keep crop classes apart."""
    from nfi_head_cv import generic_accuracy_suite
    from validate_against_nfi import accuracy_suite

    truth = np.array([11, 11, 12, 12, 15, 15])
    pred = np.array([11, 12, 12, 12, 15, 11])

    collapsed = accuracy_suite(truth, pred)
    assert collapsed["overall_accuracy_5class"] == 1.0  # the trap, proven

    suite = generic_accuracy_suite(truth, pred)
    assert suite["overall_accuracy"] == round(4 / 6, 4)
    assert 0 < suite["cohen_kappa"] < 1
    assert suite["per_class"]["vete"]["support"] == 2
    assert suite["per_class"]["korn"]["producers_accuracy"] == 1.0


def test_generic_suite_perfect_prediction():
    from nfi_head_cv import generic_accuracy_suite

    y = np.array([11, 12, 13, 21, 21])
    suite = generic_accuracy_suite(y, y.copy())
    assert suite["overall_accuracy"] == 1.0
    assert suite["cohen_kappa"] == 1.0


# --- Codex re-review findings (2026-09-01) ------------------------------


TEMPLATE_MANIFESTS = sorted(
    [OUT_DIR / f"crop-distill-{m}-job.yaml" for m in MODELS]
    + [OUT_DIR / f"distill-{m}-job.yaml" for m in MODELS]
    + [OUT_DIR / "lucas-crop-split-job.yaml",
       OUT_DIR / "distill-pinned-plots-job.yaml"])


@pytest.mark.parametrize("path", TEMPLATE_MANIFESTS, ids=lambda p: p.name)
def test_cpu_job_images_are_digest_pinned(path):
    """Zero-tolerance rule: a mutable tag can be repointed under the
    pipeline's feet. Every generated template job pins by digest."""
    doc = _job_doc(path)
    image = doc["spec"]["template"]["spec"]["containers"][0]["image"]
    if path.name.startswith("crop-distill-") or path.name == (
        "lucas-crop-split-job.yaml"
    ):
        assert image == glm.CROP_DISTILL_IMAGE
    else:
        assert image == PYTHON_IMAGE
    assert "@sha256:" in image


def test_split_qualifies_on_the_true_six_column_intersection():
    """REQUIRED_KEYS alone (s1_vv_vh) strands tessera: its dataset drops
    embedding-less tiles at init, so a frozen tile without the tessera
    key would abort crop-distill-tessera at the pinned-OOF merge."""
    from build_lucas_crop_split import required_npz_keys

    keys = required_npz_keys()
    assert "s1_vv_vh" in keys
    assert "tessera" in keys


def test_missing_tessera_tile_does_not_qualify(tmp_path):
    from build_lucas_crop_split import required_npz_keys, tile_qualifies

    keys = required_npz_keys()
    full = tmp_path / "full.npz"
    np.savez(full, s1_vv_vh=np.zeros(1), s1_enrich_v=np.int32(4),
             tessera=np.zeros(1))
    no_tess = tmp_path / "no_tess.npz"
    np.savez(no_tess, s1_vv_vh=np.zeros(1), s1_enrich_v=np.int32(4))
    stale_sar = tmp_path / "stale_sar.npz"
    np.savez(stale_sar, s1_vv_vh=np.zeros(1), s1_enrich_v=np.int32(0),
             tessera=np.zeros(1))

    assert tile_qualifies(full, keys) is True
    assert tile_qualifies(no_tess, keys) is False
    assert tile_qualifies(stale_sar, keys) is False


def _write_lucas_fixture(tmp_path, n_tiles=10, with_tessera=True,
                         forced_tiles=()):
    """A minimal PVC: n_tiles qualified npz tiles + an index parquet with
    all 11 crop classes x 5 points per tile, inside the crop window.

    The historical protocol freezes 53 tile names, while the recovered real
    L1 index has exactly 71 prior-test rows on only 24 represented tiles.
    Keep that distinction realistic without making every test hash 53 NPZs:
    one requested crop tile contributes 48 rows and 23 non-crop source rows
    contribute one represented prior tile each. The other canonical tiles
    have no co-located LUCAS row, just like the real artifact.
    """
    import pandas as pd

    n_tiles = max(n_tiles, 8)  # grouped 5-fold needs >=5 distill tile groups
    forced_tiles = tuple(forced_tiles) or (CANONICAL_PRIOR_CROP_TILE,)
    assert len(forced_tiles) == 1
    forced_crop_tile = forced_tiles[0]
    import json as _json

    canonical_prior_tiles = _json.loads(
        (REPO / "data/distill/distill_split.json").read_text()
    )["test_tiles"]
    assert forced_crop_tile in canonical_prior_tiles
    data_dir = tmp_path / "tiles"
    data_dir.mkdir(exist_ok=True)
    rows = []
    crop_tile_names = [
        *(f"tile{t:02d}" for t in range(n_tiles - 1)),
        forced_crop_tile,
    ]
    for t, name in enumerate(crop_tile_names):
        kw = {
            "s1_vv_vh": np.array([t], dtype=np.float32),
            "s1_enrich_v": np.int32(4),
        }
        if with_tessera or t != 0:  # tile00 optionally lacks the embedding
            kw["tessera"] = np.array([t], dtype=np.float32)
        np.savez(data_dir / f"{name}.npz", **kw)
        pid = 0
        for cls in range(11, 22):
            for _ in range(5):
                if name == forced_crop_tile and pid >= 48:
                    pid += 1
                    continue
                rows.append({
                    "tile_name": name,
                    "tile_path": str(data_dir / f"{name}.npz"),
                    "row": 10 + pid, "col": 10 + pid,
                    "unified_class": cls,
                    "point_id": str(t * 1000 + pid),
                    "split": "test" if name == forced_crop_tile else "train",
                })
                pid += 1
    remaining_prior_tiles = sorted(
        set(canonical_prior_tiles) - {forced_crop_tile}
    )
    assert len(remaining_prior_tiles) == 52
    for prior_idx, name in enumerate(remaining_prior_tiles[:23]):
        rows.append({
            "tile_name": name,
            "tile_path": str(data_dir / f"{name}.npz"),
            "row": 10,
            "col": 10,
            "unified_class": 1,  # prior identity, outside crop domain
            "point_id": str(900_000 + prior_idx),
            "split": "test",
        })
    index = tmp_path / "lucas_tile_index.parquet"
    pd.DataFrame(rows).to_parquet(index)
    return index, data_dir


def _run_split_builder(
    monkeypatch, index, data_dir, out_dir, *extra, default_git_sha=FIXTURE_GIT_SHA,
):
    import build_lucas_crop_split as blcs

    options = list(extra)
    if "--verify" in options or "--verify-consumer" in options:
        if "--expected-git-sha" not in options and default_git_sha is not None:
            options.extend(["--expected-git-sha", default_git_sha])
    elif "--git-sha" not in options and default_git_sha is not None:
        options.extend(["--git-sha", default_git_sha])
    monkeypatch.setattr(sys, "argv", [
        "build_lucas_crop_split.py",
        "--lucas-index", str(index),
        "--data-dir", str(data_dir),
        "--out-dir", str(out_dir),
        *options,
    ])
    try:
        blcs.main()
    except SystemExit as exc:
        # exit 0 is the frozen no-op path; anything else propagates
        if exc.code not in (None, 0):
            raise


def _refresh_artifact_hash(out_dir: Path, artifact_name: str) -> None:
    import hashlib
    import json as _json

    manifest_path = out_dir / "lucas_crop_split.MANIFEST.json"
    manifest = _json.loads(manifest_path.read_text())
    manifest["artifacts"][artifact_name] = hashlib.sha256(
        (out_dir / artifact_name).read_bytes()
    ).hexdigest()
    manifest_path.write_text(_json.dumps(manifest, indent=1))


def test_frozen_split_is_immutable(monkeypatch, tmp_path, capsys):
    """Blocker 3: a re-run (e.g. the k8s Job re-applied after its TTL
    removed the Job object) must be a no-op — never a re-freeze. Adding
    a source tile before the re-run must not move the holdout."""
    import json as _json

    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir,
                       "--git-sha", FIXTURE_GIT_SHA)
    frozen_index = (out_dir / "lucas_crop_distill_index.parquet").read_bytes()
    frozen_holdout = (
        out_dir / "lucas_crop_validator_holdout_index.parquet"
    ).read_bytes()
    frozen_split = (out_dir / "lucas_crop_split.json").read_bytes()
    manifest = _json.loads(
        (out_dir / "lucas_crop_split.MANIFEST.json").read_text())
    assert manifest["git_sha"] == FIXTURE_GIT_SHA
    assert set(manifest["artifacts"]) == {
        "lucas_crop_distill_index.parquet",
        "lucas_crop_validator_holdout_index.parquet",
        "lucas_crop_split.json",
    }
    assert (out_dir / ".lucas_crop_split.lock").exists() is False

    # a new tile lands on the PVC; the re-run must ignore it entirely
    np.savez(data_dir / "tile99.npz", s1_vv_vh=np.zeros(1),
             s1_enrich_v=np.int32(4), tessera=np.zeros(1))
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    assert "refusing to overwrite" in capsys.readouterr().out
    assert (out_dir / "lucas_crop_distill_index.parquet").read_bytes() == frozen_index
    assert (
        out_dir / "lucas_crop_validator_holdout_index.parquet"
    ).read_bytes() == frozen_holdout
    assert (out_dir / "lucas_crop_split.json").read_bytes() == frozen_split

    d = _json.loads(frozen_split)
    assert set(d["required_keys"]) >= {"s1_vv_vh", "tessera"}


def test_corrupt_freeze_is_rejected(monkeypatch, tmp_path):
    """A manifest whose artifacts are missing or hash-mismatched is a
    corrupt freeze: hard refusal with recovery — NEVER accepted as frozen
    (the pre-manifest check accepted a truncated parquet)."""
    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    split_p = out_dir / "lucas_crop_split.json"
    original = split_p.read_bytes()
    split_p.write_bytes(original[: len(original) // 2])  # truncation
    with pytest.raises(SystemExit, match="CORRUPT"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)

    split_p.write_bytes(original)
    (out_dir / "lucas_crop_distill_index.parquet").unlink()  # missing artifact
    with pytest.raises(SystemExit, match="CORRUPT"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)


def test_partial_freeze_is_rejected(monkeypatch, tmp_path):
    """Artifacts without the commit marker = interrupted freeze — neither
    side trustworthy; refuse with an explicit recovery path, not rebuild."""
    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    (out_dir / "lucas_crop_split.MANIFEST.json").unlink()
    with pytest.raises(SystemExit, match="PARTIAL"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)


@pytest.mark.parametrize("state", ["frozen", "corrupt", "partial"])
def test_freeze_recovery_requires_reviewed_uid_2000_owner_write(
    monkeypatch, tmp_path, capsys, state,
):
    """Routine storage prep preserves 0550 roots and is not an unlock path."""
    import build_lucas_crop_split as blcs

    monkeypatch.setattr(
        blcs,
        "freeze_state",
        lambda *_args, **_kwargs: (state, "injected state"),
    )
    with pytest.raises(SystemExit) as exc_info:
        blcs._guard(tmp_path)
    message = capsys.readouterr().out + str(exc_info.value)
    assert "explicitly reviewed" in message
    assert "UID 2000" in message
    assert "storage-prep" in message
    assert "cannot unlock" in message


def test_missing_holdout_index_is_a_corrupt_completed_freeze(
    monkeypatch, tmp_path,
):
    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    (out_dir / "lucas_crop_validator_holdout_index.parquet").unlink()

    with pytest.raises(
        SystemExit, match="lucas_crop_validator_holdout_index.parquet missing"
    ):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)


def test_hash_refresh_cannot_hide_key_disagreement(monkeypatch, tmp_path):
    """Codex repro: drop one parquet row AND refresh the marker hash —
    byte-integrity then passes while JSON and parquet disagree on which
    plots exist. Semantic validation must catch it: CORRUPT, and
    --verify must exit non-zero on the same state."""
    import hashlib
    import json as _json

    import pandas as pd

    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    idx_p = out_dir / "lucas_crop_distill_index.parquet"
    man_p = out_dir / "lucas_crop_split.MANIFEST.json"
    df = pd.read_parquet(idx_p)
    df.iloc[:-1].to_parquet(idx_p)  # one row gone
    m = _json.loads(man_p.read_text())
    m["artifacts"]["lucas_crop_distill_index.parquet"] = hashlib.sha256(
        idx_p.read_bytes()).hexdigest()  # refreshed marker hash
    man_p.write_text(_json.dumps(m, indent=1))

    with pytest.raises(SystemExit, match="key sets disagree"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)
    with pytest.raises(SystemExit, match="INVALID"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir, "--verify")


def test_verify_mode_accepts_a_valid_freeze(monkeypatch, tmp_path, capsys):
    """--verify is the consumers' gate (crop-distill runs it before
    extraction): exit 0 + VALID on a healthy freeze, non-zero otherwise."""
    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    _run_split_builder(monkeypatch, index, data_dir, out_dir, "--verify")
    assert "freeze VALID" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="INVALID"):
        _run_split_builder(monkeypatch, index, data_dir,
                           tmp_path / "empty", "--verify")


@pytest.mark.parametrize("damage", ["remove", "tamper"])
def test_consumer_verify_never_touches_validator_holdout(
    monkeypatch, tmp_path, damage,
):
    """Crop consumers validate only their distill-side inputs. Even a
    stat/open/hash of validator rows would breach holdout isolation."""
    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    validator = out_dir / "lucas_crop_validator_holdout_index.parquet"
    import build_lucas_crop_split as blcs

    if damage == "remove":
        validator.unlink()
    else:
        validator.write_bytes(b"validator bytes must remain opaque")

    real_capture = blcs._capture_regular_at

    def guarded_capture(directory_fd, name, **kwargs):
        if name == validator.name:
            raise AssertionError("consumer touched validator parquet")
        return real_capture(directory_fd, name, **kwargs)

    def reject_tile(*_args, **_kwargs):
        raise AssertionError("consumer touched a validator or distill tile")

    with monkeypatch.context() as guard:
        guard.setattr(blcs, "_capture_regular_at", guarded_capture)
        guard.setattr(blcs, "_capture_tile_at", reject_tile)
        _run_split_builder(
            monkeypatch,
            index,
            data_dir,
            out_dir / "crop_consumer",
            "--verify-consumer",
        )

    with pytest.raises(SystemExit, match="INVALID"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir, "--verify")


def test_consumer_verify_rejects_distill_index_tamper(monkeypatch, tmp_path):
    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    distill = out_dir / "crop_consumer/lucas_crop_distill_index.parquet"
    distill.write_bytes(distill.read_bytes() + b"tamper")

    with pytest.raises(SystemExit, match="INVALID"):
        _run_split_builder(
            monkeypatch,
            index,
            data_dir,
            out_dir / "crop_consumer",
            "--verify-consumer",
        )


@pytest.mark.parametrize("path", [
    "lucas-crop-split-job.yaml", "crop-distill-tessera-job.yaml"],
    ids=lambda v: v)
def test_job_entrypoints_are_failure_sensitive(path):
    """Every Python boundary records failures, including identity failures."""
    module = (
        crop_entrypoint if path.startswith("crop-distill") else split_entrypoint
    )
    source = inspect.getsource(module.main)
    assert source.index("runtime_claims(environment)") < source.index("try:")
    assert source.index("try:") < source.index("runtime_identity(environment)")
    assert "job.publish_failure(exit_code)" in source
    assert "return 97" in source


def test_crop_consumer_verifies_the_freeze_before_extraction():
    """Blocker 2: existence tests admit a publish window or crash-left
    pair; the consumer must run the builder's own --verify gate first."""
    source = inspect.getsource(crop_entrypoint.CropDistillJob.execute)
    verify_at = source.index('"--verify-consumer"')
    extract_at = source.index('"extract-features"')
    assert verify_at < extract_at


@pytest.mark.parametrize("model", MODELS)
def test_crop_training_never_receives_validator_holdout_index(model):
    """The full holdout parquet exists only so the verifier can prove the
    freeze. Passing its path to training would burn the independent set."""
    assert "lucas_crop_validator_holdout_index.parquet" not in _job_text(
        _crop_path(model)
    )


def test_grouped_folds_never_split_a_tile(tmp_path):
    """Round 4: LUCAS crop points cluster ~1.75/tile; point-level folds
    leak same-tile context train→test. Grouped folds must keep every
    group wholly on one side, while still predicting every point once."""
    from nfi_head_cv import make_folds

    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 3)          # 40 tiles × 3 points
    y = rng.integers(11, 14, size=len(groups))
    folds = make_folds(y, 5, groups)
    covered = np.zeros(len(y), dtype=int)
    for tr, te in folds:
        covered[te] += 1
        assert not (set(groups[tr]) & set(groups[te])), "tile straddles folds"
    assert (covered == 1).all(), "every point must be OOF-predicted once"


def test_crop_oof_is_group_folded_and_sha_stamped():
    """The baked protocol must request grouped folds + source provenance."""
    source = inspect.getsource(crop_entrypoint.CropDistillJob.execute)
    assert crop_protocol.OOF_GROUP_COLUMN == "tile_name"
    assert '"--group-col"' in source
    assert '"--git-sha"' in source
    assert "self.identity.source_git_sha" in source


@pytest.mark.parametrize("path", [
    "lucas-crop-split-job.yaml", "crop-distill-tessera-job.yaml"],
    ids=lambda v: v)
def test_scoring_deps_are_pinned_and_snapshotted(path):
    """The image's separate hash-locked environments route each command."""
    if path.startswith("crop-distill"):
        source = inspect.getsource(crop_entrypoint.CropDistillJob.execute)
        assert crop_protocol.MODEL_PYTHON == Path("/opt/venvs/model/bin/python")
        assert "MODEL_PYTHON" in source and "EXTRACT_SCRIPT" in source
        assert "SCORING_PYTHON" in source and "SCORE_SCRIPT" in source
    else:
        source = inspect.getsource(split_entrypoint.LucasCropSplitJob.execute)
        assert crop_protocol.SCORING_PYTHON == Path(
            "/opt/venvs/scoring/bin/python"
        )
        assert "SCORING_PYTHON" in source and "SPLIT_SCRIPT" in source


def test_schema_reduced_parquet_is_corrupt(monkeypatch, tmp_path):
    """Codex round-4 repro: a parquet reduced to the key columns stays
    hash-consistent after a marker refresh but is unconsumable — the
    verifier must require the full extract schema."""
    import hashlib
    import json as _json

    import pandas as pd

    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    idx_p = out_dir / "lucas_crop_distill_index.parquet"
    man_p = out_dir / "lucas_crop_split.MANIFEST.json"
    pd.read_parquet(idx_p)[["tile_name", "point_id"]].to_parquet(idx_p)
    m = _json.loads(man_p.read_text())
    m["artifacts"]["lucas_crop_distill_index.parquet"] = hashlib.sha256(
        idx_p.read_bytes()).hexdigest()
    man_p.write_text(_json.dumps(m, indent=1))

    with pytest.raises(SystemExit, match="lacks extract columns"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir, "--verify")


def test_manifest_split_holdout_count_mismatch_is_corrupt(monkeypatch, tmp_path):
    import hashlib
    import json as _json

    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    split_p = out_dir / "lucas_crop_split.json"
    man_p = out_dir / "lucas_crop_split.MANIFEST.json"
    s = _json.loads(split_p.read_text())
    s["n_holdout"] += 1
    split_p.write_text(_json.dumps(s, indent=1))
    m = _json.loads(man_p.read_text())
    m["artifacts"]["lucas_crop_split.json"] = hashlib.sha256(
        split_p.read_bytes()).hexdigest()
    man_p.write_text(_json.dumps(m, indent=1))

    with pytest.raises(SystemExit, match="n_holdout"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir, "--verify")


def test_holdout_fraction_counts_forced_tiles(monkeypatch, tmp_path):
    """Round 4: the 30% target is of ALL qualified tiles — computing it on
    the forced-reduced pool undershot by FRAC × n_forced."""
    import json as _json

    index, data_dir = _write_lucas_fixture(
        tmp_path,
        n_tiles=10,
        forced_tiles=(CANONICAL_PRIOR_CROP_TILE,),
    )
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    split = _json.loads((out_dir / "lucas_crop_split.json").read_text())
    # 10 qualified tiles → round(3.0) = 3 holdout INCLUDING the forced one.
    assert len(split["holdout_tiles"]) == 3
    assert CANONICAL_PRIOR_CROP_TILE in split["holdout_tiles"]
    assert split["forced_holdout_tiles_from_prior_split"] == [
        CANONICAL_PRIOR_CROP_TILE
    ]


def test_freeze_lock_excludes_concurrent_builders(monkeypatch, tmp_path):
    """Two racing builders must never both publish: the loser dies on the
    O_EXCL lock, loudly, before any artifact is written."""
    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / ".lucas_crop_split.lock").write_text("{}")  # holder alive

    with pytest.raises(SystemExit, match="freeze lock"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)
    assert not (out_dir / "lucas_crop_split.MANIFEST.json").exists()
    assert not (out_dir / "lucas_crop_distill_index.parquet").exists()


def test_unqualified_tile_is_excluded_from_the_freeze(monkeypatch, tmp_path):
    """End-to-end blocker-2 check: a tile missing the tessera embedding
    must appear on NEITHER side of the frozen split."""
    import json as _json

    import pandas as pd

    index, data_dir = _write_lucas_fixture(tmp_path, with_tessera=False)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    dist = pd.read_parquet(out_dir / "lucas_crop_distill_index.parquet")
    split = _json.loads((out_dir / "lucas_crop_split.json").read_text())
    assert "tile00" not in set(dist["tile_name"])
    assert "tile00" not in set(split["holdout_tiles"])


def test_split_persists_complete_partition_and_normalizes_paths(
    monkeypatch, tmp_path,
):
    """The split must identify both sides, bind the full qualified
    partition, and ignore stale/adversarial tile_path values in the source."""
    import json as _json

    import pandas as pd
    from build_lucas_crop_split import (
        EXTRACT_COLUMNS,
        KEY_DIGEST_FORMAT,
        MIN_HOLDOUT_PER_CLASS,
        _key_digest,
        _partition_digest,
        _sha256,
    )

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    source = pd.read_parquet(index)
    source["tile_path"] = "/stale-pvc/not-the-qualified-file.npz"
    source.to_parquet(index)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)

    split = _json.loads((out_dir / "lucas_crop_split.json").read_text())
    manifest = _json.loads(
        (out_dir / "lucas_crop_split.MANIFEST.json").read_text()
    )
    dist = pd.read_parquet(out_dir / "lucas_crop_distill_index.parquet")
    holdout = pd.read_parquet(
        out_dir / "lucas_crop_validator_holdout_index.parquet"
    )
    distill_keys = [
        (plot["tile_name"], plot["point_id"]) for plot in split["plots"]
    ]
    holdout_keys = [
        (plot["tile_name"], plot["point_id"])
        for plot in split["holdout_plots"]
    ]

    assert split["key_digest_format"] == KEY_DIGEST_FORMAT
    assert split["n_qualified"] == int(
        source["unified_class"].isin(range(11, 22)).sum()
    )
    assert split["n_qualified"] == split["n_distill"] + split["n_holdout"]
    assert len(set(distill_keys + holdout_keys)) == split["n_qualified"]
    assert {tile for tile, _ in distill_keys}.isdisjoint(
        {tile for tile, _ in holdout_keys}
    )
    assert split["holdout_tiles"] == sorted({tile for tile, _ in holdout_keys})
    assert tuple(dist.columns) == EXTRACT_COLUMNS
    assert tuple(holdout.columns) == EXTRACT_COLUMNS
    assert set(zip(
        holdout["tile_name"], holdout["point_id"], strict=True
    )) == set(holdout_keys)
    assert all(
        (holdout["unified_class"] == crop_class).sum()
        >= MIN_HOLDOUT_PER_CLASS
        for crop_class in range(11, 22)
    )
    expected_digests = {
        "distill_keys_sha256": _key_digest(distill_keys),
        "holdout_keys_sha256": _key_digest(holdout_keys),
        "qualified_keys_sha256": _key_digest(distill_keys + holdout_keys),
        "partition_sha256": _partition_digest(distill_keys, holdout_keys),
    }
    for field, expected in expected_digests.items():
        assert split[field] == expected
        assert manifest[field] == expected
    assert manifest["n_qualified"] == split["n_qualified"]
    assert split["git_sha"] == manifest["git_sha"] == FIXTURE_GIT_SHA
    assert split["source_index_path"] == manifest["source_index_path"]
    assert split["source_index_sha256"] == manifest["source_index_sha256"]
    assert split["source_index_sha256"] == _sha256(index)
    assert len(split["source_index_sha256"]) == 64
    assert set(dist["tile_path"]) == {
        str(data_dir.resolve() / f"{tile}.npz")
        for tile in set(dist["tile_name"])
    }


def test_duplicate_qualified_source_keys_fail_before_publication(
    monkeypatch, tmp_path,
):
    """The old builder published success and only its next invocation
    noticed duplicate keys. Reject the source before the first artifact."""
    import pandas as pd

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    source = pd.read_parquet(index)
    duplicate = source[
        (source["split"] == "train")
        & source["unified_class"].isin(range(11, 22))
    ].iloc[[0]]
    pd.concat([source, duplicate], ignore_index=True).to_parquet(index)
    out_dir = tmp_path / "out"

    with pytest.raises(SystemExit, match="duplicate keys"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)
    assert not (out_dir / "lucas_crop_distill_index.parquet").exists()
    assert not (out_dir / "lucas_crop_split.json").exists()
    assert not (out_dir / "lucas_crop_split.MANIFEST.json").exists()


def test_hash_refresh_cannot_hide_partition_metadata_tampering(
    monkeypatch, tmp_path,
):
    """Refreshing the byte hash must not make changed holdout identity,
    protocol metadata, counts, or partition digests valid."""
    import copy
    import json as _json

    from build_lucas_crop_split import freeze_state

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    split_path = out_dir / "lucas_crop_split.json"
    original_split = _json.loads(split_path.read_text())
    original_manifest = (out_dir / "lucas_crop_split.MANIFEST.json").read_bytes()
    distill_tile = original_split["plots"][0]["tile_name"]

    def changed_holdout_tiles(doc):
        doc["holdout_tiles"] = [distill_tile]

    def changed_required_keys(doc):
        doc["required_keys"] = ["s1_vv_vh"]

    def changed_truth(doc):
        doc["truth_col"] = "wrong_truth"

    def changed_window(doc):
        doc["crop_window"] = [0, 512]

    def float_window(doc):
        doc["crop_window"] = [float(doc["crop_window"][0]), doc["crop_window"][1]]

    def changed_count(doc):
        doc["n_qualified"] += 1

    def float_count(doc):
        doc["n_qualified"] = float(doc["n_qualified"])

    def float_seed(doc):
        doc["seed"] = float(doc["seed"])

    def float_trial(doc):
        doc["trial_offset"] = float(doc["trial_offset"])

    def float_min_support(doc):
        doc["min_holdout_per_class"] = float(doc["min_holdout_per_class"])

    def nonfinite_fraction(doc):
        doc["holdout_frac"] = float("nan")

    def changed_digest(doc):
        doc["partition_sha256"] = "0" * 64

    cases = (
        (changed_holdout_tiles, "holdout_tiles identity"),
        (changed_required_keys, "unexpected required_keys"),
        (changed_truth, "unexpected truth_col"),
        (changed_window, "unexpected crop_window"),
        (float_window, "malformed crop_window"),
        (changed_count, "n_qualified"),
        (float_count, "malformed n_qualified"),
        (float_seed, "non-integer seed"),
        (float_trial, "non-integer trial_offset"),
        (float_min_support, "non-integer min_holdout_per_class"),
        (nonfinite_fraction, "unexpected holdout_frac"),
        (changed_digest, "partition_sha256"),
    )
    for mutate, expected in cases:
        split_path.write_text(_json.dumps(original_split, indent=1))
        (out_dir / "lucas_crop_split.MANIFEST.json").write_bytes(
            original_manifest
        )
        changed = copy.deepcopy(original_split)
        mutate(changed)
        split_path.write_text(_json.dumps(changed, indent=1))
        _refresh_artifact_hash(out_dir, "lucas_crop_split.json")
        state, detail = freeze_state(out_dir)
        assert state == "corrupt"
        assert expected in detail


def test_hash_refresh_cannot_hide_partition_key_tampering(
    monkeypatch, tmp_path,
):
    """Exact holdout keys are part of the freeze: duplicates, tile leaks,
    and membership changes stay corrupt even after refreshing file hashes."""
    import copy
    import json as _json

    from build_lucas_crop_split import freeze_state

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    split_path = out_dir / "lucas_crop_split.json"
    original_split = _json.loads(split_path.read_text())
    original_manifest = (out_dir / "lucas_crop_split.MANIFEST.json").read_bytes()

    def duplicate_holdout_key(doc):
        doc["holdout_plots"][1] = dict(doc["holdout_plots"][0])

    def leak_distill_tile(doc):
        doc["holdout_plots"][0]["tile_name"] = doc["plots"][0]["tile_name"]

    def alter_partition_member(doc):
        doc["holdout_plots"][0]["point_id"] += 10_000_000

    for mutate, expected in (
        (duplicate_holdout_key, "duplicate plot keys"),
        (leak_distill_tile, "tile leak"),
        (alter_partition_member, "holdout_keys_sha256"),
    ):
        split_path.write_text(_json.dumps(original_split, indent=1))
        (out_dir / "lucas_crop_split.MANIFEST.json").write_bytes(
            original_manifest
        )
        changed = copy.deepcopy(original_split)
        mutate(changed)
        split_path.write_text(_json.dumps(changed, indent=1))
        _refresh_artifact_hash(out_dir, "lucas_crop_split.json")
        state, detail = freeze_state(out_dir)
        assert state == "corrupt"
        assert expected in detail


def test_hash_refreshed_nonexistent_holdout_point_is_rejected(
    monkeypatch, tmp_path,
):
    """Refreshing the parquet byte hash cannot invent a validator point
    absent from the exact holdout identity frozen in split JSON."""
    import pandas as pd
    from build_lucas_crop_split import freeze_state

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    holdout_path = out_dir / "lucas_crop_validator_holdout_index.parquet"
    holdout = pd.read_parquet(holdout_path)

    holdout.loc[holdout.index[0], "point_id"] = 99_999_999
    holdout.to_parquet(holdout_path)
    _refresh_artifact_hash(out_dir, "lucas_crop_validator_holdout_index.parquet")

    state, detail = freeze_state(out_dir, expected_git_sha=FIXTURE_GIT_SHA)
    assert state == "corrupt"
    assert "key sets disagree" in detail


def test_hash_refresh_cannot_hide_holdout_class_support_loss(
    monkeypatch, tmp_path,
):
    import pandas as pd
    from build_lucas_crop_split import freeze_state

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    holdout_path = out_dir / "lucas_crop_validator_holdout_index.parquet"
    holdout = pd.read_parquet(holdout_path)
    holdout.loc[holdout["unified_class"] == 11, "unified_class"] = 12
    holdout.to_parquet(holdout_path)
    _refresh_artifact_hash(
        out_dir, "lucas_crop_validator_holdout_index.parquet"
    )

    state, detail = freeze_state(out_dir, expected_git_sha=FIXTURE_GIT_SHA)
    assert state == "corrupt"
    assert "lacks minimum class support" in detail


def test_forced_prior_test_identity_is_exact_and_self_contained(
    monkeypatch, tmp_path,
):
    """Removing the forced tile list and its digest cannot disagree with
    the separately frozen exact forced-point identity."""
    import json as _json

    from build_lucas_crop_split import _tile_digest, freeze_state

    index, data_dir = _write_lucas_fixture(
        tmp_path,
        n_tiles=4,
        forced_tiles=(CANONICAL_PRIOR_CROP_TILE,),
    )
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    split_path = out_dir / "lucas_crop_split.json"
    manifest_path = out_dir / "lucas_crop_split.MANIFEST.json"
    split = _json.loads(split_path.read_text())
    manifest = _json.loads(manifest_path.read_text())
    assert split["forced_holdout_tiles_from_prior_split"] == [
        CANONICAL_PRIOR_CROP_TILE
    ]

    split["forced_holdout_tiles_from_prior_split"] = []
    empty_digest = _tile_digest([])
    split["forced_holdout_tiles_sha256"] = empty_digest
    manifest["forced_holdout_tiles_sha256"] = empty_digest
    split_path.write_text(_json.dumps(split, indent=1))
    manifest_path.write_text(_json.dumps(manifest, indent=1))
    _refresh_artifact_hash(out_dir, "lucas_crop_split.json")

    state, detail = freeze_state(out_dir, expected_git_sha=FIXTURE_GIT_SHA)
    assert state == "corrupt"
    assert "forced holdout" in detail


def test_verification_is_independent_of_live_source_index(monkeypatch, tmp_path):
    """The bound source SHA is historical provenance. Later source drift
    or deletion cannot make a self-contained freeze unusable."""
    import json as _json

    import pandas as pd
    from build_lucas_crop_split import freeze_state

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    split = _json.loads((out_dir / "lucas_crop_split.json").read_text())
    bound_sha = split["source_index_sha256"]
    source = pd.read_parquet(index)
    source.loc[source.index[0], "row"] += 1
    source.to_parquet(index)

    state, detail = freeze_state(out_dir, expected_git_sha=FIXTURE_GIT_SHA)
    assert state == "frozen", detail
    assert split["source_index_sha256"] == bound_sha

    index.unlink()
    state, detail = freeze_state(out_dir, expected_git_sha=FIXTURE_GIT_SHA)
    assert state == "frozen", detail


def test_git_sha_is_required_strict_and_expected_on_verify(
    monkeypatch, tmp_path, capsys,
):
    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)

    with pytest.raises(SystemExit) as exc_info:
        _run_split_builder(
            monkeypatch,
            index,
            data_dir,
            tmp_path / "missing-sha",
            default_git_sha=None,
        )
    assert exc_info.value.code == 2
    assert "--git-sha is required" in capsys.readouterr().err
    for invalid in (None, "ABC", "g" * 40, "A" * 40, "a" * 39, "a" * 41):
        extra = () if invalid is None else ("--git-sha", invalid)
        with pytest.raises(SystemExit) as exc_info:
            _run_split_builder(
                monkeypatch,
                index,
                data_dir,
                tmp_path / f"bad-{invalid}",
                *extra,
                default_git_sha=None,
            )
        assert exc_info.value.code == 2
        assert "--git-sha is required" in capsys.readouterr().err

    out_dir = tmp_path / "valid"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    with pytest.raises(SystemExit, match="does not match expected"):
        _run_split_builder(
            monkeypatch,
            index,
            data_dir,
            out_dir,
            "--verify",
            "--expected-git-sha",
            "b" * 40,
        )


def test_hash_refreshed_git_sha_corruption_is_rejected(monkeypatch, tmp_path):
    import copy
    import json as _json

    from build_lucas_crop_split import freeze_state

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    split_path = out_dir / "lucas_crop_split.json"
    manifest_path = out_dir / "lucas_crop_split.MANIFEST.json"
    original_split = _json.loads(split_path.read_text())
    original_manifest = _json.loads(manifest_path.read_text())

    cases = (
        (None, FIXTURE_GIT_SHA, "split.json has malformed git_sha"),
        ("bad", FIXTURE_GIT_SHA, "split.json has malformed git_sha"),
        ("b" * 40, FIXTURE_GIT_SHA, "git_sha mismatch"),
        ("b" * 40, "b" * 40, "does not match expected"),
    )
    for split_sha, manifest_sha, expected in cases:
        split = copy.deepcopy(original_split)
        manifest = copy.deepcopy(original_manifest)
        split["git_sha"] = split_sha
        manifest["git_sha"] = manifest_sha
        split_path.write_text(_json.dumps(split, indent=1))
        manifest_path.write_text(_json.dumps(manifest, indent=1))
        _refresh_artifact_hash(out_dir, "lucas_crop_split.json")
        state, detail = freeze_state(
            out_dir, expected_git_sha=FIXTURE_GIT_SHA
        )
        assert state == "corrupt"
        assert expected in detail


def test_source_sha_must_be_well_formed_and_mirrored(monkeypatch, tmp_path):
    import copy
    import json as _json

    from build_lucas_crop_split import freeze_state

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    split_path = out_dir / "lucas_crop_split.json"
    manifest_path = out_dir / "lucas_crop_split.MANIFEST.json"
    original_split = _json.loads(split_path.read_text())
    original_manifest = _json.loads(manifest_path.read_text())

    for split_sha, manifest_sha, expected in (
        (None, original_manifest["source_index_sha256"], "malformed"),
        ("x" * 64, original_manifest["source_index_sha256"], "malformed"),
        ("b" * 64, original_manifest["source_index_sha256"], "mismatch"),
    ):
        split = copy.deepcopy(original_split)
        manifest = copy.deepcopy(original_manifest)
        split["source_index_sha256"] = split_sha
        manifest["source_index_sha256"] = manifest_sha
        split_path.write_text(_json.dumps(split, indent=1))
        manifest_path.write_text(_json.dumps(manifest, indent=1))
        _refresh_artifact_hash(out_dir, "lucas_crop_split.json")
        state, detail = freeze_state(out_dir, expected_git_sha=FIXTURE_GIT_SHA)
        assert state == "corrupt"
        assert expected in detail


def test_manifest_mirror_types_are_exact(monkeypatch, tmp_path):
    import json as _json

    from build_lucas_crop_split import freeze_state

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    manifest_path = out_dir / "lucas_crop_split.MANIFEST.json"
    manifest = _json.loads(manifest_path.read_text())
    manifest["n_holdout"] = float(manifest["n_holdout"])
    manifest_path.write_text(_json.dumps(manifest, indent=1))

    state, detail = freeze_state(out_dir, expected_git_sha=FIXTURE_GIT_SHA)
    assert state == "corrupt"
    assert "MANIFEST.json n_holdout" in detail


def test_hash_refresh_cannot_hide_invalid_extract_values(
    monkeypatch, tmp_path,
):
    """A hash-consistent parquet is still unusable if its values violate
    the pinned crop protocol or point at anything but the qualified file."""
    import pandas as pd
    from build_lucas_crop_split import freeze_state

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    index_path = out_dir / "lucas_crop_distill_index.parquet"
    original = pd.read_parquet(index_path)
    original_manifest = (out_dir / "lucas_crop_split.MANIFEST.json").read_bytes()

    cases = (
        ("unified_class", 0, "outside crop domain"),
        ("row", 9999, "outside crop window"),
        ("col", -5, "outside crop window"),
        ("point_id", 1.5, "non-canonical point_id"),
        ("tile_path", "/missing/spoof.npz", "exact qualified file"),
    )
    for field, value, expected in cases:
        changed = original.copy()
        if field == "point_id":
            changed[field] = changed[field].astype(float)
        changed.loc[changed.index[0], field] = value
        changed.to_parquet(index_path)
        (out_dir / "lucas_crop_split.MANIFEST.json").write_bytes(
            original_manifest
        )
        _refresh_artifact_hash(out_dir, "lucas_crop_distill_index.parquet")
        state, detail = freeze_state(out_dir)
        assert state == "corrupt"
        assert expected in detail


def test_freeze_revalidates_qualified_tile_files(monkeypatch, tmp_path):
    """External tile state is part of usability even though it is not an
    output artifact: missing, empty, or no-longer-qualified files fail."""
    import json as _json

    from build_lucas_crop_split import freeze_state

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    split = _json.loads((out_dir / "lucas_crop_split.json").read_text())
    tile_name = split["holdout_tiles"][0]
    tile_path = data_dir / f"{tile_name}.npz"
    original = tile_path.read_bytes()

    tile_path.unlink()
    state, detail = freeze_state(out_dir)
    assert state == "corrupt"
    assert "missing" in detail

    tile_path.write_bytes(b"")
    state, detail = freeze_state(out_dir)
    assert state == "corrupt"
    assert "empty" in detail

    np.savez(tile_path, s1_vv_vh=np.zeros(1), s1_enrich_v=np.int32(4))
    state, detail = freeze_state(out_dir)
    assert state == "corrupt"
    assert "size changed" in detail or "required_keys" in detail

    tile_path.write_bytes(original)
    assert freeze_state(out_dir)[0] == "frozen"


def test_builder_exits_nonzero_if_post_publish_validation_fails(
    monkeypatch, tmp_path,
):
    """K8s must never record OK merely because MANIFEST was published;
    the builder's own full verifier is its final success gate."""
    import build_lucas_crop_split as blcs

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    real_freeze_state = blcs.freeze_state

    def reject_published(candidate, **kwargs):
        if (
            candidate == out_dir
            and (candidate / "lucas_crop_split.MANIFEST.json").exists()
        ):
            return "corrupt", "injected semantic failure"
        return real_freeze_state(candidate, **kwargs)

    monkeypatch.setattr(blcs, "freeze_state", reject_published)
    with pytest.raises(SystemExit, match="failed self-validation.*injected"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)
    assert (out_dir / "lucas_crop_split.MANIFEST.json").exists()
    assert not (out_dir / ".lucas_crop_split.lock").exists()


def test_root_manifest_stays_absent_until_consumer_projection_is_complete(
    monkeypatch, tmp_path,
):
    """A crash while projecting consumers leaves a partial, never frozen, root."""
    import build_lucas_crop_split as blcs

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    real_publish = blcs._publish

    def fail_mid_projection(writer, destination):
        if (
            destination.parent.name == blcs.CONSUMER_DIR_NAME
            and destination.name == blcs.SPLIT_NAME
        ):
            raise RuntimeError("injected projection crash")
        return real_publish(writer, destination)

    monkeypatch.setattr(blcs, "_publish", fail_mid_projection)
    with pytest.raises(RuntimeError, match="injected projection crash"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)

    assert not (out_dir / blcs.MANIFEST_NAME).exists()
    state, detail = blcs.freeze_state(out_dir, expected_git_sha=FIXTURE_GIT_SHA)
    assert state == "partial", detail
    assert not (out_dir / blcs.LOCK_NAME).exists()


def test_consumer_cannot_freeze_before_root_commit(monkeypatch, tmp_path):
    """A root-marker crash must leave both root and consumer uncommitted."""
    import build_lucas_crop_split as blcs

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    real_publish = blcs._publish

    def crash_before_root_commit(writer, destination):
        if destination == out_dir / blcs.MANIFEST_NAME:
            raise RuntimeError("injected root marker crash")
        return real_publish(writer, destination)

    monkeypatch.setattr(blcs, "_publish", crash_before_root_commit)
    with pytest.raises(RuntimeError, match="injected root marker crash"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)

    root_state, _ = blcs.freeze_state(
        out_dir, expected_git_sha=FIXTURE_GIT_SHA
    )
    consumer_state, _ = blcs.freeze_state(
        out_dir / blcs.CONSUMER_DIR_NAME,
        expected_git_sha=FIXTURE_GIT_SHA,
        include_validator_holdout=False,
    )
    assert root_state == "partial"
    assert consumer_state == "partial"


@pytest.mark.parametrize("target_kind", ["root-manifest", "consumer-split"])
def test_freeze_rechecks_artifact_paths_after_semantic_validation(
    monkeypatch, tmp_path, target_kind,
):
    """A late alias swap cannot pass on bytes captured before the tile sweep."""
    import shutil

    import build_lucas_crop_split as blcs

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    target = (
        out_dir / blcs.MANIFEST_NAME
        if target_kind == "root-manifest"
        else out_dir / blcs.CONSUMER_DIR_NAME / blcs.SPLIT_NAME
    )
    backing = tmp_path / f"{target_kind}-backing"
    real_validate = blcs._validate_frozen_semantics
    swapped = False

    def validate_then_swap(*args, **kwargs):
        nonlocal swapped
        problem = real_validate(*args, **kwargs)
        if not swapped and kwargs.get("include_validator_holdout", True):
            shutil.copyfile(target, backing)
            target.unlink()
            target.symlink_to(backing)
            swapped = True
        return problem

    monkeypatch.setattr(blcs, "_validate_frozen_semantics", validate_then_swap)
    state, detail = blcs.freeze_state(
        out_dir, expected_git_sha=FIXTURE_GIT_SHA
    )

    assert swapped
    assert state == "corrupt", detail
    assert "alias" in detail or "changed" in detail


@pytest.mark.parametrize("consumer_only", [False, True])
def test_final_consumer_capture_batch_rejects_late_alias_swap(
    monkeypatch, tmp_path, consumer_only,
):
    """Swapping an earlier entry while its siblings are read fails closed."""
    import shutil

    import build_lucas_crop_split as blcs

    index, data_dir = _write_lucas_fixture(tmp_path, n_tiles=4)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    consumer_dir = out_dir / blcs.CONSUMER_DIR_NAME
    target = consumer_dir / blcs.INDEX_NAME
    backing = tmp_path / f"late-consumer-index-{consumer_only}"
    real_capture = blcs._capture_regular_at
    consumer_index_captures = 0

    def capture_then_swap(directory_fd, name, **kwargs):
        nonlocal consumer_index_captures
        result = real_capture(directory_fd, name, **kwargs)
        if name == blcs.INDEX_NAME and kwargs.get("label", "").startswith(
            "consumer file"
        ):
            consumer_index_captures += 1
            if consumer_index_captures == 2:
                shutil.copyfile(target, backing)
                target.unlink()
                target.symlink_to(backing)
        return result

    monkeypatch.setattr(blcs, "_capture_regular_at", capture_then_swap)
    verify_dir = consumer_dir if consumer_only else out_dir
    state, detail = blcs.freeze_state(
        verify_dir,
        expected_git_sha=FIXTURE_GIT_SHA,
        include_validator_holdout=not consumer_only,
    )

    assert consumer_index_captures == 2
    assert state == "corrupt", detail
    assert "alias" in detail or "changed" in detail


@pytest.mark.parametrize(
    "damage, expected",
    [
        ("missing-marker", "split marker"),
        ("zero-test", "exactly 71 rows, found 0"),
        ("wrong-count", "exactly 71 rows, found 70"),
        ("duplicate-key", "duplicate logical point keys"),
        ("wrong-tile-identity", "canonical prior-test tiles.*70"),
        ("substituted-marker", "does not equal the canonical 53-tile"),
    ],
)
def test_builder_requires_canonical_prior_test_identity(
    monkeypatch, tmp_path, damage, expected,
):
    import pandas as pd

    index, data_dir = _write_lucas_fixture(tmp_path)
    source = pd.read_parquet(index)
    test_rows = source.index[source["split"] == "test"].tolist()
    if damage == "missing-marker":
        source = source.drop(columns=["split"])
    elif damage == "zero-test":
        source["split"] = "train"
    elif damage == "wrong-count":
        source.loc[test_rows[0], "split"] = "train"
    elif damage == "duplicate-key":
        first, second = test_rows[:2]
        assert source.loc[first, "tile_name"] == source.loc[second, "tile_name"]
        source.loc[second, "point_id"] = source.loc[first, "point_id"]
    elif damage == "wrong-tile-identity":
        source.loc[test_rows[-1], "tile_name"] = "not-canonical-prior-tile"
    elif damage == "substituted-marker":
        source.loc[test_rows[-1], "split"] = "train"
        replacement = source.index[source["split"] == "train"][0]
        source.loc[replacement, "split"] = "test"
    source.to_parquet(index)
    out_dir = tmp_path / "out"

    with pytest.raises(SystemExit, match=expected):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)
    assert not (out_dir / "lucas_crop_split.MANIFEST.json").exists()


@pytest.mark.parametrize(
    "bad_point_id",
    [" 123", "123 ", "+123", "-1", "01", "1.0", ""],
)
def test_builder_rejects_noncanonical_source_point_id(
    monkeypatch, tmp_path, bad_point_id,
):
    import pandas as pd

    index, data_dir = _write_lucas_fixture(tmp_path)
    source = pd.read_parquet(index)
    source.loc[source.index[0], "point_id"] = bad_point_id
    source.to_parquet(index)

    with pytest.raises(SystemExit, match="non-canonical point_id"):
        _run_split_builder(monkeypatch, index, data_dir, tmp_path / "out")


@pytest.mark.parametrize("bad_point_id", [True, 1.0, -1, None])
def test_point_id_normalizer_rejects_noninteger_types(bad_point_id):
    from build_lucas_crop_split import _point_id_values

    values, problem = _point_id_values([bad_point_id], "source")
    assert values is None
    assert "point_id" in problem


def test_fixture_models_real_tile_level_prior_identity(tmp_path):
    import pandas as pd

    index, _ = _write_lucas_fixture(tmp_path)
    source = pd.read_parquet(index)
    prior = source[source["split"] == "test"]
    assert len(prior) == 71
    assert prior["tile_name"].nunique() == 24
    assert prior["point_id"].map(type).eq(str).all()


def test_freeze_persists_canonical_prior_anchor_and_tile_bytes(
    monkeypatch, tmp_path,
):
    import hashlib
    import json as _json

    from build_lucas_crop_split import (
        PRIOR_TEST_SOURCE_REF,
        PRIOR_TEST_TILES_SHA256,
        TILE_INVENTORY_FORMAT,
        _key_digest,
        _tile_inventory_digest,
    )

    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    split = _json.loads((out_dir / "lucas_crop_split.json").read_text())
    manifest = _json.loads(
        (out_dir / "lucas_crop_split.MANIFEST.json").read_text()
    )

    prior_keys = [
        (record["tile_name"], record["point_id"])
        for record in split["prior_test_plots"]
    ]
    assert split["prior_test_point_count"] == len(prior_keys) == 71
    assert split["prior_test_tile_count"] == len(split["prior_test_tiles"]) == 53
    assert split["prior_test_source_ref"] == PRIOR_TEST_SOURCE_REF
    assert split["prior_test_tiles_sha256"] == PRIOR_TEST_TILES_SHA256
    assert split["prior_test_keys_sha256"] == _key_digest(prior_keys)
    assert manifest["prior_test_tiles_sha256"] == PRIOR_TEST_TILES_SHA256
    assert split["tile_inventory_format"] == TILE_INVENTORY_FORMAT

    for side, key_field in (
        ("distill", "plots"),
        ("validator_holdout", "holdout_plots"),
    ):
        inventory = split[f"{side}_tile_inventory"]
        expected_tiles = sorted({
            record["tile_name"] for record in split[key_field]
        })
        assert [record["tile_name"] for record in inventory] == expected_tiles
        for record in inventory:
            path = Path(record["tile_path"])
            assert record["file_name"] == f"{record['tile_name']}.npz"
            assert path == data_dir / record["file_name"]
            assert record["size"] == path.stat().st_size
            assert record["sha256"] == hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        digest_field = f"{side}_input_data_sha256"
        assert split[digest_field] == _tile_inventory_digest(inventory)
        assert manifest[digest_field] == split[digest_field]


def test_consumer_projection_is_exact_and_byte_identical(monkeypatch, tmp_path):
    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    bundle = out_dir / "crop_consumer"
    expected = {
        "lucas_crop_distill_index.parquet",
        "lucas_crop_split.json",
        "lucas_crop_split.MANIFEST.json",
    }

    assert {path.name for path in bundle.iterdir()} == expected
    assert "lucas_crop_validator_holdout_index.parquet" not in expected
    for name in expected:
        assert (bundle / name).read_bytes() == (out_dir / name).read_bytes()
    _run_split_builder(monkeypatch, index, data_dir, bundle, "--verify-consumer")


@pytest.mark.parametrize("damage", ["tamper", "partial", "extra"])
def test_full_verify_rejects_consumer_projection_damage(
    monkeypatch, tmp_path, damage,
):
    from build_lucas_crop_split import freeze_state

    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    bundle = out_dir / "crop_consumer"
    if damage == "tamper":
        projected = bundle / "lucas_crop_distill_index.parquet"
        projected.write_bytes(projected.read_bytes() + b"tamper")
    elif damage == "partial":
        (bundle / "lucas_crop_split.json").unlink()
    else:
        (bundle / "unexpected.txt").write_text("not part of projection")

    state, detail = freeze_state(out_dir, expected_git_sha=FIXTURE_GIT_SHA)
    assert state == "corrupt"
    assert "consumer projection" in detail


@pytest.mark.parametrize("damage", ["extra", "symlink"])
def test_consumer_verify_rejects_non_exact_bundle(
    monkeypatch, tmp_path, damage,
):
    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    bundle = out_dir / "crop_consumer"
    if damage == "extra":
        (bundle / "unexpected.txt").write_text("unexpected")
    else:
        projected = bundle / "lucas_crop_split.json"
        projected.unlink()
        projected.symlink_to(out_dir / "lucas_crop_split.json")

    with pytest.raises(SystemExit, match="INVALID.*consumer"):
        _run_split_builder(
            monkeypatch, index, data_dir, bundle, "--verify-consumer"
        )


def test_stale_consumer_projection_is_partial_freeze(monkeypatch, tmp_path):
    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    bundle = out_dir / "crop_consumer"
    bundle.mkdir(parents=True)
    (bundle / "lucas_crop_distill_index.parquet").write_bytes(b"partial")

    with pytest.raises(SystemExit, match="PARTIAL freeze"):
        _run_split_builder(monkeypatch, index, data_dir, out_dir)
    assert not (out_dir / "lucas_crop_split.MANIFEST.json").exists()


def test_consumer_verify_does_not_read_any_npz_bytes(monkeypatch, tmp_path):
    """The extractor verifies a tile during its one data read; this cheap
    preflight validates only the sealed distill inventory metadata."""
    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    bundle = out_dir / "crop_consumer"
    import build_lucas_crop_split as blcs

    def reject_tile(*_args, **_kwargs):
        raise AssertionError("consumer preflight touched NPZ bytes")

    with monkeypatch.context() as guard:
        guard.setattr(blcs, "_capture_tile_at", reject_tile)
        _run_split_builder(
            monkeypatch, index, data_dir, bundle, "--verify-consumer"
        )


@pytest.mark.parametrize("damage", ["same-schema", "missing", "replaced"])
def test_full_verify_binds_exact_distill_npz_bytes(
    monkeypatch, tmp_path, damage,
):
    import json as _json

    from build_lucas_crop_split import freeze_state

    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    split = _json.loads((out_dir / "lucas_crop_split.json").read_text())
    inventory = split["distill_tile_inventory"]
    target = Path(inventory[0]["tile_path"])
    if damage == "same-schema":
        np.savez(
            target,
            s1_vv_vh=np.array([9876], dtype=np.float32),
            s1_enrich_v=np.int32(4),
            tessera=np.array([9876], dtype=np.float32),
        )
    elif damage == "missing":
        target.unlink()
    else:
        target.write_bytes(Path(inventory[1]["tile_path"]).read_bytes())

    # Cheap consumer preflight does not sweep the dataset.
    _run_split_builder(
        monkeypatch,
        index,
        data_dir,
        out_dir / "crop_consumer",
        "--verify-consumer",
    )
    state, detail = freeze_state(out_dir, expected_git_sha=FIXTURE_GIT_SHA)
    assert state == "corrupt"
    assert "inventoried tile" in detail


def test_full_verify_binds_exact_validator_npz_bytes(monkeypatch, tmp_path):
    import json as _json

    from build_lucas_crop_split import freeze_state

    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    split = _json.loads((out_dir / "lucas_crop_split.json").read_text())
    target = Path(split["validator_holdout_tile_inventory"][0]["tile_path"])
    np.savez(
        target,
        s1_vv_vh=np.array([1234], dtype=np.float32),
        s1_enrich_v=np.int32(4),
        tessera=np.array([1234], dtype=np.float32),
    )

    _run_split_builder(
        monkeypatch,
        index,
        data_dir,
        out_dir / "crop_consumer",
        "--verify-consumer",
    )
    state, detail = freeze_state(out_dir, expected_git_sha=FIXTURE_GIT_SHA)
    assert state == "corrupt"
    assert "inventoried tile" in detail


@pytest.mark.parametrize("field", ["seed", "trial_offset"])
def test_full_verify_recomputes_seed_and_trial(monkeypatch, tmp_path, field):
    import json as _json

    from build_lucas_crop_split import freeze_state

    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    split_path = out_dir / "lucas_crop_split.json"
    split = _json.loads(split_path.read_text())
    split[field] = (
        split[field] + 1 if field == "seed" else (split[field] + 1) % 50
    )
    split_path.write_text(_json.dumps(split, indent=1))
    _refresh_artifact_hash(out_dir, "lucas_crop_split.json")

    state, detail = freeze_state(out_dir, expected_git_sha=FIXTURE_GIT_SHA)
    assert state == "corrupt"
    assert field in detail or "protocol seed" in detail


def test_builder_rejects_sparse_distill_class_groups(monkeypatch, tmp_path):
    import pandas as pd

    index, data_dir = _write_lucas_fixture(tmp_path)
    source = pd.read_parquet(index)
    crop_tiles = sorted({
        tile
        for tile in source.loc[
            source["unified_class"].isin(range(11, 22)), "tile_name"
        ]
        if tile != CANONICAL_PRIOR_CROP_TILE
    })
    thin_class_tiles = set(crop_tiles[:5])
    sparse = (
        (source["unified_class"] == 11)
        & ~source["tile_name"].isin(thin_class_tiles)
    )
    source.loc[sparse, "unified_class"] = 12
    source.to_parquet(index)

    with pytest.raises(SystemExit, match="scoreable grouped partition"):
        _run_split_builder(monkeypatch, index, data_dir, tmp_path / "out")


def test_full_verify_rejects_coherent_wrong_repartition(monkeypatch, tmp_path):
    """Even a fully rehashed, schema-valid alternate 70/30 partition is not
    the first valid seeded partition and therefore is not this protocol."""
    import json as _json

    import pandas as pd
    from build_lucas_crop_split import (
        EXTRACT_COLUMNS,
        KEY_DIGEST_FIELDS,
        _key_digest,
        _partition_digest,
        _tile_inventory_digest,
        freeze_state,
    )

    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    split_path = out_dir / "lucas_crop_split.json"
    manifest_path = out_dir / "lucas_crop_split.MANIFEST.json"
    split = _json.loads(split_path.read_text())
    manifest = _json.loads(manifest_path.read_text())
    dist_path = out_dir / "lucas_crop_distill_index.parquet"
    hold_path = out_dir / "lucas_crop_validator_holdout_index.parquet"
    dist = pd.read_parquet(dist_path)
    hold = pd.read_parquet(hold_path)
    forced = set(split["forced_holdout_tiles_from_prior_split"])
    dist_tile = min(set(dist["tile_name"]))
    random_hold_tile = next(
        tile for tile in sorted(set(hold["tile_name"])) if tile not in forced
    )
    new_dist = pd.concat([
        dist[dist["tile_name"] != dist_tile],
        hold[hold["tile_name"] == random_hold_tile],
    ]).sort_values(["tile_name", "point_id"]).loc[:, EXTRACT_COLUMNS]
    new_hold = pd.concat([
        hold[hold["tile_name"] != random_hold_tile],
        dist[dist["tile_name"] == dist_tile],
    ]).sort_values(["tile_name", "point_id"]).loc[:, EXTRACT_COLUMNS]
    new_dist.to_parquet(dist_path)
    new_hold.to_parquet(hold_path)

    def keys(frame):
        return [
            (str(tile), int(point))
            for tile, point in frame[["tile_name", "point_id"]].itertuples(
                index=False
            )
        ]

    distill_keys = keys(new_dist)
    holdout_keys = keys(new_hold)
    split["plots"] = [
        {"tile_name": tile, "point_id": point}
        for tile, point in distill_keys
    ]
    split["holdout_plots"] = [
        {"tile_name": tile, "point_id": point}
        for tile, point in holdout_keys
    ]
    split["holdout_tiles"] = sorted(set(new_hold["tile_name"]))
    digests = {
        "qualified_keys_sha256": _key_digest(distill_keys + holdout_keys),
        "distill_keys_sha256": _key_digest(distill_keys),
        "holdout_keys_sha256": _key_digest(holdout_keys),
        "partition_sha256": _partition_digest(distill_keys, holdout_keys),
    }
    assert set(digests) == set(KEY_DIGEST_FIELDS)
    split.update(digests)
    manifest.update(digests)

    inventory_by_tile = {
        record["tile_name"]: record
        for record in (
            split["distill_tile_inventory"]
            + split["validator_holdout_tile_inventory"]
        )
    }
    new_dist_inventory = [
        inventory_by_tile[tile] for tile in sorted(set(new_dist["tile_name"]))
    ]
    new_hold_inventory = [
        inventory_by_tile[tile] for tile in sorted(set(new_hold["tile_name"]))
    ]
    split["distill_tile_inventory"] = new_dist_inventory
    split["validator_holdout_tile_inventory"] = new_hold_inventory
    for field, value in (
        ("distill_input_data_sha256", _tile_inventory_digest(new_dist_inventory)),
        (
            "validator_holdout_input_data_sha256",
            _tile_inventory_digest(new_hold_inventory),
        ),
    ):
        split[field] = value
        manifest[field] = value
    split_path.write_text(_json.dumps(split, indent=1))
    manifest_path.write_text(_json.dumps(manifest, indent=1))
    for artifact in (
        "lucas_crop_distill_index.parquet",
        "lucas_crop_validator_holdout_index.parquet",
        "lucas_crop_split.json",
    ):
        _refresh_artifact_hash(out_dir, artifact)

    state, detail = freeze_state(out_dir, expected_git_sha=FIXTURE_GIT_SHA)
    assert state == "corrupt"
    assert "deterministic seeded selection" in detail


def test_builder_rejects_tile_path_alias(monkeypatch, tmp_path):
    index, data_dir = _write_lucas_fixture(tmp_path)
    target = data_dir / "tile00.npz"
    real = data_dir / "tile00-real.npz"
    target.rename(real)
    target.symlink_to(real.name)

    with pytest.raises(SystemExit, match="must not be a symlink"):
        _run_split_builder(monkeypatch, index, data_dir, tmp_path / "out")


def test_builder_rejects_hardlinked_tile(monkeypatch, tmp_path):
    """One inode must never stand for a frozen tile plus an external alias."""
    import os

    index, data_dir = _write_lucas_fixture(tmp_path)
    target = data_dir / "tile00.npz"
    os.link(target, data_dir / "external-alias.npz")

    with pytest.raises(SystemExit, match="link count 1"):
        _run_split_builder(monkeypatch, index, data_dir, tmp_path / "out")


def test_tile_hash_and_qualification_share_one_capture(monkeypatch, tmp_path):
    """A pathname swap after read cannot split identity from qualification."""
    import hashlib
    import os

    import build_lucas_crop_split as blcs

    data_dir = tmp_path / "tiles"
    data_dir.mkdir()
    tile = data_dir / "tile00.npz"
    replacement = data_dir / "replacement.npz"
    np.savez(
        tile,
        s1_vv_vh=np.zeros(1),
        s1_enrich_v=np.int32(4),
        tessera=np.zeros(1),
    )
    np.savez(replacement, s1_vv_vh=np.ones(1), s1_enrich_v=np.int32(0))
    original = tile.read_bytes()
    real_qualify = blcs._qualify_npz_bytes

    def swap_then_qualify(payload, keys):
        os.replace(replacement, tile)
        return real_qualify(payload, keys)

    monkeypatch.setattr(blcs, "_qualify_npz_bytes", swap_then_qualify)
    directory_fd = blcs._open_directory_tree(data_dir)
    try:
        record, qualifies = blcs._capture_tile_at(
            directory_fd,
            tile_name="tile00",
            data_dir=data_dir,
            required_keys=blcs.required_npz_keys(),
        )
    finally:
        os.close(directory_fd)

    assert qualifies is True
    assert record["size"] == len(original)
    assert record["sha256"] == hashlib.sha256(original).hexdigest()
    assert tile.read_bytes() != original


def test_tile_capture_rejects_path_swap_during_read(monkeypatch, tmp_path):
    import os

    import build_lucas_crop_split as blcs

    data_dir = tmp_path / "tiles"
    data_dir.mkdir()
    tile = data_dir / "tile00.npz"
    replacement = data_dir / "replacement.npz"
    np.savez(
        tile,
        s1_vv_vh=np.zeros(1),
        s1_enrich_v=np.int32(4),
        tessera=np.zeros(1),
    )
    np.savez(
        replacement,
        s1_vv_vh=np.ones(1),
        s1_enrich_v=np.int32(4),
        tessera=np.ones(1),
    )
    real_read = blcs.os.read
    swapped = False

    def read_then_swap(fd, amount):
        nonlocal swapped
        chunk = real_read(fd, amount)
        if chunk and not swapped:
            os.replace(replacement, tile)
            swapped = True
        return chunk

    monkeypatch.setattr(blcs.os, "read", read_then_swap)
    directory_fd = blcs._open_directory_tree(data_dir)
    try:
        with pytest.raises(blcs.UnsafeFileError, match="changed while"):
            blcs._capture_tile_at(
                directory_fd,
                tile_name="tile00",
                data_dir=data_dir,
                required_keys=blcs.required_npz_keys(),
            )
    finally:
        os.close(directory_fd)


def test_publish_does_not_follow_precreated_temp_symlink(monkeypatch, tmp_path):
    import build_lucas_crop_split as blcs

    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"do not overwrite")
    destination = tmp_path / "artifact.json"
    first_token = "1" * 32
    second_token = "2" * 32
    trap = tmp_path / f".{destination.name}.tmp.{first_token}"
    trap.symlink_to(victim)
    tokens = iter((first_token, second_token, "3" * 32))
    monkeypatch.setattr(blcs.secrets, "token_hex", lambda _size: next(tokens))

    digest = blcs._publish(
        lambda handle: handle.write(b"sealed payload"), destination
    )

    assert digest == (
        "95944372a174d98cee43988d4946c1e7"
        "eff5bf8b40b79e14202f11468889dc90"
    )
    assert destination.read_bytes() == b"sealed payload"
    assert victim.read_bytes() == b"do not overwrite"
    assert trap.is_symlink()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        blcs._publish(lambda handle: handle.write(b"new"), destination)


def test_builder_rejects_symlinked_output_parent(monkeypatch, tmp_path):
    index, data_dir = _write_lucas_fixture(tmp_path)
    real_parent = tmp_path / "real-output-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-output-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(SystemExit, match="secure output directory"):
        _run_split_builder(
            monkeypatch, index, data_dir, linked_parent / "split"
        )
    assert not (real_parent / "split").exists()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_full_verify_rejects_linked_freeze_artifacts(
    monkeypatch, tmp_path, link_kind,
):
    """Byte-identical links cannot impersonate any root or consumer file."""
    import os
    import shutil

    from build_lucas_crop_split import freeze_state

    index, data_dir = _write_lucas_fixture(tmp_path)
    pristine = tmp_path / "pristine"
    _run_split_builder(monkeypatch, index, data_dir, pristine)
    targets = (
        Path("lucas_crop_split.MANIFEST.json"),
        Path("lucas_crop_distill_index.parquet"),
        Path("lucas_crop_validator_holdout_index.parquet"),
        Path("lucas_crop_split.json"),
        Path("crop_consumer/lucas_crop_split.MANIFEST.json"),
        Path("crop_consumer/lucas_crop_distill_index.parquet"),
        Path("crop_consumer/lucas_crop_split.json"),
    )
    for position, relative in enumerate(targets):
        case = tmp_path / f"{link_kind}-{position}"
        shutil.copytree(pristine, case)
        target = case / relative
        backing = tmp_path / f"{link_kind}-backing-{position}"
        shutil.copyfile(target, backing)
        target.unlink()
        if link_kind == "symlink":
            target.symlink_to(backing)
        else:
            os.link(backing, target)

        state, detail = freeze_state(
            case, expected_git_sha=FIXTURE_GIT_SHA
        )
        assert state == "corrupt", (relative, detail)
        assert any(
            marker in detail for marker in ("alias", "link count", "symlink")
        ), (relative, detail)


@pytest.mark.parametrize("consumer_only", [False, True])
def test_freeze_verifier_opens_files_read_only(
    monkeypatch, tmp_path, consumer_only,
):
    import os

    import build_lucas_crop_split as blcs

    index, data_dir = _write_lucas_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _run_split_builder(monkeypatch, index, data_dir, out_dir)
    verify_dir = out_dir / "crop_consumer" if consumer_only else out_dir
    real_open = blcs.os.open
    mutation_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC

    def read_only_open(path, flags, *args, **kwargs):
        assert not flags & mutation_flags, (path, flags)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(blcs.os, "open", read_only_open)
    state, detail = blcs.freeze_state(
        verify_dir,
        expected_git_sha=FIXTURE_GIT_SHA,
        include_validator_holdout=not consumer_only,
    )
    assert state == "frozen", detail
