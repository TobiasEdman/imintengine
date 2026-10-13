"""Regression tests for full-state training resume."""

from pathlib import Path

import pytest
import torch

from imint.training.trainer import LULCTrainer
from imint.metrics import SEGMENTATION_METRIC_VERSION


class _Scheduler:
    def load_state_dict(self, state):
        self.state = state


def _trainer_with_linear_model() -> LULCTrainer:
    trainer = LULCTrainer.__new__(LULCTrainer)
    trainer.model = torch.nn.Linear(2, 1)
    trainer.device = torch.device("cpu")
    trainer._training_log = []
    return trainer


def _checkpoint(model, *, extra_model_state=None):
    state = dict(model.state_dict())
    state.update(extra_model_state or {})
    return {
        "metric_version": SEGMENTATION_METRIC_VERSION,
        "epoch": 25,
        "model_state_dict": state,
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "step": 100,
        "best_metric": 0.4,
        "best_miou": 0.4192,
        "best_epoch": 25,
        "patience_counter": 0,
    }


def test_resume_migrates_lazy_croma_normalizer_buffers(tmp_path: Path):
    trainer = _trainer_with_linear_model()
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    scheduler = _Scheduler()
    extras = {
        "_norm_croma.s2_mean": torch.zeros(1, 12, 1, 1),
        "_norm_croma.s2_std": torch.ones(1, 12, 1, 1),
        "_norm_croma.s1_mean": torch.zeros(1, 2, 1, 1),
        "_norm_croma.s1_std": torch.ones(1, 2, 1, 1),
    }
    ckpt = _checkpoint(trainer.model, extra_model_state=extras)
    ckpt["optimizer_state_dict"] = optimizer.state_dict()
    path = tmp_path / "last_checkpoint.pt"
    torch.save(ckpt, path)

    resumed = trainer._load_resume_checkpoint(path, optimizer, scheduler)

    assert resumed["epoch"] == 25
    assert resumed["best_miou"] == pytest.approx(0.4192)


def test_resume_still_rejects_unknown_model_keys(tmp_path: Path):
    trainer = _trainer_with_linear_model()
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    scheduler = _Scheduler()
    ckpt = _checkpoint(
        trainer.model,
        extra_model_state={"unexpected.learned_weight": torch.ones(1)},
    )
    ckpt["optimizer_state_dict"] = optimizer.state_dict()
    path = tmp_path / "last_checkpoint.pt"
    torch.save(ckpt, path)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        trainer._load_resume_checkpoint(path, optimizer, scheduler)


@pytest.mark.parametrize("version", [None, "legacy-v1", "future-v3"])
def test_incompatible_resume_does_not_mutate_state(tmp_path, version):
    trainer = _trainer_with_linear_model()
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    scheduler = _Scheduler()
    scheduler.state = {"untouched": True}
    before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    log = trainer._training_log
    checkpoint = _checkpoint(trainer.model)
    checkpoint["model_state_dict"] = {key: value + 5 for key, value in before.items()}
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] = 9.0
    checkpoint["training_log"] = {"replaced": True}
    if version is None:
        checkpoint.pop("metric_version")
    else:
        checkpoint["metric_version"] = version
    path = tmp_path / "resume.pt"
    torch.save(checkpoint, path)
    saved = path.read_bytes()
    with pytest.raises(ValueError, match="Checkpoint metric version"):
        trainer._load_resume_checkpoint(path, optimizer, scheduler)
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert scheduler.state == {"untouched": True}
    assert trainer._training_log is log
    assert path.read_bytes() == saved


@pytest.mark.parametrize("automatic", [True, False])
def test_train_preflight_rejects_before_setup(tmp_path, automatic):
    from imint.training.config import TrainingConfig

    out = tmp_path / "output"
    path = out / "last_checkpoint.pt" if automatic else tmp_path / "legacy.pt"
    path.parent.mkdir(exist_ok=True)
    torch.save({"best_metric": 0.25}, path)
    trainer = _trainer_with_linear_model()
    trainer.config = TrainingConfig(checkpoint_dir=str(out))
    if not automatic:
        trainer.config.resume_from_checkpoint = str(path)
    before = set(tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="Checkpoint metric version"):
        trainer.train(None, None)
    assert set(tmp_path.rglob("*")) == before


def test_resume_checkpoint_version_roundtrip(tmp_path):
    trainer = _trainer_with_linear_model()
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    path = tmp_path / "last_checkpoint.pt"
    trainer._save_resume_checkpoint(path, 3, optimizer, scheduler, 12, 0.8, 0.8, 2, 1)
    saved = torch.load(path, weights_only=False)
    assert saved["metric_version"] == SEGMENTATION_METRIC_VERSION
    resumed = trainer._load_resume_checkpoint(path, optimizer, scheduler)
    assert resumed["best_metric"] == 0.8
    assert resumed["best_epoch"] == 2
    assert resumed["patience_counter"] == 1


def test_unversioned_weight_only_load_keeps_selection_state(tmp_path):
    from imint.training.config import TrainingConfig

    trainer = _trainer_with_linear_model()
    trainer.config = TrainingConfig()
    log = trainer._training_log
    weights = {key: value + 1 for key, value in trainer.model.state_dict().items()}
    path = tmp_path / "historical.pt"
    torch.save({"state_dict": weights, "best_metric": 0.25}, path)
    trainer._load_spectral_checkpoint(path)
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, weights[key])
    assert trainer._training_log is log


@pytest.mark.parametrize("automatic", [True, False])
@pytest.mark.parametrize("version", [None, "legacy-v1"])
def test_constructor_refuses_before_rng_model_and_output_mutation(tmp_path, monkeypatch, automatic, version):
    import random
    import numpy as np
    from imint.training.config import TrainingConfig

    out = tmp_path / "output"
    path = out / "last_checkpoint.pt" if automatic else tmp_path / "legacy.pt"
    path.parent.mkdir(exist_ok=True)
    checkpoint = {"best_metric": 0.25}
    if version is not None:
        checkpoint["metric_version"] = version
    torch.save(checkpoint, path)
    config = TrainingConfig(checkpoint_dir=str(out))
    if not automatic:
        config.resume_from_checkpoint = str(path)
    def forbidden(*args, **kwargs):
        raise AssertionError("model or device setup started before resume validation")
    monkeypatch.setattr(LULCTrainer, "_build_model", forbidden)
    monkeypatch.setattr(LULCTrainer, "_resolve_device", forbidden)
    before_files = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    before_paths = set(tmp_path.rglob("*"))
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(ValueError, match="Checkpoint metric version"):
        LULCTrainer(config)
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_rng[0]
    np.testing.assert_array_equal(after_numpy[1], numpy_rng[1])
    assert after_numpy[2:] == numpy_rng[2:]
    torch.testing.assert_close(torch.random.get_rng_state(), torch_rng)
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert set(tmp_path.rglob("*")) == before_paths
    assert {p: p.read_bytes() for p in before_files} == before_files
