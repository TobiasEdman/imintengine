"""Regression tests for full-state training resume."""

from pathlib import Path

import pytest
import torch

from imint.training.trainer import LULCTrainer


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
