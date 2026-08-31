"""ERA5/ERA5-Land AUX wiring, cache, and experiment regressions."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from imint.training.config import TrainingConfig
from imint.training.era5_aux import (
    ERA5_API_CELL_COORD_ATOL_DEGREES,
    ERA5_REQUEST_SCHEMA,
    era5_api_cell_coords_match,
    era5_grid_context,
    fetch_era5_land_growing_season,
)
from imint.training.trainer import LULCTrainer
from imint.training.unified_dataset import ERA5_AUX_CHANNELS, UnifiedDataset
from scripts.fetch_era5_aux import _valid_sidecar, tile_context

ERA5_NAMES = (
    "era5_t2m_mean", "era5_tp_sum", "era5_swvl1_mean",
    "era5_ssrd_sum", "era5_gdd",
)


def test_era5_channels_are_opt_in_and_trailing():
    assert not (ERA5_AUX_CHANNELS & set(TrainingConfig().enabled_aux_names))
    cfg = TrainingConfig(enable_height_channel=True, enable_era5_channels=True)
    assert cfg.enabled_aux_names[0] == "height"
    assert tuple(cfg.enabled_aux_names[-5:]) == ERA5_NAMES


def _dataset(tmp_path, mode: str) -> UnifiedDataset:
    ds = UnifiedDataset.__new__(UnifiedDataset)
    ds.aux_channel_names = ERA5_NAMES
    ds.era5_mode = mode
    ds.era5_dir = tmp_path if mode == "treatment" else None
    return ds


def _sidecar_values(tile_name: str = "real.npz") -> dict:
    grid = era5_grid_context(59.3, 18.1)
    return {
        "tile_name": tile_name, "year": 2022,
        "cutoff_date": "2022-08-12", "lat": 59.3, "lon": 18.1,
        "era5_t2m_mean": 17.0, "era5_tp_sum": 450.0,
        "era5_swvl1_mean": 0.33, "era5_ssrd_sum": 4000.0,
        "era5_gdd": 1600.0,
        "era5_request_lat": grid["request_lat"],
        "era5_request_lon": grid["request_lon"],
        "era5_land_cell_lat": grid["land_cell"]["lat"],
        "era5_land_cell_lon": grid["land_cell"]["lon"],
        "era5_atmosphere_cell_lat": grid["atmosphere_cell"]["lat"],
        "era5_atmosphere_cell_lon": grid["atmosphere_cell"]["lon"],
    }


def test_treatment_changes_inputs_and_missing_treatment_fails(tmp_path):
    np.savez(tmp_path / "real.npz", **_sidecar_values())
    tile = {"year": np.asarray(2022), "dates": np.asarray([
        "2021-09-20", "2022-05-10", "2022-07-01", "2022-08-12",
    ])}
    treatment = _dataset(tmp_path, "treatment")._load_aux_channels(
        tile, 3, 4, "real.npz",
    )
    control = _dataset(tmp_path, "control")._load_aux_channels(
        tile, 3, 4, "real.npz",
    )
    assert treatment.shape == (5, 3, 4)
    assert np.allclose(treatment[:, 0, 0], [1.0] * 5)
    assert np.allclose(control, 0.0)
    with pytest.raises(FileNotFoundError, match="Missing ERA5 treatment"):
        _dataset(tmp_path, "treatment")._load_aux_channels(
            tile, 3, 4, "missing.npz",
        )


def test_treatment_rejects_weather_for_another_prithvi_location(tmp_path):
    np.savez(tmp_path / "real.npz", **_sidecar_values())
    tile = {
        "year": np.asarray(2022),
        "dates": np.asarray([
            "2021-09-20", "2022-05-10", "2022-07-01", "2022-08-12",
        ]),
    }

    with pytest.raises(ValueError, match="sidecar location mismatch"):
        _dataset(tmp_path, "treatment")._load_aux_channels(
            tile,
            3,
            4,
            "real.npz",
            expected_location=(59.4, 18.1),
        )


def test_sidecar_validation_binds_tile_location(tmp_path):
    sidecar = tmp_path / "tile.npz"
    np.savez(sidecar, **_sidecar_values("tile.npz"))
    assert _valid_sidecar(
        sidecar, "tile.npz", 2022, "2022-08-12", 59.3, 18.1,
    )
    assert not _valid_sidecar(
        sidecar, "tile.npz", 2022, "2022-08-12", 59.4, 18.1,
    )


def test_api_cell_coordinate_match_absorbs_only_representation_noise():
    assert era5_api_cell_coords_match(
        59.300003, 18.100006, 59.3, 18.1,
    )
    assert not era5_api_cell_coords_match(
        59.3 + 2 * ERA5_API_CELL_COORD_ATOL_DEGREES,
        18.1,
        59.3,
        18.1,
    )
    assert not era5_api_cell_coords_match(59.4, 18.1, 59.3, 18.1)


def test_sidecar_accepts_api_float_noise_but_keeps_request_identity_strict(
    tmp_path,
):
    tile = {
        "year": np.asarray(2022),
        "dates": np.asarray([
            "2021-09-20", "2022-05-10", "2022-07-01", "2022-08-12",
        ]),
    }
    values = _sidecar_values()
    values["era5_land_cell_lat"] = 59.300003
    values["era5_land_cell_lon"] = 18.100006
    values["era5_atmosphere_cell_lat"] = 59.250004
    values["era5_atmosphere_cell_lon"] = 18.000006
    sidecar = tmp_path / "real.npz"
    np.savez(sidecar, **values)

    assert _valid_sidecar(
        sidecar, "real.npz", 2022, "2022-08-12", 59.3, 18.1,
    )
    treatment = _dataset(tmp_path, "treatment")._load_aux_channels(
        tile, 3, 4, "real.npz",
    )
    assert treatment.shape == (5, 3, 4)

    values["era5_request_lat"] = 59.300003
    np.savez(sidecar, **values)
    assert not _valid_sidecar(
        sidecar, "real.npz", 2022, "2022-08-12", 59.3, 18.1,
    )
    with pytest.raises(ValueError, match="inconsistent grid cells"):
        _dataset(tmp_path, "treatment")._load_aux_channels(
            tile, 3, 4, "real.npz",
        )


def test_sidecar_rejects_different_api_cell(tmp_path):
    values = _sidecar_values()
    values["era5_land_cell_lat"] = 59.4
    sidecar = tmp_path / "real.npz"
    np.savez(sidecar, **values)
    assert not _valid_sidecar(
        sidecar, "real.npz", 2022, "2022-08-12", 59.3, 18.1,
    )


def _api_payloads():
    daily_times = [
        (datetime(2022, 4, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(183)
    ]
    hourly_times = [
        (datetime(2022, 4, 1) + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M")
        for i in range(4392)
    ]
    land = {
        # Open-Meteo returns float32-like coordinates even when the selected
        # canonical ERA5-Land cell is exactly 59.3, 18.1.
        "latitude": 59.300003,
        "longitude": 18.100006,
        "daily": {"time": daily_times, "temperature_2m_mean": [10.0] * 183},
        "hourly": {"time": hourly_times, "soil_moisture_0_to_7cm": [0.3] * 4392},
    }
    atmosphere = {"latitude": 59.250004, "longitude": 18.000006, "hourly": {
        "time": hourly_times, "precipitation": [1.0] * 4392,
        "shortwave_radiation": [100.0] * 4392,
    }}
    return land, atmosphere


def test_fetch_summary_is_cutoff_bounded_and_cache_is_validated(tmp_path, monkeypatch):
    land, atmosphere = _api_payloads()

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    calls = []

    def get(*args, **kwargs):
        calls.append(kwargs["params"])
        payload = land if kwargs["params"]["models"] == "era5_land" else atmosphere
        return Response(payload)

    monkeypatch.setattr("requests.get", get)
    first = fetch_era5_land_growing_season(
        59.34, 18.06, 2022, cache_dir=tmp_path, cutoff_date="2022-04-02",
    )
    second = fetch_era5_land_growing_season(
        59.31, 18.09, 2022, cache_dir=tmp_path, cutoff_date="2022-04-02",
    )
    assert first == second
    assert len(calls) == 2
    assert {call["models"] for call in calls} == {"era5_land", "era5"}
    assert all(call["cell_selection"] == "nearest" for call in calls)
    assert first["era5_t2m_mean"] == 10.0
    assert first["era5_tp_sum"] == 48.0
    assert first["era5_ssrd_sum"] == pytest.approx(17.28)
    assert first["era5_gdd"] == 10.0
    cached = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert cached["schema"] == ERA5_REQUEST_SCHEMA

    cached["atmosphere"]["hourly"]["precipitation"] = [1.0]
    next(tmp_path.glob("*.json")).write_text(json.dumps(cached))
    fetch_era5_land_growing_season(
        59.34, 18.06, 2022, cache_dir=tmp_path, cutoff_date="2022-04-02",
    )
    assert len(calls) == 4


def test_fetch_rejects_duplicate_or_unordered_timestamps(tmp_path, monkeypatch):
    land, atmosphere = _api_payloads()
    land["hourly"]["time"][1] = land["hourly"]["time"][0]

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response(
        land if kwargs["params"]["models"] == "era5_land" else atmosphere
    ))
    monkeypatch.setattr("time.sleep", lambda *_: None)
    with pytest.raises(ValueError, match="timestamps"):
        fetch_era5_land_growing_season(
            59.3, 18.1, 2022, cache_dir=tmp_path, cutoff_date="2022-04-02",
        )


def test_aux_only_training_unfreezes_multilevel_aux_path():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.decode_head = torch.nn.Linear(2, 2)
            self.lidar_branch = torch.nn.Linear(2, 2)
            self.gated_fusions = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

    trainer = LULCTrainer.__new__(LULCTrainer)
    trainer.model = Model()
    trainer._freeze_for_aux_training()
    state = {name: p.requires_grad for name, p in trainer.model.named_parameters()}
    assert all(state[name] for name in state if name.startswith(("lidar_branch", "gated_fusions")))
    assert not any(state[name] for name in state if name.startswith(("encoder", "decode_head")))


def test_aux_only_optimizer_step_preserves_every_frozen_state_entry():
    """A real update must not mutate frozen weights or normalization buffers."""

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(3, 3),
                torch.nn.BatchNorm1d(3),
                torch.nn.Dropout(p=0.5),
            )
            self.lidar_branch = torch.nn.Sequential(
                torch.nn.Linear(3, 3),
                torch.nn.ReLU(),
            )
            self.decode_head = torch.nn.Sequential(
                torch.nn.BatchNorm1d(3),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(3, 2),
            )

        def forward(self, spectral: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
            return self.decode_head(self.encoder(spectral) + self.lidar_branch(aux))

    torch.manual_seed(7)
    trainer = LULCTrainer.__new__(LULCTrainer)
    trainer.model = Model()
    trainer._freeze_for_aux_training()

    # The production loop calls model.train() at every epoch before restoring
    # AUX-only modes. Exercise that exact transition so frozen BatchNorm and
    # Dropout state cannot change during the optimizer step.
    trainer.model.train()
    trainer._set_aux_only_module_modes()

    assert not trainer.model.training
    assert not trainer.model.encoder.training
    assert not trainer.model.decode_head.training
    assert not trainer.model.encoder[2].training
    assert not trainer.model.decode_head[1].training
    assert trainer.model.lidar_branch.training

    before = {
        name: value.detach().clone()
        for name, value in trainer.model.state_dict().items()
    }
    optimizer = torch.optim.SGD(
        [parameter for parameter in trainer.model.parameters()
         if parameter.requires_grad],
        lr=0.2,
    )
    spectral = torch.tensor([
        [1.0, 0.0, -1.0],
        [0.5, 1.0, 2.0],
        [-0.5, 2.0, 1.0],
        [2.0, -1.0, 0.5],
    ])
    aux = torch.tensor([
        [0.1, 1.0, 2.0],
        [1.5, 0.2, 0.4],
        [2.0, 1.0, 0.5],
        [0.5, 2.0, 1.0],
    ])
    target = torch.zeros(4, 2)

    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(trainer.model(spectral, aux), target)
    loss.backward()
    optimizer.step()

    after = trainer.model.state_dict()
    aux_names = tuple(
        name for name in after if name.startswith("lidar_branch.")
    )
    frozen_names = tuple(name for name in after if name not in aux_names)
    assert any(not torch.equal(before[name], after[name]) for name in aux_names)
    assert all(torch.equal(before[name], after[name]) for name in frozen_names)
    assert any("running_mean" in name for name in frozen_names)
    assert any("num_batches_tracked" in name for name in frozen_names)


def test_checkpoint_expansion_copies_11_aux_and_zero_initializes_era5():
    class ConvStem(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(16, 4, kernel_size=3, bias=False)

    class AuxBranch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(ConvStem())

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lidar_branch = AuxBranch()

    trainer = LULCTrainer.__new__(LULCTrainer)
    trainer.model = Model()
    key = "lidar_branch.net.0.conv.weight"
    checkpoint_weight = torch.arange(4 * 11 * 3 * 3, dtype=torch.float32).reshape(
        4, 11, 3, 3,
    )

    expanded = trainer._expand_aux_input_conv({key: checkpoint_weight})[key]

    assert expanded.shape == (4, 16, 3, 3)
    assert torch.equal(expanded[:, :11], checkpoint_weight)
    assert torch.count_nonzero(expanded[:, 11:]) == 0


def test_tile_context_supports_lpis_year_and_filename_bbox(tmp_path):
    crop = tmp_path / "tile_500000_6500000.npz"
    np.savez(
        crop, spectral=np.zeros((24, 4, 4)), lpis_year=2022,
        dates=np.asarray([
            "2021-09-20", "2022-05-10", "2022-07-01", "2022-08-12",
        ]),
    )
    lat, lon, year, cutoff = tile_context(crop)
    assert 55 < lat < 70 and 10 < lon < 25
    assert (year, cutoff) == (2022, "2022-08-12")

    temporal = tmp_path / "tile_510000_6510000.npz"
    np.savez(temporal, spectral=np.zeros((24, 4, 4)), dates=np.asarray([
        "2021-09-20", "2022-05-10", "2022-07-01", "2022-08-12",
    ]))
    assert tile_context(temporal)[2:] == (2022, "2022-08-12")


def test_tile_context_rejects_mixed_growing_years(tmp_path):
    path = tmp_path / "tile_510000_6510000.npz"
    np.savez(path, spectral=np.zeros((24, 4, 4)), lpis_year=2022,
             dates=np.asarray(["2021-09-20", "2022-05-10", "2023-07-01", ""]))
    with pytest.raises(ValueError, match="year|Year|disagree"):
        tile_context(path)


def test_tile_context_rejects_missing_growing_dates(tmp_path):
    path = tmp_path / "tile_510000_6510000.npz"
    np.savez(path, spectral=np.zeros((24, 4, 4)), lpis_year=2022)
    with pytest.raises(ValueError, match="no spectral dates"):
        tile_context(path)


def test_control_ignores_era5_keys_embedded_in_source_tile(tmp_path):
    tile = {
        "year": np.asarray(2022),
        "dates": np.asarray([
            "2021-09-20", "2022-05-10", "2022-07-01", "2022-08-12",
        ]),
        **{name: np.full((3, 4), 1e9, dtype=np.float32) for name in ERA5_NAMES},
    }
    control = _dataset(tmp_path, "control")._load_aux_channels(
        tile, 3, 4, "embedded.npz",
    )
    assert np.array_equal(control, np.zeros((5, 3, 4), dtype=np.float32))
