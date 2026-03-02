"""
imint/analyzers/grazing.py — Grazing activity classifier (pib-ml-grazing)

CNN-biLSTM model from RISE Research Institutes of Sweden that classifies
Sentinel-2 timeseries as "active grazing" or "no activity".

Reference: https://github.com/aleksispi/pib-ml-grazing
License: MIT (RISE Research Institutes of Sweden, 2025)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model architecture — copied from pib-ml-grazing/classes.py (MIT license)
# ---------------------------------------------------------------------------

class _CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        conv_ker = 3
        conv_str = 1
        pool_ker = 2
        pool_str = 2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=conv_ker, padding=conv_str)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_ker, stride=pool_str)
        self.out_dim_factor = out_channels * 0.5 * 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.conv(x)))


class _LSTMClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        cnn_out_dim: int,
        hidden_dim: int,
        num_layers: int,
        im_height: int,
        im_width: int,
        bidir: bool = False,
    ):
        super().__init__()
        self.cnn = _CNNBlock(in_channels, cnn_out_dim)
        lstm_in_dim = int(self.cnn.out_dim_factor * im_height * im_width)
        self.lstm = nn.LSTM(
            lstm_in_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=bidir,
        )
        fc_in = hidden_dim * (2 if bidir else 1)
        self.fc = nn.Linear(fc_in, num_classes)

    def forward(self, x: torch.Tensor, seq_lens=None):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.reshape(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.reshape(batch_size, timesteps, -1)
        if seq_lens is not None:
            r_in = nn.utils.rnn.pack_padded_sequence(
                r_in, seq_lens, batch_first=True, enforce_sorted=False,
            )
        r_out, _ = self.lstm(r_in)
        if seq_lens is not None:
            r_out, _ = nn.utils.rnn.pad_packed_sequence(r_out, batch_first=True)
        r_out = r_out.reshape(batch_size * timesteps, -1)
        out = self.fc(r_out)
        out = out.reshape(batch_size, timesteps, -1)
        return out


# ---------------------------------------------------------------------------
# Default model config (matches pib-ml-grazing/config.py defaults)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "in_channels": 12,
    "num_classes": 2,
    "cnn_out_dim": 4,
    "hidden_dim": 8,
    "num_layers": 1,
    "img_size": 46,
    "bidir": True,  # biLSTM
    "pred_median_last_x": 4,
}

_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "fm" / "pib_grazing"
_DEFAULT_WEIGHTS = _WEIGHTS_DIR / "2025-07-06_10-40-23" / "2" / "train_stats" / "model_weights.pth"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GrazingPrediction:
    """Prediction result for one polygon."""
    polygon_id: str | int
    predicted_class: int          # 0 = no activity, 1 = active grazing
    class_label: str              # "No activity" or "Active grazing"
    confidence: float             # softmax probability of predicted class
    probabilities: list[float]    # [p_no_activity, p_active_grazing]


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class GrazingAnalyzer:
    """CNN-biLSTM grazing activity classifier.

    Accepts ``GrazingTimeseriesResult`` objects (from ``fetch_grazing_timeseries``)
    and predicts whether each polygon shows active grazing or not.
    """

    LABELS = {0: "No activity", 1: "Active grazing"}

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | None = None,
        config: dict | None = None,
    ):
        self.cfg = {**_DEFAULT_CONFIG, **(config or {})}
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = self._load_model(weights_path or _DEFAULT_WEIGHTS)

    def _load_model(self, weights_path: str | Path) -> _LSTMClassifier:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        model = _LSTMClassifier(
            in_channels=self.cfg["in_channels"],
            num_classes=self.cfg["num_classes"],
            cnn_out_dim=self.cfg["cnn_out_dim"],
            hidden_dim=self.cfg["hidden_dim"],
            num_layers=self.cfg["num_layers"],
            im_height=self.cfg["img_size"],
            im_width=self.cfg["img_size"],
            bidir=self.cfg["bidir"],
        )
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        print(f"  GrazingAnalyzer: loaded weights from {weights_path}")
        return model

    def _center_crop(self, data: np.ndarray) -> np.ndarray:
        """Center-crop spatial dims to img_size x img_size.

        Args:
            data: (T, C, H, W) array.

        Returns:
            (T, C, img_size, img_size) array.
        """
        sz = self.cfg["img_size"]
        _, _, H, W = data.shape
        if H <= sz and W <= sz:
            # Pad if smaller
            padded = np.zeros((data.shape[0], data.shape[1], sz, sz), dtype=data.dtype)
            y0 = (sz - H) // 2
            x0 = (sz - W) // 2
            padded[:, :, y0:y0 + H, x0:x0 + W] = data
            return padded
        y0 = (H - sz) // 2
        x0 = (W - sz) // 2
        return data[:, :, y0:y0 + sz, x0:x0 + sz]

    def predict(self, timeseries_result) -> GrazingPrediction:
        """Run inference on a single GrazingTimeseriesResult.

        Args:
            timeseries_result: Object with ``data`` (T, 12, H, W),
                ``polygon_id``, and ``dates`` attributes.

        Returns:
            GrazingPrediction with class, label, and confidence.
        """
        data = timeseries_result.data  # (T, 12, H, W) float32

        # Center crop
        data = self._center_crop(data)  # (T, 12, 46, 46)

        # Normalize per-band (mean/std computed from this polygon's data)
        means = data.mean(axis=(0, 2, 3), keepdims=True)  # (1, 12, 1, 1)
        stds = data.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        data = (data - means) / stds

        # To tensor: (1, T, 12, 46, 46)
        x = torch.from_numpy(data).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(x)  # (1, T, 2)

            # Median of last X timestep predictions (matches pib-ml-grazing inference)
            last_x = self.cfg["pred_median_last_x"]
            pred_steps = out[0, -last_x:, :]  # (X, 2)
            pred_median, _ = torch.median(pred_steps, dim=0)  # (2,)

            probs = torch.softmax(pred_median, dim=0).cpu().numpy()
            cls = int(np.argmax(probs))

        return GrazingPrediction(
            polygon_id=timeseries_result.polygon_id,
            predicted_class=cls,
            class_label=self.LABELS[cls],
            confidence=float(probs[cls]),
            probabilities=probs.tolist(),
        )

    def predict_batch(self, timeseries_results: list) -> list[GrazingPrediction]:
        """Run inference on multiple GrazingTimeseriesResult objects.

        Each polygon may have different timeseries lengths so they are
        processed individually (no batch padding needed).
        """
        predictions = []
        for ts in timeseries_results:
            try:
                pred = self.predict(ts)
                predictions.append(pred)
            except Exception as e:
                print(f"  Warning: failed on polygon {ts.polygon_id}: {e}")
                predictions.append(GrazingPrediction(
                    polygon_id=ts.polygon_id,
                    predicted_class=-1,
                    class_label="Error",
                    confidence=0.0,
                    probabilities=[0.0, 0.0],
                ))
        return predictions
