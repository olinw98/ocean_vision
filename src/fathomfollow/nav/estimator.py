from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class VelocityGRU(nn.Module):
    def __init__(self, input_size: int = 9, hidden_size: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class VelocityEstimator:
    """DriftGuard: predicts body-frame velocity from IMU (+ DVL when valid)."""

    def __init__(self, window_size: int = 20, hidden_size: int = 64) -> None:
        self.window_size = window_size
        self._model = VelocityGRU(input_size=9, hidden_size=hidden_size)
        self._device = torch.device("cpu")

    def estimate(self, imu_window: np.ndarray, dvl: np.ndarray | None) -> np.ndarray:
        win = imu_window[-self.window_size :]
        if len(win) < self.window_size:
            pad = np.zeros((self.window_size - len(win), win.shape[1]), dtype=np.float32)
            win = np.vstack([pad, win])
        feat = win.astype(np.float32)
        if dvl is not None:
            dvl_feat = np.tile(dvl.reshape(1, 3), (self.window_size, 1))
        else:
            dvl_feat = np.zeros((self.window_size, 3), dtype=np.float32)
        x = np.concatenate([feat, dvl_feat], axis=1)
        t = torch.from_numpy(x).unsqueeze(0).to(self._device)
        with torch.no_grad():
            vel = self._model(t).cpu().numpy()[0]
        return vel.astype(np.float32)

    def load(self, checkpoint: Path | str) -> None:
        path = Path(checkpoint)
        state = torch.load(path, map_location=self._device, weights_only=True)
        hidden_size = int(state["head.weight"].shape[1])
        self._model = VelocityGRU(input_size=9, hidden_size=hidden_size)
        self._model.load_state_dict(state)

    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        self._model.train()
        pred = self._model(batch_x)
        loss = nn.functional.mse_loss(pred, batch_y)
        return float(loss.item())
