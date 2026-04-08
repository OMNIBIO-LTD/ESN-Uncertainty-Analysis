"""
Ridge Regression Readout for next-state prediction.

This is the ONLY trainable part of the ESN pipeline.
It learns a linear mapping: reservoir_state → predicted_next_state.

Training is a closed-form solution (not gradient descent):
  W_out = (S^T S + alpha*I)^{-1} S^T Y

Where:
  S = matrix of reservoir states (one row per timestep)
  Y = matrix of actual next states (targets)
  alpha = regularization strength (prevents overfitting)

This solves instantly — no epochs, no learning rate tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class PredictionReadout:
    """
    Ridge regression: ESN reservoir state → predicted next EE state.

    Usage:
        readout = PredictionReadout(512, 8)
        for t in range(T-1):
            readout.collect(esn_state_t, actual_state_t_plus_1)
        metrics = readout.train()
        predicted = readout.predict(esn_state)
    """

    def __init__(self, reservoir_size: int, output_dim: int, alpha: float = 0.01):
        self.reservoir_size = reservoir_size
        self.output_dim = output_dim
        self.alpha = alpha

        self.W_out: Optional[np.ndarray] = None  # (output_dim, reservoir_size + 1)
        self._states: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def collect(self, esn_state: np.ndarray, target_next_state: np.ndarray) -> None:
        """Store one (reservoir_state, next_state) training pair."""
        self._states.append(esn_state.copy())
        self._targets.append(target_next_state.copy())

    def train(self) -> dict:
        """
        Solve ridge regression on all collected pairs.

        Returns dict with: mse, rmse, num_samples.
        """
        n = len(self._states)
        if n < 2:
            raise ValueError(f"Need >= 2 samples, got {n}")

        S = np.array(self._states, dtype=np.float64)   # (N, reservoir_size)
        Y = np.array(self._targets, dtype=np.float64)   # (N, output_dim)

        # Add bias column: S_bias = [S | 1]
        S_bias = np.hstack([S, np.ones((n, 1), dtype=np.float64)])
        dim = S_bias.shape[1]

        # Closed-form: W = (S^T S + alpha*I)^{-1} S^T Y
        A = S_bias.T @ S_bias + self.alpha * np.eye(dim, dtype=np.float64)
        B = S_bias.T @ Y
        self.W_out = np.linalg.solve(A, B).T  # (output_dim, dim)

        # Training error
        predictions = S_bias @ self.W_out.T
        mse = float(np.mean((predictions - Y) ** 2))

        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "num_samples": n,
        }

    def predict(self, esn_state: np.ndarray) -> np.ndarray:
        """Predict next state from one reservoir state. Returns (output_dim,)."""
        if self.W_out is None:
            raise RuntimeError("Not trained yet — call train() first")
        s = np.append(esn_state.astype(np.float64), 1.0)
        return (self.W_out @ s).astype(np.float32)

    def predict_batch(self, esn_states: np.ndarray) -> np.ndarray:
        """Predict next states for a batch. Returns (N, output_dim)."""
        if self.W_out is None:
            raise RuntimeError("Not trained yet — call train() first")
        n = esn_states.shape[0]
        S_bias = np.hstack([esn_states.astype(np.float64), np.ones((n, 1))])
        return (S_bias @ self.W_out.T).astype(np.float32)

    def clear(self) -> None:
        """Free collected training data."""
        self._states.clear()
        self._targets.clear()

    @property
    def num_collected(self) -> int:
        return len(self._states)

    @property
    def is_trained(self) -> bool:
        return self.W_out is not None

    def save(self, path: str) -> None:
        """Save weights to .npz."""
        if self.W_out is None:
            raise RuntimeError("Nothing to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W_out=self.W_out, reservoir_size=self.reservoir_size,
                 output_dim=self.output_dim, alpha=self.alpha)

    def load(self, path: str) -> None:
        """Load weights from .npz."""
        data = np.load(path)
        self.W_out = data["W_out"]
        self.reservoir_size = int(data["reservoir_size"])
        self.output_dim = int(data["output_dim"])
        self.alpha = float(data["alpha"])
