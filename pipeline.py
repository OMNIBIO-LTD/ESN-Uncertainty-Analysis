"""
omni3 Pipeline: ESN-based next-state prediction and uncertainty for UR5.

This is the main orchestrator. It takes UR5 episodes and:
  1. Feeds state+action (15-dim) through the ESN reservoir (512 neurons)
  2. Trains the readout to predict next state (8-dim) via ridge regression
  3. Evaluates prediction error (sigma) on test episodes
  4. Returns sigma timelines for visualization and analysis

Usage:
    pipeline = UncertaintyPipeline()
    metrics = pipeline.train(train_episodes)
    result = pipeline.evaluate(test_episode)
    print(result["stats"]["sigma_mean"])
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from omni3 import config
from omni3.core.esn import EchoStateNetwork
from omni3.core.readout import PredictionReadout
from omni3.core.uncertainty import UncertaintyEstimator, SigmaReading
from omni3.data.loader import Episode


class UncertaintyPipeline:
    """
    Full pipeline: Episode → ESN → Readout → Sigma.
    """

    def __init__(
        self,
        reservoir_size: int = config.RESERVOIR_SIZE,
        input_dim: int = config.ESN_INPUT_DIM,
        output_dim: int = config.READOUT_OUTPUT_DIM,
        ridge_alpha: float = config.RIDGE_ALPHA,
        warmup_steps: int = config.ESN_WARMUP_STEPS,
    ):
        self.warmup_steps = warmup_steps
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.esn = EchoStateNetwork(
            input_dim=input_dim,
            reservoir_size=reservoir_size,
            spectral_radius=config.SPECTRAL_RADIUS,
            input_scaling=config.INPUT_SCALING,
            leaking_rate=config.LEAKING_RATE,
            sparsity=config.RESERVOIR_SPARSITY,
            seed=config.RANDOM_SEED,
        )
        self.readout = PredictionReadout(
            reservoir_size=reservoir_size,
            output_dim=output_dim,
            alpha=ridge_alpha,
        )
        self.uncertainty = UncertaintyEstimator(
            window_size=config.SIGMA_WINDOW_SIZE,
            caution_threshold=config.SIGMA_CAUTION_THRESHOLD,
            curiosity_threshold=config.SIGMA_CURIOSITY_THRESHOLD,
        )

    def train(self, episodes: list[Episode], verbose: bool = True) -> dict:
        """
        Train ESN readout on episodes.

        For each episode:
          1. Reset ESN state
          2. Feed state(8)+action(7)=15-dim through ESN each timestep
          3. After warmup, collect (esn_state, next_actual_state) pairs
          4. Solve ridge regression (instant, closed-form)

        Returns training metrics.
        """
        self.readout.clear()

        for ep in episodes:
            self.esn.reset()
            T = min(len(ep.states), len(ep.actions))

            for t in range(T - 1):
                esn_input = np.concatenate([ep.states[t], ep.actions[t]])
                esn_state = self.esn.update(esn_input.astype(np.float32))

                if t >= self.warmup_steps:
                    self.readout.collect(esn_state, ep.states[t + 1])

        if verbose:
            print(f"  Collected {self.readout.num_collected} training samples "
                  f"from {len(episodes)} episodes")

        metrics = self.readout.train()

        if verbose:
            print(f"  Training done: MSE={metrics['mse']:.6f}, "
                  f"RMSE={metrics['rmse']:.4f}")

        return metrics

    def evaluate(self, episode: Episode) -> dict:
        """
        Run trained pipeline on one episode, compute sigma timeline.

        Returns dict with:
          - sigmas_raw: (N,) raw prediction errors
          - sigmas_norm: (N,) z-score normalized
          - zones: (N,) zone labels
          - timestamps: (N,) corresponding timestamps
          - readings: list of SigmaReading objects
          - stats: summary statistics
        """
        self.esn.reset()
        self.uncertainty.reset()

        T = min(len(episode.states), len(episode.actions))
        readings: list[SigmaReading] = []

        for t in range(T - 1):
            esn_input = np.concatenate([episode.states[t], episode.actions[t]])
            esn_state = self.esn.update(esn_input.astype(np.float32))

            if t >= self.warmup_steps and self.readout.is_trained:
                predicted = self.readout.predict(esn_state)
                reading = self.uncertainty.update(
                    predicted, episode.states[t + 1], frame_idx=t,
                )
                readings.append(reading)

        sigmas_raw = np.array([r.sigma_raw for r in readings]) if readings else np.zeros(1)
        sigmas_norm = np.array([r.sigma_normalized for r in readings]) if readings else np.zeros(1)
        zones = [r.zone for r in readings] if readings else ["normal"]
        frame_indices = [r.frame_idx for r in readings] if readings else [0]

        # Timestamps for the sigma readings
        start = self.warmup_steps + 1
        end = min(start + len(readings), len(episode.timestamps))
        ts = episode.timestamps[start:end] if len(episode.timestamps) > start else np.zeros(len(readings))

        caution_frames = [r.frame_idx for r in readings if r.zone == "caution"]
        curiosity_frames = [r.frame_idx for r in readings if r.zone == "curiosity"]

        return {
            "episode_idx": episode.index,
            "task": episode.task,
            "num_frames": T,
            "success": episode.success,
            "sigmas_raw": sigmas_raw,
            "sigmas_norm": sigmas_norm,
            "zones": zones,
            "timestamps": ts,
            "frame_indices": frame_indices,
            "readings": readings,
            "caution_frames": caution_frames,
            "curiosity_frames": curiosity_frames,
            "stats": {
                "sigma_mean": float(sigmas_raw.mean()),
                "sigma_std": float(sigmas_raw.std()),
                "sigma_max": float(sigmas_raw.max()),
                "sigma_min": float(sigmas_raw.min()),
                "caution_count": len(caution_frames),
                "curiosity_count": len(curiosity_frames),
                "caution_frac": len(caution_frames) / max(len(readings), 1),
            },
        }

    def save(self, path: str) -> None:
        """Save trained readout."""
        self.readout.save(path)

    def load(self, path: str) -> None:
        """Load trained readout."""
        self.readout.load(path)
