"""
Uncertainty estimation via ESN prediction error (sigma).

The core idea:
  sigma(t) = ||predicted_state(t+1) - actual_state(t+1)||_2

When sigma is LOW:
  → ESN predicted correctly → situation is familiar → confident

When sigma is HIGH:
  → ESN predicted wrong → something unexpected happened → uncertain
  → Robot should be CAUTIOUS (slow down) or CURIOUS (explore)

The raw sigma is normalized using a running window to produce z-scores:
  z = (sigma - mean) / std

Zones:
  normal:    z < 1.5  (business as usual)
  caution:   1.5 < z < 3.0  (slow down, be careful)
  curiosity: z > 3.0  (very unexpected — worth exploring)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SigmaReading:
    """One uncertainty measurement at a single timestep."""
    frame_idx: int
    sigma_raw: float         # L2 prediction error
    sigma_normalized: float  # z-score
    zone: str                # "normal", "caution", or "curiosity"


class UncertaintyEstimator:
    """
    Tracks prediction error and classifies into uncertainty zones.

    Args:
        window_size: Number of recent sigmas for running statistics.
        caution_threshold: z-score above which = caution zone.
        curiosity_threshold: z-score above which = curiosity zone.
    """

    def __init__(
        self,
        window_size: int = 50,
        caution_threshold: float = 1.5,
        curiosity_threshold: float = 3.0,
    ):
        self.window_size = window_size
        self.caution_threshold = caution_threshold
        self.curiosity_threshold = curiosity_threshold
        self._history: deque[float] = deque(maxlen=window_size)

    def update(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        frame_idx: int = 0,
    ) -> SigmaReading:
        """
        Compute sigma from predicted vs actual next state.

        Returns SigmaReading with raw sigma, z-score, and zone label.
        """
        sigma_raw = float(np.linalg.norm(predicted - actual))
        self._history.append(sigma_raw)

        # Running stats
        h = np.array(self._history)
        mean = float(h.mean())
        std = max(float(h.std()), 1e-8) if len(h) > 1 else 1.0

        z = (sigma_raw - mean) / std

        if z > self.curiosity_threshold:
            zone = "curiosity"
        elif z > self.caution_threshold:
            zone = "caution"
        else:
            zone = "normal"

        return SigmaReading(
            frame_idx=frame_idx,
            sigma_raw=sigma_raw,
            sigma_normalized=z,
            zone=zone,
        )

    def get_stats(self) -> dict:
        """Return running statistics."""
        h = np.array(self._history) if self._history else np.array([0.0])
        return {
            "mean": float(h.mean()),
            "std": float(h.std()) if len(h) > 1 else 0.0,
            "min": float(h.min()),
            "max": float(h.max()),
            "count": len(self._history),
        }

    def reset(self) -> None:
        """Clear history (call between episodes)."""
        self._history.clear()
