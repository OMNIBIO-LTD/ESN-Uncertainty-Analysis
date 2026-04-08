"""
Trajectory filtering for UR5 — Phase 0 anomaly detection.

Scans each episode for signs of trouble:
  1. Velocity spikes — sudden jerky EE movements
  2. Command gaps — robot not following commanded actions
  3. Gripper oscillation — repeated open/close = struggling to grasp
  4. Acceleration spikes — loss of smooth control

Each episode gets an anomaly_score = fraction of frames with anomalies.
High-scoring episodes are where ESN sigma should be validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from omni3 import config
from omni3.data.loader import Episode


@dataclass(frozen=True)
class AnomalyEvent:
    """One detected anomaly at a specific frame."""
    frame_idx: int
    timestamp: float
    anomaly_type: str    # velocity_spike, command_gap, gripper_osc, accel_spike
    severity: float      # L2 norm or count
    details: str = ""


@dataclass
class FilterResult:
    """Filtering results for one episode."""
    episode_idx: int
    task: str
    num_frames: int
    duration_s: float
    success: bool
    anomaly_score: float
    anomaly_count: int
    events: list[AnomalyEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "episode_idx": self.episode_idx,
            "task": self.task,
            "num_frames": self.num_frames,
            "duration_s": round(self.duration_s, 2),
            "success": self.success,
            "anomaly_score": round(self.anomaly_score, 4),
            "anomaly_count": self.anomaly_count,
            "events": [
                {
                    "frame": e.frame_idx,
                    "time": round(e.timestamp, 3),
                    "type": e.anomaly_type,
                    "severity": round(e.severity, 4),
                    "details": e.details,
                }
                for e in self.events
            ],
        }


def filter_episode(
    episode: Episode,
    vel_threshold: float = config.FILTER_VEL_SPIKE_THRESHOLD,
    gap_threshold: float = config.FILTER_COMMAND_GAP_THRESHOLD,
    gripper_window: int = config.FILTER_GRIPPER_OSC_WINDOW,
    gripper_min_changes: int = config.FILTER_GRIPPER_OSC_MIN_CHANGES,
) -> FilterResult:
    """
    Analyze one episode for anomalies.

    Returns FilterResult with anomaly events and score.
    """
    states = episode.states    # (T, 8)
    actions = episode.actions  # (T, 7)
    ts = episode.timestamps    # (T,)

    if len(states) < 3:
        return FilterResult(
            episode_idx=episode.index, task=episode.task,
            num_frames=episode.num_frames, duration_s=episode.duration_s,
            success=episode.success, anomaly_score=0.0, anomaly_count=0,
        )

    events: list[AnomalyEvent] = []

    # 1. Velocity spikes: large jumps in action between frames
    #    Actions are delta commands — sudden large deltas = jerky
    action_deltas = np.diff(actions, axis=0)              # (T-1, 7)
    delta_norms = np.linalg.norm(action_deltas[:, :6], axis=1)  # ignore gripper dim
    spikes = np.where(delta_norms > vel_threshold)[0]
    for idx in spikes:
        events.append(AnomalyEvent(
            frame_idx=int(idx + 1),
            timestamp=float(ts[idx + 1]) if idx + 1 < len(ts) else 0.0,
            anomaly_type="velocity_spike",
            severity=float(delta_norms[idx]),
            details=f"action_delta_L2={delta_norms[idx]:.4f}",
        ))

    # 2. Command gap: action says move, but state barely changes
    #    Compare action[:3] (delta xyz) with actual state[:3] change
    state_deltas = np.diff(states[:, :3], axis=0)         # (T-1, 3) actual EE movement
    action_pos = actions[:-1, :3]                           # (T-1, 3) commanded movement
    gaps = np.linalg.norm(action_pos - state_deltas, axis=1)
    gap_spikes = np.where(gaps > gap_threshold)[0]
    for idx in gap_spikes:
        events.append(AnomalyEvent(
            frame_idx=int(idx + 1),
            timestamp=float(ts[idx + 1]) if idx + 1 < len(ts) else 0.0,
            anomaly_type="command_gap",
            severity=float(gaps[idx]),
            details=f"pos_gap_L2={gaps[idx]:.4f}",
        ))

    # 3. Gripper oscillation: repeated open/close = struggling
    gripper = states[:, 7]  # gripper state
    gripper_deltas = np.diff(gripper)
    step = max(1, gripper_window // 2)
    for start in range(0, len(gripper_deltas) - gripper_window + 1, step):
        window = gripper_deltas[start:start + gripper_window]
        sign_changes = int(np.sum(np.diff(np.sign(window)) != 0))
        if sign_changes >= gripper_min_changes:
            mid = start + gripper_window // 2
            events.append(AnomalyEvent(
                frame_idx=int(mid),
                timestamp=float(ts[mid]) if mid < len(ts) else 0.0,
                anomaly_type="gripper_osc",
                severity=float(sign_changes),
                details=f"{sign_changes} sign changes in {gripper_window} frames",
            ))

    # 4. Acceleration spikes: 2nd derivative of EE position
    if len(states) > 3:
        pos_accel = np.diff(states[:, :3], n=2, axis=0)   # (T-2, 3)
        accel_norms = np.linalg.norm(pos_accel, axis=1)
        accel_mean = float(accel_norms.mean())
        accel_std = float(accel_norms.std())
        if accel_std > 0:
            threshold = accel_mean + 3 * accel_std
            accel_spikes = np.where(accel_norms > threshold)[0]
            for idx in accel_spikes:
                events.append(AnomalyEvent(
                    frame_idx=int(idx + 2),
                    timestamp=float(ts[idx + 2]) if idx + 2 < len(ts) else 0.0,
                    anomaly_type="accel_spike",
                    severity=float(accel_norms[idx]),
                    details=f"accel_L2={accel_norms[idx]:.4f} (thr={threshold:.4f})",
                ))

    # Score: fraction of frames with at least one anomaly
    anomaly_frames = set(e.frame_idx for e in events)
    score = len(anomaly_frames) / max(episode.num_frames, 1)

    return FilterResult(
        episode_idx=episode.index,
        task=episode.task,
        num_frames=episode.num_frames,
        duration_s=episode.duration_s,
        success=episode.success,
        anomaly_score=score,
        anomaly_count=len(events),
        events=events,
    )


def filter_episodes(
    episodes: list[Episode],
    min_score: float = config.FILTER_MIN_ANOMALY_SCORE,
    top_n: Optional[int] = None,
    verbose: bool = True,
) -> list[FilterResult]:
    """
    Filter a batch of episodes, return those with highest anomaly scores.
    """
    results = []
    for i, ep in enumerate(episodes):
        results.append(filter_episode(ep))
        if verbose and (i + 1) % 100 == 0:
            print(f"  Filtered {i + 1}/{len(episodes)} episodes...")

    # Keep only episodes above minimum score
    filtered = [r for r in results if r.anomaly_score >= min_score]
    filtered.sort(key=lambda r: r.anomaly_score, reverse=True)

    if top_n is not None:
        filtered = filtered[:top_n]

    if verbose:
        print(f"  {len(filtered)}/{len(results)} episodes above threshold "
              f"(min_score={min_score})")

    return results, filtered
