"""
Rerun visualization logger for UR5 ESN pipeline.

Logs UR5 trajectory data into Rerun's interactive timeline viewer:
  - EE position (x, y, z) as 3D points + time series
  - EE orientation as quaternion
  - Gripper state (open/close)
  - Actions (commanded deltas)
  - Sigma (prediction error) timeline
  - Anomaly event markers
  - Camera frames (if available)

All data is logged with two timelines:
  - "frame": integer frame index
  - "time": seconds from episode start
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import rerun as rr

from omni3 import config
from omni3.data.loader import Episode
from omni3.core.uncertainty import SigmaReading


class RerunLogger:
    """
    Wraps Rerun SDK for UR5 trajectory visualization.

    Usage:
        logger = RerunLogger()
        logger.init()
        logger.log_episode(episode)
        logger.log_sigma_timeline(readings)
    """

    def __init__(self, app_id: str = config.RERUN_APP_ID):
        self.app_id = app_id

    def init(self, spawn: bool = True, recording_id: Optional[str] = None) -> None:
        """Initialize Rerun recording. spawn=True opens the viewer."""
        rec_id = recording_id or config.RERUN_RECORDING_ID
        rr.init(self.app_id, recording_id=rec_id, spawn=spawn)

    def log_episode(self, episode: Episode, prefix: str = "") -> None:
        """
        Log full episode trajectory into Rerun.

        Logs EE position as 3D points, orientation, gripper, and actions
        as time series under the given prefix.
        """
        p = f"{prefix}/ep_{episode.index}" if prefix else f"ep_{episode.index}"
        states = episode.states   # (T, 8)
        actions = episode.actions  # (T, 7)
        ts = episode.timestamps

        for t in range(len(states)):
            rr.set_time_sequence("frame", t)
            rr.set_time_seconds("time", float(ts[t]) if t < len(ts) else t / config.DATASET_FPS)

            # EE position as 3D point
            rr.log(f"{p}/ee/position", rr.Points3D(
                positions=[states[t, :3].tolist()],
                radii=[0.01],
            ))

            # EE position as time series (x, y, z separate)
            rr.log(f"{p}/state/ee_x", rr.Scalars(float(states[t, 0])))
            rr.log(f"{p}/state/ee_y", rr.Scalars(float(states[t, 1])))
            rr.log(f"{p}/state/ee_z", rr.Scalars(float(states[t, 2])))

            # Orientation quaternion components
            rr.log(f"{p}/state/qw", rr.Scalars(float(states[t, 3])))
            rr.log(f"{p}/state/qx", rr.Scalars(float(states[t, 4])))
            rr.log(f"{p}/state/qy", rr.Scalars(float(states[t, 5])))
            rr.log(f"{p}/state/qz", rr.Scalars(float(states[t, 6])))

            # Gripper
            rr.log(f"{p}/state/gripper", rr.Scalars(float(states[t, 7])))

            # Actions
            if t < len(actions):
                rr.log(f"{p}/action/dx", rr.Scalars(float(actions[t, 0])))
                rr.log(f"{p}/action/dy", rr.Scalars(float(actions[t, 1])))
                rr.log(f"{p}/action/dz", rr.Scalars(float(actions[t, 2])))
                rr.log(f"{p}/action/gripper_cmd", rr.Scalars(float(actions[t, 6])))

    def log_ee_trajectory(self, episode: Episode, prefix: str = "") -> None:
        """Log the full EE path as a line strip (3D trail)."""
        p = f"{prefix}/ep_{episode.index}" if prefix else f"ep_{episode.index}"
        positions = episode.states[:, :3].tolist()
        rr.log(f"{p}/ee/trajectory", rr.LineStrips3D(
            strips=[positions],
            radii=[0.003],
        ))

    def log_sigma_timeline(
        self,
        readings: list[SigmaReading],
        timestamps: np.ndarray,
        prefix: str = "",
    ) -> None:
        """Log sigma readings as time series with zone coloring."""
        p = prefix or "esn"

        for i, reading in enumerate(readings):
            t_sec = float(timestamps[i]) if i < len(timestamps) else i / config.DATASET_FPS
            rr.set_time_sequence("frame", reading.frame_idx)
            rr.set_time_seconds("time", t_sec)

            rr.log(f"{p}/sigma_raw", rr.Scalars(reading.sigma_raw))
            rr.log(f"{p}/sigma_zscore", rr.Scalars(reading.sigma_normalized))

            # Log zone as text annotation for anomalous frames
            if reading.zone != "normal":
                rr.log(f"{p}/zone", rr.TextLog(
                    text=f"{reading.zone}: sigma={reading.sigma_raw:.4f} (z={reading.sigma_normalized:.2f})",
                    level=rr.TextLogLevel.WARN if reading.zone == "caution" else rr.TextLogLevel.ERROR,
                ))

    def log_anomaly_events(
        self,
        events: list,
        prefix: str = "anomalies",
    ) -> None:
        """Log anomaly events as text markers on the timeline."""
        for event in events:
            rr.set_time_sequence("frame", event.frame_idx)
            rr.set_time_seconds("time", event.timestamp)

            level = rr.TextLogLevel.WARN
            if event.severity > 0.05:
                level = rr.TextLogLevel.ERROR

            rr.log(f"{prefix}/{event.anomaly_type}", rr.TextLog(
                text=f"[{event.anomaly_type}] severity={event.severity:.4f} {event.details}",
                level=level,
            ))

    def log_camera_frame(
        self,
        frame_idx: int,
        timestamp: float,
        image: np.ndarray,
        camera_name: str = "overhead",
        prefix: str = "",
    ) -> None:
        """Log a single camera frame."""
        p = f"{prefix}/cameras/{camera_name}" if prefix else f"cameras/{camera_name}"
        rr.set_time_sequence("frame", frame_idx)
        rr.set_time_seconds("time", timestamp)
        rr.log(p, rr.Image(image))

    def log_prediction_comparison(
        self,
        frame_idx: int,
        timestamp: float,
        predicted: np.ndarray,
        actual: np.ndarray,
        prefix: str = "esn",
    ) -> None:
        """Log per-dimension prediction error."""
        rr.set_time_sequence("frame", frame_idx)
        rr.set_time_seconds("time", timestamp)

        for i, name in enumerate(config.STATE_NAMES):
            error = float(abs(predicted[i] - actual[i]))
            rr.log(f"{prefix}/error/{name}", rr.Scalars(error))
