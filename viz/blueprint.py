"""
Rerun viewer layout (blueprint) for UR5 ESN visualization.

Defines a multi-panel layout:
  Left:   EE position time series (x, y, z) + gripper
  Center: 3D view of EE trajectory + sigma timeline
  Right:  Anomaly log + per-dim prediction error

The blueprint is sent once at startup; Rerun handles the rest.
"""

from __future__ import annotations

import rerun as rr
import rerun.blueprint as rrb


def create_episode_blueprint(episode_idx: int = 0) -> rrb.Blueprint:
    """
    Create the Rerun viewer layout for a single episode.

    Args:
        episode_idx: Which episode to show.

    Returns:
        Blueprint to send via rr.send_blueprint().
    """
    ep = f"ep_{episode_idx}"

    return rrb.Blueprint(
        rrb.Horizontal(
            # Left panel: state time series
            rrb.Vertical(
                rrb.TimeSeriesView(
                    name="EE Position",
                    contents=[
                        f"{ep}/state/ee_x",
                        f"{ep}/state/ee_y",
                        f"{ep}/state/ee_z",
                    ],
                ),
                rrb.TimeSeriesView(
                    name="Gripper",
                    contents=[
                        f"{ep}/state/gripper",
                        f"{ep}/action/gripper_cmd",
                    ],
                ),
                row_shares=[3, 1],
            ),
            # Center panel: 3D view + sigma
            rrb.Vertical(
                rrb.Spatial3DView(
                    name="EE Trajectory (3D)",
                    contents=[
                        f"{ep}/ee/**",
                    ],
                ),
                rrb.TimeSeriesView(
                    name="Sigma (Uncertainty)",
                    contents=[
                        "esn/sigma_raw",
                        "esn/sigma_zscore",
                    ],
                ),
                row_shares=[2, 1],
            ),
            # Right panel: anomalies + prediction error
            rrb.Vertical(
                rrb.TimeSeriesView(
                    name="Prediction Error (per dim)",
                    contents=["esn/error/**"],
                ),
            ),
            column_shares=[2, 3, 2],
        ),
    )


def create_multi_episode_blueprint(episode_indices: list[int]) -> rrb.Blueprint:
    """
    Create layout for comparing multiple episodes.

    Shows sigma overlaid from all episodes in one panel.
    """
    sigma_contents = []
    for idx in episode_indices:
        sigma_contents.append(f"ep_{idx}/esn/sigma_raw")

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                *[
                    rrb.TimeSeriesView(
                        name=f"Episode {idx}",
                        contents=[f"ep_{idx}/state/**"],
                    )
                    for idx in episode_indices[:4]
                ],
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    name="Sigma Comparison",
                    contents=sigma_contents,
                ),
            ),
            column_shares=[1, 1],
        ),
    )
