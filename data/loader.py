"""
UR5 dataset loader via LeRobot.

Loads trajectories from lerobot/berkeley_autolab_ur5 (parquet + mp4)
and provides iteration over episodes as numpy arrays.

Each episode contains:
  - states: (T, 8) EE pose + gripper
  - actions: (T, 7) delta EE + gripper command
  - rewards: (T,) sparse reward (1.0 at success)
  - timestamps: (T,) seconds from episode start
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class Episode:
    """A single UR5 trajectory episode."""
    index: int
    states: np.ndarray       # (T, 8) observation.state
    actions: np.ndarray      # (T, 7) action
    rewards: np.ndarray      # (T,) next.reward
    timestamps: np.ndarray   # (T,) seconds
    task: str                # task description
    num_frames: int

    @property
    def duration_s(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def success(self) -> bool:
        return float(self.rewards.sum()) > 0.0


def load_dataset(
    repo_id: str = "lerobot/berkeley_autolab_ur5",
    episodes: Optional[list[int]] = None,
    download_videos: bool = False,
):
    """
    Load UR5 dataset via LeRobot.

    Returns a LeRobotDataset instance (parquet-backed).
    Videos are NOT downloaded by default — we use parquet for state/action.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id=repo_id,
        episodes=episodes,
        download_videos=download_videos,
    )


def get_dataset_info(dataset) -> dict:
    """Extract key metadata."""
    tasks = []
    if hasattr(dataset.meta, "tasks"):
        tasks_data = dataset.meta.tasks
        if hasattr(tasks_data, "to_dict"):
            tasks = list(tasks_data.to_dict().values())
    return {
        "repo_id": dataset.repo_id,
        "num_episodes": dataset.num_episodes,
        "num_frames": dataset.num_frames,
        "fps": dataset.fps,
        "features": list(dataset.features.keys()),
        "tasks": tasks,
    }


def _build_episode_index(hf) -> dict[int, tuple[int, int]]:
    """
    Build a map of episode_idx -> (from_idx, to_idx) by reading
    the episode_index column once. Cached on the dataset object.
    """
    ep_col = np.array(hf["episode_index"])
    index: dict[int, tuple[int, int]] = {}
    current_ep = int(ep_col[0])
    start = 0
    for i in range(1, len(ep_col)):
        if int(ep_col[i]) != current_ep:
            index[current_ep] = (start, i)
            current_ep = int(ep_col[i])
            start = i
    index[current_ep] = (start, len(ep_col))
    return index


# Module-level cache to avoid re-reading the column
_episode_index_cache: dict[int, dict[int, tuple[int, int]]] = {}


def extract_episode(dataset, episode_idx: int) -> Episode:
    """
    Extract a single episode as numpy arrays from parquet data.

    Uses hf_dataset directly to avoid video decode overhead.
    Builds an episode index on first call (fast after that).
    """
    hf = dataset.hf_dataset
    ds_id = id(hf)

    # Build index once, reuse for subsequent calls
    if ds_id not in _episode_index_cache:
        _episode_index_cache[ds_id] = _build_episode_index(hf)

    ep_index = _episode_index_cache[ds_id]
    if episode_idx not in ep_index:
        raise ValueError(f"Episode {episode_idx} not found in dataset")

    from_idx, to_idx = ep_index[episode_idx]
    ep_slice = hf.select(range(from_idx, to_idx))

    states = _stack_tensor(ep_slice["observation.state"])
    actions = _stack_tensor(ep_slice["action"])
    rewards = np.array([
        float(r.item() if hasattr(r, "item") else r)
        for r in ep_slice["next.reward"]
    ], dtype=np.float32)
    timestamps = np.array([
        float(t.item() if hasattr(t, "item") else t)
        for t in ep_slice["timestamp"]
    ], dtype=np.float32)

    # Task text from first frame
    first = hf[from_idx]
    task = str(first.get("task", ""))

    return Episode(
        index=episode_idx,
        states=states,
        actions=actions,
        rewards=rewards,
        timestamps=timestamps,
        task=task,
        num_frames=len(states),
    )


def extract_episodes(
    dataset,
    indices: Optional[list[int]] = None,
    verbose: bool = True,
) -> list[Episode]:
    """Extract multiple episodes. If indices=None, extracts all."""
    if indices is None:
        indices = list(range(dataset.num_episodes))

    episodes = []
    for i, idx in enumerate(indices):
        ep = extract_episode(dataset, idx)
        episodes.append(ep)
        if verbose and (i + 1) % 100 == 0:
            print(f"  Extracted {i + 1}/{len(indices)} episodes...")

    return episodes


def _stack_tensor(column) -> np.ndarray:
    """Stack a column of tensors/arrays into a single numpy array."""
    if isinstance(column, torch.Tensor):
        return column.detach().cpu().numpy()
    if isinstance(column, np.ndarray):
        return column
    # List of tensors
    items = []
    for item in column:
        if isinstance(item, torch.Tensor):
            items.append(item.detach().cpu().numpy())
        else:
            items.append(np.array(item))
    return np.stack(items)
