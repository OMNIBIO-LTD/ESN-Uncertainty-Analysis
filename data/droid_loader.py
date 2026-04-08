"""
DROID-100 dataset loader.

Loads lerobot/droid_100 (Franka Panda, 100 episodes, 81 success + 19 failed)
and splits into success/failure groups.

State: (7,) joint positions
Action: (7,) joint commands
FPS: 15
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch

DROID_REPO = "lerobot/droid_100"
DROID_STATE_DIM = 7
DROID_ACTION_DIM = 7
DROID_INPUT_DIM = 14  # state + action
DROID_FPS = 15
DROID_JOINT_NAMES = [
    "joint_0", "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6",
]


class DroidEpisode(TypedDict):
    index: int
    states: np.ndarray      # (T, 7)
    actions: np.ndarray     # (T, 7)
    timestamps: np.ndarray  # (T,)
    success: bool
    task: str
    num_frames: int


def load_droid_episodes(
    repo_id: str = DROID_REPO,
    verbose: bool = True,
) -> tuple[list[DroidEpisode], list[DroidEpisode]]:
    """
    Load DROID-100, return (successes, failures).

    Returns:
        Tuple of (success_episodes, failure_episodes).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, download_videos=False)
    hf = ds.hf_dataset

    # Build episode boundaries
    ep_col = np.array(hf["episode_index"])
    rewards = np.array(hf["next.reward"])

    ep_bounds: dict[int, tuple[int, int]] = {}
    cur_ep, start = int(ep_col[0]), 0
    for i in range(1, len(ep_col)):
        if int(ep_col[i]) != cur_ep:
            ep_bounds[cur_ep] = (start, i)
            cur_ep, start = int(ep_col[i]), i
    ep_bounds[cur_ep] = (start, len(ep_col))

    successes: list[DroidEpisode] = []
    failures: list[DroidEpisode] = []

    for ep_idx, (from_idx, to_idx) in ep_bounds.items():
        ep_slice = hf.select(range(from_idx, to_idx))

        states = _to_numpy(ep_slice["observation.state"])
        actions = _to_numpy(ep_slice["action"])
        success = float(rewards[from_idx:to_idx].sum()) > 0
        timestamps = np.arange(len(states), dtype=np.float32) / DROID_FPS

        first = hf[from_idx]
        task = str(first.get("task", ""))

        ep: DroidEpisode = {
            "index": int(ep_idx),
            "states": states.astype(np.float32),
            "actions": actions.astype(np.float32),
            "timestamps": timestamps,
            "success": success,
            "task": task,
            "num_frames": len(states),
        }

        if success:
            successes.append(ep)
        else:
            failures.append(ep)

    if verbose:
        print(f"  DROID-100: {len(successes)} success, {len(failures)} failed")

    return successes, failures


def load_droid_task_episodes(
    task_index: int = 5,
    repo_id: str = DROID_REPO,
    verbose: bool = True,
) -> tuple[list[DroidEpisode], list[DroidEpisode]]:
    """
    Load episodes for a single task only.

    Task 5 has 35 success + 19 failure episodes.
    Returns (successes, failures) filtered to that task.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, download_videos=False)
    hf = ds.hf_dataset

    ep_col = np.array(hf["episode_index"])
    task_col = np.array(hf["task_index"])
    rewards = np.array(hf["next.reward"])

    # Build episode boundaries
    ep_bounds: dict[int, tuple[int, int]] = {}
    cur_ep, start = int(ep_col[0]), 0
    for i in range(1, len(ep_col)):
        if int(ep_col[i]) != cur_ep:
            ep_bounds[cur_ep] = (start, i)
            cur_ep, start = int(ep_col[i]), i
    ep_bounds[cur_ep] = (start, len(ep_col))

    successes: list[DroidEpisode] = []
    failures: list[DroidEpisode] = []

    for ep_idx, (from_idx, to_idx) in ep_bounds.items():
        # Filter by task
        if int(task_col[from_idx]) != task_index:
            continue

        ep_slice = hf.select(range(from_idx, to_idx))
        states = _to_numpy(ep_slice["observation.state"])
        actions = _to_numpy(ep_slice["action"])
        success = float(rewards[from_idx:to_idx].sum()) > 0
        timestamps = np.arange(len(states), dtype=np.float32) / DROID_FPS

        ep: DroidEpisode = {
            "index": int(ep_idx),
            "states": states.astype(np.float32),
            "actions": actions.astype(np.float32),
            "timestamps": timestamps,
            "success": success,
            "task": f"task_{task_index}",
            "num_frames": len(states),
        }

        if success:
            successes.append(ep)
        else:
            failures.append(ep)

    if verbose:
        print(f"  Task {task_index}: {len(successes)} success, {len(failures)} failed")

    return successes, failures


def _to_numpy(column) -> np.ndarray:
    if isinstance(column, torch.Tensor):
        return column.detach().cpu().numpy()
    items = []
    for item in column:
        if isinstance(item, torch.Tensor):
            items.append(item.detach().cpu().numpy())
        else:
            items.append(np.array(item))
    return np.stack(items)
