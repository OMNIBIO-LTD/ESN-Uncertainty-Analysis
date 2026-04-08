#!/usr/bin/env python3
"""
Script 00: Explore the UR5 dataset.

Downloads the dataset (if not cached) and prints a summary:
  - Number of episodes, frames, FPS
  - Task descriptions
  - State/action dimensions and value ranges
  - Episode length distribution
  - Success rate

This is the first script you run — it tells you what you're working with.

Usage:
    python3.10 omni3/scripts/00_explore.py
    python3.10 omni3/scripts/00_explore.py --episodes 10  # only first 10
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from omni3 import config
from omni3.data.loader import load_dataset, get_dataset_info, extract_episode


def main():
    parser = argparse.ArgumentParser(description="Explore UR5 dataset")
    parser.add_argument("--repo", default=config.DATASET_REPO_ID)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of episodes to inspect (default: all)")
    args = parser.parse_args()

    print("=" * 60)
    print("  UR5 Dataset Explorer")
    print("=" * 60)

    # Load dataset
    print(f"\n[1/3] Loading dataset: {args.repo}")
    t0 = time.time()
    dataset = load_dataset(repo_id=args.repo, download_videos=False)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Basic info
    info = get_dataset_info(dataset)
    print(f"\n  Total episodes:  {info['num_episodes']}")
    print(f"  Total frames:    {info['num_frames']}")
    print(f"  FPS:             {info['fps']}")
    print(f"  Features:        {len(info['features'])}")

    # Tasks
    if info["tasks"]:
        print(f"\n  Tasks ({len(info['tasks'])}):")
        for t in info["tasks"]:
            if isinstance(t, dict):
                for k, v in t.items():
                    print(f"    [{k}] {v}")
            else:
                print(f"    - {t}")

    # Extract episodes for analysis
    n_eps = args.episodes or info["num_episodes"]
    n_eps = min(n_eps, info["num_episodes"])

    print(f"\n[2/3] Extracting {n_eps} episodes...")
    t0 = time.time()
    episodes = []
    for i in range(n_eps):
        episodes.append(extract_episode(dataset, i))
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{n_eps}")
    print(f"  Extracted in {time.time() - t0:.1f}s")

    # Statistics
    print(f"\n[3/3] Computing statistics...")
    lengths = [ep.num_frames for ep in episodes]
    durations = [ep.duration_s for ep in episodes]
    successes = sum(1 for ep in episodes if ep.success)

    print(f"\n{'=' * 60}")
    print(f"  DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Episodes analyzed: {n_eps}")
    print(f"  Success rate:      {successes}/{n_eps} ({100*successes/n_eps:.1f}%)")
    print(f"\n  Episode lengths (frames):")
    print(f"    Mean: {np.mean(lengths):.1f}")
    print(f"    Min:  {min(lengths)}, Max: {max(lengths)}")
    print(f"    Std:  {np.std(lengths):.1f}")
    print(f"\n  Episode duration (seconds):")
    print(f"    Mean: {np.mean(durations):.1f}s")
    print(f"    Min:  {min(durations):.1f}s, Max: {max(durations):.1f}s")

    # State/action ranges
    all_states = np.concatenate([ep.states for ep in episodes])
    all_actions = np.concatenate([ep.actions for ep in episodes])

    print(f"\n  State dimensions ({config.STATE_DIM}):")
    for i, name in enumerate(config.STATE_NAMES):
        col = all_states[:, i]
        print(f"    {name:>10s}: min={col.min():+.4f}  max={col.max():+.4f}  "
              f"mean={col.mean():+.4f}  std={col.std():.4f}")

    print(f"\n  Action dimensions ({config.ACTION_DIM}):")
    for i, name in enumerate(config.ACTION_NAMES):
        col = all_actions[:, i]
        print(f"    {name:>12s}: min={col.min():+.4f}  max={col.max():+.4f}  "
              f"mean={col.mean():+.4f}  std={col.std():.4f}")

    # Task breakdown
    task_counts: dict[str, int] = {}
    task_success: dict[str, int] = {}
    for ep in episodes:
        task_counts[ep.task] = task_counts.get(ep.task, 0) + 1
        if ep.success:
            task_success[ep.task] = task_success.get(ep.task, 0) + 1

    print(f"\n  Task breakdown:")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        succ = task_success.get(task, 0)
        label = task[:55] if len(task) > 55 else task
        print(f"    {label:55s}  {count:4d} eps  ({100*succ/count:.0f}% success)")

    print(f"\n{'=' * 60}")
    print(f"  Dataset ready. Next: python3.10 omni3/scripts/01_filter.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
