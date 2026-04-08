#!/usr/bin/env python3
"""
Script 02: Train ESN on UR5 arm data.

Trains the ESN readout to predict next EE state from current state+action.
Uses "normal" (low-anomaly) episodes for training.

The ESN reservoir has 512 neurons with FIXED random weights.
Only the readout is trained (ridge regression, instant closed-form).

Input:  state(8) + action(7) = 15-dim
Output: predicted next state(8)

Usage:
    python3.10 omni3/scripts/02_train_esn.py
    python3.10 omni3/scripts/02_train_esn.py --train-episodes 500
    python3.10 omni3/scripts/02_train_esn.py --filter-output output/filtered_episodes.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from omni3 import config
from omni3.data.loader import load_dataset, extract_episodes
from omni3.pipeline import UncertaintyPipeline


def main():
    parser = argparse.ArgumentParser(description="Train ESN on UR5 data")
    parser.add_argument("--repo", default=config.DATASET_REPO_ID)
    parser.add_argument("--train-episodes", type=int, default=None,
                        help="Number of episodes to train on (default: use filter split)")
    parser.add_argument("--filter-output", default=config.FILTER_OUTPUT_PATH,
                        help="Path to filtered_episodes.json from step 01")
    parser.add_argument("--save-model", default=os.path.join(config.MODEL_DIR, "esn_readout.npz"))
    args = parser.parse_args()

    print("=" * 60)
    print("  ESN Training: UR5 Next-State Prediction")
    print("=" * 60)
    print(f"\n  ESN: {config.ESN_INPUT_DIM}-dim input → {config.RESERVOIR_SIZE} reservoir "
          f"→ {config.READOUT_OUTPUT_DIM}-dim output")
    print(f"  Spectral radius: {config.SPECTRAL_RADIUS}")
    print(f"  Ridge alpha: {config.RIDGE_ALPHA}")

    # Determine which episodes to train on
    flagged_indices = set()
    if os.path.exists(args.filter_output):
        with open(args.filter_output) as f:
            filter_data = json.load(f)
        for ep in filter_data.get("flagged_episodes", []):
            flagged_indices.add(ep["episode_idx"])
        print(f"\n  Loaded filter results: {len(flagged_indices)} flagged episodes excluded")

    # Load dataset
    print(f"\n[1/3] Loading dataset: {args.repo}")
    dataset = load_dataset(repo_id=args.repo, download_videos=False)
    all_indices = list(range(dataset.num_episodes))

    # Split: train on normal, exclude flagged
    if args.train_episodes:
        train_indices = list(range(min(args.train_episodes, len(all_indices))))
    elif flagged_indices:
        train_indices = [i for i in all_indices if i not in flagged_indices]
    else:
        # No filter: use 80% for train
        split = int(len(all_indices) * 0.8)
        train_indices = all_indices[:split]

    print(f"  Training on {len(train_indices)} episodes "
          f"(excluded {len(flagged_indices)} flagged)")

    # Extract training episodes
    print(f"\n[2/3] Extracting training episodes...")
    t0 = time.time()
    train_episodes = extract_episodes(dataset, indices=train_indices, verbose=True)
    total_frames = sum(ep.num_frames for ep in train_episodes)
    print(f"  Extracted {len(train_episodes)} episodes ({total_frames} frames) "
          f"in {time.time() - t0:.1f}s")

    # Train
    print(f"\n[3/3] Training ESN pipeline...")
    pipeline = UncertaintyPipeline()
    t0 = time.time()
    metrics = pipeline.train(train_episodes, verbose=True)
    train_time = time.time() - t0

    # Save model
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    pipeline.save(args.save_model)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING RESULTS")
    print(f"{'=' * 60}")
    print(f"  Train time:     {train_time:.2f}s")
    print(f"  Training MSE:   {metrics['mse']:.6f}")
    print(f"  Training RMSE:  {metrics['rmse']:.4f}")
    print(f"  Samples used:   {metrics['num_samples']}")
    print(f"  Model saved:    {args.save_model}")
    print(f"\n{'=' * 60}")
    print(f"  Next: python3.10 omni3/scripts/03_validate.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
