#!/usr/bin/env python3
"""
Script 01: Filter UR5 trajectories for anomalies (Phase 0).

Scans all episodes for:
  - Velocity spikes (jerky EE movements)
  - Command gaps (robot not following commands)
  - Gripper oscillation (struggling to grasp)
  - Acceleration spikes (loss of smooth control)

Outputs filtered_episodes.json ranked by anomaly score.
Optionally visualizes results in Rerun.

Usage:
    python3.10 omni3/scripts/01_filter.py
    python3.10 omni3/scripts/01_filter.py --episodes 100  # first 100 only
    python3.10 omni3/scripts/01_filter.py --rerun          # with visualization
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from omni3 import config
from omni3.data.loader import load_dataset, extract_episodes
from omni3.analysis.filter import filter_episodes


def main():
    parser = argparse.ArgumentParser(description="Filter UR5 trajectories")
    parser.add_argument("--repo", default=config.DATASET_REPO_ID)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of episodes to filter (default: all)")
    parser.add_argument("--min-score", type=float, default=config.FILTER_MIN_ANOMALY_SCORE)
    parser.add_argument("--top-n", type=int, default=config.FILTER_TOP_N)
    parser.add_argument("--output", default=config.FILTER_OUTPUT_PATH)
    parser.add_argument("--rerun", action="store_true", help="Visualize in Rerun")
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 0: Trajectory Filtering")
    print("=" * 60)

    # Load dataset
    print(f"\n[1/3] Loading dataset: {args.repo}")
    t0 = time.time()
    dataset = load_dataset(repo_id=args.repo, download_videos=False)
    n_eps = args.episodes or dataset.num_episodes
    n_eps = min(n_eps, dataset.num_episodes)
    print(f"  Loaded in {time.time() - t0:.1f}s ({dataset.num_episodes} episodes total)")

    # Extract episodes
    print(f"\n[2/3] Extracting {n_eps} episodes...")
    t0 = time.time()
    indices = list(range(n_eps))
    episodes = extract_episodes(dataset, indices=indices, verbose=True)
    print(f"  Extracted in {time.time() - t0:.1f}s")

    # Filter
    print(f"\n[3/3] Filtering for anomalies...")
    t0 = time.time()
    all_results, filtered = filter_episodes(
        episodes, min_score=args.min_score, top_n=args.top_n,
    )
    print(f"  Filtered in {time.time() - t0:.1f}s")

    # Summary
    total_anomalies = sum(r.anomaly_count for r in all_results)
    scores = [r.anomaly_score for r in all_results]
    anomaly_types: dict[str, int] = {}
    for r in all_results:
        for e in r.events:
            anomaly_types[e.anomaly_type] = anomaly_types.get(e.anomaly_type, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"  FILTERING RESULTS")
    print(f"{'=' * 60}")
    print(f"  Episodes scanned:    {len(all_results)}")
    print(f"  Episodes flagged:    {len(filtered)} (score >= {args.min_score})")
    print(f"  Total anomaly events: {total_anomalies}")
    print(f"\n  Anomaly score distribution:")
    print(f"    Mean:   {np.mean(scores):.4f}")
    print(f"    Median: {np.median(scores):.4f}")
    print(f"    Max:    {np.max(scores):.4f}")
    print(f"    Zero-score episodes: {sum(1 for s in scores if s == 0)}")
    print(f"\n  Anomaly type breakdown:")
    for atype, count in sorted(anomaly_types.items(), key=lambda x: -x[1]):
        print(f"    {atype:20s}: {count:5d} events")

    # Top flagged episodes
    if filtered:
        print(f"\n  Top {min(10, len(filtered))} flagged episodes:")
        for r in filtered[:10]:
            task = r.task[:40] if len(r.task) > 40 else r.task
            print(f"    ep {r.episode_idx:>4d}: score={r.anomaly_score:.4f}  "
                  f"events={r.anomaly_count:3d}  "
                  f"success={'Y' if r.success else 'N'}  "
                  f"task=\"{task}\"")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data = {
        "config": {
            "vel_threshold": config.FILTER_VEL_SPIKE_THRESHOLD,
            "gap_threshold": config.FILTER_COMMAND_GAP_THRESHOLD,
            "gripper_window": config.FILTER_GRIPPER_OSC_WINDOW,
            "min_score": args.min_score,
        },
        "summary": {
            "total_episodes": len(all_results),
            "flagged_episodes": len(filtered),
            "total_anomaly_events": total_anomalies,
            "anomaly_types": anomaly_types,
        },
        "all_episodes": [r.to_dict() for r in all_results],
        "flagged_episodes": [r.to_dict() for r in filtered],
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Saved to: {args.output}")

    # Optional Rerun visualization
    if args.rerun and filtered:
        print(f"\n  Launching Rerun viewer...")
        _visualize_filtered(episodes, all_results, filtered)

    print(f"\n{'=' * 60}")
    print(f"  Next: python3.10 omni3/scripts/02_train_esn.py")
    print(f"{'=' * 60}")


def _visualize_filtered(episodes, all_results, filtered):
    """Show filtered episodes in Rerun."""
    import rerun as rr
    from omni3.viz.rerun_logger import RerunLogger

    logger = RerunLogger()
    logger.init(spawn=True, recording_id="phase0_filter")

    # Show top 5 flagged episodes
    for result in filtered[:5]:
        ep = episodes[result.episode_idx]
        logger.log_episode(ep)
        logger.log_ee_trajectory(ep)
        logger.log_anomaly_events(result.events, prefix=f"ep_{ep.index}/anomalies")

    # Show anomaly score distribution
    for i, result in enumerate(all_results):
        rr.set_time_sequence("episode", i)
        rr.log("overview/anomaly_score", rr.Scalar(result.anomaly_score))
        rr.log("overview/anomaly_count", rr.Scalar(float(result.anomaly_count)))


if __name__ == "__main__":
    main()
