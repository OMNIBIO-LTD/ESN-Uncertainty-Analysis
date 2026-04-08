#!/usr/bin/env python3
"""
Script 03: Validate ESN uncertainty signal.

KEY QUESTION: Does prediction error (sigma) spike BEFORE anomaly events?

Loads the trained ESN model, evaluates sigma on flagged episodes,
and checks whether sigma rises 0.5-2 seconds before detected anomalies.

If sigma spikes before failures → ESN is a valid uncertainty estimator.

Usage:
    python3.10 omni3/scripts/03_validate.py
    python3.10 omni3/scripts/03_validate.py --eval-episodes 0 1 2 3 4
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from omni3 import config
from omni3.data.loader import load_dataset, extract_episode
from omni3.pipeline import UncertaintyPipeline


def main():
    parser = argparse.ArgumentParser(description="Validate ESN uncertainty")
    parser.add_argument("--repo", default=config.DATASET_REPO_ID)
    parser.add_argument("--model", default=os.path.join(config.MODEL_DIR, "esn_readout.npz"))
    parser.add_argument("--filter-output", default=config.FILTER_OUTPUT_PATH)
    parser.add_argument("--eval-episodes", type=int, nargs="*", default=None,
                        help="Specific episodes to evaluate (default: flagged from filter)")
    parser.add_argument("--output-dir", default=config.VALIDATION_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print("  ESN Uncertainty Validation")
    print("=" * 60)

    # Load filter results for anomaly event locations
    filter_events: dict[int, list] = {}
    flagged_indices: list[int] = []
    if os.path.exists(args.filter_output):
        with open(args.filter_output) as f:
            filter_data = json.load(f)
        for ep in filter_data.get("flagged_episodes", []):
            idx = ep["episode_idx"]
            flagged_indices.append(idx)
            filter_events[idx] = ep.get("events", [])
        print(f"  Loaded {len(flagged_indices)} flagged episodes from Phase 0 filter")

    # Determine eval episodes
    if args.eval_episodes:
        eval_indices = args.eval_episodes
    elif flagged_indices:
        eval_indices = flagged_indices[:20]  # cap at 20
    else:
        # No filter: eval on last 20% of episodes
        dataset_temp = load_dataset(repo_id=args.repo, download_videos=False)
        n = dataset_temp.num_episodes
        eval_indices = list(range(int(n * 0.8), n))

    print(f"  Evaluating {len(eval_indices)} episodes")

    # Load model
    print(f"\n[1/3] Loading trained ESN from {args.model}")
    if not os.path.exists(args.model):
        print(f"  ERROR: Model not found! Run 02_train_esn.py first.")
        return
    pipeline = UncertaintyPipeline()
    pipeline.load(args.model)

    # Also train ESN reservoir states (need to re-run through training data)
    # The readout weights are loaded, but ESN needs fresh state per episode
    print(f"  Model loaded (readout weights)")

    # Load dataset and extract eval episodes
    print(f"\n[2/3] Extracting eval episodes...")
    dataset = load_dataset(repo_id=args.repo, download_videos=False)
    eval_episodes = [extract_episode(dataset, i) for i in eval_indices]

    # Evaluate
    print(f"\n[3/3] Computing sigma on {len(eval_episodes)} episodes...")
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for ep in eval_episodes:
        result = pipeline.evaluate(ep)
        all_results.append(result)

        s = result["stats"]
        print(f"  ep {ep.index:>4d}: sigma_mean={s['sigma_mean']:.4f} "
              f"max={s['sigma_max']:.4f} "
              f"caution={s['caution_count']} curiosity={s['curiosity_count']} "
              f"success={'Y' if ep.success else 'N'}")

        # Save sigma timeline
        np.savez(
            os.path.join(args.output_dir, f"sigma_ep_{ep.index}.npz"),
            sigmas_raw=result["sigmas_raw"],
            sigmas_norm=result["sigmas_norm"],
            timestamps=result["timestamps"],
            frame_indices=np.array(result["frame_indices"]),
        )

    # KEY ANALYSIS: Does sigma spike before anomaly events?
    print(f"\n{'=' * 60}")
    print(f"  KEY QUESTION: Does sigma spike BEFORE failures?")
    print(f"{'=' * 60}")

    fps = config.DATASET_FPS
    lookahead_s = 2.0  # check 2 seconds before each anomaly
    lookahead_frames = int(lookahead_s * fps)

    total_events = 0
    events_with_spike = 0

    for result in all_results:
        ep_idx = result["episode_idx"]
        events = filter_events.get(ep_idx, [])
        sigmas = result["sigmas_raw"]
        frame_indices = result["frame_indices"]

        if len(events) == 0 or len(sigmas) == 0:
            continue

        sigma_mean = float(sigmas.mean())
        sigma_std = float(sigmas.std()) if len(sigmas) > 1 else 1.0

        print(f"\n  Episode {ep_idx} ({len(events)} anomaly events):")

        for event in events[:5]:
            event_frame = event["frame"]
            event_type = event["type"]
            total_events += 1

            # Find sigma values in the window BEFORE the anomaly
            # Map event_frame to sigma array index
            warmup = config.ESN_WARMUP_STEPS
            if event_frame <= warmup:
                continue

            # Find closest sigma index
            sigma_idx = None
            for si, fi in enumerate(frame_indices):
                if fi >= event_frame:
                    sigma_idx = si
                    break
            if sigma_idx is None:
                sigma_idx = len(sigmas) - 1

            window_start = max(0, sigma_idx - lookahead_frames)
            window_end = sigma_idx

            if window_end <= window_start:
                continue

            pre_sigma = sigmas[window_start:window_end]
            pre_max = float(pre_sigma.max())
            z_score = (pre_max - sigma_mean) / max(sigma_std, 1e-8)

            spike = z_score > 1.5
            if spike:
                events_with_spike += 1
            icon = "[+]" if spike else "[ ]"

            print(f"    {icon} {event_type:15s} @ frame {event_frame:4d}: "
                  f"pre-sigma max={pre_max:.4f} (z={z_score:.2f})")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    if total_events > 0:
        hit_rate = 100 * events_with_spike / total_events
        print(f"  Anomaly events checked:  {total_events}")
        print(f"  Pre-spike detected:      {events_with_spike} ({hit_rate:.1f}%)")
        if hit_rate > 50:
            print(f"  --> PROMISING: sigma spikes before {hit_rate:.0f}% of anomalies")
        else:
            print(f"  --> WEAK: sigma only predicts {hit_rate:.0f}% of anomalies")
            print(f"      Consider tuning ESN parameters or using more training data")
    else:
        print(f"  No anomaly events to check (all episodes may be clean)")

    # Save validation summary
    summary = {
        "eval_episodes": len(eval_indices),
        "total_anomaly_events": total_events,
        "events_with_pre_spike": events_with_spike,
        "hit_rate": events_with_spike / max(total_events, 1),
        "episode_results": [
            {
                "episode_idx": r["episode_idx"],
                "task": r["task"],
                "success": r["success"],
                "stats": r["stats"],
            }
            for r in all_results
        ],
    }
    summary_path = os.path.join(args.output_dir, "validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to: {args.output_dir}/")

    print(f"\n{'=' * 60}")
    print(f"  Next: python3.10 omni3/scripts/04_visualize.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
