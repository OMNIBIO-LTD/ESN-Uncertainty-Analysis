#!/usr/bin/env python3
"""
Script 06: Analyze DROID-100 failures with ESN uncertainty.

DROID-100 (lerobot/droid_100) is a Franka Panda dataset with:
  - 100 episodes (81 success, 19 FAILED)
  - 47 diverse manipulation tasks
  - State (7): joint positions
  - Action (7): joint commands
  - 3 cameras at 15 FPS

THIS is where ESN uncertainty gets interesting:
  - Train on successful episodes
  - Evaluate on failed episodes
  - Does sigma spike during/before failures?

Usage:
    python3.10 omni3/scripts/06_droid_failures.py
    python3.10 omni3/scripts/06_droid_failures.py --rerun
    python3.10 omni3/scripts/06_droid_failures.py --save-rrd output/droid_failures.rrd
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from omni3.core.esn import EchoStateNetwork
from omni3.core.readout import PredictionReadout
from omni3.core.uncertainty import UncertaintyEstimator
from omni3.data.droid_loader import (
    load_droid_episodes, DROID_STATE_DIM, DROID_ACTION_DIM,
    DROID_INPUT_DIM, DROID_FPS, DROID_JOINT_NAMES, DROID_REPO,
)


def train_esn_on_droid(successes, verbose=True):
    """Train ESN on successful DROID episodes."""
    esn = EchoStateNetwork(
        input_dim=DROID_INPUT_DIM,
        reservoir_size=512,
        spectral_radius=0.95,
        input_scaling=0.3,
        leaking_rate=0.3,
        seed=42,
    )
    readout = PredictionReadout(
        reservoir_size=512,
        output_dim=DROID_STATE_DIM,
        alpha=0.01,
    )

    warmup = 10

    for ep in successes:
        esn.reset()
        states = ep["states"]
        actions = ep["actions"]
        T = min(len(states), len(actions))

        for t in range(T - 1):
            inp = np.concatenate([states[t], actions[t]]).astype(np.float32)
            esn_state = esn.update(inp)
            if t >= warmup:
                readout.collect(esn_state, states[t + 1])

    if verbose:
        print(f"  Collected {readout.num_collected} training samples "
              f"from {len(successes)} successful episodes")

    metrics = readout.train()

    if verbose:
        print(f"  Training done: MSE={metrics['mse']:.6f}, RMSE={metrics['rmse']:.4f}")

    return esn, readout, metrics


def evaluate_episode(esn, readout, ep_data, warmup=10):
    """Evaluate sigma on one episode."""
    esn.reset()
    uncertainty = UncertaintyEstimator(window_size=30, caution_threshold=1.5, curiosity_threshold=3.0)

    states = ep_data["states"]
    actions = ep_data["actions"]
    T = min(len(states), len(actions))

    readings = []
    for t in range(T - 1):
        inp = np.concatenate([states[t], actions[t]]).astype(np.float32)
        esn_state = esn.update(inp)

        if t >= warmup and readout.is_trained:
            predicted = readout.predict(esn_state)
            reading = uncertainty.update(predicted, states[t + 1], frame_idx=t)
            readings.append(reading)

    sigmas_raw = np.array([r.sigma_raw for r in readings]) if readings else np.zeros(1)
    sigmas_norm = np.array([r.sigma_normalized for r in readings]) if readings else np.zeros(1)

    return {
        "episode_idx": ep_data["index"],
        "task": ep_data["task"],
        "success": ep_data["success"],
        "num_frames": T,
        "sigmas_raw": sigmas_raw,
        "sigmas_norm": sigmas_norm,
        "readings": readings,
        "stats": {
            "sigma_mean": float(sigmas_raw.mean()),
            "sigma_std": float(sigmas_raw.std()),
            "sigma_max": float(sigmas_raw.max()),
            "caution_count": sum(1 for r in readings if r.zone == "caution"),
            "curiosity_count": sum(1 for r in readings if r.zone == "curiosity"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="DROID-100 failure analysis")
    parser.add_argument("--rerun", action="store_true", help="Visualize in Rerun")
    parser.add_argument("--save-rrd", default=None)
    parser.add_argument("--output-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "droid"))
    args = parser.parse_args()

    print("=" * 60)
    print("  DROID-100: Success vs Failure Analysis")
    print("=" * 60)

    # Load data
    print(f"\n[1/4] Loading DROID-100 dataset...")
    t0 = time.time()
    successes, failures = load_droid_episodes()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Successful: {len(successes)} episodes")
    print(f"  Failed:     {len(failures)} episodes")
    print(f"  State dim:  {DROID_STATE_DIM} (Franka joint positions)")
    print(f"  Action dim: {DROID_ACTION_DIM} (joint commands)")

    # Show failed episode tasks
    print(f"\n  Failed episode tasks:")
    for ep in failures:
        print(f"    ep {ep['index']:>3d}: {ep['num_frames']:3d} frames  \"{ep['task'][:55]}\"")

    # Train on successes
    print(f"\n[2/4] Training ESN on {len(successes)} successful episodes...")
    t0 = time.time()
    esn, readout, metrics = train_esn_on_droid(successes)
    print(f"  Train time: {time.time() - t0:.2f}s")

    # Evaluate on BOTH successes and failures
    print(f"\n[3/4] Evaluating sigma on all episodes...")
    success_results = []
    failure_results = []

    print(f"\n  --- Successful episodes ---")
    for ep in successes[:20]:  # sample 20 successes
        result = evaluate_episode(esn, readout, ep)
        success_results.append(result)

    print(f"\n  --- Failed episodes ---")
    for ep in failures:
        result = evaluate_episode(esn, readout, ep)
        failure_results.append(result)
        s = result["stats"]
        print(f"  ep {ep['index']:>3d}: sigma_mean={s['sigma_mean']:.4f} "
              f"max={s['sigma_max']:.4f} "
              f"caution={s['caution_count']} curiosity={s['curiosity_count']} "
              f"\"{ep['task'][:40]}\"")

    # KEY COMPARISON: sigma in successes vs failures
    print(f"\n{'=' * 60}")
    print(f"  KEY RESULT: Success vs Failure Sigma")
    print(f"{'=' * 60}")

    success_means = [r["stats"]["sigma_mean"] for r in success_results]
    failure_means = [r["stats"]["sigma_mean"] for r in failure_results]
    success_maxes = [r["stats"]["sigma_max"] for r in success_results]
    failure_maxes = [r["stats"]["sigma_max"] for r in failure_results]

    print(f"\n  Sigma MEAN:")
    print(f"    Successes: {np.mean(success_means):.4f} +/- {np.std(success_means):.4f}")
    print(f"    Failures:  {np.mean(failure_means):.4f} +/- {np.std(failure_means):.4f}")

    mean_ratio = np.mean(failure_means) / max(np.mean(success_means), 1e-8)
    print(f"    Ratio:     {mean_ratio:.2f}x")

    print(f"\n  Sigma MAX:")
    print(f"    Successes: {np.mean(success_maxes):.4f} +/- {np.std(success_maxes):.4f}")
    print(f"    Failures:  {np.mean(failure_maxes):.4f} +/- {np.std(failure_maxes):.4f}")

    max_ratio = np.mean(failure_maxes) / max(np.mean(success_maxes), 1e-8)
    print(f"    Ratio:     {max_ratio:.2f}x")

    if mean_ratio > 1.2:
        print(f"\n  --> PROMISING: Failed episodes have {mean_ratio:.1f}x higher sigma")
        print(f"      ESN uncertainty DOES distinguish success from failure")
    elif mean_ratio > 1.0:
        print(f"\n  --> MARGINAL: Failed episodes have slightly higher sigma ({mean_ratio:.2f}x)")
    else:
        print(f"\n  --> NO SIGNAL: Sigma is similar for success and failure")

    # Caution/curiosity comparison
    success_caution = [r["stats"]["caution_count"] for r in success_results]
    failure_caution = [r["stats"]["caution_count"] for r in failure_results]

    print(f"\n  Caution zone events:")
    print(f"    Successes: {np.mean(success_caution):.1f} avg per episode")
    print(f"    Failures:  {np.mean(failure_caution):.1f} avg per episode")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "dataset": DROID_REPO,
        "num_successes": len(successes),
        "num_failures": len(failures),
        "training_metrics": metrics,
        "sigma_comparison": {
            "success_mean": float(np.mean(success_means)),
            "failure_mean": float(np.mean(failure_means)),
            "ratio": float(mean_ratio),
            "success_max_mean": float(np.mean(success_maxes)),
            "failure_max_mean": float(np.mean(failure_maxes)),
        },
        "failed_episodes": [
            {
                "episode_idx": r["episode_idx"],
                "task": r["task"],
                "stats": r["stats"],
            }
            for r in failure_results
        ],
    }

    summary_path = os.path.join(args.output_dir, "droid_failure_analysis.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to: {summary_path}")

    # Rerun visualization
    if args.rerun or args.save_rrd:
        print(f"\n[4/4] Launching Rerun visualization...")
        _visualize_droid(successes, failures, esn, readout,
                         success_results, failure_results, args)
    else:
        print(f"\n  Run with --rerun or --save-rrd to visualize")

    print(f"\n{'=' * 60}")
    print(f"  Analysis complete!")
    print(f"{'=' * 60}")


def _visualize_droid(successes, failures, esn, readout,
                     success_results, failure_results, args):
    """Visualize success vs failure in Rerun."""
    import rerun as rr
    import rerun.blueprint as rrb

    if args.save_rrd:
        rr.init("droid_failure_analysis", recording_id="droid_failures")
        rr.save(args.save_rrd)
    else:
        rr.init("droid_failure_analysis", recording_id="droid_failures", spawn=True)

    # Log a few successes
    for i, (ep, result) in enumerate(zip(successes[:3], success_results[:3])):
        prefix = f"success/ep_{ep['index']}"
        _log_droid_episode(ep, result, prefix)

    # Log ALL failures
    for ep, result in zip(failures, failure_results):
        prefix = f"failure/ep_{ep['index']}"
        _log_droid_episode(ep, result, prefix)

    # Log sigma comparison as overview
    for i, result in enumerate(success_results):
        rr.set_time_sequence("episode", i)
        rr.log("overview/success_sigma", rr.Scalars(result["stats"]["sigma_mean"]))

    for i, result in enumerate(failure_results):
        rr.set_time_sequence("episode", len(success_results) + i)
        rr.log("overview/failure_sigma", rr.Scalars(result["stats"]["sigma_mean"]))

    # Blueprint
    if failure_results:
        fail_ep = failure_results[0]["episode_idx"]
        bp = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.TimeSeriesView(
                        name="Failed Episode Joints",
                        contents=[f"failure/ep_{fail_ep}/joints/**"],
                    ),
                    rrb.TimeSeriesView(
                        name="Failed Episode Sigma",
                        contents=[f"failure/ep_{fail_ep}/sigma/**"],
                    ),
                    row_shares=[2, 1],
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(
                        name="Sigma Overview",
                        contents=["overview/**"],
                    ),
                ),
                column_shares=[2, 1],
            ),
        )
        rr.send_blueprint(bp)

    if args.save_rrd:
        print(f"  Saved to: {args.save_rrd}")


def _log_droid_episode(ep_data, result, prefix):
    """Log one DROID episode to Rerun."""
    import rerun as rr

    states = ep_data["states"]
    T = ep_data["num_frames"]
    timestamps = ep_data["timestamps"]

    for t in range(T):
        rr.set_time_sequence("frame", t)
        rr.set_time_seconds("time", float(timestamps[t]))

        for j in range(min(7, states.shape[1])):
            rr.log(f"{prefix}/joints/j{j}", rr.Scalars(float(states[t, j])))

    # Log sigma
    for reading in result["readings"]:
        rr.set_time_sequence("frame", reading.frame_idx)
        rr.set_time_seconds("time", float(timestamps[reading.frame_idx]) if reading.frame_idx < len(timestamps) else 0)
        rr.log(f"{prefix}/sigma/raw", rr.Scalars(reading.sigma_raw))
        rr.log(f"{prefix}/sigma/zscore", rr.Scalars(reading.sigma_normalized))

        if reading.zone != "normal":
            rr.log(f"{prefix}/sigma/zone", rr.TextLog(
                text=f"{reading.zone}: sigma={reading.sigma_raw:.4f}",
                level=rr.TextLogLevel.WARN if reading.zone == "caution" else rr.TextLogLevel.ERROR,
            ))


if __name__ == "__main__":
    main()
