#!/usr/bin/env python3
"""
Script 08: Task-Specific ESN Analysis — DROID-100 Task 5.

Addresses Rob's feedback:
  1. Task-specific: train + eval within ONE task (no cross-task noise)
  2. Learning curve: repeated visits decrease sigma (generative model proof)
  3. Per-joint/temporal/classification — now task-specific

Task 5 has 35 success + 19 failure episodes of the same Franka manipulation task.

Usage:
    python3.10 omni3/scripts/08_task_specific.py
    python3.10 omni3/scripts/08_task_specific.py --n-runs 10
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
    load_droid_task_episodes, DROID_STATE_DIM, DROID_INPUT_DIM,
    DROID_JOINT_NAMES,
)
from omni3.analysis.task_analysis import run_incremental_learning_curve, plot_learning_curve
from omni3.analysis.phase1 import (
    compare_per_joint, compute_temporal_profiles,
    classify_episodes, plot_all,
)


def train_esn(successes, verbose=True):
    """Train ESN on success episodes."""
    esn = EchoStateNetwork(
        input_dim=DROID_INPUT_DIM, reservoir_size=512,
        spectral_radius=0.95, input_scaling=0.3,
        leaking_rate=0.3, seed=42,
    )
    readout = PredictionReadout(reservoir_size=512, output_dim=DROID_STATE_DIM, alpha=0.01)

    for ep in successes:
        esn.reset()
        T = min(len(ep["states"]), len(ep["actions"]))
        for t in range(T - 1):
            inp = np.concatenate([ep["states"][t], ep["actions"][t]]).astype(np.float32)
            esn_state = esn.update(inp)
            if t >= 10:
                readout.collect(esn_state, ep["states"][t + 1])

    if verbose:
        print(f"  Collected {readout.num_collected} samples from {len(successes)} episodes")
    metrics = readout.train()
    if verbose:
        print(f"  RMSE = {metrics['rmse']:.4f}")
    return esn, readout, metrics


def evaluate_all(esn, readout, episodes):
    """Evaluate sigma on a list of episodes."""
    results = []
    for ep in episodes:
        esn.reset()
        unc = UncertaintyEstimator(window_size=30)
        T = min(len(ep["states"]), len(ep["actions"]))

        readings = []
        for t in range(T - 1):
            inp = np.concatenate([ep["states"][t], ep["actions"][t]]).astype(np.float32)
            esn_state = esn.update(inp)
            if t >= 10 and readout.is_trained:
                predicted = readout.predict(esn_state)
                reading = unc.update(predicted, ep["states"][t + 1], frame_idx=t)
                readings.append(reading)

        sigmas = np.array([r.sigma_raw for r in readings]) if readings else np.zeros(1)
        results.append({
            "episode_idx": ep["index"],
            "task": ep["task"],
            "success": ep["success"],
            "sigmas_raw": sigmas,
            "readings": readings,
            "stats": {
                "sigma_mean": float(sigmas.mean()),
                "sigma_std": float(sigmas.std()),
                "sigma_max": float(sigmas.max()),
                "caution_count": sum(1 for r in readings if r.zone == "caution"),
                "curiosity_count": sum(1 for r in readings if r.zone == "curiosity"),
            },
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Task-Specific ESN Analysis")
    parser.add_argument("--task-index", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Number of shuffle runs for learning curve")
    parser.add_argument("--n-validation", type=int, default=5,
                        help="Validation episodes held out for learning curve")
    parser.add_argument("--output-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output", "task_specific"))
    args = parser.parse_args()

    print("=" * 60)
    print("  Task-Specific ESN Analysis")
    print("=" * 60)

    # ── Load task-specific data ───────────────────────────────────
    print(f"\n[1/5] Loading Task {args.task_index} episodes...")
    t0 = time.time()
    successes, failures = load_droid_task_episodes(task_index=args.task_index)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    s_lens = [ep["num_frames"] for ep in successes]
    f_lens = [ep["num_frames"] for ep in failures]
    print(f"  Success: {len(successes)} eps (avg {np.mean(s_lens):.0f} frames)")
    print(f"  Failure: {len(failures)} eps (avg {np.mean(f_lens):.0f} frames)")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS A: Incremental Learning Curve
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  ANALYSIS A: Incremental Learning Curve")
    print(f"{'=' * 60}")
    print(f"  Does sigma decrease as the ESN sees more of this task?")
    print(f"  Training incrementally: 1 → {len(successes) - args.n_validation} episodes")
    print(f"  Validation set: {args.n_validation} held-out episodes")
    print(f"  Runs: {args.n_runs} (different shuffles for error bars)")

    t0 = time.time()
    lc_result = run_incremental_learning_curve(
        successes,
        n_validation=args.n_validation,
        n_runs=args.n_runs,
    )
    lc_time = time.time() - t0
    print(f"  Computed in {lc_time:.1f}s")

    # Print learning curve table
    sigma_mean = lc_result["sigma_mean"]
    n_eps = lc_result["n_episodes"]
    print(f"\n  {'N eps':>6s}  {'Sigma':>8s}  {'RMSE':>8s}")
    print(f"  {'-' * 26}")
    for step in [0, 4, 9, 14, 19, 24, 29]:
        if step < len(sigma_mean):
            s = sigma_mean[step]
            r = lc_result["rmse_mean"][step]
            print(f"  {n_eps[step]:>6d}  {s:>8.4f}  {r:>8.4f}")

    start_sigma = sigma_mean[0]
    end_sigma = sigma_mean[-1]
    if not (np.isnan(start_sigma) or np.isnan(end_sigma)):
        decrease = (1 - end_sigma / start_sigma) * 100
        print(f"\n  Sigma decreased by {decrease:.1f}% from 1 to {n_eps[-1]} episodes")
        if decrease > 10:
            print(f"  --> CONFIRMED: Repeated visits decrease uncertainty")
            print(f"      The ESN IS building a generative model")
        elif decrease > 0:
            print(f"  --> MARGINAL decrease — ESN learns slowly on this task")
        else:
            print(f"  --> NO decrease — check ESN parameters")

    # Generate learning curve plot
    os.makedirs(args.output_dir, exist_ok=True)
    lc_path = plot_learning_curve(lc_result, os.path.join(args.output_dir, "learning_curve.png"))
    print(f"\n  Plot saved: {lc_path}")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS B: Task-Specific Phase 1
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  ANALYSIS B: Task-Specific Per-Joint / Temporal / Classification")
    print(f"{'=' * 60}")

    # Train on ALL task 5 successes
    print(f"\n[2/5] Training ESN on {len(successes)} task-specific successes...")
    t0 = time.time()
    esn, readout, metrics = train_esn(successes)
    print(f"  Train time: {time.time() - t0:.1f}s")

    # Evaluate
    print(f"\n[3/5] Evaluating...")
    success_results = evaluate_all(esn, readout, successes)
    failure_results = evaluate_all(esn, readout, failures)

    # Sigma comparison
    s_means = [r["stats"]["sigma_mean"] for r in success_results]
    f_means = [r["stats"]["sigma_mean"] for r in failure_results]
    ratio = np.mean(f_means) / max(np.mean(s_means), 1e-8)
    print(f"\n  Task-specific sigma ratio: {ratio:.2f}x (failure/success)")
    print(f"  (Previous all-task ratio was 1.35x)")

    # Per-joint
    print(f"\n[4/5] Per-joint analysis...")
    joint_result = compare_per_joint(esn, readout, successes, failures)
    print(f"\n  {'Joint':<12s}  {'Success':>10s}  {'Failure':>10s}  {'Ratio':>8s}")
    print(f"  {'-' * 44}")
    for i, name in enumerate(DROID_JOINT_NAMES):
        s = joint_result["success_mean"][i]
        f = joint_result["failure_mean"][i]
        r = joint_result["ratios"][i]
        marker = " <-- BEST" if r == joint_result["max_ratio"] else ""
        print(f"  {name:<12s}  {s:>10.4f}  {f:>10.4f}  {r:>7.2f}x{marker}")

    # Temporal
    temporal_result = compute_temporal_profiles(esn, readout, successes, failures)
    div = temporal_result["divergence_pct"]
    print(f"\n  Temporal divergence: {div:.0f}% completion")

    # Classification
    class_result = classify_episodes(success_results, failure_results)
    auc = class_result["auc"]
    m = class_result["metrics"]
    print(f"\n  Classification AUC: {auc:.3f} (task-specific)")
    print(f"  Optimal threshold: sigma_mean >= {class_result['optimal_threshold']:.4f}")
    print(f"  Precision: {m['precision']:.2f}, Recall: {m['recall']:.2f}, F1: {m['f1']:.2f}")

    # Generate Phase 1 plots (task-specific)
    print(f"\n[5/5] Generating plots...")
    plot_paths = plot_all(joint_result, temporal_result, class_result, args.output_dir)
    for p in plot_paths:
        # Rename to task5_ prefix
        base = os.path.basename(p)
        new_name = f"task5_{base}"
        new_path = os.path.join(args.output_dir, new_name)
        os.rename(p, new_path)
        print(f"  Saved: {new_path}")

    # Save results
    results = {
        "task_index": args.task_index,
        "n_success": len(successes),
        "n_failure": len(failures),
        "learning_curve": {
            "n_episodes": lc_result["n_episodes"],
            "sigma_mean": lc_result["sigma_mean"].tolist(),
            "sigma_start": float(start_sigma),
            "sigma_end": float(end_sigma),
            "decrease_pct": float(decrease) if not np.isnan(start_sigma) else 0,
            "n_runs": args.n_runs,
        },
        "sigma_ratio": float(ratio),
        "per_joint": {
            "joint_names": DROID_JOINT_NAMES,
            "ratios": joint_result["ratios"].tolist(),
            "most_predictive": joint_result["most_predictive_joint"],
            "max_ratio": joint_result["max_ratio"],
        },
        "temporal": {
            "divergence_pct": temporal_result["divergence_pct"],
        },
        "classification": {
            "auc": class_result["auc"],
            "optimal_threshold": class_result["optimal_threshold"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "accuracy": m["accuracy"],
        },
        "training": {
            "rmse": metrics["rmse"],
            "num_samples": metrics["num_samples"],
        },
    }

    json_path = os.path.join(args.output_dir, "task5_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  TASK-SPECIFIC SUMMARY (Task {args.task_index})")
    print(f"{'=' * 60}")
    print(f"  Learning curve:    {decrease:.1f}% sigma decrease over {n_eps[-1]} episodes")
    print(f"  Sigma ratio:       {ratio:.2f}x (failure/success)")
    print(f"  Most predictive:   {joint_result['most_predictive_joint']} ({joint_result['max_ratio']:.2f}x)")
    print(f"  Temporal diverge:  {div:.0f}% completion")
    print(f"  Classification:    AUC={auc:.3f}, F1={m['f1']:.2f}")
    print(f"\n  All results in: {args.output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
