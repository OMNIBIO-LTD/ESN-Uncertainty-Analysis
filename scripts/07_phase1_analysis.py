#!/usr/bin/env python3
"""
Script 07: Phase 1 — Deep Sigma Analysis.

Three analyses to understand WHY and HOW sigma distinguishes success from failure:

  1. PER-JOINT SIGMA
     Which of the 7 Franka joints best predicts failure?
     Output: bar chart comparing per-joint error in success vs failure

  2. TEMPORAL SIGMA PROFILE
     Does sigma rise BEFORE the failure endpoint, or only at the end?
     Output: line chart of sigma over normalized episode time (0-100%)

  3. SUCCESS/FAILURE CLASSIFICATION
     Can sigma_mean alone classify whether an episode succeeded?
     Output: ROC curve with AUC score

Usage:
    python3.10 omni3/scripts/07_phase1_analysis.py
    python3.10 omni3/scripts/07_phase1_analysis.py --save-rrd output/phase1.rrd
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
    DROID_INPUT_DIM, DROID_FPS, DROID_JOINT_NAMES,
)
from omni3.analysis.phase1 import (
    compare_per_joint,
    compute_temporal_profiles,
    classify_episodes,
    plot_all,
)


def train_esn(successes, verbose=True):
    """Train ESN on successful episodes."""
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
        states, actions = ep["states"], ep["actions"]
        T = min(len(states), len(actions))
        for t in range(T - 1):
            inp = np.concatenate([states[t], actions[t]]).astype(np.float32)
            esn_state = esn.update(inp)
            if t >= warmup:
                readout.collect(esn_state, states[t + 1])

    if verbose:
        print(f"  Collected {readout.num_collected} samples from {len(successes)} episodes")
    metrics = readout.train()
    if verbose:
        print(f"  RMSE = {metrics['rmse']:.4f}")
    return esn, readout, metrics


def evaluate_all(esn, readout, episodes, warmup=10):
    """Evaluate sigma on a list of episodes. Returns list of result dicts."""
    results = []
    for ep in episodes:
        esn.reset()
        unc = UncertaintyEstimator(window_size=30)
        states, actions = ep["states"], ep["actions"]
        T = min(len(states), len(actions))

        readings = []
        for t in range(T - 1):
            inp = np.concatenate([states[t], actions[t]]).astype(np.float32)
            esn_state = esn.update(inp)
            if t >= warmup and readout.is_trained:
                predicted = readout.predict(esn_state)
                reading = unc.update(predicted, states[t + 1], frame_idx=t)
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
    parser = argparse.ArgumentParser(description="Phase 1: Deep Sigma Analysis")
    parser.add_argument("--output-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "phase1"))
    parser.add_argument("--save-rrd", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 1: Deep Sigma Analysis")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    print(f"\n[1/5] Loading DROID-100...")
    t0 = time.time()
    successes, failures = load_droid_episodes()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Train ESN ─────────────────────────────────────────────────
    print(f"\n[2/5] Training ESN on {len(successes)} successful episodes...")
    t0 = time.time()
    esn, readout, metrics = train_esn(successes)
    print(f"  Train time: {time.time() - t0:.1f}s")

    # ── Evaluate all episodes ─────────────────────────────────────
    print(f"\n[3/5] Evaluating sigma on all episodes...")
    success_results = evaluate_all(esn, readout, successes)
    failure_results = evaluate_all(esn, readout, failures)
    print(f"  Success: {len(success_results)} episodes")
    print(f"  Failure: {len(failure_results)} episodes")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 1: Per-Joint Sigma
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  ANALYSIS 1: Per-Joint Prediction Error")
    print(f"{'=' * 60}")
    print(f"  Which of the 7 Franka joints best predicts failure?")

    t0 = time.time()
    joint_result = compare_per_joint(esn, readout, successes, failures)
    print(f"  Computed in {time.time() - t0:.1f}s\n")

    print(f"  {'Joint':<12s}  {'Success':>10s}  {'Failure':>10s}  {'Ratio':>8s}")
    print(f"  {'-' * 44}")
    for i, name in enumerate(joint_result["joint_names"]):
        s = joint_result["success_mean"][i]
        f = joint_result["failure_mean"][i]
        r = joint_result["ratios"][i]
        marker = " <-- BEST" if r == joint_result["max_ratio"] else ""
        print(f"  {name:<12s}  {s:>10.4f}  {f:>10.4f}  {r:>7.2f}x{marker}")

    print(f"\n  Most predictive joint: {joint_result['most_predictive_joint']} "
          f"({joint_result['max_ratio']:.2f}x failure/success ratio)")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 2: Temporal Sigma Profile
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  ANALYSIS 2: Temporal Sigma Profile")
    print(f"{'=' * 60}")
    print(f"  Does sigma rise BEFORE the failure endpoint?")

    t0 = time.time()
    temporal_result = compute_temporal_profiles(esn, readout, successes, failures)
    print(f"  Computed in {time.time() - t0:.1f}s\n")

    div = temporal_result["divergence_pct"]
    if div < 100:
        print(f"  Failure sigma diverges at {div:.0f}% of episode completion")
        print(f"  This means the ESN detects trouble at {div:.0f}% — before the episode ends!")
        remaining = 100 - div
        print(f"  --> {remaining:.0f}% of the episode remains as warning time")
    else:
        print(f"  No clear divergence point found")
        print(f"  Failure sigma is elevated throughout but doesn't clearly diverge")

    # Show profile at key points
    print(f"\n  Sigma at key points:")
    for pct in [0, 25, 50, 75, 100]:
        idx = min(int(pct / 100 * (temporal_result["num_bins"] - 1)), temporal_result["num_bins"] - 1)
        s_val = temporal_result["success_avg"][idx]
        f_val = temporal_result["failure_avg"][idx]
        ratio = f_val / max(s_val, 1e-8)
        print(f"    {pct:3d}%: success={s_val:.4f}  failure={f_val:.4f}  ratio={ratio:.2f}x")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 3: Classification
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  ANALYSIS 3: Success/Failure Classification")
    print(f"{'=' * 60}")
    print(f"  Can sigma_mean alone classify success vs failure?")

    class_result = classify_episodes(success_results, failure_results)

    auc = class_result["auc"]
    m = class_result["metrics"]
    thr = class_result["optimal_threshold"]

    print(f"\n  AUC = {auc:.3f}", end="")
    if auc > 0.8:
        print(f"  (STRONG classifier)")
    elif auc > 0.65:
        print(f"  (MODERATE classifier)")
    elif auc > 0.5:
        print(f"  (WEAK classifier)")
    else:
        print(f"  (NO better than random)")

    print(f"\n  Optimal threshold: sigma_mean >= {thr:.4f} → predict FAILURE")
    print(f"  At this threshold:")
    print(f"    Precision: {m['precision']:.2f} ({m['tp']} true failures / {m['tp']+m['fp']} predicted)")
    print(f"    Recall:    {m['recall']:.2f} ({m['tp']} caught / {m['tp']+m['fn']} actual failures)")
    print(f"    F1 Score:  {m['f1']:.2f}")
    print(f"    Accuracy:  {m['accuracy']:.2f}")
    print(f"\n    Confusion: TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")

    # ── Generate plots ────────────────────────────────────────────
    print(f"\n[4/5] Generating plots...")
    plot_paths = plot_all(joint_result, temporal_result, class_result, args.output_dir)
    for p in plot_paths:
        print(f"  Saved: {p}")

    # ── Save results JSON ─────────────────────────────────────────
    results = {
        "per_joint": {
            "joint_names": joint_result["joint_names"],
            "success_mean": joint_result["success_mean"].tolist(),
            "failure_mean": joint_result["failure_mean"].tolist(),
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

    json_path = os.path.join(args.output_dir, "phase1_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # ── Rerun visualization ───────────────────────────────────────
    if args.save_rrd:
        print(f"\n[5/5] Saving Rerun recording...")
        _save_rerun(args.save_rrd, joint_result, temporal_result,
                     success_results, failure_results)
        print(f"  Saved: {args.save_rrd}")
    else:
        print(f"\n  Run with --save-rrd output/phase1.rrd to save Rerun recording")

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  PHASE 1 SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Most predictive joint: {joint_result['most_predictive_joint']} ({joint_result['max_ratio']:.2f}x ratio)")
    print(f"  Divergence point:      {temporal_result['divergence_pct']:.0f}% episode completion")
    print(f"  Classification AUC:    {auc:.3f}")
    print(f"  Optimal threshold:     sigma_mean >= {thr:.4f}")
    print(f"  F1 score:              {m['f1']:.2f}")
    print(f"\n  Plots saved to: {args.output_dir}/")
    print(f"{'=' * 60}")


def _save_rerun(rrd_path, joint_result, temporal_result, success_results, failure_results):
    """Save Phase 1 results to Rerun."""
    import rerun as rr
    import rerun.blueprint as rrb

    rr.init("phase1_analysis", recording_id="phase1")
    rr.save(rrd_path)

    # Log temporal profiles as time series
    x_pct = temporal_result["x_pct"]
    for i in range(len(x_pct)):
        rr.set_time_sequence("pct", i)
        rr.log("temporal/success_avg", rr.Scalars(float(temporal_result["success_avg"][i])))
        rr.log("temporal/failure_avg", rr.Scalars(float(temporal_result["failure_avg"][i])))

    # Log per-episode sigma timelines for a few failures
    for result in failure_results[:5]:
        ep_idx = result["episode_idx"]
        for reading in result["readings"]:
            rr.set_time_sequence("frame", reading.frame_idx)
            rr.log(f"failure/ep_{ep_idx}/sigma", rr.Scalars(reading.sigma_raw))

    for result in success_results[:5]:
        ep_idx = result["episode_idx"]
        for reading in result["readings"]:
            rr.set_time_sequence("frame", reading.frame_idx)
            rr.log(f"success/ep_{ep_idx}/sigma", rr.Scalars(reading.sigma_raw))


if __name__ == "__main__":
    main()
