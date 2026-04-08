"""
Phase 1 analysis: Per-joint sigma, temporal profiles, classification.

Three analyses to deepen the Phase 0 finding (1.35x sigma ratio):

1. Per-joint sigma — which joints drive the failure signal?
2. Temporal profile — does sigma rise BEFORE the failure endpoint?
3. Classification — ROC/AUC: can sigma alone classify success/failure?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from omni3.core.esn import EchoStateNetwork
from omni3.core.readout import PredictionReadout
from omni3.core.uncertainty import UncertaintyEstimator
from omni3.data.droid_loader import DroidEpisode, DROID_JOINT_NAMES


# ── Analysis 1: Per-Joint Errors ─────────────────────────────────


def compute_per_joint_errors(
    esn: EchoStateNetwork,
    readout: PredictionReadout,
    ep_data: DroidEpisode,
    warmup: int = 10,
) -> np.ndarray:
    """
    Compute per-joint absolute prediction error for one episode.

    Returns:
        (N, 7) array where N = frames after warmup, 7 = joints.
        Each value = |predicted_joint[j] - actual_joint[j]|
    """
    esn.reset()
    states = ep_data["states"]
    actions = ep_data["actions"]
    T = min(len(states), len(actions))

    errors = []
    for t in range(T - 1):
        inp = np.concatenate([states[t], actions[t]]).astype(np.float32)
        esn_state = esn.update(inp)

        if t >= warmup and readout.is_trained:
            predicted = readout.predict(esn_state)
            actual = states[t + 1]
            per_joint = np.abs(predicted - actual)
            errors.append(per_joint)

    return np.array(errors) if errors else np.zeros((1, states.shape[1]))


def compare_per_joint(
    esn: EchoStateNetwork,
    readout: PredictionReadout,
    successes: list[DroidEpisode],
    failures: list[DroidEpisode],
) -> dict:
    """
    Compare per-joint errors between success and failure groups.

    Returns dict with per-joint means and failure/success ratios.
    """
    success_errors = []
    for ep in successes:
        errs = compute_per_joint_errors(esn, readout, ep)
        success_errors.append(errs.mean(axis=0))  # mean per joint across time
    success_mean = np.mean(success_errors, axis=0)  # mean across episodes

    failure_errors = []
    for ep in failures:
        errs = compute_per_joint_errors(esn, readout, ep)
        failure_errors.append(errs.mean(axis=0))
    failure_mean = np.mean(failure_errors, axis=0)

    ratios = failure_mean / np.maximum(success_mean, 1e-8)

    return {
        "joint_names": DROID_JOINT_NAMES,
        "success_mean": success_mean,
        "failure_mean": failure_mean,
        "ratios": ratios,
        "most_predictive_joint": DROID_JOINT_NAMES[int(np.argmax(ratios))],
        "max_ratio": float(ratios.max()),
    }


# ── Analysis 2: Temporal Sigma Profiles ──────────────────────────


def build_temporal_profile(
    sigmas: np.ndarray,
    num_bins: int = 100,
) -> np.ndarray:
    """
    Resample a sigma timeline to fixed-length bins (0-100% completion).

    Args:
        sigmas: (N,) raw sigma values for one episode.
        num_bins: Number of output bins.

    Returns:
        (num_bins,) resampled sigma profile.
    """
    if len(sigmas) < 2:
        return np.zeros(num_bins)

    # Interpolate to num_bins evenly spaced points
    x_orig = np.linspace(0, 1, len(sigmas))
    x_new = np.linspace(0, 1, num_bins)
    return np.interp(x_new, x_orig, sigmas)


def compute_temporal_profiles(
    esn: EchoStateNetwork,
    readout: PredictionReadout,
    successes: list[DroidEpisode],
    failures: list[DroidEpisode],
    num_bins: int = 100,
    warmup: int = 10,
) -> dict:
    """
    Build averaged temporal sigma profiles for success vs failure.

    Returns dict with success_profile, failure_profile, divergence_point.
    """
    success_profiles = []
    for ep in successes:
        sigmas = _get_sigma_timeline(esn, readout, ep, warmup)
        profile = build_temporal_profile(sigmas, num_bins)
        success_profiles.append(profile)

    failure_profiles = []
    for ep in failures:
        sigmas = _get_sigma_timeline(esn, readout, ep, warmup)
        profile = build_temporal_profile(sigmas, num_bins)
        failure_profiles.append(profile)

    success_avg = np.mean(success_profiles, axis=0)
    failure_avg = np.mean(failure_profiles, axis=0)
    success_std = np.std(success_profiles, axis=0)
    failure_std = np.std(failure_profiles, axis=0)

    # Find divergence point: where failure consistently exceeds success
    diff = failure_avg - success_avg
    divergence_idx = _find_divergence(diff, threshold=0.01)
    divergence_pct = divergence_idx / num_bins * 100

    return {
        "success_avg": success_avg,
        "failure_avg": failure_avg,
        "success_std": success_std,
        "failure_std": failure_std,
        "x_pct": np.linspace(0, 100, num_bins),
        "divergence_pct": float(divergence_pct),
        "num_bins": num_bins,
    }


def _get_sigma_timeline(
    esn: EchoStateNetwork,
    readout: PredictionReadout,
    ep_data: DroidEpisode,
    warmup: int = 10,
) -> np.ndarray:
    """Get raw sigma values for one episode."""
    esn.reset()
    uncertainty = UncertaintyEstimator(window_size=30)
    states = ep_data["states"]
    actions = ep_data["actions"]
    T = min(len(states), len(actions))

    sigmas = []
    for t in range(T - 1):
        inp = np.concatenate([states[t], actions[t]]).astype(np.float32)
        esn_state = esn.update(inp)
        if t >= warmup and readout.is_trained:
            predicted = readout.predict(esn_state)
            sigma = float(np.linalg.norm(predicted - states[t + 1]))
            sigmas.append(sigma)

    return np.array(sigmas) if sigmas else np.zeros(1)


def _find_divergence(diff: np.ndarray, threshold: float = 0.01) -> int:
    """Find first index where diff stays above threshold for 10+ consecutive bins."""
    run = 0
    for i, d in enumerate(diff):
        if d > threshold:
            run += 1
            if run >= 10:
                return i - 9
        else:
            run = 0
    return len(diff)  # no divergence found


# ── Analysis 3: Classification ───────────────────────────────────


def classify_episodes(
    success_results: list[dict],
    failure_results: list[dict],
) -> dict:
    """
    ROC curve + AUC using sigma_mean as classifier.

    Args:
        success_results: List of evaluate results for success episodes.
        failure_results: List of evaluate results for failure episodes.

    Returns dict with thresholds, TPR, FPR, AUC, optimal threshold, metrics.
    """
    # Labels: 1 = failure (positive class), 0 = success
    scores = []
    labels = []
    for r in success_results:
        scores.append(r["stats"]["sigma_mean"])
        labels.append(0)
    for r in failure_results:
        scores.append(r["stats"]["sigma_mean"])
        labels.append(1)

    scores = np.array(scores)
    labels = np.array(labels)

    # Sort thresholds
    thresholds = np.sort(np.unique(scores))
    # Add boundaries
    thresholds = np.concatenate([[scores.min() - 0.01], thresholds, [scores.max() + 0.01]])

    tpr_list = []
    fpr_list = []
    for thr in thresholds:
        predicted_pos = scores >= thr
        tp = np.sum(predicted_pos & (labels == 1))
        fp = np.sum(predicted_pos & (labels == 0))
        fn = np.sum(~predicted_pos & (labels == 1))
        tn = np.sum(~predicted_pos & (labels == 0))

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)

    # AUC via trapezoidal rule (sort by FPR)
    order = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[order]
    tpr_sorted = tpr_arr[order]
    auc = float(np.trapz(tpr_sorted, fpr_sorted))

    # Optimal threshold (Youden's J = TPR - FPR, maximize)
    j_scores = tpr_arr - fpr_arr
    best_idx = int(np.argmax(j_scores))
    optimal_thr = float(thresholds[best_idx])

    # Metrics at optimal threshold
    predicted = scores >= optimal_thr
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / len(labels)

    return {
        "thresholds": thresholds,
        "tpr": tpr_arr,
        "fpr": fpr_arr,
        "auc": auc,
        "optimal_threshold": optimal_thr,
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
    }


# ── Plotting ─────────────────────────────────────────────────────


def plot_all(
    joint_result: dict,
    temporal_result: dict,
    classification_result: dict,
    output_dir: str,
) -> list[str]:
    """Generate all Phase 1 matplotlib plots. Returns list of saved paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. Per-joint comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(joint_result["joint_names"]))
    w = 0.35
    ax.bar(x - w / 2, joint_result["success_mean"], w, label="Success", color="#2ecc71", alpha=0.8)
    ax.bar(x + w / 2, joint_result["failure_mean"], w, label="Failure", color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Joint")
    ax.set_ylabel("Mean Prediction Error")
    ax.set_title("Per-Joint Prediction Error: Success vs Failure")
    ax.set_xticks(x)
    ax.set_xticklabels(joint_result["joint_names"], rotation=45, ha="right")
    ax.legend()

    # Add ratio labels
    for i, ratio in enumerate(joint_result["ratios"]):
        y = max(joint_result["success_mean"][i], joint_result["failure_mean"][i])
        ax.text(i, y * 1.05, f"{ratio:.2f}x", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    p = str(Path(output_dir) / "per_joint_comparison.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 2. Temporal sigma profile
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pct = temporal_result["x_pct"]
    s_avg = temporal_result["success_avg"]
    f_avg = temporal_result["failure_avg"]
    s_std = temporal_result["success_std"]
    f_std = temporal_result["failure_std"]

    ax.plot(x_pct, s_avg, color="#2ecc71", linewidth=2, label="Success (avg)")
    ax.fill_between(x_pct, s_avg - s_std, s_avg + s_std, color="#2ecc71", alpha=0.15)
    ax.plot(x_pct, f_avg, color="#e74c3c", linewidth=2, label="Failure (avg)")
    ax.fill_between(x_pct, f_avg - f_std, f_avg + f_std, color="#e74c3c", alpha=0.15)

    div = temporal_result["divergence_pct"]
    if div < 100:
        ax.axvline(div, color="orange", linestyle="--", linewidth=1.5,
                   label=f"Divergence @ {div:.0f}%")

    ax.set_xlabel("Episode Completion (%)")
    ax.set_ylabel("Sigma (Prediction Error)")
    ax.set_title("Temporal Sigma Profile: When Does Failure Diverge?")
    ax.legend()
    plt.tight_layout()
    p = str(Path(output_dir) / "temporal_sigma_profile.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 3. ROC curve
    fig, ax = plt.subplots(figsize=(6, 6))
    # Sort by FPR for proper curve
    order = np.argsort(classification_result["fpr"])
    fpr = classification_result["fpr"][order]
    tpr = classification_result["tpr"][order]
    auc = classification_result["auc"]

    ax.plot(fpr, tpr, color="#3498db", linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.3, label="Random (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#3498db")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Can Sigma Classify Success vs Failure?")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    plt.tight_layout()
    p = str(Path(output_dir) / "roc_curve.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    return paths
