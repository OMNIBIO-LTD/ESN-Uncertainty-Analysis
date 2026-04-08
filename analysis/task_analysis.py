"""
Task-specific ESN analysis for DROID-100 Task 5.

Two key analyses:

1. Incremental Learning Curve
   Train on 1, 2, ..., N episodes. Sigma should decrease — proving
   the ESN builds a generative model (repeated visits → less uncertainty).

2. Task-specific plots (per-joint, temporal, classification)
   Same Phase 1 analyses, but within a single task.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from omni3.core.esn import EchoStateNetwork
from omni3.core.readout import PredictionReadout
from omni3.data.droid_loader import (
    DroidEpisode, DROID_INPUT_DIM, DROID_STATE_DIM, DROID_JOINT_NAMES,
)


def _precompute_reservoir_states(
    episodes: list[DroidEpisode],
    esn: EchoStateNetwork,
    warmup: int = 10,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Run all episodes through the ESN once and store reservoir states + targets.

    Returns:
        (all_states, all_targets) — lists of (N_i, 512) and (N_i, 7) arrays.
    """
    all_states = []
    all_targets = []

    for ep in episodes:
        esn.reset()
        states_list = []
        targets_list = []
        T = min(len(ep["states"]), len(ep["actions"]))

        for t in range(T - 1):
            inp = np.concatenate([ep["states"][t], ep["actions"][t]]).astype(np.float32)
            esn_state = esn.update(inp)
            if t >= warmup:
                states_list.append(esn_state.copy())
                targets_list.append(ep["states"][t + 1].copy())

        all_states.append(np.array(states_list) if states_list else np.zeros((0, esn.reservoir_size)))
        all_targets.append(np.array(targets_list) if targets_list else np.zeros((0, DROID_STATE_DIM)))

    return all_states, all_targets


def run_incremental_learning_curve(
    successes: list[DroidEpisode],
    n_validation: int = 5,
    n_runs: int = 5,
    warmup: int = 10,
    seed: int = 42,
) -> dict:
    """
    Train ESN on 1..N episodes incrementally, measure validation sigma.

    For each run:
      1. Shuffle success episodes
      2. Hold out n_validation as validation set
      3. Train on 1, 2, ..., (total - n_validation) episodes
      4. At each step, solve ridge regression and measure sigma on validation

    Returns dict with n_episodes, sigma curves, and RMSE curves.
    """
    n_total = len(successes)
    n_train_max = n_total - n_validation

    all_sigma_curves = []   # (n_runs, n_train_max)
    all_rmse_curves = []

    for run in range(n_runs):
        rng = np.random.RandomState(seed + run)
        indices = rng.permutation(n_total)
        train_indices = indices[:n_train_max]
        val_indices = indices[n_train_max:]

        train_eps = [successes[i] for i in train_indices]
        val_eps = [successes[i] for i in val_indices]

        # Pre-compute reservoir states (ESN is deterministic for same seed)
        esn = EchoStateNetwork(
            input_dim=DROID_INPUT_DIM, reservoir_size=512,
            spectral_radius=0.95, input_scaling=0.3,
            leaking_rate=0.3, seed=42,
        )

        train_res_states, train_res_targets = _precompute_reservoir_states(train_eps, esn, warmup)
        val_res_states, val_res_targets = _precompute_reservoir_states(val_eps, esn, warmup)

        sigma_curve = []
        rmse_curve = []

        # Incrementally accumulate training pairs
        readout = PredictionReadout(reservoir_size=512, output_dim=DROID_STATE_DIM, alpha=0.01)

        for n in range(n_train_max):
            # Add pairs from episode n
            for s, t in zip(train_res_states[n], train_res_targets[n]):
                readout.collect(s, t)

            if readout.num_collected < 2:
                sigma_curve.append(float("nan"))
                rmse_curve.append(float("nan"))
                continue

            # Re-train on all accumulated pairs
            readout.W_out = None
            metrics = readout.train()
            rmse_curve.append(metrics["rmse"])

            # Evaluate on validation set
            val_sigmas = []
            for vs, vt in zip(val_res_states, val_res_targets):
                if len(vs) == 0:
                    continue
                preds = readout.predict_batch(vs)
                errors = np.linalg.norm(preds - vt, axis=1)
                val_sigmas.extend(errors.tolist())

            sigma_curve.append(float(np.mean(val_sigmas)) if val_sigmas else float("nan"))

        all_sigma_curves.append(sigma_curve)
        all_rmse_curves.append(rmse_curve)

    sigma_arr = np.array(all_sigma_curves)  # (n_runs, n_train_max)
    rmse_arr = np.array(all_rmse_curves)

    return {
        "n_episodes": list(range(1, n_train_max + 1)),
        "sigma_curves": sigma_arr,
        "sigma_mean": np.nanmean(sigma_arr, axis=0),
        "sigma_std": np.nanstd(sigma_arr, axis=0),
        "rmse_curves": rmse_arr,
        "rmse_mean": np.nanmean(rmse_arr, axis=0),
        "n_runs": n_runs,
        "n_validation": n_validation,
        "n_train_max": n_train_max,
    }


def plot_learning_curve(result: dict, output_path: str) -> str:
    """Plot sigma vs number of training episodes (the money plot)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    n_eps = result["n_episodes"]
    sigma_mean = result["sigma_mean"]
    sigma_std = result["sigma_std"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: sigma learning curve
    ax1.plot(n_eps, sigma_mean, color="#e74c3c", linewidth=2, label="Validation Sigma")
    ax1.fill_between(n_eps, sigma_mean - sigma_std, sigma_mean + sigma_std,
                     color="#e74c3c", alpha=0.15)
    ax1.set_xlabel("Number of Training Episodes")
    ax1.set_ylabel("Mean Prediction Error (Sigma)")
    ax1.set_title("Learning Curve: Repeated Visits Decrease Sigma")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add annotation for decrease
    if len(sigma_mean) > 1:
        start_val = sigma_mean[0]
        end_val = sigma_mean[-1]
        if not (np.isnan(start_val) or np.isnan(end_val)):
            decrease_pct = (1 - end_val / start_val) * 100
            ax1.annotate(
                f"{decrease_pct:.0f}% decrease",
                xy=(n_eps[-1], end_val),
                xytext=(n_eps[-1] * 0.6, (start_val + end_val) / 2),
                arrowprops=dict(arrowstyle="->", color="#e74c3c"),
                fontsize=11, fontweight="bold", color="#e74c3c",
            )

    # Right: RMSE learning curve
    rmse_mean = result["rmse_mean"]
    rmse_std = np.nanstd(result["rmse_curves"], axis=0)
    ax2.plot(n_eps, rmse_mean, color="#3498db", linewidth=2, label="Training RMSE")
    ax2.fill_between(n_eps, rmse_mean - rmse_std, rmse_mean + rmse_std,
                     color="#3498db", alpha=0.15)
    ax2.set_xlabel("Number of Training Episodes")
    ax2.set_ylabel("Training RMSE")
    ax2.set_title("Training Error vs Data Amount")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
