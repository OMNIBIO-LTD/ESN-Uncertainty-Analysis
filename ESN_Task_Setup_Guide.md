# ESN Per-Task Setup & Replication Guide

This document outlines the standard workflow for running Task-Specific Echo State Network (ESN) analysis. This setup ensures that training and evaluation are isolated to a single task to prevent cross-task noise, and it validates the generative capabilities of the ESN through learning curves and uncertainty (sigma) metrics.

## 1. Overview
The core script for this workflow is `08_task_specific.py`. It performs the following analyses on a specified task:
- **Incremental Learning Curve:** Proves that the ESN builds a generative model by showing that uncertainty (sigma) decreases as it sees more success episodes.
- **Task-Specific Phase 1 Analysis:**
  - **Sigma Comparison:** Compares average uncertainty between success and failure episodes.
  - **Per-Joint Analysis:** Identifies which joints are most predictive of failure.
  - **Temporal Divergence:** Measures at what percentage of task completion failures begin to diverge from successes.
  - **Classification:** Evaluates how well sigma can be used to classify successes vs. failures (AUC, F1 score).

## 2. How to Run the Analysis

To run the analysis for a specific task, use the `08_task_specific.py` script. 

### Basic Command:
```bash
python3.10 omni3/scripts/08_task_specific.py --task-index <TASK_ID>
```

### Advanced Usage (with custom parameters):
```bash
python3.10 omni3/scripts/08_task_specific.py --task-index <TASK_ID> --n-runs 10 --n-validation 5
```
**Arguments:**
- `--task-index`: The ID of the task you want to analyze (default is 5).
- `--n-runs`: Number of shuffle runs for the learning curve to generate robust error bars.
- `--n-validation`: Number of validation episodes held out for the learning curve.
- `--output-dir`: Where to save the generated plots and JSON results (defaults to `omni3/output/task_specific/`).

## 3. Outputs & Artifacts

After running the script, results are saved in the output directory (`omni3/output/task_specific/` by default). You should review the following artifacts:

- **`task<ID>_results.json`**: Contains all raw metrics (learning curve stats, sigma ratios, per-joint ratios, temporal divergence, and classification metrics).
- **`learning_curve.png`**: Visual proof of the generative model. Look for a steady decrease in sigma across episodes.
- **`task<ID>_per_joint_comparison.png`**: Shows which joints exhibit the highest uncertainty during failures.
- **`task<ID>_temporal_sigma_profile.png`**: Displays how uncertainty evolves over time throughout the episode.
- **`task<ID>_roc_curve.png`**: ROC curve for success/failure classification based on sigma thresholding.

## 4. Key Metrics to Evaluate

When interpreting the results for a new task, pay close attention to the console summary and JSON outputs:

1. **Learning Curve (Sigma Decrease):** A significant percentage decrease (>10%) confirms the ESN is successfully building a generative model of the task. If it's 0% or marginal, you may need to tune ESN hyperparameters (spectral radius, leaking rate, etc.) for that specific task.
2. **Sigma Ratio:** The ratio of failure sigma to success sigma. A higher ratio (e.g., >1.35x) indicates strong distinguishability.
3. **Most Predictive Joint:** Identifies the bottleneck or point of failure in the manipulation task.
4. **Classification AUC:** A high AUC indicates that uncertainty alone is a strong predictor of task success/failure.

## 5. Replicating for New Tasks

To replicate this for any new task:
1. Ensure the task data is available and loadable via `load_droid_task_episodes(task_index=YOUR_TASK_ID)`.
2. Run the script targeting your new task index.
3. Review the learning curve to ensure the ESN is actually learning the task dynamics.
4. Share the generated plots and JSON summaries with the team for review.