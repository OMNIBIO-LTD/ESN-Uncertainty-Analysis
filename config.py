"""
omni3 Configuration — ESN Uncertainty Estimator for UR5 Robotic Arm.

All tunable parameters in one place.

Robot: Berkeley Autolab UR5 (6-DOF + gripper)
Dataset: lerobot/berkeley_autolab_ur5
  - 1,000 episodes, 97,939 frames at 5 FPS
  - 5 manipulation tasks (sweep, pour, pick-place)
  - State (8): end-effector [x,y,z] + orientation [qw,qx,qy,qz] + gripper
  - Action (7): delta [dx,dy,dz] + delta orientation [3] + gripper command
  - Cameras: image (overhead), hand_image (wrist), image_with_depth
"""

import os

# ── Python interpreter ────────────────────────────────────────────
PYTHON_BIN = "/usr/bin/python3.10"

# ── Dataset ───────────────────────────────────────────────────────
DATASET_REPO_ID = "lerobot/berkeley_autolab_ur5"
DATASET_FPS = 5

# ── UR5 End-Effector State ────────────────────────────────────────
# observation.state: (8,) — Cartesian EE pose + gripper
#   [0:3]  position (x, y, z) in meters
#   [3:7]  orientation (qw, qx, qy, qz) quaternion
#   [7]    gripper openness (0 = closed, 1 = open)
STATE_DIM = 8
STATE_NAMES = [
    "ee_x", "ee_y", "ee_z",
    "ee_qw", "ee_qx", "ee_qy", "ee_qz",
    "gripper",
]

# action: (7,) — delta EE + gripper command
#   [0:3]  delta position (dx, dy, dz)
#   [3:6]  delta orientation (3 values)
#   [6]    gripper command (0 or 1)
ACTION_DIM = 7
ACTION_NAMES = [
    "delta_x", "delta_y", "delta_z",
    "delta_rot_0", "delta_rot_1", "delta_rot_2",
    "gripper_cmd",
]

# ── Cameras ───────────────────────────────────────────────────────
CAMERA_KEYS = [
    "observation.images.image",             # overhead (480, 640, 3)
    "observation.images.hand_image",        # wrist (480, 640, 3)
    "observation.images.image_with_depth",  # overhead + depth (480, 640, 3)
]

# ── Tasks in dataset ─────────────────────────────────────────────
TASKS = [
    "sweep the green cloth to the left side of the table",
    "put the ranch bottle into the pot",
    "pick up the blue cup and put it into the brown cup",
    "take the tiger out of the red bowl and put it in the blue bowl",
    "put the marker into the bowl",
]

# ── Trajectory Filtering (Phase 0) ──────────────────────────────
# Velocity spike: L2 norm of action delta between consecutive frames
FILTER_VEL_SPIKE_THRESHOLD = 0.03
# Command gap: large deviation between action and actual state change
# UR5 uses delta position control — some gap is normal, only flag large ones
FILTER_COMMAND_GAP_THRESHOLD = 0.05
# Gripper oscillation: sign changes in gripper delta over window
FILTER_GRIPPER_OSC_WINDOW = 8
FILTER_GRIPPER_OSC_MIN_CHANGES = 3
# Minimum anomaly score to flag an episode
FILTER_MIN_ANOMALY_SCORE = 0.05
# Top N episodes by anomaly score
FILTER_TOP_N = 50

# ── ESN (Echo State Network) ────────────────────────────────────
# Input = concat[state(8), action(7)] = 15
ESN_INPUT_DIM = STATE_DIM + ACTION_DIM  # 15
RESERVOIR_SIZE = 512
SPECTRAL_RADIUS = 0.95
INPUT_SCALING = 0.3
LEAKING_RATE = 0.3
RESERVOIR_SPARSITY = 0.1  # 10% non-zero connections
RANDOM_SEED = 42

# ── Readout (Ridge Regression) ──────────────────────────────────
# Predict next observation.state (8,)
READOUT_OUTPUT_DIM = STATE_DIM  # 8
RIDGE_ALPHA = 0.01
ESN_WARMUP_STEPS = 10  # discard first N ESN states (transient)

# ── Uncertainty (Sigma) ─────────────────────────────────────────
# sigma = L2 norm of (predicted - actual) next state
SIGMA_WINDOW_SIZE = 50
SIGMA_CAUTION_THRESHOLD = 1.5   # z-score: above this = caution
SIGMA_CURIOSITY_THRESHOLD = 3.0  # z-score: above this = explore

# ── Rerun Visualization ─────────────────────────────────────────
RERUN_APP_ID = "omni3_ur5_esn"
RERUN_RECORDING_ID = "phase0"

# ── Output Paths ─────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
FILTER_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "filtered_episodes.json")
VALIDATION_DIR = os.path.join(OUTPUT_DIR, "validation")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
