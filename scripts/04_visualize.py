#!/usr/bin/env python3
"""
Script 04: Interactive Rerun visualization of UR5 ESN pipeline.

Opens Rerun viewer with:
  - 3D EE trajectory
  - State time series (x, y, z, gripper)
  - Sigma (prediction error) timeline
  - Anomaly markers
  - Per-dimension prediction error
  - Camera frames (optional, requires video download)

Usage:
    python3.10 omni3/scripts/04_visualize.py
    python3.10 omni3/scripts/04_visualize.py --episodes 0 1 2
    python3.10 omni3/scripts/04_visualize.py --episodes 0 --with-cameras
    python3.10 omni3/scripts/04_visualize.py --save-rrd output/viz.rrd
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import rerun as rr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from omni3 import config
from omni3.data.loader import load_dataset, extract_episode
from omni3.pipeline import UncertaintyPipeline
from omni3.viz.rerun_logger import RerunLogger
from omni3.viz.blueprint import create_episode_blueprint


def main():
    parser = argparse.ArgumentParser(description="Rerun visualization for UR5 ESN")
    parser.add_argument("--repo", default=config.DATASET_REPO_ID)
    parser.add_argument("--model", default=os.path.join(config.MODEL_DIR, "esn_readout.npz"))
    parser.add_argument("--filter-output", default=config.FILTER_OUTPUT_PATH)
    parser.add_argument("--episodes", type=int, nargs="*", default=[0],
                        help="Episode indices to visualize")
    parser.add_argument("--with-cameras", action="store_true",
                        help="Include camera frames (downloads video)")
    parser.add_argument("--save-rrd", default=None,
                        help="Save Rerun recording to .rrd file")
    args = parser.parse_args()

    print("=" * 60)
    print("  Rerun Visualization: UR5 ESN Pipeline")
    print("=" * 60)

    # Initialize Rerun
    logger = RerunLogger()
    if args.save_rrd:
        rr.init(config.RERUN_APP_ID, recording_id="phase0_viz")
        rr.save(args.save_rrd)
        print(f"  Recording to: {args.save_rrd}")
    else:
        logger.init(spawn=True, recording_id="phase0_viz")
        print(f"  Rerun viewer launched")

    # Load filter results
    filter_events: dict[int, list] = {}
    if os.path.exists(args.filter_output):
        with open(args.filter_output) as f:
            filter_data = json.load(f)
        for ep in filter_data.get("all_episodes", []):
            idx = ep["episode_idx"]
            filter_events[idx] = ep.get("events", [])

    # Load pipeline (if model exists)
    pipeline = None
    if os.path.exists(args.model):
        pipeline = UncertaintyPipeline()
        pipeline.load(args.model)
        print(f"  ESN model loaded from {args.model}")
    else:
        print(f"  No model found — skipping sigma visualization")
        print(f"  (Run 02_train_esn.py first to include sigma)")

    # Load dataset
    print(f"\n  Loading episodes {args.episodes}...")
    download_vids = args.with_cameras
    dataset = load_dataset(
        repo_id=args.repo,
        episodes=args.episodes if download_vids else None,
        download_videos=download_vids,
    )

    # Send blueprint for first episode
    bp = create_episode_blueprint(args.episodes[0])
    rr.send_blueprint(bp)

    # Visualize each episode
    for ep_idx in args.episodes:
        print(f"\n  Visualizing episode {ep_idx}...")
        t0 = time.time()

        episode = extract_episode(dataset, ep_idx)
        prefix = f"ep_{ep_idx}"

        # 1. Log trajectory (state time series + 3D path)
        logger.log_episode(episode, prefix="")
        logger.log_ee_trajectory(episode, prefix="")
        print(f"    Logged trajectory ({episode.num_frames} frames)")

        # 2. Log anomaly events
        from omni3.analysis.filter import AnomalyEvent
        events_raw = filter_events.get(ep_idx, [])
        events = [
            AnomalyEvent(
                frame_idx=e["frame"],
                timestamp=e["time"],
                anomaly_type=e["type"],
                severity=e["severity"],
                details=e.get("details", ""),
            )
            for e in events_raw
        ]
        if events:
            logger.log_anomaly_events(events, prefix=f"{prefix}/anomalies")
            print(f"    Logged {len(events)} anomaly events")

        # 3. Log sigma timeline (if model available)
        if pipeline is not None:
            result = pipeline.evaluate(episode)
            logger.log_sigma_timeline(
                result["readings"],
                result["timestamps"],
                prefix="esn",
            )

            # Log per-dim prediction error for detailed analysis
            pipeline.esn.reset()
            pipeline.uncertainty.reset()
            T = min(len(episode.states), len(episode.actions))
            for t in range(T - 1):
                esn_input = np.concatenate([episode.states[t], episode.actions[t]])
                esn_state = pipeline.esn.update(esn_input.astype(np.float32))
                if t >= config.ESN_WARMUP_STEPS:
                    predicted = pipeline.readout.predict(esn_state)
                    actual = episode.states[t + 1]
                    ts_val = float(episode.timestamps[t]) if t < len(episode.timestamps) else t / config.DATASET_FPS
                    logger.log_prediction_comparison(t, ts_val, predicted, actual)

            s = result["stats"]
            print(f"    Logged sigma: mean={s['sigma_mean']:.4f} max={s['sigma_max']:.4f} "
                  f"caution={s['caution_count']} curiosity={s['curiosity_count']}")

        # 4. Log camera frames (if requested)
        if args.with_cameras:
            _log_camera_frames(dataset, ep_idx, episode, logger, prefix)

        print(f"    Done in {time.time() - t0:.1f}s")

    print(f"\n{'=' * 60}")
    print(f"  Visualization complete!")
    if args.save_rrd:
        print(f"  Recording saved to: {args.save_rrd}")
        print(f"  Open with: rerun {args.save_rrd}")
    else:
        print(f"  Rerun viewer is open — scrub the timeline to explore")
    print(f"{'=' * 60}")


def _log_camera_frames(dataset, ep_idx, episode, logger, prefix):
    """Log camera frames for an episode (requires video download)."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        # Re-load with videos for this episode
        ds_vid = LeRobotDataset(
            repo_id=config.DATASET_REPO_ID,
            episodes=[ep_idx],
            download_videos=True,
        )

        n_frames = min(episode.num_frames, 50)  # cap to avoid slow decode
        step = max(1, episode.num_frames // n_frames)

        for t in range(0, episode.num_frames, step):
            try:
                sample = ds_vid[t]
                for cam_key in ["observation.images.image", "observation.images.hand_image"]:
                    if cam_key in sample:
                        img = sample[cam_key]
                        if hasattr(img, "numpy"):
                            img = img.numpy()
                        if img.ndim == 3 and img.shape[0] in (1, 3):
                            img = np.transpose(img, (1, 2, 0))
                        if img.dtype in (np.float32, np.float64):
                            img = (img * 255).clip(0, 255).astype(np.uint8)

                        cam_name = cam_key.split(".")[-1]
                        ts_val = float(episode.timestamps[t]) if t < len(episode.timestamps) else t / config.DATASET_FPS
                        logger.log_camera_frame(t, ts_val, img, cam_name, prefix)
            except Exception as e:
                pass  # skip frames that fail to decode

        print(f"    Logged camera frames (every {step} frames)")
    except Exception as e:
        print(f"    Camera frames skipped: {e}")


if __name__ == "__main__":
    main()
