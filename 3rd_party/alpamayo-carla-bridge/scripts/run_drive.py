#!/usr/bin/env python3
"""Main driving loop — connect to CARLA, run Alpamayo inference, record, compose video.

Usage::

    python scripts/run_drive.py                     # defaults from config.yaml
    python scripts/run_drive.py --config my.yaml    # custom config
    python scripts/run_drive.py --max-steps 120     # override max steps
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import carla

# Ensure the repo root is on sys.path so ``bridge`` is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from bridge.alpamayo_inference import AlpamayoInference  # noqa: E402
from bridge.camera_manager import CameraManager  # noqa: E402
from bridge.carla_client import CarlaClient  # noqa: E402
from bridge.ego_motion import EgoMotionTracker  # noqa: E402
from bridge.recorder import Recorder  # noqa: E402
from bridge.trajectory_controller import TrajectoryController  # noqa: E402
from bridge.utils import load_config, setup_logging  # noqa: E402
from bridge.video_composer import VideoComposer  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Alpamayo–CARLA driving bridge")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "config.yaml"),
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps from config",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip automatic video composition after the run",
    )
    parser.add_argument(
        "--recording-dir",
        type=str,
        default=str(_REPO_ROOT / "recordings"),
        help="Base directory for recordings",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the driving loop."""
    args = parse_args()
    logger = setup_logging()

    # ---- Load config ----
    config = load_config(args.config)
    max_steps: int = args.max_steps or config.get("run", {}).get("max_steps", 60)
    logger.info("Max steps: %d", max_steps)

    carla_client: CarlaClient | None = None
    camera_mgr: CameraManager | None = None
    recorder: Recorder | None = None

    try:
        # ---- 1. Connect to CARLA ----
        carla_client = CarlaClient(config)
        world = carla_client.connect()

        # ---- 2. Spawn ego vehicle ----
        vehicle = carla_client.spawn_vehicle()

        # ---- 3. Attach cameras ----
        camera_mgr = CameraManager(world, vehicle, config)
        camera_mgr.spawn_all()

        # ---- 4. Attach collision sensor ----
        carla_client.attach_collision_sensor()

        # ---- 5. Load Alpamayo model ----
        alpamayo = AlpamayoInference(config)
        alpamayo.load()

        # ---- 6. Init controller ----
        controller = TrajectoryController(config=config)

        # ---- 7. Init recorder ----
        recorder = Recorder(config, base_dir=args.recording_dir)
        spawn_info = carla_client.get_spawn_transform()
        world_name = world.get_map().name
        recorder.initialise(spawn_info, world_name)

        # ---- 8. Init ego motion tracker ----
        ego_tracker = EgoMotionTracker(max_history=20)

        # Initial tick to populate sensor buffers.
        carla_client.tick()
        time.sleep(0.5)

        # ---- 9. Driving loop ----
        step = 0
        logger.info("=== Starting driving loop (max %d steps) ===", max_steps)

        while step < max_steps and not carla_client.crashed:
            step += 1
            t0 = time.time()

            # a. Collect frames
            frames = camera_mgr.collect_frames()
            if not frames:
                logger.warning("No frames collected at step %d — skipping.", step)
                carla_client.tick()
                continue

            # b. Update ego motion
            ego_state = ego_tracker.update(vehicle)

            # c. Run Alpamayo inference
            output = alpamayo.predict(frames, ego_tracker.get_history_list())
            trajectory = output.trajectory
            reasoning = output.reasoning

            # d. PID controller
            cmd = controller.compute(
                trajectory,
                ego_state.speed_kmh,
                ego_state.yaw,
            )

            # e. Record step
            recorder.save_step(
                step=step,
                frames=frames,
                predicted_trajectory=trajectory,
                reasoning_trace=reasoning,
                ego_state=ego_state.to_dict(),
                control=cmd.to_dict(),
            )

            # f. Apply vehicle control
            control = carla.VehicleControl(
                throttle=float(cmd.throttle),
                steer=float(cmd.steer),
                brake=float(cmd.brake),
            )
            vehicle.apply_control(control)

            # g. Tick
            carla_client.tick()

            dt = time.time() - t0
            logger.info(
                "Step %3d/%d | speed=%.1f km/h | steer=%.3f | throttle=%.3f | dt=%.2fs",
                step,
                max_steps,
                ego_state.speed_kmh,
                cmd.steer,
                cmd.throttle,
                dt,
            )

        # ---- 10. Wrap up ----
        if carla_client.crashed:
            logger.warning("Vehicle crashed at step %d. Stopping.", step)
        else:
            logger.info("Reached max steps (%d). Stopping.", max_steps)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)

    finally:
        # Finalise recording
        if recorder is not None and recorder.enabled:
            recorder.finalise()

        # Destroy CARLA actors
        if camera_mgr is not None:
            camera_mgr.destroy()
        if carla_client is not None:
            carla_client.destroy()

    # ---- Auto-compose video ----
    if recorder is not None and recorder.enabled and not args.no_video:
        logger.info("Composing video from recording…")
        try:
            composer = VideoComposer(recorder.session_dir, config)
            video_path = composer.compose()
            logger.info("Video saved: %s", video_path)
        except Exception as exc:
            logger.error("Video composition failed: %s", exc)

    logger.info("Done.")


if __name__ == "__main__":
    main()
