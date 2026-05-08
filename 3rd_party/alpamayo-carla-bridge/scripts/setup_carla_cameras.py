#!/usr/bin/env python3
"""Camera setup utility — spawn the vehicle, attach cameras, save preview snapshots.

Useful for iterating on camera positions without running the full driving loop.

Usage::

    python scripts/setup_carla_cameras.py
    python scripts/setup_carla_cameras.py --config my.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from bridge.camera_manager import CameraManager  # noqa: E402
from bridge.carla_client import CarlaClient  # noqa: E402
from bridge.utils import load_config, setup_logging  # noqa: E402


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Preview CARLA camera placements")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_REPO_ROOT / "camera_preview"),
        help="Directory to save preview images",
    )
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    carla_client: CarlaClient | None = None
    camera_mgr: CameraManager | None = None

    try:
        carla_client = CarlaClient(config)
        world = carla_client.connect()
        vehicle = carla_client.spawn_vehicle()

        camera_mgr = CameraManager(world, vehicle, config)
        camera_mgr.spawn_all()

        # Let the sensors warm up.
        for _ in range(5):
            carla_client.tick()
            time.sleep(0.1)

        # Collect one set of frames.
        frames = camera_mgr.collect_frames()
        for cam_name, frame_array in frames.items():
            out_path = output_dir / f"{cam_name}.jpg"
            img = Image.fromarray(frame_array)
            img.save(out_path, "JPEG", quality=95)
            logger.info("Saved preview: %s (%dx%d)", out_path, frame_array.shape[1], frame_array.shape[0])

        logger.info("All previews saved to %s", output_dir)

    except Exception as exc:
        logger.error("Error: %s", exc, exc_info=True)

    finally:
        if camera_mgr is not None:
            camera_mgr.destroy()
        if carla_client is not None:
            carla_client.destroy()


if __name__ == "__main__":
    main()
