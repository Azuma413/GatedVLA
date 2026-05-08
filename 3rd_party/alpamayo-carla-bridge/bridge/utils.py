"""Shared utility functions for the Alpamayo–CARLA bridge."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load and return the YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with a consistent format.

    Args:
        level: Logging level (default: INFO).

    Returns:
        The root logger instance.
    """
    fmt = "[%(asctime)s] %(levelname)-8s %(name)-25s %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)
    logger = logging.getLogger("alpamayo_bridge")
    logger.setLevel(level)
    return logger


def carla_rotation_to_yaw_rad(rotation: Any) -> float:
    """Convert a CARLA Rotation to yaw in radians.

    Args:
        rotation: A ``carla.Rotation`` object.

    Returns:
        Yaw angle in radians (counter-clockwise positive).
    """
    return math.radians(rotation.yaw)


def speed_from_velocity(velocity: Any) -> float:
    """Compute scalar speed in km/h from a CARLA 3D velocity vector.

    Args:
        velocity: A ``carla.Vector3D`` with x, y, z components (m/s).

    Returns:
        Speed in km/h.
    """
    speed_ms = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    return speed_ms * 3.6


def world_to_ego(
    world_x: float,
    world_y: float,
    ego_x: float,
    ego_y: float,
    ego_yaw_rad: float,
) -> tuple[float, float]:
    """Transform a world-frame point into the ego vehicle's local frame.

    Args:
        world_x: World X coordinate of the point.
        world_y: World Y coordinate of the point.
        ego_x: Ego vehicle world X.
        ego_y: Ego vehicle world Y.
        ego_yaw_rad: Ego vehicle yaw in radians.

    Returns:
        Tuple (local_x, local_y) in the ego frame (forward, left).
    """
    dx = world_x - ego_x
    dy = world_y - ego_y
    cos_yaw = math.cos(ego_yaw_rad)
    sin_yaw = math.sin(ego_yaw_rad)
    local_x = dx * cos_yaw + dy * sin_yaw
    local_y = -dx * sin_yaw + dy * cos_yaw
    return local_x, local_y


def ego_to_world(
    local_x: float,
    local_y: float,
    ego_x: float,
    ego_y: float,
    ego_yaw_rad: float,
) -> tuple[float, float]:
    """Transform a point from ego-local frame to world frame.

    Args:
        local_x: Forward distance in ego frame.
        local_y: Leftward distance in ego frame.
        ego_x: Ego vehicle world X.
        ego_y: Ego vehicle world Y.
        ego_yaw_rad: Ego vehicle yaw in radians.

    Returns:
        Tuple (world_x, world_y).
    """
    cos_yaw = math.cos(ego_yaw_rad)
    sin_yaw = math.sin(ego_yaw_rad)
    world_x = ego_x + local_x * cos_yaw - local_y * sin_yaw
    world_y = ego_y + local_x * sin_yaw + local_y * cos_yaw
    return world_x, world_y
