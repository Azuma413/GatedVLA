"""Ego-vehicle motion state tracker."""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class EgoState:
    """Snapshot of the ego vehicle's kinematic state at one time step.

    All angles are in **degrees** (matching CARLA conventions) unless noted.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    speed_kmh: float = 0.0
    heading: float = 0.0  # yaw in degrees
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Serialise to a plain dict (for JSONL recording)."""
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "z": round(self.z, 4),
            "speed_kmh": round(self.speed_kmh, 2),
            "heading": round(self.heading, 4),
            "yaw": round(self.yaw, 4),
            "pitch": round(self.pitch, 4),
            "roll": round(self.roll, 4),
        }


@dataclass
class EgoMotionTracker:
    """Maintains a rolling history of ego states for feeding into Alpamayo.

    Alpamayo expects ego-motion context (recent positions / velocities).
    This class reads the CARLA vehicle's transform and velocity each tick
    and keeps the last *N* states.

    Attributes:
        history: Deque of recent :class:`EgoState` snapshots.
        max_history: Maximum number of past states to retain.
    """

    max_history: int = 20
    history: deque[EgoState] = field(default_factory=lambda: deque(maxlen=20))

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.max_history)

    def update(self, vehicle: Any) -> EgoState:
        """Read the vehicle's current state from CARLA and append to history.

        Args:
            vehicle: A ``carla.Actor`` representing the ego vehicle.

        Returns:
            The newly recorded :class:`EgoState`.
        """
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        state = EgoState(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z,
            speed_kmh=speed_ms * 3.6,
            heading=transform.rotation.yaw,
            yaw=transform.rotation.yaw,
            pitch=transform.rotation.pitch,
            roll=transform.rotation.roll,
        )
        self.history.append(state)
        return state

    def get_history_list(self) -> List[Dict[str, float]]:
        """Return the full history as a list of dicts.

        Returns:
            List of serialised ego states (oldest first).
        """
        return [s.to_dict() for s in self.history]

    @property
    def current(self) -> EgoState:
        """Return the most recent ego state.

        Raises:
            IndexError: If no state has been recorded yet.
        """
        if not self.history:
            raise IndexError("No ego states recorded yet.")
        return self.history[-1]
