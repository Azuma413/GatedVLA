"""PID-based trajectory controller for lateral and longitudinal vehicle control."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class PIDState:
    """Internal state of a single PID controller.

    Attributes:
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
        integral: Accumulated integral term.
        prev_error: Error from the previous step.
    """

    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.1
    integral: float = 0.0
    prev_error: float = 0.0

    def step(self, error: float, dt: float = 0.1) -> float:
        """Compute a single PID update.

        Args:
            error: Current error signal.
            dt: Time step in seconds.

        Returns:
            PID output value.
        """
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

    def reset(self) -> None:
        """Reset accumulated state."""
        self.integral = 0.0
        self.prev_error = 0.0


@dataclass
class VehicleCommand:
    """Desired vehicle control output.

    Attributes:
        throttle: Throttle value in [0, 1].
        steer: Steering value in [-1, 1] (positive = right).
        brake: Brake value in [0, 1].
    """

    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Serialise to a plain dict."""
        return {
            "throttle": round(self.throttle, 4),
            "steer": round(self.steer, 4),
            "brake": round(self.brake, 4),
        }


@dataclass
class TrajectoryController:
    """Dual-PID controller that converts trajectory waypoints into vehicle commands.

    The **lateral PID** drives the steering angle towards the nearest waypoint's
    lateral offset.  The **longitudinal PID** controls throttle/brake to track
    a target speed derived from the trajectory spacing.

    Args:
        config: Full config dict (expects ``controller`` section).
    """

    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ctrl = self.config.get("controller", {})

        lat_gains = ctrl.get("lateral_pid", [1.0, 0.0, 0.1])
        lon_gains = ctrl.get("longitudinal_pid", [0.5, 0.0, 0.05])

        self._lateral_pid = PIDState(kp=lat_gains[0], ki=lat_gains[1], kd=lat_gains[2])
        self._longitudinal_pid = PIDState(kp=lon_gains[0], ki=lon_gains[1], kd=lon_gains[2])

        self._target_speed_kmh: float = ctrl.get("target_speed_kmh", 30.0)
        self._max_throttle: float = ctrl.get("max_throttle", 0.6)
        self._max_brake: float = ctrl.get("max_brake", 0.8)
        self._max_steer: float = ctrl.get("max_steer", 0.7)
        self._dt: float = self.config.get("carla", {}).get("fixed_delta", 0.1)

    # ------------------------------------------------------------------
    # Main control method
    # ------------------------------------------------------------------

    def compute(
        self,
        trajectory: List[List[float]],
        current_speed_kmh: float,
        ego_yaw_deg: float,
    ) -> VehicleCommand:
        """Compute throttle, steer, and brake from the predicted trajectory.

        The controller looks at a target waypoint a few steps ahead (rather
        than the very next one) for smoother steering.

        Args:
            trajectory: List of ``[x, y]`` waypoints in ego frame
                        (x = forward, y = left).
            current_speed_kmh: Current ego speed in km/h.
            ego_yaw_deg: Current ego yaw in degrees (unused for ego-frame traj).

        Returns:
            A :class:`VehicleCommand`.
        """
        if not trajectory:
            logger.warning("Empty trajectory — applying gentle brake.")
            return VehicleCommand(throttle=0.0, steer=0.0, brake=0.3)

        # ------ Lateral control (steering) ------
        # Pick a look-ahead waypoint (~0.5 s ahead at 10 Hz = index 5).
        look_ahead_idx = min(5, len(trajectory) - 1)
        target_wp = trajectory[look_ahead_idx]
        target_x = target_wp[0]  # forward
        target_y = target_wp[1]  # left (positive = left)

        # Lateral error: positive target_y means the waypoint is to the left,
        # so we should steer left (negative steer in CARLA convention → left).
        # However CARLA steer positive = right.  So error = -target_y.
        lateral_error = -target_y

        # Normalise by forward distance to get an angular-like quantity.
        if abs(target_x) > 0.5:
            lateral_error = math.atan2(lateral_error, target_x)

        steer_raw = self._lateral_pid.step(lateral_error, self._dt)
        steer = max(-self._max_steer, min(self._max_steer, steer_raw))

        # ------ Longitudinal control (speed) ------
        # Desired speed from trajectory spacing (clamped to target).
        desired_speed = self._target_speed_from_trajectory(trajectory)
        speed_error = desired_speed - current_speed_kmh

        accel_raw = self._longitudinal_pid.step(speed_error, self._dt)

        if accel_raw >= 0:
            throttle = min(accel_raw / self._target_speed_kmh, self._max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(accel_raw) / self._target_speed_kmh, self._max_brake)

        cmd = VehicleCommand(throttle=throttle, steer=steer, brake=brake)
        logger.debug(
            "Control: throttle=%.3f, steer=%.3f, brake=%.3f (speed=%.1f, desired=%.1f)",
            cmd.throttle,
            cmd.steer,
            cmd.brake,
            current_speed_kmh,
            desired_speed,
        )
        return cmd

    def reset(self) -> None:
        """Reset PID state (call between runs)."""
        self._lateral_pid.reset()
        self._longitudinal_pid.reset()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _target_speed_from_trajectory(self, trajectory: List[List[float]]) -> float:
        """Estimate desired speed from the trajectory's average step distance.

        Args:
            trajectory: Waypoints in ego frame.

        Returns:
            Target speed in km/h, clamped to the configured maximum.
        """
        if len(trajectory) < 2:
            return self._target_speed_kmh

        # Average distance between consecutive waypoints.
        total_dist = 0.0
        for i in range(1, min(len(trajectory), 20)):
            dx = trajectory[i][0] - trajectory[i - 1][0]
            dy = trajectory[i][1] - trajectory[i - 1][1]
            total_dist += math.sqrt(dx * dx + dy * dy)

        avg_step = total_dist / min(len(trajectory) - 1, 19)
        # At 10 Hz, distance per step = speed(m/s) * 0.1
        implied_speed_ms = avg_step / self._dt
        implied_speed_kmh = implied_speed_ms * 3.6

        return min(implied_speed_kmh, self._target_speed_kmh)
