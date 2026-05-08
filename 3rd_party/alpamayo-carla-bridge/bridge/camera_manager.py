"""Camera sensor management for the CARLA ego vehicle."""

from __future__ import annotations

import logging
import queue
from typing import Any, Dict, List, Optional

import carla
import numpy as np

logger = logging.getLogger(__name__)

# Camera names in canonical order (matches Alpamayo's expected input order).
CAMERA_ORDER: List[str] = ["front_left", "front_wide", "front_right", "front_tele"]


class CameraManager:
    """Spawns and manages multiple RGB camera sensors attached to the ego vehicle.

    Each camera stores its latest frame in a thread-safe queue so the main loop
    can retrieve all four images synchronously after a ``world.tick()``.

    Attributes:
        cameras: Mapping from camera name to the CARLA sensor actor.
        resolution: Tuple ``(width, height)`` shared by all cameras.
    """

    def __init__(
        self,
        world: carla.World,
        vehicle: carla.Actor,
        config: Dict[str, Any],
    ) -> None:
        """Configure cameras from the ``cameras`` section of the config.

        Args:
            world: The CARLA world handle (needed for blueprint library).
            vehicle: The ego vehicle actor to attach cameras to.
            config: Full parsed config dict.
        """
        self._world = world
        self._vehicle = vehicle
        self._cam_cfg = config["cameras"]

        res = self._cam_cfg.get("resolution", [800, 600])
        self.resolution: tuple[int, int] = (int(res[0]), int(res[1]))

        self.cameras: Dict[str, carla.Actor] = {}
        self._queues: Dict[str, queue.Queue[np.ndarray]] = {}
        self._actors: List[carla.Actor] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def spawn_all(self) -> None:
        """Spawn all four cameras defined in the config and start listening."""
        for cam_name in CAMERA_ORDER:
            cam_params = self._cam_cfg.get(cam_name)
            if cam_params is None:
                logger.warning("Camera '%s' not found in config — skipping.", cam_name)
                continue
            self._spawn_camera(cam_name, cam_params)
        logger.info("Spawned %d cameras: %s", len(self.cameras), list(self.cameras.keys()))

    def collect_frames(self) -> Dict[str, np.ndarray]:
        """Retrieve the latest frame from every camera (blocking per queue).

        Should be called once after each ``world.tick()`` in synchronous mode.

        Returns:
            Dict mapping camera name → numpy array of shape ``(H, W, 3)`` in RGB.
        """
        frames: Dict[str, np.ndarray] = {}
        for cam_name in CAMERA_ORDER:
            if cam_name not in self._queues:
                continue
            try:
                frames[cam_name] = self._queues[cam_name].get(timeout=5.0)
            except queue.Empty:
                logger.error("Timeout waiting for frame from '%s'.", cam_name)
        return frames

    def destroy(self) -> None:
        """Stop listening and destroy all camera actors."""
        for actor in self._actors:
            try:
                actor.stop()
                if actor.is_alive:
                    actor.destroy()
            except Exception as exc:
                logger.debug("Could not destroy camera actor: %s", exc)
        self._actors.clear()
        self.cameras.clear()
        self._queues.clear()
        logger.info("All cameras destroyed.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _spawn_camera(self, name: str, params: Dict[str, Any]) -> None:
        """Spawn a single RGB camera sensor.

        Args:
            name: Logical name (e.g. ``"front_wide"``).
            params: Dict with keys ``fov, x, y, z, pitch, yaw``.
        """
        bp = self._world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.resolution[0]))
        bp.set_attribute("image_size_y", str(self.resolution[1]))
        bp.set_attribute("fov", str(params.get("fov", 90)))

        transform = carla.Transform(
            carla.Location(
                x=float(params.get("x", 1.5)),
                y=float(params.get("y", 0.0)),
                z=float(params.get("z", 1.6)),
            ),
            carla.Rotation(
                pitch=float(params.get("pitch", 0)),
                yaw=float(params.get("yaw", 0)),
                roll=0,
            ),
        )

        sensor = self._world.spawn_actor(bp, transform, attach_to=self._vehicle)
        self._actors.append(sensor)
        self.cameras[name] = sensor

        # Per-camera frame queue (depth=1 — we only care about the latest frame).
        q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._queues[name] = q

        def _on_image(image: carla.Image, _q: queue.Queue = q) -> None:
            """Convert raw CARLA image to a numpy RGB array and enqueue."""
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))[:, :, :3]  # BGRA → BGR
            array = array[:, :, ::-1].copy()  # BGR → RGB
            # Drop stale frames — keep only the latest.
            if _q.full():
                try:
                    _q.get_nowait()
                except queue.Empty:
                    pass
            _q.put(array)

        sensor.listen(_on_image)
        logger.debug(
            "Camera '%s' spawned: fov=%s, pos=(%.1f, %.1f, %.1f), yaw=%.0f",
            name,
            params.get("fov"),
            params.get("x", 1.5),
            params.get("y", 0.0),
            params.get("z", 1.6),
            params.get("yaw", 0),
        )
