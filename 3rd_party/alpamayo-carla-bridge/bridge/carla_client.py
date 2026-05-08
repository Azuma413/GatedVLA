"""CARLA client wrapper — connection, world setup, ego-vehicle spawning."""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional

import carla

logger = logging.getLogger(__name__)


class CarlaClient:
    """Manages the connection to the CARLA server and the ego vehicle lifecycle.

    Attributes:
        client: The underlying ``carla.Client`` instance.
        world: The active ``carla.World``.
        vehicle: The spawned ego vehicle actor (or ``None``).
        collision_sensor: Attached collision sensor (or ``None``).
        crashed: Set to ``True`` when a collision is detected.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise with configuration but do **not** connect yet.

        Args:
            config: Full parsed config dict (expects ``carla`` and ``vehicle`` keys).
        """
        self._cfg_carla = config["carla"]
        self._cfg_vehicle = config["vehicle"]
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Actor] = None
        self.collision_sensor: Optional[carla.Actor] = None
        self.crashed: bool = False
        self._original_settings: Optional[carla.WorldSettings] = None
        self._actors_to_destroy: list[carla.Actor] = []

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> carla.World:
        """Connect to the CARLA server and load the requested world.

        Returns:
            The ``carla.World`` handle.
        """
        host = self._cfg_carla["host"]
        port = self._cfg_carla["port"]
        timeout = self._cfg_carla.get("timeout", 30)

        logger.info("Connecting to CARLA at %s:%d (timeout=%ds)…", host, port, timeout)
        self.client = carla.Client(host, port)
        self.client.set_timeout(float(timeout))

        target_world = self._cfg_carla.get("world", "Town03")
        available = [m.split("/")[-1] for m in self.client.get_available_maps()]
        if target_world not in available:
            logger.warning(
                "Map '%s' not available (%s). Using current world.", target_world, available
            )
            self.world = self.client.get_world()
        else:
            logger.info("Loading world '%s'…", target_world)
            self.world = self.client.load_world(target_world)

        self._setup_sync_mode()
        logger.info("Connected. World: %s", self.world.get_map().name)
        return self.world

    # ------------------------------------------------------------------
    # Synchronous mode
    # ------------------------------------------------------------------

    def _setup_sync_mode(self) -> None:
        """Enable synchronous mode with a fixed time-step if configured."""
        if not self._cfg_carla.get("sync_mode", True):
            return

        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._cfg_carla.get("fixed_delta", 0.1)
        self.world.apply_settings(settings)
        logger.info(
            "Synchronous mode ON (fixed_delta=%.3f s)",
            settings.fixed_delta_seconds,
        )

    # ------------------------------------------------------------------
    # Spawn ego vehicle
    # ------------------------------------------------------------------

    def spawn_vehicle(self) -> carla.Actor:
        """Spawn the ego vehicle at a random available spawn point.

        Returns:
            The spawned ``carla.Actor``.

        Raises:
            RuntimeError: If no spawn points are available or spawning fails.
        """
        bp_library = self.world.get_blueprint_library()
        bp_name = self._cfg_vehicle.get("blueprint", "vehicle.tesla.model3")
        bp = bp_library.find(bp_name)

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available on this map.")

        spawn_point = random.choice(spawn_points)
        logger.info(
            "Spawning '%s' at (%.1f, %.1f, %.1f)",
            bp_name,
            spawn_point.location.x,
            spawn_point.location.y,
            spawn_point.location.z,
        )

        self.vehicle = self.world.spawn_actor(bp, spawn_point)
        self._actors_to_destroy.append(self.vehicle)
        logger.info("Ego vehicle spawned: id=%d", self.vehicle.id)
        return self.vehicle

    # ------------------------------------------------------------------
    # Collision sensor
    # ------------------------------------------------------------------

    def attach_collision_sensor(self) -> carla.Actor:
        """Attach a collision sensor to the ego vehicle.

        Sets ``self.crashed = True`` upon any collision event.

        Returns:
            The collision sensor actor.

        Raises:
            RuntimeError: If no vehicle has been spawned yet.
        """
        if self.vehicle is None:
            raise RuntimeError("Spawn the vehicle before attaching sensors.")

        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        transform = carla.Transform(carla.Location(x=0, y=0, z=0))
        self.collision_sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        self._actors_to_destroy.append(self.collision_sensor)

        def _on_collision(event: carla.SensorData) -> None:  # type: ignore[override]
            other = event.other_actor
            logger.warning("COLLISION with %s (id=%d)", other.type_id, other.id)
            self.crashed = True

        self.collision_sensor.listen(_on_collision)
        logger.info("Collision sensor attached.")
        return self.collision_sensor

    # ------------------------------------------------------------------
    # World tick helper
    # ------------------------------------------------------------------

    def tick(self) -> int:
        """Advance the CARLA world by one simulation step.

        Returns:
            The frame number after the tick.
        """
        return self.world.tick()

    # ------------------------------------------------------------------
    # Spawn point info (for metadata)
    # ------------------------------------------------------------------

    def get_spawn_transform(self) -> Dict[str, float]:
        """Return the ego vehicle's current transform as a plain dict.

        Returns:
            Dict with keys ``x, y, z, yaw, pitch, roll``.
        """
        t = self.vehicle.get_transform()
        return {
            "x": t.location.x,
            "y": t.location.y,
            "z": t.location.z,
            "yaw": t.rotation.yaw,
            "pitch": t.rotation.pitch,
            "roll": t.rotation.roll,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Destroy all spawned actors and restore world settings."""
        logger.info("Cleaning up %d actors…", len(self._actors_to_destroy))
        for actor in reversed(self._actors_to_destroy):
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception as exc:
                logger.debug("Could not destroy actor %s: %s", actor.id, exc)
        self._actors_to_destroy.clear()

        if self._original_settings is not None and self.world is not None:
            try:
                self.world.apply_settings(self._original_settings)
                logger.info("Original world settings restored.")
            except Exception as exc:
                logger.debug("Could not restore settings: %s", exc)
