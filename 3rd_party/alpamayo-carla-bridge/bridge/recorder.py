"""Per-step recording of frames, telemetry, and metadata."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class Recorder:
    """Saves camera frames and telemetry data for each driving step.

    Directory layout::

        recordings/{session_id}/
        ├── frames/
        │   ├── step_0001_front_left.jpg
        │   ├── step_0001_front_wide.jpg
        │   └── ...
        ├── data.jsonl
        └── metadata.json

    Args:
        config: Full config dict (used for metadata snapshot).
        base_dir: Root directory under which session folders are created.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        base_dir: str | Path = "recordings",
    ) -> None:
        self._config = config
        self._enabled: bool = config.get("recording", {}).get("enabled", True)

        self._session_id: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._session_dir: Path = Path(base_dir) / self._session_id
        self._frames_dir: Path = self._session_dir / "frames"
        self._data_path: Path = self._session_dir / "data.jsonl"
        self._metadata_path: Path = self._session_dir / "metadata.json"

        self._data_file: Optional[Any] = None
        self._start_time: float = time.time()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """Return the unique session identifier."""
        return self._session_id

    @property
    def session_dir(self) -> Path:
        """Return the session directory path."""
        return self._session_dir

    @property
    def enabled(self) -> bool:
        """Whether recording is active."""
        return self._enabled

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialise(
        self,
        spawn_info: Dict[str, Any],
        world_name: str = "",
    ) -> Path:
        """Create the session directory and write initial metadata.

        Args:
            spawn_info: Spawn-point dict (from ``CarlaClient.get_spawn_transform``).
            world_name: Name of the CARLA world / map.

        Returns:
            Path to the session directory.
        """
        if not self._enabled:
            logger.info("Recording disabled — skipping initialisation.")
            return self._session_dir

        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._frames_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "session_id": self._session_id,
            "start_time_utc": datetime.now(timezone.utc).isoformat(),
            "start_time_unix": self._start_time,
            "world": world_name,
            "spawn_point": spawn_info,
            "config": self._config,
        }
        with open(self._metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self._data_file = open(self._data_path, "w")  # noqa: SIM115
        logger.info("Recording initialised: %s", self._session_dir)
        return self._session_dir

    def save_step(
        self,
        step: int,
        frames: Dict[str, np.ndarray],
        predicted_trajectory: List[List[float]],
        reasoning_trace: str,
        ego_state: Dict[str, float],
        control: Dict[str, float],
    ) -> None:
        """Save one driving step's data.

        Args:
            step: Step number (1-indexed).
            frames: Camera name → RGB numpy array.
            predicted_trajectory: List of ``[x, y]`` waypoints.
            reasoning_trace: Text from Alpamayo.
            ego_state: Serialised ego state dict.
            control: Serialised vehicle control dict.
        """
        if not self._enabled:
            return

        # Save frames as JPEG.
        for cam_name, frame in frames.items():
            filename = f"step_{step:04d}_{cam_name}.jpg"
            filepath = self._frames_dir / filename
            img = Image.fromarray(frame)
            img.save(filepath, "JPEG", quality=90)

        # Append JSONL record.
        record = {
            "step": step,
            "timestamp": time.time(),
            "ego_state": ego_state,
            "predicted_trajectory": predicted_trajectory,
            "actual_position": [ego_state.get("x", 0.0), ego_state.get("y", 0.0)],
            "reasoning_trace": reasoning_trace,
            "control": control,
        }
        if self._data_file is not None:
            self._data_file.write(json.dumps(record, default=str) + "\n")
            self._data_file.flush()

    def finalise(self) -> None:
        """Close the data file and log summary."""
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None

        elapsed = time.time() - self._start_time
        logger.info(
            "Recording finalised: %s (%.1f s elapsed)",
            self._session_dir,
            elapsed,
        )
