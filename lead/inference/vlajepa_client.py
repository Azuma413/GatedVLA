"""Thin websocket client wrapper for a VLA-JEPA driving policy server.

The VLA-JEPA model runs in its own process/environment (``vlajepa.sif`` or the
on-prem inference server) behind ``deployment/model_server/server_policy.py``.
This client (running inside ``lead.sif`` alongside CARLA) sends the current
multi-view camera frames + navigation instruction and receives the predicted
future ego-motion deltas ``[dx, dy, dyaw]`` per step. Waypoint integration and
PID control stay on the LEAD side (see ``vlajepa_agent.py``).
"""

import logging
import sys
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)


def _ensure_vlajepa_on_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    vlajepa_root = project_root / "3rd_party" / "VLA-JEPA"
    path_str = str(vlajepa_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class VLAJEPAClient:
    """Websocket client for the VLA-JEPA policy server (reuses VLA-JEPA's protocol)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 10093) -> None:
        _ensure_vlajepa_on_path()
        from deployment.model_server.tools.websocket_policy_client import (  # noqa: PLC0415
            WebsocketClientPolicy,
        )

        LOG.info("Connecting to VLA-JEPA policy server at %s:%d", host, port)
        self._client = WebsocketClientPolicy(host=host, port=port)

    def predict_deltas(self, images_rgb: list[np.ndarray], instruction: str) -> np.ndarray:
        """Return predicted ego-motion deltas of shape ``[chunk_len, 3]`` (dx, dy, dyaw).

        Args:
            images_rgb: multi-view frames as HxWx3 uint8 RGB arrays, in the training
                video-key order [front_wide, cross_left, front_tele, cross_right].
            instruction: navigation command text.
        """
        obs = {
            "batch_images": [images_rgb],  # batch of 1, list of views
            "instructions": [instruction],
        }
        response = self._client.infer(obs)
        if not response.get("ok", False):
            raise RuntimeError(f"VLA-JEPA server error: {response.get('error')}")
        normalized_actions = np.asarray(response["data"]["normalized_actions"], dtype=np.float32)
        # [B, chunk_len, action_dim] -> [chunk_len, 3]
        return normalized_actions[0]

    def close(self) -> None:
        self._client.close()
