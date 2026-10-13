"""Acceleration / curvature action representation (Alpamayo 1.5 unicycle model) for ego-frame trajectories.

``encode``: future ego-frame positions and headings + current speed -> per-step (acceleration [m/s²],
curvature [1/m]) with Alpamayo's regularised least-squares fit (smooth heading, velocity, acceleration, curvature;
curvature is 0 below 0.6 m/s).
``decode``: controls + current speed -> positions by integrating the unicycle model from the origin (heading 0).
Both work on un-normalised controls; normalisation is done by the caller.
"""

from __future__ import annotations

import numpy as np
import torch

from .unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace


def _yaw_to_rot(yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    rot = torch.zeros(*yaw.shape, 3, 3, dtype=yaw.dtype, device=yaw.device)
    rot[..., 0, 0], rot[..., 0, 1], rot[..., 1, 0], rot[..., 1, 1], rot[..., 2, 2] = c, -s, s, c, 1.0
    return rot


class UnicycleCodec:
    def __init__(self, horizon: int, dt: float = 0.1):
        self.horizon = horizon
        self.space = UnicycleAccelCurvatureActionSpace(dt=dt, n_waypoints=horizon).double()

    def encode(self, future_xy: np.ndarray, future_yaw: np.ndarray, v0) -> np.ndarray:
        """future_xy (..., N, 2), future_yaw (..., N) relative to the current heading, v0 (...) -> (..., N, 2)."""
        xy = torch.as_tensor(np.asarray(future_xy), dtype=torch.float64)
        yaw = torch.as_tensor(np.asarray(future_yaw), dtype=torch.float64)
        v0 = torch.as_tensor(np.asarray(v0), dtype=torch.float64)
        lead = xy.shape[:-2]
        fut_xyz = torch.cat([xy, torch.zeros(*xy.shape[:-1], 1, dtype=xy.dtype)], dim=-1)
        hist_xyz = torch.zeros(*lead, 1, 3, dtype=xy.dtype)
        hist_rot = _yaw_to_rot(torch.zeros(*lead, 1, dtype=xy.dtype))
        squeeze = len(lead) == 0
        if squeeze:  # the solvers expect a batch dimension
            fut_xyz, hist_xyz, hist_rot, yaw, v0 = fut_xyz[None], hist_xyz[None], hist_rot[None], yaw[None], v0[None]
        action = self.space.traj_to_action(hist_xyz, hist_rot, fut_xyz, _yaw_to_rot(yaw), t0_states={"v": v0})
        return (action[0] if squeeze else action).numpy()

    def decode(self, controls, v0) -> torch.Tensor:
        """controls (B, N, 2) [accel, curvature], v0 (B,) -> positions (B, N, 2) at t0 + dt .. t0 + N·dt."""
        controls = torch.as_tensor(controls, dtype=torch.float64)
        v0 = torch.as_tensor(v0, dtype=torch.float64, device=controls.device)
        hist_xyz = torch.zeros(controls.shape[0], 1, 3, dtype=torch.float64, device=controls.device)
        xyz, _ = self.space.to(controls.device).action_to_traj(controls, hist_xyz, None, t0_states={"v": v0})
        return xyz[..., :2]
