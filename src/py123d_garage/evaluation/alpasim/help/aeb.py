"""
Camera-based emergency braking on top of the served plan: a COCO detector finds road users in the front camera,
their box bottoms are cast onto the ground plane of the rig, and the plan is re-timed along its own path when the
nearest one in the ego's corridor is closing in faster than a comfortable stop allows.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from py123d.datatypes import FThetaCameraMetadata

from py123d_garage.config.schema.evaluation.alpasim_config import AebConfig

# COCO classes the ego must not drive into.
_OBSTACLE_LABELS = frozenset({"person", "bicycle", "car", "motorcycle", "bus", "train", "truck"})
# Ground samples along each box's bottom edge.
_NUM_EDGE_SAMPLES = 7
_MAX_RANGE_M = 80.0
# Time grid of a re-timed plan, fine enough for the braking ramp.
_RETIME_INTERVAL_S = 0.1


@dataclasses.dataclass
class LeadTrack:
    """The nearest in-corridor obstacle of one session over time."""

    times_s: list[float] = dataclasses.field(default_factory=list)
    distances_m: list[float] = dataclasses.field(default_factory=list)

    def update(self, time_s: float, distance_m: float | None, config: AebConfig) -> None:
        if distance_m is None:
            self.times_s, self.distances_m = [], []
            return
        if self.distances_m and abs(distance_m - self.distances_m[-1]) > config.track_jump_m:
            self.times_s, self.distances_m = [], []
        self.times_s.append(time_s)
        self.distances_m.append(distance_m)
        keep = [i for i, t in enumerate(self.times_s) if t >= time_s - config.track_window_s]
        self.times_s = [self.times_s[i] for i in keep]
        self.distances_m = [self.distances_m[i] for i in keep]

    def closing_speed_mps(self, config: AebConfig) -> float | None:
        """Rate the gap shrinks at, from a line fit; None until the track is long enough."""
        if len(self.times_s) < config.min_track_samples:
            return None
        times = np.asarray(self.times_s) - self.times_s[-1]
        slope = np.polyfit(times, np.asarray(self.distances_m), 1)[0]
        return float(-slope)


class ObstacleDetector:
    def __init__(self, config: AebConfig, device: torch.device) -> None:
        from transformers import AutoModelForObjectDetection

        self._config = config
        self._device = device
        assert config.detector_dir is not None, "the detector loads only with emergency braking enabled"
        self._model = AutoModelForObjectDetection.from_pretrained(Path(config.detector_dir)).to(device).eval()
        id2label = self._model.config.id2label
        self._obstacle_ids = torch.tensor(
            [int(i) for i, name in id2label.items() if name in _OBSTACLE_LABELS],
            device=device,
        )

    @torch.no_grad()
    def detect(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
        """Obstacle boxes (N, 4) as x1, y1, x2, y2 pixels of the RGB image."""
        height, width = image.shape[:2]
        pixels = torch.from_numpy(image).to(self._device).permute(2, 0, 1)[None].float() / 255.0
        size = self._config.input_size
        pixels = torch.nn.functional.interpolate(pixels, size=(size, size), mode="bilinear", antialias=True)
        outputs = self._model(pixel_values=pixels)
        scores = outputs.logits[0].sigmoid()[:, self._obstacle_ids].max(dim=-1).values
        keep = scores > self._config.score_threshold
        cx, cy, w, h = outputs.pred_boxes[0][keep].unbind(-1)
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        boxes = boxes * torch.tensor([width, height, width, height], device=self._device)
        return boxes.double().cpu().numpy()


def ground_points_rig(
    boxes: npt.NDArray[np.float64],
    metadata: FThetaCameraMetadata,
    edge_shrink: float = 0.0,
) -> list[npt.NDArray[np.float64]]:
    """Per box, the rig-frame ground points (M, 2) under its bottom edge; boxes above the horizon get none."""
    intrinsics = metadata.intrinsics
    pose = metadata.camera_to_imu_se3
    rotation = pose.rotation_matrix
    origin = np.array([pose.x, pose.y, pose.z])
    bw_poly = intrinsics.bw_poly
    result = []
    for x1, _, x2, y2 in boxes:
        margin = edge_shrink * (x2 - x1)
        us = np.linspace(x1 + margin, x2 - margin, _NUM_EDGE_SAMPLES)
        du, dv = us - intrinsics.cx, np.full_like(us, y2 - intrinsics.cy)
        radius = np.hypot(du, dv)
        theta = np.polynomial.polynomial.polyval(radius, bw_poly)
        phi = np.arctan2(dv, du)
        rays_cam = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
        rays = rays_cam @ rotation.T
        down = rays[:, 2] < -1e-3
        scale = -origin[2] / rays[down, 2]
        points = origin[None, :2] + scale[:, None] * rays[down, :2]
        points = points[np.linalg.norm(points, axis=-1) < _MAX_RANGE_M]
        result.append(points)
    return result


def lead_distance_m(
    obstacles_rig: list[npt.NDArray[np.float64]],
    path_rig: npt.NDArray[np.float64],
    config: AebConfig,
) -> float | None:
    """Distance along the planned path from the front bumper to the nearest obstacle inside the corridor."""
    path = _extended_path(path_rig)
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=-1)
    arclength = np.concatenate([[0.0], np.cumsum(lengths)])
    nearest = None
    for points in obstacles_rig:
        if len(points) == 0:
            continue
        # Project every point on every segment.
        rel = points[:, None, :] - path[None, :-1, :]
        frac = np.clip((rel * segments[None]).sum(-1) / np.maximum(lengths**2, 1e-9)[None], 0.0, 1.0)
        foot = path[None, :-1, :] + frac[..., None] * segments[None]
        lateral = np.linalg.norm(points[:, None, :] - foot, axis=-1)
        best = lateral.argmin(axis=1)
        rows = np.arange(len(points))
        inside = lateral[rows, best] < config.corridor_half_width_m
        if not inside.any():
            continue
        along = arclength[best] + frac[rows, best] * lengths[best]
        distance = float(along[inside].min()) - config.front_overhang_m
        if distance > -1.0 and (nearest is None or distance < nearest):
            nearest = distance
    return nearest


def _extended_path(path_rig: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The path from the rig origin, continued straight to the detection range."""
    path = np.vstack([[[0.0, 0.0]], path_rig])
    keep = np.concatenate([[True], np.linalg.norm(np.diff(path, axis=0), axis=-1) > 1e-3])
    path = path[keep]
    heading = path[-1] - path[-2] if len(path) > 1 else np.array([1.0, 0.0])
    heading = heading / max(float(np.linalg.norm(heading)), 1e-9)
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=-1).sum()) if len(path) > 1 else 0.0
    extra = max(_MAX_RANGE_M - length, 0.0)
    return np.vstack([path, path[-1] + heading * extra]) if extra > 0 else path


def speed_cap(
    distance_m: float | None,
    closing_mps: float | None,
    speed_mps: float,
    config: AebConfig,
) -> tuple[float, float] | None:
    """
    (deceleration, floor speed) the plan must not exceed: braking at the deceleration down to the lead's speed;
    a zero deceleration only forbids accelerating. None leaves the plan alone.
    """
    if distance_m is None or closing_mps is None or closing_mps <= 0.0:
        return None
    lead_speed = max(speed_mps - closing_mps, 0.0)
    gap = distance_m - config.stop_gap_m - closing_mps * config.latency_s
    required = closing_mps**2 / (2.0 * max(gap, 0.1))
    if gap <= 0.0 or required > config.trigger_decel_mps2:
        return min(max(1.2 * required, 3.0), config.max_decel_mps2), lead_speed
    if distance_m / closing_mps < config.no_accel_ttc_s:
        return 0.0, speed_mps
    return None


def retime_plan(
    times_s: npt.NDArray[np.float64],
    positions_xy: npt.NDArray[np.float64],
    yaws: npt.NDArray[np.float64],
    elapsed_s: float,
    ego_xy: npt.NDArray[np.float64],
    speed_mps: float,
    decel_mps2: float,
    floor_mps: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    The rest of a plan from elapsed_s on, slowed along its own path: it starts where the ego projects onto the path
    and covers what the plan still meant to cover, but never more than a ramp from speed_mps down at decel_mps2 to
    floor_mps allows (a zero deceleration holds speed_mps). Returns times from now, positions and yaws.
    """
    arclength = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(positions_xy, axis=0), axis=-1))])
    taus = np.arange(0.0, max(times_s[-1] - elapsed_s, 0.0) + 1e-9, _RETIME_INTERVAL_S)
    wanted = np.interp(elapsed_s + taus, times_s, arclength) - np.interp(elapsed_s, times_s, arclength)
    if decel_mps2 > 0.0:
        braking = np.minimum(taus, max(speed_mps - floor_mps, 0.0) / decel_mps2)
        cap = speed_mps * braking - 0.5 * decel_mps2 * braking**2 + floor_mps * (taus - braking)
    else:
        cap = speed_mps * taus
    distance = _project_on_path(ego_xy, positions_xy, arclength) + np.minimum(wanted, cap)
    xs = np.interp(distance, arclength, positions_xy[:, 0])
    ys = np.interp(distance, arclength, positions_xy[:, 1])
    new_yaws = np.interp(distance, arclength, np.unwrap(yaws))
    return taus, np.stack([xs, ys], axis=-1), new_yaws


def _project_on_path(
    point: npt.NDArray[np.float64],
    path: npt.NDArray[np.float64],
    arclength: npt.NDArray[np.float64],
) -> float:
    """Arclength of the point's foot on the polyline."""
    if len(path) < 2:
        return 0.0
    segments = np.diff(path, axis=0)
    lengths_sq = np.maximum((segments**2).sum(-1), 1e-12)
    frac = np.clip(((point - path[:-1]) * segments).sum(-1) / lengths_sq, 0.0, 1.0)
    foot = path[:-1] + frac[:, None] * segments
    best = int(np.linalg.norm(foot - point, axis=-1).argmin())
    return float(arclength[best] + frac[best] * math.sqrt(lengths_sq[best]))
