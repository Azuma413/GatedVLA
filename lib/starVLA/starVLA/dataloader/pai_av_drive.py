# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
PhysicalAI-Autonomous-Vehicles (PAI-AV) front-camera dataset for driving policies (QwenDrivePI).

Reads the raw HF release directly (no conversion):
    clip_index.parquet, metadata/feature_presence.parquet
    labels/egomotion/egomotion.chunk_XXXX.zip            -> {clip}.egomotion.parquet (100 Hz, local world frame)
    camera/camera_front_wide_120fov/..chunk_XXXX.zip     -> {clip}.camera_front_wide_120fov.mp4 (+ timestamps)

Sample layout (DriveMoE DrivePi0 design, 10 Hz):
    image  : [front(t0 - 0.1 s), front(t0)]                     (history first, current last)
    state  : (state_horizon=5, 10) at t0-0.4 .. t0, per step
             [speed, acc_x, acc_y, acc_z, angvel_x, angvel_y, angvel_z, heading_rel, target_x, target_y]
             acc / angvel in the ego (rig) frame of that step, heading relative to yaw(t0),
             target point in the ego frame at t0 (repeated per step).
    action : (action_horizon=10, 2) ego-frame (x forward, y left) positions at t0 + 0 .. t0 + 0.9 s.
PAI-AV has no route, so the target point is the first future ego position >= ``target_distance`` m
(Euclidean) from the current position, or the last recorded position if the clip ends earlier.
state / action are normalised to [-1, 1] with per-dimension q01/q99 statistics (see ``compute_norm_stats``).
"""

from __future__ import annotations

import io
import json
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

FRONT_CAMERA = "camera_front_wide_120fov"
STATE_NAMES = [
    "speed", "acc_x", "acc_y", "acc_z", "angvel_x", "angvel_y", "angvel_z", "heading_rel", "target_x", "target_y",
]
ACTION_NAMES = ["x", "y"]
US = 1e-6


def budget_size(w: int, h: int, max_pixels: int, factor: int) -> tuple:
    """(W, H) with aspect kept, sides multiples of ``factor`` and W*H <= max_pixels (same rule as QwenDrivePI)."""
    scale = min(1.0, (max_pixels / float(w * h)) ** 0.5)
    return max(factor, int(w * scale // factor) * factor), max(factor, int(h * scale // factor) * factor)


# ──────────────────────────────────────────────────────────────────────
#  Raw file access
# ──────────────────────────────────────────────────────────────────────
def load_clip_table(data_root: str, split: Optional[str] = None) -> pd.DataFrame:
    """Valid clips that have egomotion and the front wide camera: columns clip_id, chunk, split."""
    root = Path(data_root)
    clips = pd.read_parquet(root / "clip_index.parquet").reset_index()
    presence = pd.read_parquet(root / "metadata" / "feature_presence.parquet").reset_index()
    clips = clips.merge(presence[["clip_id", "egomotion", FRONT_CAMERA]], on="clip_id")
    clips = clips[clips.clip_is_valid & clips.egomotion & clips[FRONT_CAMERA]]
    if split is not None:
        clips = clips[clips.split == split]
    return clips[["clip_id", "chunk", "split"]].sort_values(["chunk", "clip_id"]).reset_index(drop=True)


class ClipStore:
    """Lazy, per-process access to egomotion parquet and front-camera video inside the chunk zips."""

    def __init__(self, data_root: str, max_open_zips: int = 8):
        self.root = Path(data_root)
        self.max_open_zips = max_open_zips
        self._zips: "OrderedDict[str, zipfile.ZipFile]" = OrderedDict()

    def __getstate__(self):  # zip handles are not shared across dataloader workers
        state = self.__dict__.copy()
        state["_zips"] = OrderedDict()
        return state

    def _zip(self, rel: str) -> zipfile.ZipFile:
        if rel in self._zips:
            self._zips.move_to_end(rel)
            return self._zips[rel]
        zf = zipfile.ZipFile(self.root / rel)
        self._zips[rel] = zf
        if len(self._zips) > self.max_open_zips:
            self._zips.popitem(last=False)[1].close()
        return zf

    def egomotion(self, clip_id: str, chunk: int) -> pd.DataFrame:
        zf = self._zip(f"labels/egomotion/egomotion.chunk_{chunk:04d}.zip")
        return pd.read_parquet(io.BytesIO(zf.read(f"{clip_id}.egomotion.parquet")))

    def _camera_zip(self, chunk: int) -> zipfile.ZipFile:
        return self._zip(f"camera/{FRONT_CAMERA}/{FRONT_CAMERA}.chunk_{chunk:04d}.zip")

    def frame_timestamps(self, clip_id: str, chunk: int) -> np.ndarray:
        zf = self._camera_zip(chunk)
        df = pd.read_parquet(io.BytesIO(zf.read(f"{clip_id}.{FRONT_CAMERA}.timestamps.parquet")))
        return df["timestamp"].to_numpy(dtype=np.int64)

    def decode_frames(self, clip_id: str, chunk: int, frame_indices: Sequence[int], pixel_budgets: Sequence[int],
                      factor: int = 32):
        """Decode the given frame indices (any order); frame i is resized (aspect kept, sides multiple of
        ``factor``) to at most ``pixel_budgets[i]`` pixels."""
        zf = self._camera_zip(chunk)
        wanted = {int(i): k for k, i in enumerate(frame_indices)}
        out: List[Optional[Image.Image]] = [None] * len(frame_indices)
        with zf.open(f"{clip_id}.{FRONT_CAMERA}.mp4") as fh, av.open(fh) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            fps = float(stream.average_rate)
            sizes = [budget_size(stream.width, stream.height, b, factor) for b in pixel_budgets]
            first = min(wanted)
            container.seek(int(first / fps / stream.time_base), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                idx = int(round(frame.time * fps))
                if idx in wanted:
                    w, h = sizes[wanted[idx]]
                    out[wanted[idx]] = frame.reformat(width=w, height=h, format="rgb24").to_image()
                if idx >= max(wanted):
                    break
        missing = [i for i, img in zip(frame_indices, out) if img is None]
        if missing:
            raise RuntimeError(f"{clip_id}: could not decode frames {missing}")
        return out


# ──────────────────────────────────────────────────────────────────────
#  Geometry
# ──────────────────────────────────────────────────────────────────────
class EgoTrack:
    """Egomotion of one clip with ego-frame quantities, interpolable at arbitrary timestamps (µs)."""

    def __init__(self, df: pd.DataFrame):
        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        self.t = df["timestamp"].to_numpy(dtype=np.float64)
        self.xy = df[["x", "y"]].to_numpy(dtype=np.float64)
        rot = Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64))
        self.yaw = np.unwrap(rot.as_euler("ZYX")[:, 0])
        vel = df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)  # local world frame
        acc = df[["ax", "ay", "az"]].to_numpy(dtype=np.float64)  # local world frame
        self.speed = np.linalg.norm(vel[:, :2], axis=1)
        self.acc_body = rot.inv().apply(acc)
        # body-frame angular velocity from consecutive orientations
        dt = np.diff(self.t) * US
        omega = (rot[:-1].inv() * rot[1:]).as_rotvec() / np.maximum(dt, 1e-6)[:, None]
        self.angvel_body = np.concatenate([omega, omega[-1:]], axis=0)

    @property
    def t_min(self) -> float:
        return float(self.t[0])

    @property
    def t_max(self) -> float:
        return float(self.t[-1])

    def _interp(self, arr: np.ndarray, tq: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return np.interp(tq, self.t, arr)
        return np.stack([np.interp(tq, self.t, arr[:, k]) for k in range(arr.shape[1])], axis=-1)

    def pose(self, t0: float):
        return self._interp(self.xy, np.array([t0]))[0], float(np.interp(t0, self.t, self.yaw))

    def to_ego(self, xy: np.ndarray, t0: float) -> np.ndarray:
        """World xy -> ego frame at t0 (x forward, y left)."""
        p0, yaw0 = self.pose(t0)
        c, s = np.cos(yaw0), np.sin(yaw0)
        d = xy - p0
        return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=-1)

    def positions_ego(self, tq: np.ndarray, t0: float) -> np.ndarray:
        return self.to_ego(self._interp(self.xy, tq), t0)

    def target_point(self, t0: float, distance: float) -> np.ndarray:
        p0, _ = self.pose(t0)
        future = self.t > t0
        xy = self.xy[future]
        if len(xy) == 0:
            return np.zeros(2)
        far = np.nonzero(np.linalg.norm(xy - p0, axis=1) >= distance)[0]
        point = xy[far[0]] if len(far) else xy[-1]
        return self.to_ego(point, t0)

    def heading_change(self, t0: float, horizon_s: float) -> float:
        t1 = min(t0 + horizon_s / US, self.t_max)
        return float(np.interp(t1, self.t, self.yaw) - np.interp(t0, self.t, self.yaw))


def classify_maneuver(track: EgoTrack, t0: float, horizon_s: float = 3.0, turn_deg: float = 15.0) -> str:
    dyaw = np.degrees(track.heading_change(t0, horizon_s))
    if dyaw > turn_deg:
        return "left"
    if dyaw < -turn_deg:
        return "right"
    return "straight"


# ──────────────────────────────────────────────────────────────────────
#  Sample construction (raw, un-normalised)
# ──────────────────────────────────────────────────────────────────────
class DriveSampler:
    """Builds raw state / action / meta for a clip at time t0 (µs)."""

    def __init__(
        self,
        state_horizon: int = 5,
        action_horizon: int = 10,
        dt: float = 0.1,
        image_offsets: Sequence[float] = (0.1, 0.0),
        target_distance: float = 25.0,
        context_past_s: float = 2.0,
        context_future_s: float = 5.0,
    ):
        self.state_horizon = state_horizon
        self.action_horizon = action_horizon
        self.dt = dt
        self.image_offsets = sorted(image_offsets, reverse=True)  # oldest first, 0.0 (current) last
        self.target_distance = target_distance
        self.context_past_s = context_past_s
        self.context_future_s = context_future_s

    def valid_range(self, track: EgoTrack, frame_ts: Optional[np.ndarray] = None):
        lo = track.t_min + (self.state_horizon - 1) * self.dt / US
        hi = track.t_max - (self.action_horizon - 1) * self.dt / US
        if frame_ts is not None:
            lo = max(lo, frame_ts[0] + max(self.image_offsets) / US)
            hi = min(hi, frame_ts[-1])
        return lo, hi

    def state_action(self, track: EgoTrack, t0: float):
        t_hist = t0 + (np.arange(self.state_horizon) - (self.state_horizon - 1)) * self.dt / US
        t_fut = t0 + np.arange(self.action_horizon) * self.dt / US
        _, yaw0 = track.pose(t0)
        target = track.target_point(t0, self.target_distance)
        state = np.concatenate(
            [
                track._interp(track.speed, t_hist)[:, None],
                track._interp(track.acc_body, t_hist),
                track._interp(track.angvel_body, t_hist),
                (np.interp(t_hist, track.t, track.yaw) - yaw0)[:, None],
                np.repeat(target[None], self.state_horizon, axis=0),
            ],
            axis=1,
        )
        action = track.positions_ego(t_fut, t0)
        return state.astype(np.float32), action.astype(np.float32), target.astype(np.float32)

    def context(self, track: EgoTrack, t0: float) -> Dict[str, np.ndarray]:
        """Longer past / future ego trajectory for plotting (clipped to the recording)."""
        t_past = t0 - np.arange(int(self.context_past_s / self.dt), -1, -1) * self.dt / US
        t_fut = t0 + np.arange(int(self.context_future_s / self.dt) + 1) * self.dt / US
        t_past = t_past[t_past >= track.t_min]
        t_fut = t_fut[t_fut <= track.t_max]
        return {
            "past_xy": track.positions_ego(t_past, t0).astype(np.float32),
            "future_xy": track.positions_ego(t_fut, t0).astype(np.float32),
        }

    def frame_indices(self, frame_ts: np.ndarray, t0: float) -> List[int]:
        return [int(np.argmin(np.abs(frame_ts - (t0 - off / US)))) for off in self.image_offsets]


# ──────────────────────────────────────────────────────────────────────
#  Normalisation (q01 / q99 -> [-1, 1], as DriveMoE and starVLA)
# ──────────────────────────────────────────────────────────────────────
class Normalizer:
    def __init__(self, stats: dict):
        self.stats = stats
        self._s = {
            k: (np.asarray(v["q01"], np.float32), np.asarray(v["q99"], np.float32))
            for k, v in stats.items()
            if isinstance(v, dict) and "q01" in v
        }

    @classmethod
    def from_file(cls, path: str) -> "Normalizer":
        with open(path) as f:
            return cls(json.load(f))

    def normalize(self, key: str, x: np.ndarray) -> np.ndarray:
        lo, hi = self._s[key]
        return (2.0 * (x - lo) / np.maximum(hi - lo, 1e-6) - 1.0).astype(np.float32)

    def unnormalize(self, key: str, x: np.ndarray) -> np.ndarray:
        lo, hi = self._s[key]
        return ((np.asarray(x, np.float32) + 1.0) / 2.0 * (hi - lo) + lo).astype(np.float32)


def compute_norm_stats(data_root: str, clips: pd.DataFrame, sampler: DriveSampler, t0s_per_clip: int = 8, seed: int = 0):
    """Per-dimension q01/q99 of state and action over randomly sampled t0s (egomotion only)."""
    rng = np.random.default_rng(seed)
    store = ClipStore(data_root)
    states, actions = [], []
    for row in clips.itertuples():
        try:
            track = EgoTrack(store.egomotion(row.clip_id, row.chunk))
        except Exception:
            continue
        lo, hi = sampler.valid_range(track)
        if hi <= lo:
            continue
        for t0 in rng.uniform(lo, hi, size=t0s_per_clip):
            s, a, _ = sampler.state_action(track, t0)
            states.append(s)
            actions.append(a)
    states = np.concatenate(states).reshape(-1, len(STATE_NAMES))
    actions = np.concatenate(actions).reshape(-1, len(ACTION_NAMES))
    pct = lambda x, q: np.percentile(x, q, axis=0).tolist()  # noqa: E731
    return {
        "state": {"q01": pct(states, 1), "q99": pct(states, 99), "names": STATE_NAMES},
        "action": {"q01": pct(actions, 1), "q99": pct(actions, 99), "names": ACTION_NAMES},
        "num_samples": int(len(actions) // sampler.action_horizon),
    }


def build_eval_index(
    data_root: str, clips: pd.DataFrame, sampler: DriveSampler, per_class: int, seed: int = 0, step_s: float = 1.0
) -> List[dict]:
    """Fixed open-loop eval samples balanced over {left, right, straight} (one t0 per clip and class)."""
    rng = np.random.default_rng(seed)
    store = ClipStore(data_root)
    buckets: Dict[str, List[dict]] = {"left": [], "right": [], "straight": []}
    for row in clips.sample(frac=1.0, random_state=seed).itertuples():
        if all(len(b) >= per_class for b in buckets.values()):
            break
        try:
            track = EgoTrack(store.egomotion(row.clip_id, row.chunk))
            frame_ts = store.frame_timestamps(row.clip_id, row.chunk)
        except Exception:
            continue
        lo, hi = sampler.valid_range(track, frame_ts)
        if hi <= lo:
            continue
        by_class: Dict[str, List[float]] = {}
        for t0 in np.arange(lo, hi, step_s / US):
            by_class.setdefault(classify_maneuver(track, t0), []).append(float(t0))
        for name, t0s in by_class.items():
            if len(buckets[name]) < per_class:
                buckets[name].append(
                    {"clip_id": row.clip_id, "chunk": int(row.chunk), "t0": float(rng.choice(t0s)), "maneuver": name}
                )
    samples = []
    for k in range(per_class):  # interleave classes so any prefix stays balanced
        for name in ("left", "right", "straight"):
            if k < len(buckets[name]):
                samples.append(buckets[name][k])
    return samples


# ──────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────
class PAIAVDriveDataset(Dataset):
    """
    Train mode (``samples=None``): one item per clip, t0 drawn uniformly at random on every access.
    Eval mode (``samples=[{clip_id, chunk, t0, ...}]``): fixed samples, deterministic.
    Items follow the QwenDrivePI schema (image / state / action) plus a ``meta`` dict.
    """

    def __init__(
        self,
        data_root: str,
        normalizer: Normalizer,
        split: str = "train",
        samples: Optional[List[dict]] = None,
        sampler: Optional[DriveSampler] = None,
        current_image_pixels: int = 921600,
        history_image_pixels: int = 174080,
        image_factor: int = 32,
        max_clips: Optional[int] = None,
        max_retries: int = 10,
    ):
        self.data_root = data_root
        self.normalizer = normalizer
        self.sampler = sampler or DriveSampler()
        self.samples = samples
        if samples is None:
            clips = load_clip_table(data_root, split)
            if max_clips:
                clips = clips.sample(n=min(max_clips, len(clips)), random_state=0)
            self.clips = clips.reset_index(drop=True)
        self.store = ClipStore(data_root)
        self.current_image_pixels = current_image_pixels
        self.history_image_pixels = history_image_pixels
        self.image_factor = image_factor
        self.max_retries = max_retries

    def __len__(self) -> int:
        return len(self.samples) if self.samples is not None else len(self.clips)

    def _build(self, clip_id: str, chunk: int, t0: Optional[float], rng: np.random.Generator) -> dict:
        track = EgoTrack(self.store.egomotion(clip_id, chunk))
        frame_ts = self.store.frame_timestamps(clip_id, chunk)
        lo, hi = self.sampler.valid_range(track, frame_ts)
        if hi <= lo:
            raise ValueError(f"{clip_id}: no valid t0")
        if t0 is None:
            t0 = float(rng.uniform(lo, hi))
        state, action, target = self.sampler.state_action(track, t0)

        idx = self.sampler.frame_indices(frame_ts, t0)
        budgets = [self.history_image_pixels] * (len(idx) - 1) + [self.current_image_pixels]
        images = self.store.decode_frames(clip_id, chunk, idx, budgets, self.image_factor)
        meta = {
            "clip_id": clip_id,
            "t0": t0,
            "maneuver": classify_maneuver(track, t0),
            "action_xy": action,
            "target_xy": target,
            **self.sampler.context(track, t0),
        }
        return {
            "image": images,
            "state": self.normalizer.normalize("state", state),
            "action": self.normalizer.normalize("action", action),
            "meta": meta,
        }

    def __getitem__(self, i: int) -> dict:
        if self.samples is not None:
            s = self.samples[i]
            item = self._build(s["clip_id"], int(s["chunk"]), float(s["t0"]), np.random.default_rng(0))
            item["meta"]["maneuver"] = s.get("maneuver", item["meta"]["maneuver"])
            return item
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        for attempt in range(self.max_retries):
            row = self.clips.iloc[i if attempt == 0 else int(rng.integers(len(self.clips)))]
            try:
                return self._build(row.clip_id, int(row.chunk), None, rng)
            except Exception as e:  # corrupt / too-short clip: resample another clip
                last_error = e
        raise RuntimeError(f"failed to build a sample after {self.max_retries} attempts: {last_error}")


def collate_examples(batch: List[dict]) -> List[dict]:
    return batch
