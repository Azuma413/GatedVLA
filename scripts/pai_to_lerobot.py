#!/usr/bin/env python3
"""Convert a PAI-style CARLA dataset into a GR00T/LeRobot v2.1 dataset for VLA-JEPA.

Input: the PAI layout produced by ``scripts/export_py123d_to_pai.py``
(``clip_index.parquet``, ``camera/<cam>/<cam>.chunk_XXXX.zip``,
``labels/egomotion/...``, ``labels/navigation_command/...``).

Output: a LeRobot v2.1 dataset (one episode per PAI clip) with the GR00T
``meta/modality.json`` that VLA-JEPA's ``gr00t_lerobot`` dataloader consumes:

    meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json}
    data/chunk-000/episode_XXXXXX.parquet
    videos/chunk-000/observation.images.<cam>/episode_XXXXXX.mp4

Driving action representation (see USAGE_VLAJEPA.md), aligned with Alpamayo 1.5:
  Each frame stores ``action`` = the unicycle control ``[accel, curvature]``
  (m/s^2, 1/m) taking the ego from that frame to the next, computed with
  Alpamayo 1.5's ``UnicycleAccelCurvatureActionSpace``. The VLA future-action
  chunk of length H is H consecutive controls, which the eval client rolls out
  (``action_to_traj``, seeded with the current speed) into ego-frame waypoints
  for LEAD's PID controller. accel and curvature live on very different scales
  (std ~3.5 vs ~0.02), so the dataloader normalizes each per-dim (mean/std);
  the raw (unnormalized) controls are stored here and stats are computed later.

Runtime: designed to run inside ``containers/lead.sif`` (numpy/pandas/pyarrow/cv2
+ system ffmpeg). GPU is not required.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("pai_to_lerobot")

# PAI camera name -> LeRobot video feature name. Order matters: VLA-JEPA feeds
# every video key to the VLM as an image, and uses keys[0] and keys[2] as the
# two world-model views (front_wide + front_tele).
CAMERA_MAP = [
    ("camera_front_wide_120fov", "front_wide"),
    ("camera_cross_left_120fov", "cross_left"),
    ("camera_front_tele_30fov", "front_tele"),
    ("camera_cross_right_120fov", "cross_right"),
]

COMMAND_TEXT_BY_VALUE = {
    0: "Unknown",
    1: "Turn left",
    2: "Turn right",
    3: "Go straight",
    4: "Follow the lane",
    5: "Change lane left",
    6: "Change lane right",
}

FPS = 10.0
CHUNKS_SIZE = 1000  # all episodes land in chunk 0 for our dataset sizes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pai-dir", required=True, type=Path, help="PAI dataset root (from export_py123d_to_pai.py).")
    p.add_argument("--output-dir", required=True, type=Path, help="Destination LeRobot dataset root.")
    p.add_argument("--robot-type", default="carla_ego", help="robot_type string written into info.json/meta.")
    p.add_argument("--max-clips", default=None, type=int, help="Optional cap for smoke tests.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def quat_to_yaw(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray) -> np.ndarray:
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


# Alpamayo 1.5 unicycle action space (accel, curvature). Imported lazily so this
# script still imports without alpamayo1.5 on the path (e.g. for --help).
_ACTION_SPACE = None
CURVATURE_BOUND = 0.2  # matches Alpamayo's curvature_bounds; clamp for stability


def _action_space():
    global _ACTION_SPACE
    if _ACTION_SPACE is None:
        import sys

        alpamayo15 = Path(__file__).resolve().parents[1] / "3rd_party" / "alpamayo1.5" / "src"
        if str(alpamayo15) not in sys.path:
            sys.path.insert(0, str(alpamayo15))
        from alpamayo1_5.action_space.unicycle_accel_curvature import (
            UnicycleAccelCurvatureActionSpace,
        )

        # Raw (unnormalized) controls: mean=0, std=1. Normalization is applied in
        # the dataloader from dataset_statistics. dt = 1/FPS (10 Hz -> 0.1 s).
        _ACTION_SPACE = UnicycleAccelCurvatureActionSpace(dt=1.0 / FPS)
    return _ACTION_SPACE


def compute_accel_curvature(x: np.ndarray, y: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Per-frame unicycle control ``[accel, curvature]`` (last frame = 0).

    Uses Alpamayo 1.5's inversion (``_v_to_a`` / ``_theta_v_a_to_kappa``), the exact
    inverse of the eval-time ``action_to_traj`` rollout, so the stored controls
    reconstruct the trajectory. Speed is derived from the ego positions.
    """
    import torch

    n = len(x)
    out = np.zeros((n, 2), dtype=np.float32)
    if n < 3:
        return out
    dt = 1.0 / FPS
    # Instantaneous speed per frame from position deltas (last frame padded).
    speed = np.hypot(np.diff(x), np.diff(y)) / dt  # (n-1,)
    speed = np.concatenate([speed, speed[-1:]])  # (n,)
    yaw_unwrapped = np.unwrap(yaw)

    u = _action_space()
    v_t = torch.tensor(speed, dtype=torch.float64)
    theta_t = torch.tensor(yaw_unwrapped, dtype=torch.float64)
    accel = u._v_to_a(v_t)  # (n-1,)
    kappa = u._theta_v_a_to_kappa(theta_t, v_t, accel)  # (n-1,)
    kappa = torch.clamp(kappa, -CURVATURE_BOUND, CURVATURE_BOUND)
    out[:-1, 0] = accel.numpy().astype(np.float32)
    out[:-1, 1] = kappa.numpy().astype(np.float32)
    return out


def read_zip_member_parquet(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member) as fh:
            return pd.read_parquet(io.BytesIO(fh.read()))


def extract_zip_member(zip_path: Path, member: str, dest: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member) as fh, open(dest, "wb") as out:
            out.write(fh.read())


def nearest_rows(source_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Return, for each query timestamp, the index of the nearest source timestamp."""
    order = np.argsort(source_ts)
    sorted_ts = source_ts[order]
    pos = np.searchsorted(sorted_ts, query_ts)
    pos = np.clip(pos, 1, len(sorted_ts) - 1)
    left = sorted_ts[pos - 1]
    right = sorted_ts[pos]
    choose_left = (query_ts - left) <= (right - query_ts)
    idx = np.where(choose_left, pos - 1, pos)
    return order[idx]


def reencode_h264(src_mp4: Path, dst_mp4: Path) -> None:
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src_mp4),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-g", "1", "-an",
        str(dst_mp4),
    ]
    subprocess.run(cmd, check=True)


def build_clip_frame_table(pai_dir: Path, clip_id: str, chunk: int, tmp: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Return per-frame table (ts, x, y, yaw, command) and extracted camera mp4 paths."""
    # 1. canonical frame timeline = front_wide camera frames
    cam_videos: dict[str, Path] = {}
    frame_ts = None
    for pai_name, new_name in CAMERA_MAP:
        cam_zip = pai_dir / "camera" / pai_name / f"{pai_name}.chunk_{chunk:04d}.zip"
        ts_member = f"{clip_id}.{pai_name}.timestamps.parquet"
        video_member = f"{clip_id}.{pai_name}.mp4"
        ts_df = read_zip_member_parquet(cam_zip, ts_member).sort_values("frame_index")
        if frame_ts is None:
            frame_ts = ts_df["timestamp"].to_numpy(dtype=np.int64)
        raw_mp4 = tmp / f"{clip_id}.{new_name}.raw.mp4"
        extract_zip_member(cam_zip, video_member, raw_mp4)
        cam_videos[new_name] = raw_mp4

    assert frame_ts is not None
    n = len(frame_ts)

    # 2. egomotion aligned to frame timestamps
    ego_zip = pai_dir / "labels" / "egomotion" / f"egomotion.chunk_{chunk:04d}.zip"
    ego_df = read_zip_member_parquet(ego_zip, f"{clip_id}.egomotion.parquet")
    ego_ts = ego_df["timestamp"].to_numpy(dtype=np.int64)
    sel = nearest_rows(ego_ts, frame_ts)
    x = ego_df["x"].to_numpy(dtype=np.float64)[sel]
    y = ego_df["y"].to_numpy(dtype=np.float64)[sel]
    yaw = quat_to_yaw(
        ego_df["qx"].to_numpy(dtype=np.float64)[sel],
        ego_df["qy"].to_numpy(dtype=np.float64)[sel],
        ego_df["qz"].to_numpy(dtype=np.float64)[sel],
        ego_df["qw"].to_numpy(dtype=np.float64)[sel],
    )

    # 3. navigation command aligned to frame timestamps (per-frame command text)
    command_text = np.array(["Follow the lane"] * n, dtype=object)
    nav_zip = pai_dir / "labels" / "navigation_command" / f"navigation_command.chunk_{chunk:04d}.zip"
    if nav_zip.exists():
        try:
            nav_df = read_zip_member_parquet(nav_zip, f"{clip_id}.navigation_command.parquet")
            nav_ts = nav_df["timestamp_us"].to_numpy(dtype=np.int64)
            nav_sel = nearest_rows(nav_ts, frame_ts)
            if "command_text" in nav_df.columns:
                command_text = nav_df["command_text"].to_numpy(dtype=object)[nav_sel]
            else:
                near = nav_df["command_near"].to_numpy(dtype=np.int64)[nav_sel]
                command_text = np.array(
                    [COMMAND_TEXT_BY_VALUE.get(int(v), "Unknown") for v in near], dtype=object
                )
        except KeyError:
            LOG.warning("Clip %s: navigation_command member missing, defaulting command text", clip_id)

    table = pd.DataFrame(
        {
            "timestamp_us": frame_ts,
            "x": x,
            "y": y,
            "yaw": yaw,
            "command_text": command_text,
        }
    )
    return table, cam_videos


def write_info_json(output_dir: Path, robot_type: str, total_episodes: int, total_frames: int,
                    total_tasks: int, video_shape: tuple[int, int]) -> None:
    height, width = video_shape
    features: dict = {
        "action": {"dtype": "float32", "shape": [2], "names": ["accel", "curvature"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for _pai, new_name in CAMERA_MAP:
        features[f"observation.images.{new_name}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": FPS,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
            },
        }
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(CAMERA_MAP),
        "total_chunks": 1,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)


def write_modality_json(output_dir: Path) -> None:
    modality = {
        "state": {},
        "action": {
            "accel": {"start": 0, "end": 1, "original_key": "action"},
            "curvature": {"start": 1, "end": 2, "original_key": "action"},
        },
        "video": {
            new_name: {"original_key": f"observation.images.{new_name}"}
            for _pai, new_name in CAMERA_MAP
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }
    with open(output_dir / "meta" / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s %(message)s")
    args = parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} not empty; use --overwrite.")
        import shutil

        shutil.rmtree(args.output_dir)
    (args.output_dir / "meta").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    clip_index = pd.read_parquet(args.pai_dir / "clip_index.parquet")
    clip_ids = list(clip_index.index)
    if args.max_clips is not None:
        clip_ids = clip_ids[: args.max_clips]
    LOG.info("Converting %d PAI clips -> %s", len(clip_ids), args.output_dir)

    # global task vocabulary
    task_to_index: dict[str, int] = {}

    episodes: list[dict] = []
    total_frames = 0
    video_shape: tuple[int, int] | None = None
    global_index = 0

    with tempfile.TemporaryDirectory(prefix="pai2lerobot_") as tmpdir:
        tmp = Path(tmpdir)
        for ep_idx, clip_id in enumerate(clip_ids):
            chunk = int(clip_index.loc[clip_id, "chunk"])
            LOG.info("clip %d/%d: %s", ep_idx + 1, len(clip_ids), clip_id)
            table, cam_videos = build_clip_frame_table(args.pai_dir, str(clip_id), chunk, tmp)
            n = len(table)

            actions = compute_accel_curvature(
                table["x"].to_numpy(), table["y"].to_numpy(), table["yaw"].to_numpy()
            )

            # task indices
            task_indices = []
            for text in table["command_text"].tolist():
                text = str(text)
                if text not in task_to_index:
                    task_to_index[text] = len(task_to_index)
                task_indices.append(task_to_index[text])

            frame_df = pd.DataFrame(
                {
                    "action": list(actions.astype(np.float32)),
                    "timestamp": (np.arange(n, dtype=np.float32) / FPS),
                    "frame_index": np.arange(n, dtype=np.int64),
                    "episode_index": np.full(n, ep_idx, dtype=np.int64),
                    "index": np.arange(global_index, global_index + n, dtype=np.int64),
                    "task_index": np.array(task_indices, dtype=np.int64),
                }
            )
            global_index += n
            frame_df.to_parquet(
                args.output_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet",
                index=False,
            )

            # videos (re-encode to h264 for torchvision_av)
            for _pai, new_name in CAMERA_MAP:
                dst = (
                    args.output_dir
                    / "videos"
                    / "chunk-000"
                    / f"observation.images.{new_name}"
                    / f"episode_{ep_idx:06d}.mp4"
                )
                reencode_h264(cam_videos[new_name], dst)
                if video_shape is None:
                    import cv2

                    cap = cv2.VideoCapture(str(dst))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    video_shape = (height, width)

            episodes.append(
                {
                    "episode_index": ep_idx,
                    "length": n,
                    "tasks": sorted({str(t) for t in table["command_text"].tolist()}),
                }
            )
            total_frames += n

    if video_shape is None:
        raise RuntimeError("No clips converted; nothing to write.")

    # meta/episodes.jsonl
    with open(args.output_dir / "meta" / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    # meta/tasks.jsonl (ordered by task_index)
    with open(args.output_dir / "meta" / "tasks.jsonl", "w") as f:
        for task, idx in sorted(task_to_index.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    write_info_json(
        args.output_dir,
        robot_type=args.robot_type,
        total_episodes=len(episodes),
        total_frames=total_frames,
        total_tasks=len(task_to_index),
        video_shape=video_shape,
    )
    write_modality_json(args.output_dir)
    LOG.info(
        "Done. %d episodes, %d frames, %d tasks. video=%s",
        len(episodes), total_frames, len(task_to_index), video_shape,
    )


if __name__ == "__main__":
    main()
