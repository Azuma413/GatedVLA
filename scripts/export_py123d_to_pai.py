#!/usr/bin/env python3
"""Export LEAD/123D Arrow logs to an Alpamayo-readable Physical AI AV layout.

This converter is intentionally focused on Alpamayo SFT inputs:
camera videos, frame timestamps, egomotion, clip_index, feature_presence, and
features.csv. If present beside each Arrow log, navigation_command.parquet is
exported as a PAI label for text-conditioned Alpamayo SFT.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.datatypes import CameraID, Timestamp

LOG = logging.getLogger("export_py123d_to_pai")

PAI_CAMERA_ORDER = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]

REQUIRED_CAMERA_MAPPING = {
    CameraID.FTCAM_L0: "camera_cross_left_120fov",
    CameraID.FTCAM_F0: "camera_front_wide_120fov",
    CameraID.FTCAM_R0: "camera_cross_right_120fov",
    CameraID.FTCAM_TELE_F0: "camera_front_tele_30fov",
}

COMMAND_TEXT_BY_VALUE = {
    0: "Unknown",
    1: "Turn left",
    2: "Turn right",
    3: "Go straight",
    4: "Follow the lane",
    5: "Change lane left",
    6: "Change lane right",
}


@dataclass(frozen=True)
class ClipExport:
    clip_id: str
    chunk: int
    split: str
    start_timestamp: int
    end_timestamp: int
    event_t0s: np.ndarray
    present_features: set[str]
    camera_zip_members: dict[str, dict[str, str]]
    egomotion_zip_members: dict[str, str]
    navigation_command_zip_members: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", required=True, type=Path, help="Root containing 123D Arrow log directories.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination PAI-style dataset root.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="PAI split name.")
    parser.add_argument("--chunk-size", default=128, type=int, help="Number of clips per chunk zip.")
    parser.add_argument("--fps", default=10.0, type=float, help="Output MP4 frame rate.")
    parser.add_argument("--history-seconds", default=1.6, type=float, help="Safe history margin for event_t0s.")
    parser.add_argument("--future-seconds", default=6.4, type=float, help="Safe future margin for event_t0s.")
    parser.add_argument("--event-stride-seconds", default=1.0, type=float, help="Spacing between generated t0 samples.")
    parser.add_argument("--max-logs", default=None, type=int, help="Optional cap for smoke tests.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before exporting.",
    )
    return parser.parse_args()


def find_log_dirs(logs_root: Path) -> list[Path]:
    if (logs_root / "sync.arrow").exists():
        return [logs_root]
    return sorted(path.parent for path in logs_root.rglob("sync.arrow"))


def safe_clip_id(log_dir: Path) -> str:
    return log_dir.name.replace("/", "_").replace(" ", "_")


def timestamp_us(ts: Timestamp | int) -> int:
    return ts.time_us if isinstance(ts, Timestamp) else int(ts)


def build_event_t0s(
    timestamps: list[int],
    history_seconds: float,
    future_seconds: float,
    stride_seconds: float,
) -> np.ndarray:
    if not timestamps:
        return np.asarray([], dtype=np.int64)
    start = timestamps[0] + int(history_seconds * 1_000_000)
    end = timestamps[-1] - int(future_seconds * 1_000_000)
    stride = int(stride_seconds * 1_000_000)
    if end < start:
        return np.asarray([], dtype=np.int64)
    return np.arange(start, end + 1, stride, dtype=np.int64)


def pose_to_egomotion_row(scene: ArrowSceneAPI, timestamp: int) -> dict[str, float | int]:
    ego_state = scene.get_ego_state_se3_at_timestamp(timestamp, criteria="nearest")
    if ego_state is None:
        raise RuntimeError(f"No ego_state_se3 near timestamp {timestamp}")

    imu = ego_state.imu_se3
    dyn = ego_state.dynamic_state_se3
    if dyn is None:
        vx = vy = vz = ax = ay = az = 0.0
    else:
        vel = dyn.velocity_3d
        acc = dyn.acceleration_3d
        vx, vy, vz = float(vel.x), float(vel.y), float(vel.z)
        ax, ay, az = float(acc.x), float(acc.y), float(acc.z)

    return {
        "timestamp": int(timestamp),
        "x": float(imu.x),
        "y": float(imu.y),
        "z": float(imu.z),
        "qx": float(imu.qx),
        "qy": float(imu.qy),
        "qz": float(imu.qz),
        "qw": float(imu.qw),
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "ax": ax,
        "ay": ay,
        "az": az,
    }


def add_planar_curvature(egomotion_df: pd.DataFrame) -> pd.DataFrame:
    """Approximate path curvature from ego pose yaw over traveled distance."""
    if len(egomotion_df) < 3:
        egomotion_df["curvature"] = 0.0
        return egomotion_df

    x = egomotion_df["x"].to_numpy(dtype=np.float64)
    y = egomotion_df["y"].to_numpy(dtype=np.float64)
    qx = egomotion_df["qx"].to_numpy(dtype=np.float64)
    qy = egomotion_df["qy"].to_numpy(dtype=np.float64)
    qz = egomotion_df["qz"].to_numpy(dtype=np.float64)
    qw = egomotion_df["qw"].to_numpy(dtype=np.float64)

    yaw = np.unwrap(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    ds = np.hypot(np.diff(x), np.diff(y))
    distance = np.concatenate([[0.0], np.cumsum(ds)])

    if distance[-1] <= 1e-6 or len(np.unique(distance)) < 3:
        curvature = np.zeros_like(distance)
    else:
        curvature = np.gradient(yaw, distance, edge_order=1)
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

    egomotion_df["curvature"] = curvature.astype(np.float32)
    return egomotion_df


def write_video_from_frames(
    scene: ArrowSceneAPI,
    camera_id: CameraID,
    timestamps: list[int],
    output_path: Path,
    fps: float,
) -> pd.DataFrame:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    written_timestamps: list[int] = []

    try:
        for ts in timestamps:
            cam = scene.get_camera_at_timestamp(ts, camera_id=camera_id, criteria="nearest")
            if cam is None:
                continue
            rgb = cam.image
            if writer is None:
                height, width = rgb.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open MP4 writer for {output_path}")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            written_timestamps.append(ts)
    finally:
        if writer is not None:
            writer.release()

    if not written_timestamps:
        raise RuntimeError(f"No frames written for camera {camera_id}")
    # The CoC autolabeler video loader needs BOTH a frame-index column and a
    # timestamp column to map keyframe timestamps to mp4 frames; frame_index is the
    # written order (== frame position in the mp4).
    return pd.DataFrame(
        {
            "frame_index": np.arange(len(written_timestamps), dtype=np.int64),
            "timestamp": np.asarray(written_timestamps, dtype=np.int64),
        }
    )


def add_file_to_zip(zip_path: Path, source_path: Path, arcname: str) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(source_path, arcname)


def find_navigation_command_sidecar(log_dir: Path) -> Path | None:
    candidates = [
        log_dir / "navigation_command.parquet",
        log_dir.with_suffix(".navigation_command.parquet"),
        log_dir.parent / f"{log_dir.name}.navigation_command.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def normalize_navigation_command_df(command_df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "timestamp" in command_df.columns and "timestamp_us" not in command_df.columns:
        rename_map["timestamp"] = "timestamp_us"
    command_df = command_df.rename(columns=rename_map).copy()

    if "timestamp_us" not in command_df.columns:
        raise ValueError("navigation command parquet must contain timestamp_us or timestamp")
    if "command_near" not in command_df.columns:
        raise ValueError("navigation command parquet must contain command_near")

    if "command_far" not in command_df.columns:
        command_df["command_far"] = command_df["command_near"]
    if "next_command" not in command_df.columns:
        command_df["next_command"] = command_df["command_far"]
    if "command_text" not in command_df.columns:
        command_df["command_text"] = command_df["command_near"].map(
            lambda value: COMMAND_TEXT_BY_VALUE.get(int(value), "Unknown")
        )

    columns = ["timestamp_us", "command_near", "command_far", "next_command", "command_text"]
    command_df = command_df[columns].sort_values("timestamp_us")
    command_df["timestamp_us"] = command_df["timestamp_us"].astype(np.int64)
    for column in ["command_near", "command_far", "next_command"]:
        command_df[column] = command_df[column].astype(np.int64)
    command_df["command_text"] = command_df["command_text"].astype(str)
    return command_df


def export_clip(
    log_dir: Path,
    output_dir: Path,
    split: str,
    chunk: int,
    args: argparse.Namespace,
) -> ClipExport | None:
    scene = ArrowSceneAPI(log_dir)
    clip_id = safe_clip_id(log_dir)
    timestamps = [timestamp_us(ts) for ts in scene.get_all_iteration_timestamps()]
    if not timestamps:
        LOG.warning("Skipping %s: no sync timestamps", log_dir)
        return None

    event_t0s = build_event_t0s(
        timestamps,
        history_seconds=args.history_seconds,
        future_seconds=args.future_seconds,
        stride_seconds=args.event_stride_seconds,
    )
    if len(event_t0s) == 0:
        LOG.warning("Skipping %s: clip is too short for Alpamayo margins", log_dir)
        return None

    present_features: set[str] = set()
    camera_zip_members: dict[str, dict[str, str]] = {}
    egomotion_zip_members: dict[str, str] = {}
    navigation_command_zip_members: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix="py123d_to_pai_") as tmp:
        tmp_path = Path(tmp)

        ego_df = pd.DataFrame([pose_to_egomotion_row(scene, ts) for ts in timestamps])
        ego_df = add_planar_curvature(ego_df)
        ego_tmp = tmp_path / f"{clip_id}.egomotion.parquet"
        ego_df.to_parquet(ego_tmp, index=False)
        ego_arc = f"{clip_id}.egomotion.parquet"
        ego_zip = output_dir / "labels" / "egomotion" / f"egomotion.chunk_{chunk:04d}.zip"
        add_file_to_zip(ego_zip, ego_tmp, ego_arc)
        present_features.add("egomotion")
        egomotion_zip_members["egomotion"] = ego_arc

        command_sidecar = find_navigation_command_sidecar(log_dir)
        if command_sidecar is None:
            LOG.warning("Clip %s has no navigation_command.parquet sidecar", clip_id)
        else:
            command_df = normalize_navigation_command_df(pd.read_parquet(command_sidecar))
            command_tmp = tmp_path / f"{clip_id}.navigation_command.parquet"
            command_df.to_parquet(command_tmp, index=False)
            command_arc = f"{clip_id}.navigation_command.parquet"
            command_zip = (
                output_dir
                / "labels"
                / "navigation_command"
                / f"navigation_command.chunk_{chunk:04d}.zip"
            )
            add_file_to_zip(command_zip, command_tmp, command_arc)
            present_features.add("navigation_command")
            navigation_command_zip_members["navigation_command"] = command_arc

        available_cameras = scene.get_camera_metadatas()
        missing_cameras = [
            camera_id
            for camera_id in REQUIRED_CAMERA_MAPPING
            if camera_id not in available_cameras
        ]
        if missing_cameras:
            raise RuntimeError(
                f"Clip {clip_id} is missing Alpamayo camera ids {missing_cameras}. "
                "Collect with target_dataset=CARLA_ALPAMAYO_4CAMERAS instead of converting legacy logs."
            )
        inverse_mapping = {pai_name: camera_id for camera_id, pai_name in REQUIRED_CAMERA_MAPPING.items()}

        for pai_name in PAI_CAMERA_ORDER:
            camera_id = inverse_mapping.get(pai_name)
            if camera_id is None:
                raise RuntimeError(f"Internal camera mapping error for {pai_name}")
            video_tmp = tmp_path / f"{clip_id}.{pai_name}.mp4"
            ts_df = write_video_from_frames(scene, camera_id, timestamps, video_tmp, args.fps)
            ts_tmp = tmp_path / f"{clip_id}.{pai_name}.timestamps.parquet"
            ts_df.to_parquet(ts_tmp, index=False)

            video_arc = f"{clip_id}.{pai_name}.mp4"
            ts_arc = f"{clip_id}.{pai_name}.timestamps.parquet"
            camera_zip = output_dir / "camera" / pai_name / f"{pai_name}.chunk_{chunk:04d}.zip"
            add_file_to_zip(camera_zip, video_tmp, video_arc)
            add_file_to_zip(camera_zip, ts_tmp, ts_arc)
            present_features.add(pai_name)
            camera_zip_members[pai_name] = {"video": video_arc, "frame_timestamps": ts_arc}

    return ClipExport(
        clip_id=clip_id,
        chunk=chunk,
        split=split,
        start_timestamp=int(timestamps[0]),
        end_timestamp=int(timestamps[-1]),
        event_t0s=event_t0s,
        present_features=present_features,
        camera_zip_members=camera_zip_members,
        egomotion_zip_members=egomotion_zip_members,
        navigation_command_zip_members=navigation_command_zip_members,
    )


def write_metadata(output_dir: Path, clips: Iterable[ClipExport]) -> None:
    clips = list(clips)
    clip_index = pd.DataFrame(
        [
            {
                "clip_id": clip.clip_id,
                "chunk": clip.chunk,
                "split": clip.split,
                "start_timestamp": clip.start_timestamp,
                "end_timestamp": clip.end_timestamp,
                "event_t0s": clip.event_t0s,
            }
            for clip in clips
        ]
    ).set_index("clip_id")
    clip_index.to_parquet(output_dir / "clip_index.parquet")

    features = ["egomotion", "navigation_command", *PAI_CAMERA_ORDER]
    feature_rows = []
    for feature in features:
        if feature == "egomotion":
            filename_template = "labels/egomotion/egomotion.chunk_{chunk:04d}.zip"
        elif feature == "navigation_command":
            filename_template = (
                "labels/navigation_command/navigation_command.chunk_{chunk:04d}.zip"
            )
        else:
            filename_template = f"camera/{feature}/{feature}.chunk_{{chunk:04d}}.zip"
        clip_files_in_zip = {}
        for clip in clips:
            if feature == "egomotion" and "egomotion" in clip.present_features:
                clip_files_in_zip[clip.clip_id] = clip.egomotion_zip_members
            elif (
                feature == "navigation_command"
                and "navigation_command" in clip.present_features
            ):
                clip_files_in_zip[clip.clip_id] = clip.navigation_command_zip_members
            elif feature in clip.camera_zip_members:
                clip_files_in_zip[clip.clip_id] = clip.camera_zip_members[feature]
        feature_rows.append(
            {
                "feature": feature,
                "filename_template": filename_template,
                "chunk_filename_template": filename_template,
                "clip_files_in_zip": json.dumps(clip_files_in_zip),
            }
        )
    pd.DataFrame(feature_rows).to_csv(output_dir / "features.csv", index=False)

    presence = pd.DataFrame(
        [
            {feature: feature in clip.present_features for feature in features}
            for clip in clips
        ],
        index=[clip.clip_id for clip in clips],
    )
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    presence.to_parquet(metadata_dir / "feature_presence.parquet")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s %(message)s")
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} is not empty. Use --overwrite or choose a fresh output directory."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_dirs = find_log_dirs(args.logs_root)
    if args.max_logs is not None:
        log_dirs = log_dirs[: args.max_logs]
    if not log_dirs:
        raise FileNotFoundError(f"No 123D log directories with sync.arrow found under {args.logs_root}")

    LOG.info("Exporting %d 123D logs to %s", len(log_dirs), args.output_dir)
    clips: list[ClipExport] = []
    for index, log_dir in enumerate(log_dirs):
        chunk = index // args.chunk_size
        LOG.info("Exporting clip %d/%d: %s", index + 1, len(log_dirs), log_dir)
        clip = export_clip(log_dir, args.output_dir, args.split, chunk, args)
        if clip is not None:
            clips.append(clip)

    if not clips:
        raise RuntimeError("No clips were exported.")

    write_metadata(args.output_dir, clips)
    LOG.info("Done. Exported %d clips.", len(clips))


if __name__ == "__main__":
    main()
