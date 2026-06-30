# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Optional, Sequence, Set, Tuple

import cv2
import meta_action.utils.io as io_utils
import numpy as np
import pandas as pd
import tqdm
from meta_action.utils.constant import (
    DELTA_TIMESTAMP,
    META_ACTION2TEXT,
    META_ACTION_LANE,
    META_ACTION_LATERAL,
    META_ACTION_LONGITUDINAL,
    NUM_FRAMES_VIS,
    PANO_VIDEO,
    VIS_FPS,
)

logger = logging.getLogger(__name__)


def _read_clip_list_file(path: Optional[str]) -> Optional[Set[str]]:
    """Load clip IDs from a text file.

    Args:
        path: Path to a file with one clip ID per line.

    Returns:
        A set of clip IDs, or None when `path` is None, missing, or empty.
    """
    if path is None:
        return None
    if not os.path.exists(path):
        logger.warning("clip_list not found at %s", path)
        return None
    with open(path, encoding="utf-8") as f:
        ids = [line.strip() for line in f.readlines()]
    ids = [cid for cid in ids if cid]
    return set(ids) if ids else None


def load_lane_map_status_track(lane_debug_dir: Optional[str], clip_id: str) -> List[str]:
    """Build per-frame lane-map availability captions from lane debug csv."""
    track = ["Lane Map: Unknown" for _ in range(NUM_FRAMES_VIS)]
    if not lane_debug_dir:
        return ["Lane Map: N/A" for _ in range(NUM_FRAMES_VIS)]

    debug_path = os.path.join(lane_debug_dir, f"{clip_id}_ego_lane_prepare_debug.csv")
    if not os.path.exists(debug_path):
        return ["Lane Map: N/A" for _ in range(NUM_FRAMES_VIS)]

    try:
        df = pd.read_csv(debug_path)
    except Exception:
        return ["Lane Map: N/A" for _ in range(NUM_FRAMES_VIS)]

    if len(df) == 0:
        return ["Lane Map: N/A" for _ in range(NUM_FRAMES_VIS)]

    # 10Hz lane debug to 30fps video: 1 lane ts ~= 3 frames.
    frames_per_lane_ts = max(1, int(round(float(VIS_FPS) * float(DELTA_TIMESTAMP))))
    for fi in range(NUM_FRAMES_VIS):
        ts = min(int(fi / frames_per_lane_ts), len(df) - 1)
        lane_id = df.iloc[ts].get("lane_centerline_id", np.nan)
        lane_offset = df.iloc[ts].get("lane_lateral_offset_m", np.nan)
        unavailable = pd.isna(lane_id) or pd.isna(lane_offset)
        if unavailable:
            track[fi] = "Lane Map: UNAVAILABLE"
        else:
            track[fi] = "Lane Map: OK"
    return track


def load_lane_confidence_track(lane_debug_dir: Optional[str], clip_id: str) -> List[str]:
    """Build per-frame lane confidence captions from lane debug csv."""
    track = ["Lane Conf: N/A" for _ in range(NUM_FRAMES_VIS)]
    if not lane_debug_dir:
        return track

    debug_path = os.path.join(lane_debug_dir, f"{clip_id}_ego_lane_prepare_debug.csv")
    if not os.path.exists(debug_path):
        return track

    try:
        df = pd.read_csv(debug_path)
    except Exception:
        return track

    if len(df) == 0:
        return track

    has_score = "lane_confidence_score" in df.columns
    has_label = "lane_confidence_label" in df.columns
    if not has_score and not has_label:
        return track

    frames_per_lane_ts = max(1, int(round(float(VIS_FPS) * float(DELTA_TIMESTAMP))))
    for fi in range(NUM_FRAMES_VIS):
        ts = min(int(fi / frames_per_lane_ts), len(df) - 1)
        score_val = df.iloc[ts].get("lane_confidence_score", np.nan)
        lbl_val = df.iloc[ts].get("lane_confidence_label", "")
        lbl = str(lbl_val) if not pd.isna(lbl_val) else ""
        if pd.isna(score_val):
            if lbl:
                track[fi] = f"Lane Conf: {lbl}"
            continue
        track[fi] = f"Lane Conf: {float(score_val):.2f} ({lbl or 'n/a'})"
    return track


def add_caption_to_video(
    video_path: str,
    output_path: str,
    captions: Sequence[Sequence[str]],
    debug_values: Sequence[Sequence[float]],
    start_frame_idx: int,
    total_frames: int,
    output_downsample: float = 1.0,
) -> None:
    """Render captions and debug overlays on top of a video clip.

    Args:
        video_path: Input video path.
        output_path: Output video path.
        captions: Caption tracks indexed by frame; first 3 tracks are timeline inputs.
        debug_values: Two numeric tracks `[velocity_values, heading_values]`.
        start_frame_idx: Start frame index in the input video.
        total_frames: Number of frames to render.
        output_downsample: Spatial downsample scale for output video frames.
            Must be in `(0, 1]`; `1.0` keeps original resolution.

    Returns:
        None.
    """
    cap = cv2.VideoCapture(video_path)
    out = None
    try:
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {video_path}")

        # Get video properties.
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps != VIS_FPS:
            raise ValueError(f"FPS mismatch: expected {VIS_FPS}, got {fps}.")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = float(output_downsample)
        if not (0.0 < scale <= 1.0):
            raise ValueError("output_downsample must be in (0, 1].")
        out_width = max(1, int(round(width * scale)))
        out_height = max(1, int(round(height * scale)))

        # Define codec and create output writer.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        # Seek to first overlapping frame.
        if start_frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame_idx))

        # Process each frame in the overlap window.
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Add caption text for each caption track.
            types_caption_count = 1
            start_margin = 15
            caption_interval = 75
            fontsize = 2

            for caption_list_tmp in captions:
                if frame_idx < len(caption_list_tmp):
                    cv2.putText(
                        frame,
                        caption_list_tmp[frame_idx],
                        (start_margin, caption_interval * types_caption_count),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontsize,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                types_caption_count += 1

            # Draw debug overlays only when valid data exists.
            velocity_values = debug_values[0] if len(debug_values) > 0 else []
            heading_values = debug_values[1] if len(debug_values) > 1 else []
            if len(velocity_values) > 0:
                draw_velocity_plot(velocity_values, frame, frame_idx)
            if len(heading_values) > 0 and len(velocity_values) == len(heading_values):
                draw_heading_plot(
                    heading_values,
                    frame,
                    frame_idx,
                    velocity_values=velocity_values,
                )

            draw_timeline(frame_idx, frame, captions[0], captions[1], captions[2], total_frames)

            if frame_idx >= total_frames - 1:
                break

            # Write rendered frame.
            if scale < 1.0:
                frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
            out.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        if out is not None:
            out.release()


def stack_videos(video_paths: Sequence[str], output_path: str, layout: str = "horizontal") -> None:
    """Stack multiple videos into one output video.

    Args:
        video_paths: Input videos to stack in listed order.
        output_path: Output stacked video path.
        layout: "horizontal" or "vertical".

    Returns:
        None.
    """
    cap_list = [cv2.VideoCapture(path) for path in video_paths]
    out = None
    try:
        # Use first video as reference for frame size.
        frame_width = int(cap_list[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap_list[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Compute output dimensions from layout.
        if layout == "horizontal":
            out_width = frame_width * len(video_paths)
            out_height = frame_height
        elif layout == "vertical":
            out_width = frame_width
            out_height = frame_height * len(video_paths)
        else:
            raise ValueError("Invalid layout. Use 'horizontal' or 'vertical'.")

        # Create output writer.
        out = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (out_width, out_height)
        )

        while True:
            frames = []
            for cap in cap_list:
                ret, frame = cap.read()
                if not ret:
                    break  # Stop if any video ends.
                frames.append(frame)

            if len(frames) != len(cap_list):
                break

            if layout == "horizontal":
                stacked_frame = np.hstack(frames)
            else:
                stacked_frame = np.vstack(frames)

            out.write(stacked_frame)

            # Allow manual early exit while previewing.
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        for cap in cap_list:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


def _draw_dashed_horizontal_line(
    img: np.ndarray,
    x0: int,
    x1: int,
    y: int,
    color: Tuple[int, int, int],
    thickness: int,
    dash_len: int = 10,
    gap_len: int = 6,
) -> None:
    """Draw a dashed horizontal line on an image.

    Args:
        img: Target frame.
        x0: Start x coordinate.
        x1: End x coordinate.
        y: Y coordinate.
        color: BGR color.
        thickness: Line thickness in pixels.
        dash_len: Dash segment length in pixels.
        gap_len: Gap length in pixels.

    Returns:
        None.
    """
    if x0 > x1:
        x0, x1 = x1, x0
    x = int(x0)
    while x < x1:
        x_end = min(x + dash_len, x1)
        cv2.line(img, (x, int(y)), (int(x_end), int(y)), color, thickness)
        x += dash_len + gap_len


def draw_velocity_plot(
    velocity_values: Sequence[float],
    frame: np.ndarray,
    frame_idx: int,
    size: int = 300,
    margin: int = 50,
) -> None:
    """Draw a velocity-vs-time plot in the lower-left of the frame.

    Visual elements:
    - Dashed horizontal grid lines at 0/5/10/15/20/25/30 m/s.
    - Solid white velocity curve.
    - Current-frame vertical cursor and speed text label.

    Args:
        velocity_values: Velocity per frame in m/s.
        frame: Target frame in BGR format.
        frame_idx: Current frame index for cursor/label.
        size: Plot width/height bound in pixels.
        margin: Plot margin from frame edges in pixels.

    Returns:
        None.
    """
    if frame is None or len(frame.shape) < 2:
        return

    if len(velocity_values) == 0:
        return

    height, width = frame.shape[:2]

    plot_w = min(size, max(10, width - 2 * margin))
    plot_h = min(size, max(10, height - 2 * margin))

    x0 = margin
    y1 = height - margin
    x1 = x0 + plot_w
    y0 = y1 - plot_h

    # Colors and thicknesses
    color_axis = (180, 180, 180)
    color_main = (255, 255, 255)  # white
    thickness_main = 2
    thickness_minor = 1
    thickness_vert = 1

    # Value range [0, 30] m/s
    v_min, v_max = 0.0, 30.0

    # Background box (optional subtle border)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 60), 1)

    # Y grid lines and labels
    ticks = [0, 5, 10, 15, 20, 25, 30]
    for tick in ticks:
        # Map velocity to y
        y = int(y1 - (float(tick) - v_min) / (v_max - v_min) * plot_h)
        # Choose thickness: main for 0 m/s
        t = thickness_main if tick == 0 else thickness_minor
        _draw_dashed_horizontal_line(frame, x0, x1, y, color_axis, t)
        # Label near the left end inside the plot area
        label = str(tick)
        cv2.putText(
            frame,
            label,
            (x0 + 3, max(y - 2, y0 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color_axis,
            1,
            cv2.LINE_AA,
        )

    # Velocity polyline
    n = len(velocity_values)
    if n >= 2:
        pts = []
        for i, v in enumerate(velocity_values):
            # Clamp
            v_clamped = float(max(v_min, min(v_max, v)))
            # Map index to x across the width
            x = int(x0 + (i * (plot_w - 1)) / (n - 1))
            # Map velocity to y
            y = int(y1 - (v_clamped - v_min) / (v_max - v_min) * plot_h)
            pts.append([x, y])
        pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts_np], isClosed=False, color=color_main, thickness=thickness_main)

    # Current index vertical line and label
    if n >= 1:
        idx = int(max(0, min(len(velocity_values) - 1, frame_idx)))
        x_cur = int(x0 + (idx * (plot_w - 1)) / max(1, (n - 1)))
        cv2.line(frame, (x_cur, y0), (x_cur, y1), color_main, thickness_vert)

        v_cur = float(velocity_values[idx])
        v_cur_clamped = float(max(v_min, min(v_max, v_cur)))
        y_cur = int(y1 - (v_cur_clamped - v_min) / (v_max - v_min) * plot_h)
        # Place the label slightly above the polyline point, clamped within the plot
        text = f"{v_cur:.1f} m/s"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = max(x0 + 2, min(x1 - text_size[0] - 2, x_cur - text_size[0] // 2))
        text_y = max(y0 + text_size[1] + 2, y_cur - 6)
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_main,
            1,
            cv2.LINE_AA,
        )


def _draw_dashed_vertical_line(
    img: np.ndarray,
    x: int,
    y0: int,
    y1: int,
    color: Tuple[int, int, int],
    thickness: int,
    dash_len: int = 10,
    gap_len: int = 6,
) -> None:
    """Draw a dashed vertical line on an image.

    Args:
        img: Target frame.
        x: X coordinate.
        y0: Start y coordinate.
        y1: End y coordinate.
        color: BGR color.
        thickness: Line thickness in pixels.
        dash_len: Dash segment length in pixels.
        gap_len: Gap length in pixels.

    Returns:
        None.
    """
    if y0 > y1:
        y0, y1 = y1, y0
    y = int(y0)
    while y < y1:
        y_end = min(y + dash_len, y1)
        cv2.line(img, (int(x), y), (int(x), int(y_end)), color, thickness)
        y += dash_len + gap_len


def draw_heading_plot(
    heading_values: Sequence[float],
    frame: np.ndarray,
    frame_idx: int,
    size: int = 300,
    margin: int = 50,
    velocity_values: Optional[Sequence[float]] = None,
) -> None:
    """Draw heading and heading-rate plots in the lower-right corner.

    Visual elements:
    - White heading curve (rad).
    - Red heading-rate curve (rad/m), derived from wrapped heading delta and traveled distance.
    - Current-frame vertical cursor, value labels, and legend.

    Args:
        heading_values: Heading per frame in radians.
        frame: Target frame in BGR format.
        frame_idx: Current frame index for cursor/labels.
        size: Plot width/height bound in pixels.
        margin: Plot margin from frame edges in pixels.
        velocity_values: Velocity per frame used to compute heading rate in rad/m.

    Returns:
        None.
    """
    if frame is None or len(frame.shape) < 2:
        raise ValueError("Invalid frame passed to draw_heading_plot")

    if len(heading_values) == 0:
        raise ValueError("heading_values must be provided and non-empty for rad/m plot")

    if velocity_values is None:
        raise ValueError("velocity_values must be provided to compute rad/m")

    height, width = frame.shape[:2]

    plot_w = min(size, max(10, width - 2 * margin))
    plot_h = min(size, max(10, height - 2 * margin))

    # Lower-right corner placement
    x1 = width - margin
    y1 = height - margin
    x0 = x1 - plot_w
    y0 = y1 - plot_h

    # Colors and thicknesses
    color_axis = (180, 180, 180)
    color_heading = (255, 255, 255)  # white
    color_rate = (0, 0, 255)  # red (BGR)
    thickness_main = 2
    thickness_minor = 1
    thickness_vert = 1
    axis_text_scale = 0.4
    rate_text_scale = 0.5
    text_thickness = 1

    # Border rectangle (optional subtle)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 60), 1)

    # Prepare data
    h = np.asarray(heading_values, dtype=np.float64)
    n = len(h)
    if n < 1:
        return

    # Axis ranges
    h_min, h_max = -np.pi, np.pi
    # Compute rate of heading change in rad/m (wrapped dh divided by per-frame distance)
    legend_unit = "rad/m"
    if n >= 2:
        dh = np.arctan2(np.sin(h[1:] - h[:-1]), np.cos(h[1:] - h[:-1]))
        if len(velocity_values) != n:
            raise ValueError("velocity_values must be the same length as heading_values")
        v = np.asarray(velocity_values, dtype=np.float64)
        d_per_frame = np.maximum(v, 0.0) / float(VIS_FPS)
        d = d_per_frame[1:]
        rate = dh / np.maximum(d, 1e-6)
        rate = np.concatenate(([0.0], rate))
    else:
        rate = np.zeros_like(h)

    # Use fixed rate range for secondary Y-axis (rad/m)
    r_min, r_max = -0.08, 0.08

    # Draw heading grid lines at -pi, -pi/2, 0, pi/2, pi (left-side labels)
    ticks_heading = [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi]
    y_ticks_for_right = []
    for t in ticks_heading:
        y = int(y1 - (float(t) - h_min) / (h_max - h_min) * plot_h)
        _draw_dashed_horizontal_line(frame, x0, x1, y, color_axis, thickness_minor)
        y_ticks_for_right.append(y)
        label = f"{t:.2f}"
        cv2.putText(
            frame,
            label,
            (x0 + 3, max(y - 2, y0 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            axis_text_scale,
            color_axis,
            text_thickness,
            cv2.LINE_AA,
        )

    # Right-side labels (red) for rad/m using the same 5 dashed lines
    right_labels = ["0.08", "0.04", "0.00", "-0.04", "-0.08"]
    # Sort y positions from top to bottom to align with labels order
    y_sorted = sorted(y_ticks_for_right)
    for y, lab in zip(y_sorted, right_labels):
        text_size, _ = cv2.getTextSize(
            lab, cv2.FONT_HERSHEY_SIMPLEX, rate_text_scale, text_thickness
        )
        text_w, text_h = text_size
        tx = max(x0, x1 - text_w - 3)
        ty = max(y0 + text_h + 2, min(y1 - 2, y + text_h // 2))
        cv2.putText(
            frame,
            lab,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            rate_text_scale,
            (0, 0, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    # Helper: map index to x
    def idx_to_x(i: int) -> int:
        if n <= 1:
            return x0
        return int(x0 + (i * (plot_w - 1)) / (n - 1))

    # Heading polyline (white)
    if n >= 2:
        pts_h = []
        for i in range(n):
            h_clamped = float(max(h_min, min(h_max, h[i])))
            x = idx_to_x(i)
            y = int(y1 - (h_clamped - h_min) / (h_max - h_min) * plot_h)
            pts_h.append([x, y])
        pts_h_np = np.array(pts_h, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            frame,
            [pts_h_np],
            isClosed=False,
            color=color_heading,
            thickness=thickness_main,
        )

    # Angular velocity polyline (red), mapped to its own range on the same axes
    if n >= 2:
        pts_r = []
        for i in range(n):
            r_clamped = float(max(r_min, min(r_max, rate[i])))
            x = idx_to_x(i)
            # Map rate to y using r_min..r_max range onto the same plot area
            y = int(y1 - (r_clamped - r_min) / (r_max - r_min) * plot_h)
            pts_r.append([x, y])
        pts_r_np = np.array(pts_r, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            frame,
            [pts_r_np],
            isClosed=False,
            color=color_rate,
            thickness=thickness_minor,
        )

    # Current frame vertical line
    idx = int(max(0, min(n - 1, frame_idx)))
    x_cur = idx_to_x(idx)
    cv2.line(frame, (x_cur, y0), (x_cur, y1), color_heading, thickness_vert)

    # Current value labels (optional, small)
    h_cur = float(h[idx])
    h_cur_clamped = float(max(h_min, min(h_max, h_cur)))
    y_h_cur = int(y1 - (h_cur_clamped - h_min) / (h_max - h_min) * plot_h)
    txt_h = f"{h_cur:.2f} rad"
    sz_h, _ = cv2.getTextSize(txt_h, cv2.FONT_HERSHEY_SIMPLEX, axis_text_scale, text_thickness)
    tx_h = max(x0 + 2, min(x1 - sz_h[0] - 2, x_cur - sz_h[0] // 2))
    ty_h = max(y0 + sz_h[1] + 2, y_h_cur - 6)
    cv2.putText(
        frame,
        txt_h,
        (tx_h, ty_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        axis_text_scale,
        color_heading,
        text_thickness,
        cv2.LINE_AA,
    )

    r_cur = float(rate[idx])
    # Map rate to y for positioning label
    r_cur_clamped = float(max(r_min, min(r_max, r_cur)))
    y_r_cur = int(y1 - (r_cur_clamped - r_min) / (r_max - r_min) * plot_h)
    txt_r = f"{r_cur:.2f} {legend_unit}"
    sz_r, _ = cv2.getTextSize(txt_r, cv2.FONT_HERSHEY_SIMPLEX, rate_text_scale, text_thickness)
    tx_r = max(x0 + 2, min(x1 - sz_r[0] - 2, x_cur - sz_r[0] // 2))
    ty_r = min(y1 - 2, y_r_cur + sz_r[1] + 6)
    cv2.putText(
        frame,
        txt_r,
        (tx_r, ty_r),
        cv2.FONT_HERSHEY_SIMPLEX,
        rate_text_scale,
        color_rate,
        text_thickness,
        cv2.LINE_AA,
    )

    # Legend (top-left inside the plot)
    legend_x = x0 + 6
    legend_y = y0 + 14
    # heading (white)
    cv2.rectangle(
        frame,
        (legend_x, legend_y - 8),
        (legend_x + 10, legend_y - 2),
        color_heading,
        -1,
    )
    cv2.putText(
        frame,
        "heading (rad)",
        (legend_x + 14, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        axis_text_scale,
        color_heading,
        text_thickness,
        cv2.LINE_AA,
    )
    # rate (red)
    legend_y2 = legend_y + 14
    cv2.rectangle(frame, (legend_x, legend_y2 - 8), (legend_x + 10, legend_y2 - 2), color_rate, -1)
    cv2.putText(
        frame,
        f"ang vel ({legend_unit})",
        (legend_x + 14, legend_y2),
        cv2.FONT_HERSHEY_SIMPLEX,
        rate_text_scale,
        color_rate,
        text_thickness,
        cv2.LINE_AA,
    )


def draw_timeline(
    frame_idx: int,
    frame: np.ndarray,
    captions_longitudinal: Sequence[str],
    captions_lateral: Sequence[str],
    captions_lane: Sequence[str],
    total_frames: int,
    size: int = 300,
    margin: int = 50,
) -> None:
    """Draw longitudinal/lateral/lane caption timelines and a frame cursor.

    Visual elements:
    - Three horizontal tracks (longitudinal, lateral, lane).
    - Dashed vertical ticks at label change points.
    - Current-frame vertical cursor and elapsed-time label.

    Args:
        frame_idx: Current frame index.
        frame: Target frame in BGR format.
        captions_longitudinal: Longitudinal caption values by frame.
        captions_lateral: Lateral caption values by frame.
        captions_lane: Lane caption values by frame.
        total_frames: Number of frames used for x-axis normalization.
        size: Plot width/height bound in pixels.
        margin: Plot margin from frame edges in pixels.

    Returns:
        None.
    """
    if frame is None or len(frame.shape) < 2:
        return

    height, width = frame.shape[:2]
    plot_w = min(size, max(10, width - 2 * margin))
    plot_h = min(size, max(10, height - 2 * margin))
    # Position at top-right with given margins
    x1 = width - margin
    x0 = x1 - plot_w
    y0 = margin
    y1 = y0 + plot_h

    color_white = (255, 255, 255)
    thick = 2
    thin = 1

    # Track y positions (top to bottom: longitudinal, lateral, lane)
    y_long = int(y0 + 0.25 * plot_h)
    y_lat = int(y0 + 0.50 * plot_h)
    y_lane = int(y0 + 0.75 * plot_h)

    # Draw boundary vertical lines (start/end)
    cv2.line(frame, (x0, y0), (x0, y1), color_white, thick)
    cv2.line(frame, (x1, y0), (x1, y1), color_white, thick)

    # Draw the three horizontal tracks
    cv2.line(frame, (x0, y_long), (x1, y_long), color_white, thick)
    cv2.line(frame, (x0, y_lat), (x1, y_lat), color_white, thick)
    cv2.line(frame, (x0, y_lane), (x1, y_lane), color_white, thick)

    # Labels above the left end
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    cv2.putText(
        frame,
        "Longitudinal",
        (x0 + 3, max(y_long - 6, y0 + 12)),
        font,
        font_scale,
        color_white,
        thin,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Lateral",
        (x0 + 3, max(y_lat - 6, y0 + 12)),
        font,
        font_scale,
        color_white,
        thin,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Lane",
        (x0 + 3, max(y_lane - 6, y0 + 12)),
        font,
        font_scale,
        color_white,
        thin,
        cv2.LINE_AA,
    )

    # Determine change points for each track
    def compute_change_indices(labels: Sequence[str]) -> List[int]:
        if labels is None or len(labels) == 0:
            return []
        n = min(len(labels), total_frames)
        idxs = []
        prev = labels[0]
        for i in range(1, n):
            if labels[i] != prev:
                idxs.append(i)
                prev = labels[i]
        return idxs

    changes_long = compute_change_indices(captions_longitudinal)
    changes_lat = compute_change_indices(captions_lateral)
    changes_lane = compute_change_indices(captions_lane)

    # Map index to x
    def frame_index_to_x(i: int) -> int:
        if total_frames <= 1:
            return x0
        return int(x0 + (i * (plot_w - 1)) / (total_frames - 1))

    # Draw dashed vertical ticks (20% plot height) centered on each track line
    seg_half = int(0.1 * plot_h)
    for i in changes_long:
        x = frame_index_to_x(i)
        _draw_dashed_vertical_line(
            frame, x, y_long - seg_half, y_long + seg_half, color_white, thin
        )
    for i in changes_lat:
        x = frame_index_to_x(i)
        _draw_dashed_vertical_line(frame, x, y_lat - seg_half, y_lat + seg_half, color_white, thin)
    for i in changes_lane:
        x = frame_index_to_x(i)
        _draw_dashed_vertical_line(
            frame, x, y_lane - seg_half, y_lane + seg_half, color_white, thin
        )

    # Current frame vertical cursor across the entire plot height
    x_cur = frame_index_to_x(frame_idx)
    cv2.line(frame, (x_cur, y0), (x_cur, y1), color_white, thin)

    # Draw milliseconds label at the bottom of the plot area (centered)
    ms_since_start = int(round((float(frame_idx) / float(VIS_FPS)) * 1000.0))
    text = f"{ms_since_start} ms"
    text_size, _ = cv2.getTextSize(text, font, font_scale, thin)
    text_w, text_h = text_size
    tx = max(x0, x0 + (plot_w - text_w) // 2)
    ty = min(y1 - 2, y1 - 2)  # a couple pixels above bottom line
    cv2.putText(frame, text, (tx, ty), font, font_scale, color_white, thin, cv2.LINE_AA)


def _empty_agent_dataframe() -> pd.DataFrame:
    """Create an empty standardized agent dataframe.

    Returns:
        Empty dataframe with required columns and expected dtypes.
    """
    return pd.DataFrame(
        {
            "scene_ts": np.array([], dtype=np.int64),
            "rel_time_seconds": np.array([], dtype=np.float64),
            "v": np.array([], dtype=np.float64),
            "heading": np.array([], dtype=np.float64),
        }
    )


def get_agent_data(cache_root: Optional[str], clip_id: str) -> pd.DataFrame:
    """Load agent data for a clip from cache and return standardized dataframe.

    Args:
        cache_root: Root path containing per-clip cached agent data.
        clip_id: Clip ID used to locate cache file.

    Returns:
        Dataframe with `scene_ts`, `rel_time_seconds`, `v`, `heading`.
        Returns an empty standardized dataframe when data is missing.
    """
    if cache_root is None:
        return _empty_agent_dataframe()

    agent_feather_path = os.path.join(cache_root, clip_id, "agent_data_dt0.10.feather")

    if not os.path.exists(agent_feather_path):
        logger.warning("agent data not found at %s", agent_feather_path)
        return _empty_agent_dataframe()

    df = pd.read_feather(agent_feather_path)

    num_rows = len(df)
    scene_ts = np.arange(num_rows, dtype=np.int64)
    rel_time_seconds = scene_ts.astype(np.float64) * float(DELTA_TIMESTAMP)

    vx = df["vx"].to_numpy(dtype=np.float64)
    vy = df["vy"].to_numpy(dtype=np.float64)
    v = np.sqrt(vx * vx + vy * vy)

    heading = df["heading"].to_numpy(dtype=np.float64)

    result = pd.DataFrame(
        {
            "scene_ts": scene_ts,
            "rel_time_seconds": rel_time_seconds,
            "v": v,
            "heading": heading,
        }
    )

    return result


def load_clip_start_end(
    parquet_root: Optional[str], clip_id: str
) -> Tuple[Optional[int], Optional[int]]:
    """Load start/end epoch timestamps from clip parquet metadata.

    Args:
        parquet_root: Root directory containing `<clip_id>/clip.parquet`.
        clip_id: Clip ID.

    Returns:
        `(start_micros, end_micros)` in epoch microseconds, or `(None, None)` when missing.
    """
    if parquet_root is None:
        logger.warning("parquet root is not provided for clip %s", clip_id)
        return None, None

    parquet_path = os.path.join(parquet_root, clip_id, "clip.parquet")
    if not os.path.exists(parquet_path):
        return None, None

    df = pd.read_parquet(parquet_path)

    key = df["key"].iloc[0]

    start_micros = int(key["time_range"]["start_micros"])
    end_micros = int(key["time_range"]["end_micros"])

    return start_micros, end_micros


def load_video_start_end(video_path: str) -> Tuple[Optional[int], Optional[int]]:
    """Load start/end epoch timestamps from a video `.timestamps` sidecar.

    Args:
        video_path: Source video path.

    Returns:
        `(video_start, video_end)` in epoch microseconds, or `(None, None)` when missing.
    """
    timestamp_path = video_path + ".timestamps"
    if timestamp_path is None or not os.path.exists(timestamp_path):
        logger.warning("video not found at %s", timestamp_path)
        return None, None

    with open(timestamp_path) as file:
        timestamps = file.readlines()

    video_start = int(timestamps[0].split("\t")[1])
    video_end = int(timestamps[-1].split("\t")[1])

    return video_start, video_end


def process_clip(
    meta_action_file: str,
    video_root: str,
    output_root: str,
    meta_action_root: str,
    clip_to_vis_list: Optional[Set[str]],
    cache_root: Optional[str],
    parquet_root: Optional[str],
    golden_values_df: Optional[pd.DataFrame],
    lane_debug_dir: Optional[str],
    output_downsample: float,
) -> None:
    """Create visualization output for one clip.

    Args:
        meta_action_file: Path to clip-level meta-action output.
        video_root: Root directory of raw videos.
        output_root: Output directory for rendered videos.
        meta_action_root: Directory with meta-action files.
        clip_to_vis_list: Optional clip allowlist.
        cache_root: Optional root for cached agent trajectory files.
        parquet_root: Optional root for clip parquet files.
        golden_values_df: Optional GT dataframe for overlay tracks.
        output_downsample: Spatial downsample scale for output video frames.

    Returns:
        None. Skips clips with missing/no-overlap data.
    """
    clip_id = meta_action_file.split("/")[-1].split(".")[0]
    if clip_to_vis_list is not None and clip_id not in clip_to_vis_list:
        return

    # Resolve source video path for the clip (single front camera or pano stack).
    if PANO_VIDEO:
        video_paths_postfixes = [
            ".camera_cross_left_120fov.mp4",
            ".camera_front_wide_120fov.mp4",
            ".camera_cross_right_120fov.mp4",
        ]
        video_paths = [
            os.path.join(
                video_root,
                clip_id.split("_")[0],
                f"{clip_id.split('_')[0]}{postfix}",
            )
            for postfix in video_paths_postfixes
        ]
        video_path = pano_video_path = os.path.join(output_root, f"{clip_id}_pano.mp4")
        stack_videos(video_paths, pano_video_path, layout="horizontal")
    else:
        video_path = os.path.join(
            video_root,
            clip_id[:4],  # only take the first 4 letters as the data folder
            clip_id.split("_")[0],
            "recordings/**/camera_front_wide_120fov.mp4",
        )

        video_path = glob.glob(video_path, recursive=True)
        video_path = video_path[0]  # Path to the input video

    output_path = os.path.join(output_root, f"{clip_id}.mp4")

    video_start, video_end = load_video_start_end(video_path)

    clip_start, clip_end = load_clip_start_end(parquet_root, clip_id)

    # Load meta-action outputs.
    # - Pre-smoothing format: json
    # - Post-smoothing format: txt
    try:
        # the json format pre-smoothing
        meta_action_file = os.path.join(meta_action_root, f"{clip_id}.json")
        with open(meta_action_file) as file:
            meta_action_list = json.load(file)
    except FileNotFoundError:
        # the txt format post-smoothing
        meta_action_file = os.path.join(meta_action_root, f"{clip_id}.txt")
        with open(meta_action_file) as file:
            meta_action_list = [line.strip() for line in file]

    # Initialize per-frame caption tracks for predicted labels.
    captions_longitu: List[str] = ["Longitudinal: None" for _ in range(NUM_FRAMES_VIS)]
    captions_lateral: List[str] = ["Lateral: None" for _ in range(NUM_FRAMES_VIS)]
    captions_lane: List[str] = ["Lane Control: None" for _ in range(NUM_FRAMES_VIS)]

    # Initialize optional per-frame caption tracks for GT labels.
    gt_longitu: List[str] = ["GT Longitudinal: None" for _ in range(NUM_FRAMES_VIS)]
    gt_lateral: List[str] = ["GT Lateral: None" for _ in range(NUM_FRAMES_VIS)]
    gt_lane: List[str] = ["GT Lane Control: None" for _ in range(NUM_FRAMES_VIS)]
    lane_map_status: List[str] = load_lane_map_status_track(lane_debug_dir, clip_id)
    lane_conf_status: List[str] = load_lane_confidence_track(lane_debug_dir, clip_id)

    for meta_action_str in meta_action_list:
        # Decode pre-smoothing format.
        try:
            meta_action_cls, others = meta_action_str.split("(")
            agent_name, others = others.split(" at ")
            time_start, others = others.split("-")
            time_end = others.split(")")[0]

        # Decode post-smoothing format.
        # example:   Decelerate - Agent:<ego>, Start:0, End:52
        except ValueError:
            meta_action_cls, others = meta_action_str.split(" - ")
            agent_name, time_start, time_end = others.split(", ")
            agent_name = agent_name.split("<")[-1].split(">")[0]
            time_start = time_start.split(":")[-1]
            time_end = time_end.split(":")[-1]

        # it might include other objects besides ego
        if agent_name == "ego":
            time_start_in_sec = int(time_start) * DELTA_TIMESTAMP
            time_end_in_sec = int(time_end) * DELTA_TIMESTAMP
            frame_to_start = int(time_start_in_sec * VIS_FPS)
            frame_to_end = int(time_end_in_sec * VIS_FPS)

            # Map class to human-readable text (for pre-smoothing) or use class directly.
            try:
                text: str = META_ACTION2TEXT[meta_action_cls]
            # to accommodate smoothing
            except KeyError:
                text: str = meta_action_cls

            # convert meta action class to caption
            for frame_index in range(frame_to_start, frame_to_end):
                try:
                    if text in META_ACTION_LONGITUDINAL:
                        captions_longitu[frame_index] = f"Longitudinal: {text}"
                    elif text in META_ACTION_LATERAL:
                        captions_lateral[frame_index] = f"Lateral: {text}"
                    elif text in META_ACTION_LANE:
                        captions_lane[frame_index] = f"Lane Control: {text}"
                except IndexError:
                    logger.warning("frame %s is beyond what we need to visualize", frame_index)

    # Fill GT caption tracks, if provided.
    if golden_values_df is not None:
        try:
            gt_rows = golden_values_df[golden_values_df["clip_id"] == clip_id]
        except Exception:
            gt_rows = None
        if gt_rows is not None and len(gt_rows) > 0:
            for _, row in gt_rows.iterrows():
                try:
                    start_ms = int(row.get("segment_start_ms", 0))
                    end_ms = int(row.get("segment_end_ms", 0))
                    # Convert ms to frame indices at VIS_FPS
                    start_f = int(round((start_ms / 1000.0) * float(VIS_FPS)))
                    end_f = int(round((end_ms / 1000.0) * float(VIS_FPS)))
                    start_f = max(0, min(start_f, NUM_FRAMES_VIS))
                    end_f = max(0, min(end_f, NUM_FRAMES_VIS))
                    if end_f <= start_f:
                        continue
                    lon_lbl = str(row.get("gt_longitudinal", "None"))
                    lat_lbl = str(row.get("gt_lateral", "None"))
                    lane_lbl = str(row.get("gt_lane", "None"))
                    for fi in range(start_f, end_f):
                        if lon_lbl and lon_lbl != "None":
                            gt_longitu[fi] = f"GT Longitudinal: {lon_lbl}"
                        if lat_lbl and lat_lbl != "None":
                            gt_lateral[fi] = f"GT Lateral: {lat_lbl}"
                        if lane_lbl and lane_lbl != "None":
                            gt_lane[fi] = f"GT Lane Control: {lane_lbl}"
                except Exception:
                    continue

    # Prepare optional debug overlays (velocity and heading) from cached agent data.
    agent_df = get_agent_data(cache_root, clip_id)
    debug_velocity = []
    debug_heading = []
    if len(agent_df) > 0:
        # Sort by time in seconds from agent data
        df_sorted = agent_df.sort_values("rel_time_seconds").reset_index(drop=True)

        # Target timestamps for each video frame: 0..(NUM_FRAMES_VIS-1) / VIS_FPS
        target_times = np.arange(NUM_FRAMES_VIS, dtype=np.float64) / float(VIS_FPS)

        # Set index to rel_time_seconds for time-based interpolation
        df_indexed = df_sorted.set_index("rel_time_seconds")

        # Reindex to target timestamps and interpolate along the time index
        df_reindexed = df_indexed.reindex(target_times)
        df_interp = df_reindexed.interpolate(method="index", limit_direction="both").ffill().bfill()

        # Extract as float lists aligned with target_times
        debug_velocity = df_interp["v"].astype(np.float64).to_numpy().tolist()
        debug_heading = df_interp["heading"].astype(np.float64).to_numpy().tolist()

    # Align video frames, captions, and debug arrays by epoch timestamps.
    epoch_per_frame_us = int(round(1e6 / float(VIS_FPS)))

    # Fallback: use clip-relative indexing when timestamp boundaries are missing.
    if video_start is None or video_end is None or clip_start is None or clip_end is None:
        start_frame_idx = 0
        # limit by available captions/debug length
        total_frames = min(
            NUM_FRAMES_VIS,
            len(captions_longitu),
            len(debug_velocity) if len(debug_velocity) > 0 else NUM_FRAMES_VIS,
        )
        cap_long = captions_longitu[:total_frames]
        cap_lat = captions_lateral[:total_frames]
        cap_lane = captions_lane[:total_frames]
        cap_lane_map = lane_map_status[:total_frames]
        cap_lane_conf = lane_conf_status[:total_frames]
        cap_long_gt = gt_longitu[:total_frames]
        cap_lat_gt = gt_lateral[:total_frames]
        cap_lane_gt = gt_lane[:total_frames]
        dbg_vel = debug_velocity[:total_frames] if len(debug_velocity) > 0 else []
        dbg_head = debug_heading[:total_frames] if len(debug_heading) > 0 else []
    else:
        overlap_start = max(video_start, clip_start)
        overlap_end = min(video_end, clip_end)
        if overlap_end <= overlap_start:
            logger.warning("no overlap between video and clip for %s", clip_id)
            return

        # Map overlap to video frame indices
        start_frame_idx = max(0, int(round((overlap_start - video_start) / epoch_per_frame_us)))
        end_frame_idx_video = max(0, int(round((overlap_end - video_start) / epoch_per_frame_us)))
        frames_video = max(0, end_frame_idx_video - start_frame_idx + 1)

        # Map overlap to clip-relative frame indices (for captions/debug arrays)
        start_idx_clip = max(0, int(round((overlap_start - clip_start) / epoch_per_frame_us)))
        end_idx_clip = max(0, int(round((overlap_end - clip_start) / epoch_per_frame_us)))
        frames_clip = max(0, end_idx_clip - start_idx_clip + 1)

        # Determine total frames to render while respecting overlap and available arrays.
        total_frames = max(
            0,
            min(
                frames_video,
                frames_clip,
                NUM_FRAMES_VIS - start_idx_clip,
                len(captions_longitu) - start_idx_clip,
            ),
        )

        if total_frames == 0:
            logger.warning("zero-length overlap for %s", clip_id)
            return

        cap_long = captions_longitu[start_idx_clip : start_idx_clip + total_frames]
        cap_lat = captions_lateral[start_idx_clip : start_idx_clip + total_frames]
        cap_lane = captions_lane[start_idx_clip : start_idx_clip + total_frames]
        cap_lane_map = lane_map_status[start_idx_clip : start_idx_clip + total_frames]
        cap_lane_conf = lane_conf_status[start_idx_clip : start_idx_clip + total_frames]

        cap_long_gt = gt_longitu[start_idx_clip : start_idx_clip + total_frames]
        cap_lat_gt = gt_lateral[start_idx_clip : start_idx_clip + total_frames]
        cap_lane_gt = gt_lane[start_idx_clip : start_idx_clip + total_frames]

        if len(debug_velocity) > 0 and len(debug_heading) > 0:
            dbg_vel = debug_velocity[start_idx_clip : start_idx_clip + total_frames]
            dbg_head = debug_heading[start_idx_clip : start_idx_clip + total_frames]
        else:
            dbg_vel, dbg_head = [], []

    # Render final visualization video for this clip.
    captions_to_draw = [cap_long, cap_lat, cap_lane, cap_lane_map, cap_lane_conf]
    if golden_values_df is not None:
        captions_to_draw += [cap_long_gt, cap_lat_gt, cap_lane_gt]

    add_caption_to_video(
        video_path,
        output_path,
        captions_to_draw,
        debug_values=[dbg_vel, dbg_head],
        start_frame_idx=start_frame_idx,
        total_frames=total_frames,
        output_downsample=output_downsample,
    )

    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    argparser = argparse.ArgumentParser(
        description="Render meta-action overlays on clip videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "--video_root",
        type=str,
        required=True,
        help="Root directory containing source videos for clips.",
    )
    argparser.add_argument(
        "--meta_action_dir",
        type=str,
        required=True,
        help="Directory containing per-clip meta-action outputs (.json/.txt).",
    )
    argparser.add_argument(
        "--vis_dir",
        type=str,
        required=True,
        help="Output directory for rendered visualization videos.",
    )
    argparser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help=(
            "Path to trajdata cache directory. Each clip is expected under "
            "`<cache_dir>/<dataset_name>/<clip_id>/` with cached agent data files."
        ),
    )
    argparser.add_argument(
        "--parquet_dir",
        type=str,
        required=True,
        help=(
            "Path to the parquet data root used for timestamp alignment "
            "(contains per-clip parquet records)."
        ),
    )
    argparser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of worker threads used to process clips in parallel.",
    )
    argparser.add_argument(
        "--clip_to_vis_list",
        type=str,
        default=None,
        help="Optional comma-separated clip IDs to visualize.",
    )
    argparser.add_argument(
        "--clip_list_path",
        type=str,
        default=None,
        help="Optional path to file containing clip IDs to visualize (one per line).",
    )
    argparser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Optional path to the ground-truth CSV file.",
    )
    argparser.add_argument(
        "--lane_debug_dir",
        type=str,
        default=None,
        help="Optional directory containing <clip_id>_ego_lane_prepare_debug.csv files.",
    )
    argparser.add_argument(
        "--output_downsample",
        type=float,
        default=1.0,
        help=(
            "Spatial downsample scale for final visualized video in (0, 1]. "
            "Example: 0.5 halves width and height."
        ),
    )

    # parse the argument
    args = argparser.parse_args()

    if args.clip_list_path is not None and args.clip_to_vis_list is not None:
        raise ValueError("Supply only one of clip_list_path or clip_to_vis_list")
    if not (0.0 < float(args.output_downsample) <= 1.0):
        raise ValueError("--output_downsample must be in (0, 1].")

    video_root = args.video_root
    output_root = args.vis_dir
    meta_action_root = args.meta_action_dir
    cache_root = args.cache_dir
    parquet_root = args.parquet_dir

    clip_list_ids = None
    if args.clip_list_path is not None or args.clip_to_vis_list is not None:
        clip_list_ids = (
            _read_clip_list_file(args.clip_list_path)
            if args.clip_list_path
            else set(args.clip_to_vis_list.split(","))
        )

    golden_values_df = pd.read_csv(args.gt) if args.gt else None

    io_utils.mkdir_if_missing(output_root)

    # “freeze” save_dir and clip_id_to_file into the function
    worker = partial(
        process_clip,
        video_root=video_root,
        output_root=output_root,
        meta_action_root=meta_action_root,
        clip_to_vis_list=clip_list_ids,
        cache_root=cache_root,
        parquet_root=parquet_root,
        golden_values_df=golden_values_df,
        lane_debug_dir=args.lane_debug_dir,
        output_downsample=args.output_downsample,
    )

    # retrieve the clip id and filter to the requested list if provided
    meta_action_file_list, num_clips = io_utils.load_list_from_folder(meta_action_root)
    if clip_list_ids is not None and len(clip_list_ids) > 0:
        filtered = []
        for path in meta_action_file_list:
            cid = os.path.splitext(os.path.basename(path))[0]
            if cid in clip_list_ids:
                filtered.append(path)
        meta_action_file_list = filtered
        num_clips = len(meta_action_file_list)
    logger.info("number of clips with results is %s", num_clips)

    # Using ThreadPoolExecutor to create and manage a pool of threads
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Map the function to the items in the list and execute them concurrently
        list(
            tqdm.tqdm(
                executor.map(worker, meta_action_file_list),
                total=len(meta_action_file_list),
            )
        )
