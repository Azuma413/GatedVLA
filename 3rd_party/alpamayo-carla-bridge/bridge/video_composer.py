"""Composite video renderer from recorded driving sessions.

Layout::

    ┌──────────────────────────────────────────────┐
    │               front_tele (Zoom)              │
    ├─────────────┬──────────────┬─────────────────┤
    │ front_left  │  front_wide  │  front_right    │
    ├─────────────┴──────────────┴─────────────────┤
    │ BEV Trajectory Map        │ Reasoning Trace  │
    │ blue=predicted            │ (typewriter)     │
    │ red=actual, green=ego     │                  │
    ├───────────────────────────┴──────────────────┤
    │ Status Bar: Speed │ Step X/60 │ Timer        │
    └──────────────────────────────────────────────┘
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — no display needed.
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Colours (BGR for OpenCV, RGB for Pillow).
_BLUE_RGB = (60, 120, 255)
_RED_RGB = (255, 60, 60)
_GREEN_RGB = (60, 220, 60)
_WHITE_RGB = (255, 255, 255)
_DARK_BG_RGB = (30, 30, 35)
_STATUS_BG_RGB = (20, 20, 25)


class VideoComposer:
    """Reads a recording session directory and produces a composite MP4 video.

    Args:
        session_dir: Path to ``recordings/{session_id}/``.
        config: Full config dict (for video dimensions, FPS).
    """

    def __init__(
        self,
        session_dir: str | Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._session_dir = Path(session_dir)
        self._frames_dir = self._session_dir / "frames"
        self._data_path = self._session_dir / "data.jsonl"
        self._metadata_path = self._session_dir / "metadata.json"
        self._output_path = self._session_dir / "output.mp4"

        rec_cfg = (config or {}).get("recording", {})
        self._width: int = rec_cfg.get("video_width", 1920)
        self._height: int = rec_cfg.get("video_height", 1080)
        self._fps: int = rec_cfg.get("fps", 10)

        run_cfg = (config or {}).get("run", {})
        self._max_steps: int = run_cfg.get("max_steps", 60)

        # Try to load a nice font; fall back to default.
        self._font: Optional[ImageFont.FreeTypeFont] = None
        self._font_small: Optional[ImageFont.FreeTypeFont] = None
        _font_candidates = [
            # Windows
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            # macOS
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        for font_path in _font_candidates:
            try:
                self._font = ImageFont.truetype(font_path, 20)
                self._font_small = ImageFont.truetype(font_path, 16)
                logger.debug("Using font: %s", font_path)
                break
            except (OSError, IOError):
                continue
        if self._font is None:
            logger.debug("No TrueType font found — using default bitmap font.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compose(self) -> Path:
        """Read the recording and produce the composite video.

        Returns:
            Path to the output MP4 file.
        """
        records = self._load_records()
        if not records:
            logger.error("No records found in %s", self._data_path)
            raise FileNotFoundError(f"No data.jsonl in {self._session_dir}")

        metadata = self._load_metadata()
        logger.info(
            "Composing video: %d steps, %dx%d @ %d FPS",
            len(records),
            self._width,
            self._height,
            self._fps,
        )

        # Collect all actual positions for the full BEV plot.
        all_actual_positions: List[List[float]] = [
            r.get("actual_position", [0, 0]) for r in records
        ]

        # Write frames to a temp directory and encode with ffmpeg.
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, record in enumerate(records):
                frame = self._render_frame(record, idx, len(records), all_actual_positions)
                out_path = Path(tmpdir) / f"frame_{idx:05d}.png"
                frame.save(str(out_path))

            self._encode_video(tmpdir)

        logger.info("Video written: %s", self._output_path)
        return self._output_path

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def _render_frame(
        self,
        record: Dict[str, Any],
        frame_idx: int,
        total_frames: int,
        all_actual_positions: List[List[float]],
    ) -> Image.Image:
        """Render a single composite frame.

        Args:
            record: Parsed JSONL record for this step.
            frame_idx: Zero-based frame index.
            total_frames: Total number of frames.
            all_actual_positions: All actual positions for BEV history.

        Returns:
            A PIL Image of size ``(width, height)``.
        """
        canvas = Image.new("RGB", (self._width, self._height), _DARK_BG_RGB)
        step = record.get("step", frame_idx + 1)

        # --- Layout dimensions ---
        # Row 1: front_tele (full width, 30% height)
        # Row 2: 3 cams side-by-side (full width, 25% height)
        # Row 3: BEV + Reasoning (full width, 35% height)
        # Row 4: Status bar (full width, 10% height)
        row1_h = int(self._height * 0.30)
        row2_h = int(self._height * 0.25)
        row3_h = int(self._height * 0.35)
        row4_h = self._height - row1_h - row2_h - row3_h

        # Row 1: front_tele
        tele_img = self._load_frame_image(step, "front_tele")
        if tele_img is not None:
            tele_img = tele_img.resize((self._width, row1_h), Image.LANCZOS)
            canvas.paste(tele_img, (0, 0))
        self._draw_label(canvas, "TELE (30° FOV)", 10, 5)

        # Row 2: three cameras side by side
        cam_w = self._width // 3
        y2 = row1_h
        for i, cam_name in enumerate(["front_left", "front_wide", "front_right"]):
            cam_img = self._load_frame_image(step, cam_name)
            if cam_img is not None:
                cam_img = cam_img.resize((cam_w, row2_h), Image.LANCZOS)
                canvas.paste(cam_img, (i * cam_w, y2))
            label = cam_name.replace("front_", "").upper()
            self._draw_label(canvas, label, i * cam_w + 5, y2 + 3)

        # Row 3: BEV (left 55%) + Reasoning (right 45%)
        y3 = row1_h + row2_h
        bev_w = int(self._width * 0.55)
        reason_w = self._width - bev_w

        bev_img = self._render_bev(
            record.get("predicted_trajectory", []),
            all_actual_positions[: frame_idx + 1],
            record.get("ego_state", {}),
            bev_w,
            row3_h,
        )
        canvas.paste(bev_img, (0, y3))

        reason_img = self._render_reasoning(
            record.get("reasoning_trace", ""),
            reason_w,
            row3_h,
        )
        canvas.paste(reason_img, (bev_w, y3))

        # Row 4: Status bar
        y4 = row1_h + row2_h + row3_h
        status_img = self._render_status_bar(
            record.get("ego_state", {}),
            step,
            total_frames,
            record.get("timestamp", 0),
            self._width,
            row4_h,
        )
        canvas.paste(status_img, (0, y4))

        return canvas

    # ------------------------------------------------------------------
    # Sub-renderers
    # ------------------------------------------------------------------

    def _load_frame_image(self, step: int, cam_name: str) -> Optional[Image.Image]:
        """Load a saved camera frame JPEG.

        Args:
            step: Step number.
            cam_name: Camera name.

        Returns:
            PIL Image or None if not found.
        """
        path = self._frames_dir / f"step_{step:04d}_{cam_name}.jpg"
        if path.exists():
            return Image.open(path).convert("RGB")
        return None

    def _render_bev(
        self,
        predicted_traj: List[List[float]],
        actual_positions: List[List[float]],
        ego_state: Dict[str, float],
        width: int,
        height: int,
    ) -> Image.Image:
        """Render a bird's-eye-view trajectory map using matplotlib.

        Args:
            predicted_traj: Predicted waypoints ``[[x,y], ...]`` in ego frame.
            actual_positions: Historical actual positions in world frame.
            ego_state: Current ego state dict.
            width: Output pixel width.
            height: Output pixel height.

        Returns:
            PIL Image of the BEV plot.
        """
        dpi = 100
        fig_w = width / dpi
        fig_h = height / dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig.patch.set_facecolor("#1e1e23")
        ax.set_facecolor("#1e1e23")

        # Actual path (world frame — plot as-is).
        if actual_positions:
            xs = [p[0] for p in actual_positions]
            ys = [p[1] for p in actual_positions]
            ax.plot(xs, ys, color="#ff3c3c", linewidth=1.5, alpha=0.8, label="Actual")

        # Current ego position.
        ego_x = ego_state.get("x", 0)
        ego_y = ego_state.get("y", 0)
        ax.scatter([ego_x], [ego_y], color="#3cdc3c", s=80, zorder=5, label="Ego")

        # Predicted trajectory (ego-frame → world-frame for plotting).
        if predicted_traj:
            import math

            yaw_rad = math.radians(ego_state.get("yaw", 0))
            cos_y = math.cos(yaw_rad)
            sin_y = math.sin(yaw_rad)
            pred_xs = []
            pred_ys = []
            for wp in predicted_traj:
                wx = ego_x + wp[0] * cos_y - wp[1] * sin_y
                wy = ego_y + wp[0] * sin_y + wp[1] * cos_y
                pred_xs.append(wx)
                pred_ys.append(wy)
            ax.plot(pred_xs, pred_ys, color="#3c78ff", linewidth=2, alpha=0.9, label="Predicted")

        ax.legend(fontsize=8, loc="upper left", facecolor="#2a2a30", edgecolor="gray", labelcolor="white")
        ax.tick_params(colors="gray", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("gray")
        ax.set_title("Bird's Eye View", color="white", fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.2, color="gray")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB").resize((width, height), Image.LANCZOS)
        return img

    def _render_reasoning(self, text: str, width: int, height: int) -> Image.Image:
        """Render the reasoning trace panel with semi-transparent background.

        Args:
            text: Reasoning text string.
            width: Pixel width.
            height: Pixel height.

        Returns:
            PIL Image.
        """
        img = Image.new("RGB", (width, height), (25, 25, 30))
        draw = ImageDraw.Draw(img)

        # Title
        font = self._font or ImageFont.load_default()
        font_small = self._font_small or ImageFont.load_default()

        draw.text((10, 8), "Reasoning Trace", fill=_WHITE_RGB, font=font)

        # Typewriter-style text (wrap manually).
        margin = 10
        y_offset = 35
        max_chars_per_line = max(1, (width - 2 * margin) // 9)  # rough estimate

        # Simple word wrap.
        words = text.split()
        lines: List[str] = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars_per_line:
                current_line = f"{current_line} {word}" if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        for line in lines:
            if y_offset + 20 > height - 10:
                draw.text((margin, y_offset), "…", fill=(180, 180, 180), font=font_small)
                break
            draw.text((margin, y_offset), line, fill=(200, 200, 210), font=font_small)
            y_offset += 20

        return img

    def _render_status_bar(
        self,
        ego_state: Dict[str, float],
        step: int,
        total_steps: int,
        timestamp: float,
        width: int,
        height: int,
    ) -> Image.Image:
        """Render the bottom status bar.

        Args:
            ego_state: Current ego state.
            step: Current step number.
            total_steps: Total steps in the recording.
            timestamp: Unix timestamp.
            width: Pixel width.
            height: Pixel height.

        Returns:
            PIL Image.
        """
        img = Image.new("RGB", (width, height), _STATUS_BG_RGB)
        draw = ImageDraw.Draw(img)
        font = self._font or ImageFont.load_default()

        speed = ego_state.get("speed_kmh", 0.0)
        elapsed = step * 0.1  # Approximate at 10 Hz

        parts = [
            f"Speed: {speed:.1f} km/h",
            f"Step: {step}/{self._max_steps}",
            f"Time: {elapsed:.1f}s",
        ]
        text = "   │   ".join(parts)

        # Center the text.
        y_center = max(0, (height - 22) // 2)
        draw.text((20, y_center), text, fill=_WHITE_RGB, font=font)

        return img

    # ------------------------------------------------------------------
    # Label helper
    # ------------------------------------------------------------------

    def _draw_label(
        self,
        canvas: Image.Image,
        text: str,
        x: int,
        y: int,
    ) -> None:
        """Draw a small translucent label on the canvas."""
        draw = ImageDraw.Draw(canvas)
        font = self._font_small or ImageFont.load_default()
        draw.text((x, y), text, fill=(220, 220, 220), font=font)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_records(self) -> List[Dict[str, Any]]:
        """Load all JSONL records from the session."""
        if not self._data_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with open(self._data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _load_metadata(self) -> Dict[str, Any]:
        """Load the session metadata."""
        if not self._metadata_path.exists():
            return {}
        with open(self._metadata_path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Video encoding
    # ------------------------------------------------------------------

    def _encode_video(self, frames_dir: str) -> None:
        """Encode PNG frames into an H.264 MP4 via ffmpeg.

        Args:
            frames_dir: Directory containing ``frame_XXXXX.png`` files.
        """
        # Use Path / operator for cross-platform path separators.
        input_pattern = str(Path(frames_dir) / "frame_%05d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(self._fps),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "20",
            str(self._output_path),
        ]
        logger.info("Running ffmpeg: %s", " ".join(cmd))
        # On Windows, suppress the ffmpeg console window.
        kwargs: Dict[str, Any] = {"capture_output": True, "text": True}
        import platform
        if platform.system() == "Windows":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        result = subprocess.run(cmd, **kwargs)  # noqa: S603
        if result.returncode != 0:
            logger.error("ffmpeg stderr: %s", result.stderr)
            raise RuntimeError(f"ffmpeg failed (rc={result.returncode}): {result.stderr[:500]}")
