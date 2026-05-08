#!/usr/bin/env python3
"""Standalone video composer — re-render a composite video from an existing recording.

Usage::

    python scripts/compose_video.py recordings/20260311_194000
    python scripts/compose_video.py recordings/20260311_194000 --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from bridge.utils import load_config, setup_logging  # noqa: E402
from bridge.video_composer import VideoComposer  # noqa: E402


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Compose video from a recorded session")
    parser.add_argument(
        "session_dir",
        type=str,
        help="Path to the recording session directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "config.yaml"),
        help="Path to config.yaml (for video dimensions / FPS)",
    )
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        logger.error("Session directory not found: %s", session_dir)
        sys.exit(1)

    composer = VideoComposer(session_dir, config)
    video_path = composer.compose()
    logger.info("Video saved: %s", video_path)


if __name__ == "__main__":
    main()
