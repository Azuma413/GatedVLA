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
import logging
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


def _read_clip_list_file(path: Optional[str]) -> Optional[Set[str]]:
    """Load clip IDs from a text file (one clip ID per line)."""
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        ids = [line.strip() for line in f.readlines()]
    ids = [cid for cid in ids if cid]
    return set(ids) if ids else None


def _extract_start_end_from_timestamp_df(
    timestamp_df: pd.DataFrame,
) -> Tuple[Optional[int], Optional[int]]:
    """Return first/last valid timestamp (microseconds) from a timestamp parquet table."""
    if timestamp_df is None or len(timestamp_df) == 0:
        return None, None

    ts_col = "timestamp"
    if ts_col not in timestamp_df.columns:
        return None, None

    values = pd.to_numeric(timestamp_df[ts_col], errors="coerce").dropna()
    if len(values) == 0:
        return None, None

    return int(values.iloc[0]), int(values.iloc[-1])


def _ensure_clip_camera_extracted(
    physical_ai_root: Path, prep_video_root: Path, clip_id: str, chunk_id: int
) -> bool:
    """Extract one clip's camera mp4 (+ optional timestamps) from the chunk zip.

    Returns:
        True when files were extracted, False when the prepared files already existed.
    """
    # Keep the output layout aligned with scripts/vis_video.py input expectation.
    target_dir = prep_video_root / clip_id[:4] / clip_id / "recordings" / "camera_front_wide_120fov"
    target_dir.mkdir(parents=True, exist_ok=True)

    out_mp4 = target_dir / "camera_front_wide_120fov.mp4"
    out_ts_txt = target_dir / "camera_front_wide_120fov.mp4.timestamps"

    if out_mp4.exists() and out_ts_txt.exists():
        return False

    zip_path = (
        physical_ai_root
        / "camera"
        / "camera_front_wide_120fov"
        / f"camera_front_wide_120fov.chunk_{chunk_id:04d}.zip"
    )
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing camera zip: {zip_path}")

    member_mp4 = f"{clip_id}.camera_front_wide_120fov.mp4"
    member_ts = f"{clip_id}.camera_front_wide_120fov.timestamps.parquet"

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = set(zf.namelist())
        if member_mp4 not in members:
            raise FileNotFoundError(f"{member_mp4} not found in {zip_path}")

        with zf.open(member_mp4, "r") as src, open(out_mp4, "wb") as dst:
            dst.write(src.read())

        if member_ts in members:
            with zf.open(member_ts, "r") as f:
                ts_df = pd.read_parquet(f)
            start_us, end_us = _extract_start_end_from_timestamp_df(ts_df)

            with open(out_ts_txt, "w", encoding="utf-8") as f:
                if start_us is not None and end_us is not None and len(ts_df) > 1:
                    ts_vals = (
                        pd.to_numeric(ts_df[ts_df.columns[0]], errors="coerce")
                        .dropna()
                        .astype(int)
                        .tolist()
                    )
                    for idx, ts in enumerate(ts_vals):
                        f.write(f"{idx}\t{ts}\n")

    return True


def main() -> None:
    """Prepare PAI inputs, then delegate rendering to scripts/vis_video.py."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare PAI camera videos for selected clips, then call scripts/vis_video.py "
            "to render meta-action overlays."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--physical_ai_root",
        type=str,
        required=True,
        help=(
            "Root directory of the Physical-AI dataset. Expected to contain "
            "`clip_index.parquet` and camera zip files under "
            "`camera/camera_front_wide_120fov/`."
        ),
    )
    parser.add_argument(
        "--meta_action_dir",
        type=str,
        required=True,
        help=(
            "Directory containing per-clip meta-action outputs to visualize (for example, "
            "`<clip_id>.json` or `<clip_id>.txt`). Clip IDs are inferred from filenames."
        ),
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        required=True,
        help="Output directory where rendered visualization videos will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help=(
            "Path to trajdata cache root (or the `pai` subdirectory). "
            "Expected cached file per clip: `agent_data_dt0.10.feather`."
        ),
    )
    parser.add_argument(
        "--work_root",
        type=str,
        required=True,
        help=(
            "Working directory for temporary extracted camera data and generated "
            "intermediate files."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers passed through to scripts/vis_video.py.",
    )
    parser.add_argument(
        "--clip_to_vis_list",
        type=str,
        default=None,
        help=(
            "Optional comma-separated clip IDs to visualize, for example "
            "`id1,id2,id3`. Ignored if --clip_list_path is provided."
        ),
    )
    parser.add_argument(
        "--clip_list_path",
        type=str,
        default=None,
        help=(
            "Optional text file with one clip ID per line. If provided, this takes "
            "priority over --clip_to_vis_list."
        ),
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Optional path to ground-truth labels to forward to scripts/vis_video.py.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    # Resolve all user paths early so later error messages are explicit.
    physical_ai_root = Path(args.physical_ai_root).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    prep_video_root = work_root / "video_root"
    prep_video_root.mkdir(parents=True, exist_ok=True)

    cache_root_arg = Path(args.cache_dir).expanduser().resolve()
    cache_root = cache_root_arg
    if cache_root_arg.name != "pai" and (cache_root_arg / "pai").exists():
        cache_root = cache_root_arg / "pai"

    logger.info("Preparing PAI visualization inputs under %s.", work_root)

    clip_index_path = physical_ai_root / "clip_index.parquet"
    logger.info("Loading clip index from %s.", clip_index_path)
    clip_index_df = pd.read_parquet(clip_index_path)

    if "chunk" not in clip_index_df.columns:
        raise KeyError(f"chunk column not found in {clip_index_path}")
    # clip_index parquet should be indexed by clip_id for .loc[clip_id] lookups.
    if "clip_id" in clip_index_df.columns and clip_index_df.index.name != "clip_id":
        clip_index_df = clip_index_df.set_index("clip_id")

    meta_action_dir = Path(args.meta_action_dir).expanduser().resolve()
    if not meta_action_dir.exists():
        raise FileNotFoundError(f"meta_action_dir not found: {meta_action_dir}")

    # Collect candidate clip IDs from model output filenames.
    clip_ids_from_results = set()
    for file_name in os.listdir(meta_action_dir):
        if file_name.endswith(".json") or file_name.endswith(".txt"):
            clip_ids_from_results.add(Path(file_name).stem)
    logger.info(
        "Found %d clips with meta-action outputs in %s.",
        len(clip_ids_from_results),
        meta_action_dir,
    )

    clip_list_ids = None
    if args.clip_list_path is not None or args.clip_to_vis_list is not None:
        clip_list_ids = (
            _read_clip_list_file(args.clip_list_path)
            if args.clip_list_path
            else set(args.clip_to_vis_list.split(","))
        )

    # Optional filtering by user-provided clip list.
    clip_ids = sorted(clip_ids_from_results)
    if clip_list_ids:
        clip_ids = [cid for cid in clip_ids if cid in clip_list_ids]
        logger.info("Filtered to %d clips from the requested clip list.", len(clip_ids))

    if len(clip_ids) == 0:
        raise RuntimeError("No clip IDs found for visualization.")

    # Keep only clips that have cached ego trajectory data required by vis_video.py.
    num_selected_clips = len(clip_ids)
    clip_ids = [
        cid for cid in clip_ids if (cache_root / cid / "agent_data_dt0.10.feather").exists()
    ]
    num_missing_cache = num_selected_clips - len(clip_ids)
    if num_missing_cache > 0:
        logger.warning(
            "Skipping %d clips without trajdata cache under %s.",
            num_missing_cache,
            cache_root,
        )
    if len(clip_ids) == 0:
        raise RuntimeError(
            f"No clips with cache found under {cache_root}. "
            "Run labeling/cache first or pass the correct --cache_dir."
        )

    logger.info("Preparing camera videos for %d clips under %s.", len(clip_ids), prep_video_root)
    prepared_clip_ids = []
    num_extracted = 0
    num_already_prepared = 0
    num_missing_from_index = 0
    progress_log_interval = max(1, len(clip_ids) // 10)
    with (
        logging_redirect_tqdm(),
        tqdm(clip_ids, desc="Preparing PAI videos", unit="clip", dynamic_ncols=True) as progress,
    ):
        for processed_count, clip_id in enumerate(progress, start=1):
            if clip_id not in clip_index_df.index:
                num_missing_from_index += 1
                logger.warning(
                    "Skipping %s because it is missing from %s.",
                    clip_id,
                    clip_index_path,
                )
            else:
                chunk_id = int(clip_index_df.loc[clip_id, "chunk"])
                extracted = _ensure_clip_camera_extracted(
                    physical_ai_root, prep_video_root, clip_id, chunk_id
                )
                if extracted:
                    num_extracted += 1
                else:
                    num_already_prepared += 1
                prepared_clip_ids.append(clip_id)
            progress.set_postfix(
                extracted=num_extracted,
                ready=num_already_prepared,
                missing=num_missing_from_index,
            )
            if processed_count == len(clip_ids) or processed_count % progress_log_interval == 0:
                logger.info(
                    "PAI video prep progress: %d/%d clips (extracted=%d, ready=%d, missing=%d).",
                    processed_count,
                    len(clip_ids),
                    num_extracted,
                    num_already_prepared,
                    num_missing_from_index,
                )

    logger.info("Finished preparing PAI camera videos.")
    logger.info("Extracted videos: %d.", num_extracted)
    logger.info("Already prepared videos: %d.", num_already_prepared)
    if num_missing_from_index > 0:
        logger.warning("Skipped clips missing from clip index: %d.", num_missing_from_index)
    if len(prepared_clip_ids) == 0:
        raise RuntimeError("No camera videos could be prepared for visualization.")
    clip_ids = prepared_clip_ids

    # Write a temporary clip list so downstream vis script can reuse its existing CLI.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir=str(work_root)
    ) as tmp_file:
        clip_list_file = tmp_file.name
        for clip_id in clip_ids:
            tmp_file.write(f"{clip_id}\n")

    # Delegate actual rendering logic to the original visualization script.
    cmd = [
        "python",
        "scripts/vis_video.py",
        "--video_root",
        str(prep_video_root),
        "--meta_action_dir",
        str(meta_action_dir),
        "--vis_dir",
        str(Path(args.vis_dir).expanduser().resolve()),
        "--cache_dir",
        str(cache_root),
        "--parquet_dir",
        str(physical_ai_root),
        "--num_workers",
        str(args.num_workers),
        "--clip_list_path",
        str(clip_list_file),
    ]

    if args.gt is not None:
        cmd.extend(["--gt", args.gt])

    logger.info("Starting visualization rendering for %d clips.", len(clip_ids))
    subprocess.run(cmd, check=True)
    logger.info("Finished visualization rendering.")


if __name__ == "__main__":
    main()
