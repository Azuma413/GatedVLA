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

"""Extract only PAI camera videos referenced by a keyframe index JSON.

The CoC video loader expects flattened PAI videos under:

    <video_dir>/camera/<clip_id>.camera_front_wide_120fov.mp4

This helper reads a keyframe/index JSON, collects the referenced clip IDs, and
extracts only those videos from PAI camera zip archives. The archives may be
chunk zips such as ``camera_front_wide_120fov.chunk_0000.zip`` or per-clip zips.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from tqdm import tqdm

UUID_PREFIX_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


@dataclass(frozen=True)
class ArchiveMemberRef:
    """Reference to one file inside a zip archive and its flattened output name."""

    clip_id: str
    zip_path: Path
    member_name: str
    target_name: str


def parse_meta_action_filter(values: list[str] | None) -> set[str] | None:
    """Parse CLI meta-action filters into a normalized set, or None for all."""
    if values is None:
        return None

    selected = set()
    for value in values:
        raw = str(value).strip()
        if raw.lower() in {"", "all", "none", "null"}:
            return None

        # Accept common forms:
        #   --meta-action-filter go_straight turn_left
        #   --meta-action-filter go_straight,turn_left
        #   --meta-action-filter "[go_straight, turn_left]"
        raw = raw.strip("[]")
        for part in re.split(r"[,\s]+", raw):
            action = part.strip().strip("'\"")
            if action:
                selected.add(action.lower())

    return selected or None


def normalize_clip_id(value: Any, camera_name: str) -> str | None:
    """Normalize a clip ID or filename-like value to the bare clip ID."""
    if value is None:
        return None

    name = Path(str(value)).name.strip()
    if not name:
        return None

    for suffix in (
        ".json",
        ".yaml",
        ".yml",
        ".txt",
        f".{camera_name}.mp4.timestamps.parquet",
        f".{camera_name}.mp4.timestamps",
        f".{camera_name}.timestamps.parquet",
        f".{camera_name}.timestamps",
        f".{camera_name}.mp4",
        f"_{camera_name}.mp4.timestamps.parquet",
        f"_{camera_name}.mp4.timestamps",
        f"_{camera_name}.timestamps.parquet",
        f"_{camera_name}.timestamps",
        f"_{camera_name}.mp4",
        f".{camera_name}",
        f"_{camera_name}",
        "_fpv.mp4",
        ".mp4",
    ):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    match = UUID_PREFIX_RE.match(name)
    if match:
        return match.group(0)
    return name


def iter_segment_entries(data: Any, bucket: str | None = None) -> Iterator[tuple[str | None, Any]]:
    """Yield (bucket, entry) pairs from common keyframe index JSON shapes."""
    if isinstance(data, list):
        for item in data:
            yield bucket, item
        return

    if not isinstance(data, dict):
        return

    if "clip_id" in data or "filename" in data:
        yield bucket, data
        return

    for key, value in data.items():
        next_bucket = str(key)
        if isinstance(value, list):
            for item in value:
                yield next_bucket, item
        elif isinstance(value, dict):
            yield from iter_segment_entries(value, next_bucket)


def entry_action_matches(bucket: str | None, entry: Any, selected_actions: set[str] | None) -> bool:
    """Return whether an index entry matches the optional meta-action filter."""
    if selected_actions is None:
        return True

    candidates = []
    if bucket is not None:
        candidates.append(bucket)
    if isinstance(entry, dict):
        candidates.extend(
            str(entry[key]) for key in ("meta_action", "action") if entry.get(key) is not None
        )

    return any(candidate.lower() in selected_actions for candidate in candidates)


def entry_clip_id(entry: Any, camera_name: str) -> str | None:
    """Extract a clip ID from one index entry."""
    if isinstance(entry, str):
        return normalize_clip_id(entry, camera_name)
    if not isinstance(entry, dict):
        return None

    for key in ("clip_id", "filename", "file_name", "video_id"):
        clip_id = normalize_clip_id(entry.get(key), camera_name)
        if clip_id:
            return clip_id
    return None


def load_clip_ids(
    index_file: Path, camera_name: str, selected_actions: set[str] | None
) -> list[str]:
    """Load and deduplicate clip IDs referenced by an index JSON."""
    with index_file.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    clip_ids = {
        clip_id
        for bucket, entry in iter_segment_entries(data)
        if entry_action_matches(bucket, entry, selected_actions)
        for clip_id in [entry_clip_id(entry, camera_name)]
        if clip_id
    }
    return sorted(clip_ids)


def target_name_for_member(member_name: str, clip_id: str, camera_name: str) -> str | None:
    """Map an archive member to the flattened output filename, if relevant."""
    basename = PurePosixPath(member_name).name
    if not basename:
        return None

    expected_video = f"{clip_id}.{camera_name}.mp4"
    sidecar_suffixes = [
        ".timestamps.parquet",
        ".timestamps",
        ".mp4.timestamps.parquet",
        ".mp4.timestamps",
    ]
    expected_sidecars = {f"{clip_id}.{camera_name}{suffix}" for suffix in sidecar_suffixes}

    if basename == expected_video or basename in expected_sidecars:
        return basename

    if basename.endswith(".mp4") and (
        basename.startswith(clip_id)
        or basename == f"{camera_name}.mp4"
        or basename.startswith(f"{camera_name}.")
    ):
        return expected_video

    for suffix in sidecar_suffixes:
        if basename.endswith(suffix) and (
            basename.startswith(clip_id) or basename.startswith(camera_name)
        ):
            return f"{clip_id}.{camera_name}{suffix}"

    return None


def infer_clip_id_from_member(
    member_name: str, requested_clip_ids: set[str], camera_name: str
) -> str | None:
    """Infer which requested clip ID an archive member belongs to."""
    path = PurePosixPath(member_name)
    candidate_parts = [path.name, *reversed(path.parent.parts)]

    for part in candidate_parts:
        clip_id = normalize_clip_id(part, camera_name)
        if clip_id in requested_clip_ids:
            return clip_id

    return None


def build_archive_member_map(
    zip_dir: Path, requested_clip_ids: set[str], camera_name: str
) -> dict[str, list[ArchiveMemberRef]]:
    """Map requested clip IDs to matching members in PAI chunk/per-clip zips."""
    member_map: dict[str, list[ArchiveMemberRef]] = {clip_id: [] for clip_id in requested_clip_ids}
    zip_paths = sorted(
        path for path in zip_dir.iterdir() if path.is_file() and path.suffix.lower() == ".zip"
    )

    for zip_index, zip_path in enumerate(zip_paths, start=1):
        if zip_index % 100 == 0:
            logging.info("Indexed archive members from %d/%d zips.", zip_index, len(zip_paths))

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                members = zip_file.infolist()
        except zipfile.BadZipFile:
            logging.warning("Skip bad zip file: %s", zip_path)
            continue

        for member in members:
            if member.is_dir():
                continue

            clip_id = infer_clip_id_from_member(member.filename, requested_clip_ids, camera_name)
            if clip_id is None:
                continue

            target_name = target_name_for_member(member.filename, clip_id, camera_name)
            if target_name is None:
                continue

            member_map[clip_id].append(
                ArchiveMemberRef(
                    clip_id=clip_id,
                    zip_path=zip_path,
                    member_name=member.filename,
                    target_name=target_name,
                )
            )

    return member_map


def copy_zip_member(zip_file: zipfile.ZipFile, member: zipfile.ZipInfo, target_path: Path) -> None:
    """Copy one zip member to a target path via a temporary file."""
    temp_path = target_path.with_name(f".{target_path.name}.tmp")
    with zip_file.open(member, "r") as source, temp_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    os.replace(temp_path, target_path)


def group_members_by_zip(
    member_refs: Iterator[ArchiveMemberRef],
) -> dict[Path, list[ArchiveMemberRef]]:
    """Group member references by containing zip path."""
    grouped: dict[Path, list[ArchiveMemberRef]] = {}
    for member_ref in member_refs:
        grouped.setdefault(member_ref.zip_path, []).append(member_ref)
    return grouped


def extract_member_refs(
    refs_by_zip: dict[Path, list[ArchiveMemberRef]],
    camera_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> tuple[int, int]:
    """Extract selected archive members.

    Returns:
        A tuple of (extracted_count, skipped_existing_count).
    """
    extracted_count = 0
    skipped_existing_count = 0
    total_members = sum(len(member_refs) for member_refs in refs_by_zip.values())
    progress_desc = "Checking archive members" if dry_run else "Extracting archive members"

    with tqdm(
        total=total_members,
        desc=progress_desc,
        unit="file",
        dynamic_ncols=True,
    ) as progress:
        for zip_path, member_refs in refs_by_zip.items():
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_file:
                    for member_ref in member_refs:
                        try:
                            target_path = camera_dir / member_ref.target_name
                            if target_path.exists() and not overwrite:
                                skipped_existing_count += 1
                                continue

                            logging.debug(
                                "Extract %s:%s -> %s",
                                zip_path,
                                member_ref.member_name,
                                target_path,
                            )
                            if not dry_run:
                                copy_zip_member(
                                    zip_file,
                                    zip_file.getinfo(member_ref.member_name),
                                    target_path,
                                )
                            extracted_count += 1
                        finally:
                            progress.update(1)
            except zipfile.BadZipFile:
                logging.warning("Skip bad zip file during extraction: %s", zip_path)
                progress.update(len(member_refs))
                continue

    return extracted_count, skipped_existing_count


def count_existing_videos(
    clip_ids: list[str], camera_dir: Path, camera_name: str, overwrite: bool
) -> tuple[list[str], int]:
    """Return clip IDs that still need extraction plus already-present count."""
    pending_clip_ids = []
    skipped_existing_videos = 0

    for clip_id in clip_ids:
        target_path = camera_dir / f"{clip_id}.{camera_name}.mp4"
        if target_path.exists() and not overwrite:
            skipped_existing_videos += 1
        else:
            pending_clip_ids.append(clip_id)

    return pending_clip_ids, skipped_existing_videos


def select_members_to_extract(
    member_map: dict[str, list[ArchiveMemberRef]],
    clip_ids: list[str],
    camera_name: str,
) -> tuple[list[ArchiveMemberRef], list[str], list[str]]:
    """Select members and identify missing clips/video members."""
    refs_to_extract = []
    missing_zips = []
    missing_video_members = []

    for clip_id in clip_ids:
        member_refs = member_map.get(clip_id, [])
        if not member_refs:
            missing_zips.append(clip_id)
            continue

        expected_video = f"{clip_id}.{camera_name}.mp4"
        if not any(member_ref.target_name == expected_video for member_ref in member_refs):
            zip_names = sorted({member_ref.zip_path.name for member_ref in member_refs})
            missing_video_members.append(f"{clip_id} ({', '.join(zip_names[:3])})")
            continue

        refs_to_extract.extend(member_refs)

    return refs_to_extract, missing_zips, missing_video_members


def remove_duplicate_targets(member_refs: list[ArchiveMemberRef]) -> list[ArchiveMemberRef]:
    """Keep one archive member for each output target name."""
    deduplicated = []
    seen_targets = set()
    for member_ref in sorted(
        member_refs,
        key=lambda ref: (ref.target_name, ref.zip_path.name, ref.member_name),
    ):
        if member_ref.target_name in seen_targets:
            continue
        seen_targets.add(member_ref.target_name)
        deduplicated.append(member_ref)
    return deduplicated


def log_missing(label: str, values: list[str]) -> None:
    """Log a compact warning for missing clips."""
    if not values:
        return
    logging.warning("%s for %d clips.", label, len(values))
    logging.warning("First affected clips: %s", ", ".join(values[:10]))


def ensure_output_dir(camera_dir: Path, dry_run: bool) -> None:
    """Create the output camera directory unless this is a dry run."""
    if not dry_run:
        camera_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract only PAI camera videos referenced by a keyframe/index JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index-file",
        type=Path,
        required=True,
        help="Keyframe/index JSON, for example segments_relative_timestamp_sampled.json.",
    )
    parser.add_argument(
        "--video-zip-dir",
        "--zip-dir",
        type=Path,
        required=True,
        help="Directory containing PAI camera zip files, either chunk zips or per-clip zips.",
    )
    parser.add_argument(
        "--output-video-root",
        "--output-root",
        type=Path,
        required=True,
        help="Destination video_dir root. Extracted files are written under <root>/camera.",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="camera_front_wide_120fov",
        help="PAI camera stream name to extract.",
    )
    parser.add_argument(
        "--meta-action-filter",
        nargs="+",
        default=None,
        help=(
            "Optional meta-action names to keep. Omit this flag to extract all clips "
            "referenced by the index file."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted videos and sidecar timestamp files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be extracted without writing files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if any requested clip zip or video member is missing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    """Run selective extraction."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    if not args.index_file.exists():
        raise FileNotFoundError(f"Index file not found: {args.index_file}")
    if not args.video_zip_dir.is_dir():
        raise NotADirectoryError(f"Video zip directory not found: {args.video_zip_dir}")

    selected_actions = parse_meta_action_filter(args.meta_action_filter)
    clip_ids = load_clip_ids(args.index_file, args.camera_name, selected_actions)
    if not clip_ids:
        logging.warning("No clip IDs found in %s.", args.index_file)
        return 0

    logging.info("Loaded %d unique clip IDs from %s.", len(clip_ids), args.index_file)
    if selected_actions is not None:
        logging.info("Applied meta-action filter: %s.", ", ".join(sorted(selected_actions)))

    camera_dir = args.output_video_root / "camera"
    ensure_output_dir(camera_dir, args.dry_run)
    pending_clip_ids, skipped_existing_videos = count_existing_videos(
        clip_ids, camera_dir, args.camera_name, args.overwrite
    )

    logging.info("Skipped existing videos: %d.", skipped_existing_videos)
    logging.info("Need to extract videos for %d clips.", len(pending_clip_ids))
    if not pending_clip_ids:
        logging.info("All requested videos already exist. Nothing to extract.")
        return 0

    logging.info("Indexing zip archive members under %s.", args.video_zip_dir)
    member_map = build_archive_member_map(
        args.video_zip_dir, set(pending_clip_ids), args.camera_name
    )
    refs_to_extract, missing_zips, missing_video_members = select_members_to_extract(
        member_map, pending_clip_ids, args.camera_name
    )
    refs_to_extract = remove_duplicate_targets(refs_to_extract)
    refs_by_zip = group_members_by_zip(iter(refs_to_extract))

    logging.info(
        "Matched %d archive members across %d zip files.",
        len(refs_to_extract),
        len(refs_by_zip),
    )

    extracted_files, skipped_existing_files = extract_member_refs(
        refs_by_zip,
        camera_dir=camera_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    logging.info("Finished selective extraction.")
    logging.info("Extracted files: %d.", extracted_files)
    logging.info("Skipped existing videos: %d.", skipped_existing_videos)
    logging.info("Skipped existing files inside processed zips: %d.", skipped_existing_files)

    log_missing("Missing zip members", missing_zips)
    log_missing("Missing matching video members", missing_video_members)

    if args.strict and (missing_zips or missing_video_members):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
