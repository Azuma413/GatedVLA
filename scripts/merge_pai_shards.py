#!/usr/bin/env python3
"""Merge per-shard PAI exports into one coherent PAI dataset.

``export_py123d_to_pai.py --num-shards N --shard-index i`` writes each shard to
its own directory with chunk numbers restarting at 0. This script relocates each
shard's chunk zips into a single output directory under a global chunk offset and
concatenates the metadata (clip_index, feature_presence) so the result is
byte-for-byte the same layout a single-process export would have produced.

Zip files are moved (same-filesystem rename), so no data is duplicated.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

PAI_CAMERA_ORDER = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]
FEATURES = ["egomotion", "navigation_command", *PAI_CAMERA_ORDER]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-root", required=True, type=Path, help="Directory containing shard_XXXX subdirs.")
    p.add_argument("--output-dir", required=True, type=Path, help="Destination merged PAI dataset root.")
    p.add_argument("--num-shards", required=True, type=int)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def feature_zip_paths(root: Path, feature: str, chunk: int) -> Path:
    if feature == "egomotion":
        return root / "labels" / "egomotion" / f"egomotion.chunk_{chunk:04d}.zip"
    if feature == "navigation_command":
        return root / "labels" / "navigation_command" / f"navigation_command.chunk_{chunk:04d}.zip"
    return root / "camera" / feature / f"{feature}.chunk_{chunk:04d}.zip"


def filename_template(feature: str) -> str:
    if feature == "egomotion":
        return "labels/egomotion/egomotion.chunk_{chunk:04d}.zip"
    if feature == "navigation_command":
        return "labels/navigation_command/navigation_command.chunk_{chunk:04d}.zip"
    return f"camera/{feature}/{feature}.chunk_{{chunk:04d}}.zip"


def clip_members(feature: str, clip_id: str) -> dict[str, str]:
    if feature == "egomotion":
        return {"egomotion": f"{clip_id}.egomotion.parquet"}
    if feature == "navigation_command":
        return {"navigation_command": f"{clip_id}.navigation_command.parquet"}
    return {"video": f"{clip_id}.{feature}.mp4", "frame_timestamps": f"{clip_id}.{feature}.timestamps.parquet"}


def write_features_csv(output_dir: Path, clip_index: pd.DataFrame, presence: pd.DataFrame) -> None:
    rows = []
    for feature in FEATURES:
        clip_files: dict[str, dict[str, str]] = {}
        for clip_id in clip_index.index:
            if bool(presence.loc[clip_id, feature]):
                clip_files[str(clip_id)] = clip_members(feature, str(clip_id))
        rows.append(
            {
                "feature": feature,
                "filename_template": filename_template(feature),
                "chunk_filename_template": filename_template(feature),
                "clip_files_in_zip": json.dumps(clip_files),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "features.csv", index=False)


def main() -> None:
    args = parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{out} is not empty. Use --overwrite.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    clip_indexes: list[pd.DataFrame] = []
    presences: list[pd.DataFrame] = []
    chunk_offset = 0
    total_clips = 0

    for shard in range(args.num_shards):
        shard_dir = args.shard_root / f"shard_{shard:04d}"
        clip_index_path = shard_dir / "clip_index.parquet"
        if not clip_index_path.exists():
            print(f"[merge] shard {shard}: no clip_index.parquet (empty shard), skipping")
            continue

        clip_index = pd.read_parquet(clip_index_path)
        presence = pd.read_parquet(shard_dir / "metadata" / "feature_presence.parquet")
        n_chunks = int(clip_index["chunk"].max()) + 1

        # Relocate every chunk zip under the global offset.
        for feature in FEATURES:
            for local_chunk in range(n_chunks):
                src = feature_zip_paths(shard_dir, feature, local_chunk)
                if not src.exists():
                    continue
                dst = feature_zip_paths(out, feature, local_chunk + chunk_offset)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

        clip_index = clip_index.copy()
        clip_index["chunk"] = clip_index["chunk"].astype(int) + chunk_offset
        clip_indexes.append(clip_index)
        presences.append(presence)
        chunk_offset += n_chunks
        total_clips += len(clip_index)
        print(f"[merge] shard {shard}: {len(clip_index)} clips, {n_chunks} chunks -> global chunks {chunk_offset - n_chunks}..{chunk_offset - 1}")

    if not clip_indexes:
        raise RuntimeError("No shard produced any clips.")

    merged_clip_index = pd.concat(clip_indexes)
    merged_presence = pd.concat(presences)
    merged_clip_index.to_parquet(out / "clip_index.parquet")
    (out / "metadata").mkdir(parents=True, exist_ok=True)
    merged_presence.to_parquet(out / "metadata" / "feature_presence.parquet")
    write_features_csv(out, merged_clip_index, merged_presence)

    print(f"[merge] Done. Merged {total_clips} clips across {chunk_offset} chunks into {out}")


if __name__ == "__main__":
    main()
