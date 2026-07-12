#!/usr/bin/env python3
"""Build a VideoFolderDataset-style corpus for VLA-JEPA stage-1 (V-JEPA2 world-model
pretraining) from a LeRobot CARLA dataset.

`starVLA/dataloader/video_datasets.py::VideoFolderDataset` expects:
  - `video_dir`: a *flat* folder of video files named `<int_id>.<ext>`.
  - `text_file`: a `;`-separated CSV whose 1st column is the int id and 2nd the caption
    (read positionally, so a header row is required — it is consumed as column names).

Each LeRobot episode -> one video (single camera; the model duplicates it into 2 views)
+ the episode's nav-command task list joined into one instruction string.

Videos are symlinked (not copied) to keep this cheap on the ~100GB dataset.
"""
import argparse
import csv
import json
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lerobot-dir", required=True, type=Path, help="LeRobot dataset root.")
    p.add_argument("--out-dir", required=True, type=Path, help="Destination corpus root.")
    p.add_argument("--camera", default="front_wide", help="LeRobot camera key to use as the video.")
    args = p.parse_args()

    clips_dir = args.out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "instructions.csv"

    cam_dir = args.lerobot_dir / "videos" / "chunk-000" / f"observation.images.{args.camera}"
    if not cam_dir.is_dir():
        raise SystemExit(f"camera video dir not found: {cam_dir}")

    rows = []
    n = 0
    with open(args.lerobot_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            ep = json.loads(line)
            idx = int(ep["episode_index"])
            src = (cam_dir / f"episode_{idx:06d}.mp4").resolve()
            if not src.exists():
                print(f"[warn] missing video for episode {idx}: {src}")
                continue
            dst = clips_dir / f"{idx}.mp4"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
            tasks = ep.get("tasks") or ["Follow the lane"]
            # de-dup while preserving order, then join into one instruction
            seen, uniq = set(), []
            for t in tasks:
                if t not in seen:
                    seen.add(t)
                    uniq.append(t)
            rows.append((idx, ", ".join(uniq)))
            n += 1

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "caption"])  # header (consumed positionally by the loader)
        w.writerows(rows)

    print(f"[done] {n} videos -> {clips_dir}")
    print(f"[done] captions -> {csv_path}")


if __name__ == "__main__":
    main()
