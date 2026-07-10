#!/usr/bin/env python
"""Build a small PAI subset containing only the given clip_ids, for CoC
verification. trajdata's UnifiedDataset preprocesses *every* clip in the PAI
data_dir before any scene-list filter, so scoping a small run requires a data_dir
with only the wanted clips. This copies the filtered clip_index / feature_presence
and rewrites each per-chunk feature zip to contain only the selected clips' members.

Usage:
  python scripts/aurora/_make_pai_subset.py \
    --src data/carla_alpamayo_pai --dst data/carla_alpamayo_pai_verify \
    --clip-list logs/coc_verify_cliplist.txt
"""
import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

FEATURES = [
    ("camera/camera_cross_left_120fov", "camera_cross_left_120fov"),
    ("camera/camera_front_wide_120fov", "camera_front_wide_120fov"),
    ("camera/camera_cross_right_120fov", "camera_cross_right_120fov"),
    ("camera/camera_front_tele_30fov", "camera_front_tele_30fov"),
    ("labels/egomotion", "egomotion"),
    ("labels/navigation_command", "navigation_command"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--clip-list", required=True)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    clip_ids = [c.strip() for c in Path(args.clip_list).read_text().splitlines() if c.strip()]
    clip_set = set(clip_ids)
    print(f"subset: {len(clip_ids)} clips -> {dst}")

    dst.mkdir(parents=True, exist_ok=True)

    # clip_index.parquet (index = clip_id) -> filtered
    ci = pd.read_parquet(src / "clip_index.parquet")
    ci_sub = ci.loc[ci.index.isin(clip_set)]
    missing = clip_set - set(ci_sub.index)
    if missing:
        raise SystemExit(f"clip_ids not found in clip_index: {sorted(missing)}")
    ci_sub.to_parquet(dst / "clip_index.parquet")
    chunks = sorted(ci_sub["chunk"].unique().tolist())
    print(f"  chunks involved: {chunks}")

    # features.csv (verbatim) and metadata/feature_presence.parquet (filtered).
    shutil.copy2(src / "features.csv", dst / "features.csv")
    (dst / "metadata").mkdir(exist_ok=True)
    fp = pd.read_parquet(src / "metadata" / "feature_presence.parquet")
    fp_idx = fp.set_index(fp.columns[0]) if fp.index.name is None and "clip_id" in fp.columns else fp
    if "clip_id" in fp.columns:
        fp_sub = fp[fp["clip_id"].isin(clip_set)]
    else:
        fp_sub = fp.loc[fp.index.isin(clip_set)]
    fp_sub.to_parquet(dst / "metadata" / "feature_presence.parquet")
    print(f"  feature_presence rows: {len(fp_sub)}")

    # Rewrite each per-chunk zip with only the selected clips' members.
    for rel, feat in FEATURES:
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for ch in chunks:
            name = f"{feat}.chunk_{ch:04d}.zip"
            src_zip = src / rel / name
            if not src_zip.exists():
                print(f"  WARN missing {src_zip}")
                continue
            kept = 0
            with zipfile.ZipFile(src_zip) as zin, zipfile.ZipFile(
                dst / rel / name, "w", zipfile.ZIP_STORED
            ) as zout:
                for m in zin.namelist():
                    cid = m.split(".", 1)[0]
                    if cid in clip_set:
                        zout.writestr(m, zin.read(m))
                        kept += 1
            print(f"  {rel}/{name}: kept {kept} members")

    print("done.")


if __name__ == "__main__":
    main()
