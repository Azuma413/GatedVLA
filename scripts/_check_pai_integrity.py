#!/usr/bin/env python3
"""Standalone integrity check for a merged PAI dataset (no reference needed)."""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CAMS = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]


def members_for(feature, clip_id):
    if feature == "egomotion":
        return [f"{clip_id}.egomotion.parquet"]
    if feature == "navigation_command":
        return [f"{clip_id}.navigation_command.parquet"]
    return [f"{clip_id}.{feature}.mp4", f"{clip_id}.{feature}.timestamps.parquet"]


def zip_for(root, feature, chunk):
    if feature in ("egomotion", "navigation_command"):
        return root / "labels" / feature / f"{feature}.chunk_{chunk:04d}.zip"
    return root / "camera" / feature / f"{feature}.chunk_{chunk:04d}.zip"


def main():
    root = Path(sys.argv[1])
    ci = pd.read_parquet(root / "clip_index.parquet")
    pres = pd.read_parquet(root / "metadata" / "feature_presence.parquet")
    print(f"clips: {len(ci)}   unique clip_ids: {ci.index.nunique()}   chunks: {ci['chunk'].min()}..{ci['chunk'].max()} ({ci['chunk'].nunique()} distinct)")
    print(f"feature_presence rows: {len(pres)}   columns: {list(pres.columns)}")

    ok = True
    if ci.index.nunique() != len(ci):
        print("FAIL duplicate clip_ids"); ok = False
    if set(ci.index) != set(pres.index):
        print("FAIL clip_index vs feature_presence id mismatch"); ok = False

    features = ["egomotion", "navigation_command", *CAMS]
    # spot-check up to 30 random clips
    rng = np.random.default_rng(0)
    sample = rng.choice(np.array(ci.index), size=min(30, len(ci)), replace=False)
    bad = 0
    for clip_id in sample:
        chunk = int(ci.loc[clip_id, "chunk"])
        for feature in features:
            if not bool(pres.loc[clip_id, feature]):
                continue
            zp = zip_for(root, feature, chunk)
            if not zp.exists():
                print(f"FAIL missing {zp}"); bad += 1; continue
            with zipfile.ZipFile(zp) as zf:
                names = set(zf.namelist())
            for m in members_for(feature, clip_id):
                if m not in names:
                    print(f"FAIL {m} not in {zp}"); bad += 1
    print(f"spot-checked {len(sample)} clips: {'all members resolve' if bad == 0 else str(bad) + ' problems'}")
    if bad:
        ok = False

    # presence coverage: every clip should have all 4 cameras + egomotion + nav
    for feature in features:
        n = int(pres[feature].sum())
        print(f"  present[{feature}] = {n}/{len(pres)}")
        if n != len(pres):
            print(f"  NOTE: {feature} missing for {len(pres) - n} clips")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
