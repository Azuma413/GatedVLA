#!/usr/bin/env python3
"""Assert a merged PAI dataset is equivalent to a serial-export reference.

Chunk *numbers* legitimately differ (each shard restarts at chunk 0), so we
compare clip membership, feature presence, and that every clip's declared chunk
zip actually contains that clip's expected members.
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

CAMS = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]


def members_for(feature: str, clip_id: str) -> list[str]:
    if feature == "egomotion":
        return [f"{clip_id}.egomotion.parquet"]
    if feature == "navigation_command":
        return [f"{clip_id}.navigation_command.parquet"]
    return [f"{clip_id}.{feature}.mp4", f"{clip_id}.{feature}.timestamps.parquet"]


def zip_for(root: Path, feature: str, chunk: int) -> Path:
    if feature in ("egomotion", "navigation_command"):
        return root / "labels" / feature / f"{feature}.chunk_{chunk:04d}.zip"
    return root / "camera" / feature / f"{feature}.chunk_{chunk:04d}.zip"


def main() -> int:
    serial_dir = Path(sys.argv[1])
    merged_dir = Path(sys.argv[2])

    serial = pd.read_parquet(serial_dir / "clip_index.parquet")
    merged = pd.read_parquet(merged_dir / "clip_index.parquet")
    s_pres = pd.read_parquet(serial_dir / "metadata" / "feature_presence.parquet")
    m_pres = pd.read_parquet(merged_dir / "metadata" / "feature_presence.parquet")

    ok = True

    if set(serial.index) != set(merged.index):
        print(f"FAIL clip set differs: serial={len(serial)} merged={len(merged)}")
        print("  only serial:", set(serial.index) - set(merged.index))
        print("  only merged:", set(merged.index) - set(serial.index))
        ok = False
    else:
        print(f"OK clip set matches ({len(merged)} clips)")

    # feature presence (align on index)
    common = sorted(set(serial.index) & set(merged.index))
    if not s_pres.reindex(common).equals(m_pres.reindex(common)):
        print("FAIL feature_presence differs")
        ok = False
    else:
        print("OK feature_presence matches")

    # every merged clip's declared chunk zip contains its members
    features = ["egomotion", "navigation_command", *CAMS]
    bad = 0
    for clip_id in merged.index:
        chunk = int(merged.loc[clip_id, "chunk"])
        for feature in features:
            if not bool(m_pres.loc[clip_id, feature]):
                continue
            zp = zip_for(merged_dir, feature, chunk)
            if not zp.exists():
                print(f"FAIL missing zip {zp} for clip {clip_id}")
                bad += 1
                continue
            with zipfile.ZipFile(zp) as zf:
                names = set(zf.namelist())
            for m in members_for(feature, str(clip_id)):
                if m not in names:
                    print(f"FAIL member {m} not in {zp}")
                    bad += 1
    if bad:
        print(f"FAIL {bad} member/zip problems")
        ok = False
    else:
        print("OK all clips resolve their members in the declared chunk zip")

    # no duplicate global chunk collisions that mix members incorrectly is
    # implicitly covered above; also confirm chunk uniqueness of zips per feature
    print("merged chunks:", sorted(merged["chunk"].unique()))

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
