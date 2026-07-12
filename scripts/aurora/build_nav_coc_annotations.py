#!/usr/bin/env python
"""Build the nav+CoC annotation JSON that drives Alpamayo 1.5 Stage-1/2 nav SFT.

`PAIDatasetWithNav` (alpamayo.data.pai_nav) is driven by a JSON list of samples,
each `{clip_id, t0_relative, nav_text, cot}`. This script produces that file by
joining, for every CoC-labeled segment:

  * clip_id / t0_relative   from the CoC yaml filename (`event_start_timestamp`,
                            clip-relative microseconds == load_physical_aiavdataset t0_us)
  * nav_text                from the PAI `navigation_command` label (`command_text`
                            at the timestamp nearest t0, e.g. "Turn left")
  * cot                     from the CoC yaml
                            `final_content.ego_behavior_schema.effect_on_ego_behavior`

A segment is kept only if the Alpamayo window fits the clip:
    start + num_history_steps*time_step  <=  t0  <=  end - num_future_steps*time_step
(defaults 16*0.1=1.6s history, 64*0.1=6.4s future), matching the clip_index
`event_t0s` grid. This mirrors the export's own "clip too short" skips.

Usage:
    python scripts/aurora/build_nav_coc_annotations.py \
        --pai-dir data/carla_alpamayo_pai \
        --coc-dir outputs/coc_autolabel_aurora_full/coc_labels_8b \
        --out data/carla_alpamayo_pai/nav_coc_annotations.json
"""
import argparse
import io
import json
import os
import re
import zipfile

import pandas as pd
import yaml

# libyaml C loader: ~10-50x faster than the pure-python SafeLoader, which matters
# across ~11k 60KB cot yamls (each carries a large prompt/model_usage block).
_Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

_COT_RE = re.compile(r"^cot_(\d+)\.yaml$")


def load_label(path: str) -> str | None:
    """Return the CoC ego-behavior label string, or None if absent/empty."""
    with open(path) as f:
        d = yaml.load(f, Loader=_Loader)
    try:
        s = d["final_content"]["ego_behavior_schema"]["effect_on_ego_behavior"]
    except (KeyError, TypeError):
        return None
    if not isinstance(s, str):
        return None
    s = s.strip()
    return s or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pai-dir", required=True)
    ap.add_argument("--coc-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-history-steps", type=int, default=16)
    ap.add_argument("--num-future-steps", type=int, default=64)
    ap.add_argument("--time-step", type=float, default=0.1)
    args = ap.parse_args()

    hist_us = int(args.num_history_steps * args.time_step * 1e6)
    fut_us = int(args.num_future_steps * args.time_step * 1e6)

    clip_index = pd.read_parquet(os.path.join(args.pai_dir, "clip_index.parquet"))
    # index is clip_id; keep chunk/start/end
    meta = clip_index[["chunk", "start_timestamp", "end_timestamp"]].to_dict("index")

    nav_root = os.path.join(args.pai_dir, "labels", "navigation_command")
    _zip_cache: dict[int, zipfile.ZipFile] = {}
    _nav_cache: dict[str, pd.DataFrame] = {}

    def nav_df(clip_id: str, chunk: int) -> pd.DataFrame | None:
        if clip_id in _nav_cache:
            return _nav_cache[clip_id]
        if chunk not in _zip_cache:
            zp = os.path.join(nav_root, f"navigation_command.chunk_{chunk:04d}.zip")
            _zip_cache[chunk] = zipfile.ZipFile(zp)
        z = _zip_cache[chunk]
        member = f"{clip_id}.navigation_command.parquet"
        if member not in z.namelist():
            _nav_cache[clip_id] = None
            return None
        df = pd.read_parquet(io.BytesIO(z.read(member))).sort_values("timestamp_us")
        _nav_cache[clip_id] = df
        return df

    def nav_text_at(df: pd.DataFrame, t0: int) -> str:
        i = (df["timestamp_us"] - t0).abs().idxmin()
        return str(df.loc[i, "command_text"])

    samples = []
    stats = {
        "coc_clips": 0, "clip_not_in_index": 0, "no_nav": 0,
        "segments_total": 0, "window_skip": 0, "no_label": 0, "kept": 0,
    }
    navtext_counts: dict[str, int] = {}

    for clip_id in sorted(os.listdir(args.coc_dir)):
        cdir = os.path.join(args.coc_dir, clip_id)
        if not os.path.isdir(cdir):
            continue
        stats["coc_clips"] += 1
        if clip_id not in meta:
            stats["clip_not_in_index"] += 1
            continue
        m = meta[clip_id]
        chunk = int(m["chunk"])
        start, end = int(m["start_timestamp"]), int(m["end_timestamp"])
        df = nav_df(clip_id, chunk)
        if df is None:
            stats["no_nav"] += 1
            continue
        for fname in sorted(os.listdir(cdir)):
            mm = _COT_RE.match(fname)
            if not mm:
                continue
            stats["segments_total"] += 1
            t0 = int(mm.group(1))
            if not (start + hist_us <= t0 <= end - fut_us):
                stats["window_skip"] += 1
                continue
            label = load_label(os.path.join(cdir, fname))
            if label is None:
                stats["no_label"] += 1
                continue
            nav_text = nav_text_at(df, t0)
            navtext_counts[nav_text] = navtext_counts.get(nav_text, 0) + 1
            samples.append(
                {"clip_id": clip_id, "t0_relative": t0, "nav_text": nav_text, "cot": label}
            )
            stats["kept"] += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(samples, f, indent=1)

    n_clips = len({s["clip_id"] for s in samples})
    print("=== nav+CoC annotation build ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  unique clips kept: {n_clips}")
    print("  nav_text distribution:")
    for k, v in sorted(navtext_counts.items(), key=lambda x: -x[1]):
        print(f"    {k!r}: {v}")
    print(f"Wrote {len(samples)} samples -> {args.out}")


if __name__ == "__main__":
    main()
