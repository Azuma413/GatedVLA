"""Scene index for figure-only evaluation: PAI-AV val clips whose reasoning annotations (``reasoning/
ood_reasoning.parquet``, chain-of-causation text per event) mention traffic lights, stop signs or intersections.

One sample per matching event, t0 = the event's start timestamp clamped to the sampler's valid range. The
``maneuver`` field holds the event cluster and ``caption`` the annotation text (shown in the figure titles).

    python examples/drive/PAI-AV/prepare_scene_index.py --data_root <PAI-AV> \
        --stats_path outputs/qwendrive_pi/meta_h25_v2tp/norm_stats.json \
        --out outputs/qwendrive_pi/meta_h25_v2tp/scene_samples_val.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from starVLA.dataloader.pai_av_drive import ClipStore, DriveSampler, EgoTrack, Normalizer, PAIAVDriveDataset, load_clip_table

PATTERN = re.compile(r"traffic light|red light|green light|yellow light|signal|stop sign|intersection", re.I)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--stats_path", required=True, help="norm_stats.json of the run (its sampler settings are used)")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max_samples", type=int, default=48)
    args = p.parse_args()

    stats = json.loads(Path(args.stats_path).read_text())
    sampler_cfg = stats["sampler"]
    sampler = DriveSampler(**{k: v for k, v in sampler_cfg.items() if k in DriveSampler.__init__.__code__.co_varnames})
    reasoning = pd.read_parquet(Path(args.data_root) / "reasoning" / "ood_reasoning.parquet")
    reasoning = reasoning[reasoning.split == args.split]
    chunks = load_clip_table(args.data_root, args.split)
    chunks = dict(zip(chunks.clip_id, chunks.chunk))
    store = ClipStore(args.data_root)

    samples = []
    for clip_id, row in reasoning.iterrows():
        if clip_id not in chunks:
            continue
        if not isinstance(row.events, str):  # clips without event annotations
            continue
        events = [e for e in json.loads(row.events) if PATTERN.search(str(e.get("coc", "")))]
        if not events:
            continue
        chunk = int(chunks[clip_id])
        try:
            track = EgoTrack(store.egomotion(clip_id, chunk))
            lo, hi = sampler.valid_range(track, store.frame_timestamps(clip_id, chunk))
        except Exception as e:  # missing / corrupt clip
            print(f"skip {clip_id}: {e}")
            continue
        if hi <= lo:
            continue
        for e in events:
            t0 = min(max(float(e["event_start_timestamp"]), lo), hi)
            samples.append({"clip_id": clip_id, "chunk": chunk, "t0": t0, "maneuver": str(row.event_cluster),
                            "caption": str(e["coc"])})
    # keep only samples that load (frames near the clip end can fail to decode)
    ds = PAIAVDriveDataset(args.data_root, Normalizer(stats), samples=samples, sampler=sampler)
    ok = []
    for i, s in enumerate(samples):
        try:
            ds[i]
            ok.append(s)
        except Exception as e:
            print(f"drop {s['clip_id']} t0={s['t0']:.0f}: {e}")
    samples = ok[: args.max_samples]
    Path(args.out).write_text(json.dumps(samples, indent=1))
    print(f"{len(samples)} scene samples from {len({s['clip_id'] for s in samples})} clips -> {args.out}")


if __name__ == "__main__":
    main()
