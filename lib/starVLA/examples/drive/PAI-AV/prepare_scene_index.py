"""Intersection scenes for the eval figures: one sample per maneuver (left / right / straight) from PAI-AV val clips
whose reasoning annotations (``reasoning/ood_reasoning.parquet``, chain-of-causation text per event) mention traffic
lights or intersections. Plotted next to the regular eval figures (eval_bev/<maneuver>), not part of the metrics.

Per clip, t0 is scanned over the valid range (0.5 s steps) and classified like the eval samples (heading change over
3 s); per maneuver, candidates whose annotations mention a traffic light come first, then the ones closest to an
event start; among the first few that load, the brightest current frame is kept (mask overlays are unreadable at
night).

    python examples/drive/PAI-AV/prepare_scene_index.py --data_root <PAI-AV> \
        --stats_path outputs/qwendrive_pi/meta_h25_v2tp/norm_stats.json \
        --out outputs/qwendrive_pi/meta_h25_v2tp/scene_samples_val.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from starVLA.dataloader.pai_av_drive import (
    ClipStore, DriveSampler, EgoTrack, Normalizer, PAIAVDriveDataset, classify_maneuver, load_clip_table,
)

PATTERN = re.compile(r"traffic light|red light|green light|yellow light|stop sign|intersection", re.I)
LIGHT = re.compile(r"traffic light|red light|green light|yellow light", re.I)
MANEUVERS = ("left", "right", "straight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--stats_path", required=True, help="norm_stats.json of the run (its sampler settings are used)")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--try_per_maneuver", type=int, default=6)
    args = p.parse_args()

    stats = json.loads(Path(args.stats_path).read_text())
    sampler = DriveSampler(**{k: v for k, v in stats["sampler"].items() if k in DriveSampler.__init__.__code__.co_varnames})
    reasoning = pd.read_parquet(Path(args.data_root) / "reasoning" / "ood_reasoning.parquet")
    reasoning = reasoning[reasoning.split == args.split]
    chunks = load_clip_table(args.data_root, args.split)
    chunks = dict(zip(chunks.clip_id, chunks.chunk))
    store = ClipStore(args.data_root)

    cands = {m: [] for m in MANEUVERS}
    for clip_id, row in reasoning.iterrows():
        if clip_id not in chunks or not isinstance(row.events, str):
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
        light = any(LIGHT.search(str(e["coc"])) for e in events)
        starts = np.array([float(e["event_start_timestamp"]) for e in events])
        for t0 in np.arange(lo, hi, 0.5e6):
            m = classify_maneuver(track, t0)
            k = int(np.argmin(np.abs(starts - t0)))
            cands[m].append({"clip_id": clip_id, "chunk": chunk, "t0": float(t0), "maneuver": m,
                             "caption": str(events[k]["coc"]), "event_cluster": str(row.event_cluster),
                             "_key": (not light, abs(starts[k] - t0))})

    picked = []
    for m in MANEUVERS:
        seen, ranked = set(), []
        for c in sorted(cands[m], key=lambda c: c["_key"]):
            if c["clip_id"] not in seen:  # one candidate per clip
                seen.add(c["clip_id"])
                ranked.append({k: v for k, v in c.items() if k != "_key"})
        best, best_b = None, -1.0
        for c in ranked[: args.try_per_maneuver]:
            try:
                img = PAIAVDriveDataset(args.data_root, Normalizer(stats), samples=[c], sampler=sampler)[0]["image"][-1]
            except Exception as e:  # frames at the clip end can fail to decode
                print(f"drop {c['clip_id']} t0={c['t0']:.0f}: {e}")
                continue
            b = float(np.asarray(img.convert("L").resize((64, 36))).mean())
            if b > best_b:
                best, best_b = c, b
        if best is None:
            print(f"no {m} scene found")
            continue
        print(f"{m}: {best['clip_id'][:8]} t0={best['t0'] * 1e-6:.1f}s  {best['caption'][:90]}")
        picked.append(best)
    Path(args.out).write_text(json.dumps(picked, indent=1))
    print(f"{len(picked)} scenes -> {args.out}")


if __name__ == "__main__":
    main()
