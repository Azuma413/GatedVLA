"""Training-data statistics for the longitudinal behaviour: how often the target point is the clip's last recorded
position instead of a point >= target_distance ahead (it then marks where the ego ends up, e.g. a stop), and how often
the 2.5 s future decelerates / stops, sampled like training (random clips, uniform t0 over the valid range).

    python examples/drive/PAI-AV/analyze_target_fallback.py --data_root <PAI-AV> \
        --stats_path outputs/qwendrive_pi/meta_h25_v2tp/norm_stats.json --clips 3000
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from starVLA.dataloader.pai_av_drive import ClipStore, DriveSampler, EgoTrack, load_clip_table

US = 1e-6


def worker(args):
    data_root, rows, sampler_kw, t0s, seed = args
    sampler = DriveSampler(**sampler_kw)
    rng, store, out = np.random.default_rng(seed), ClipStore(data_root), []
    for row in rows:
        try:
            track = EgoTrack(store.egomotion(row[0], row[1]))
            lo, hi = sampler.valid_range(track)
        except Exception:
            continue
        if hi <= lo:
            continue
        for t0 in rng.uniform(lo, hi, size=t0s):
            p0, _ = track.pose(t0)
            future = track.xy[track.t > t0]
            fallback = not (np.linalg.norm(future - p0, axis=1) >= sampler.target_distance).any()
            target = track.target_point(t0, sampler.target_distance)
            v0 = float(np.interp(t0, track.t, track.speed))
            v1 = float(np.interp(t0 + sampler.action_horizon * sampler.dt / US, track.t, track.speed))
            out.append((fallback, float(np.linalg.norm(target)), v0, v1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--stats_path", required=True)
    p.add_argument("--clips", type=int, default=3000)
    p.add_argument("--t0s_per_clip", type=int, default=4)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    sc = json.loads(Path(args.stats_path).read_text())["sampler"]
    kw = {k: v for k, v in sc.items() if k in DriveSampler.__init__.__code__.co_varnames}
    clips = load_clip_table(args.data_root, "train").sample(n=args.clips, random_state=0)
    rows = list(zip(clips.clip_id, clips.chunk.astype(int)))
    shards = [rows[i :: args.workers] for i in range(args.workers)]
    with Pool(args.workers) as pool:
        res = [r for part in pool.map(worker, [(args.data_root, s, kw, args.t0s_per_clip, i) for i, s in enumerate(shards)])
               for r in part]
    fb, dist, v0, v1 = (np.array(x) for x in zip(*res))
    dv = v1 - v0
    print(f"samples {len(fb)} from {args.clips} clips")
    print(f"target = last recorded position (fallback): {fb.mean() * 100:.1f}%  (target distance median "
          f"{np.median(dist[fb]) if fb.any() else float('nan'):.1f} m when fallback, {np.median(dist[~fb]):.1f} m otherwise)")
    for name, sel in (("stopped at t0 (v0 < 0.5 m/s)", v0 < 0.5), ("decelerates > 2 m/s over 2.5 s", dv < -2),
                      ("decelerates > 4 m/s", dv < -4), ("comes to a stop (v0 > 2, v_end < 0.5)", (v0 > 2) & (v1 < 0.5)),
                      ("accelerates > 2 m/s", dv > 2)):
        print(f"{name}: {sel.mean() * 100:.1f}% of samples, fallback within them {fb[sel].mean() * 100 if sel.any() else 0:.1f}%")


if __name__ == "__main__":
    main()
