"""Distribution of driving segments (stopped / straight / turns / ...) in PAI-AV, from egomotion only.

Every clip is sampled on a 1 s grid; each time point t is labelled from the ego motion over [t, t + W] (W = 3 s,
4 s for lane changes), first matching rule wins:
    reverse      body-frame forward velocity < -0.3 m/s at t
    stopped      max speed over the window < 0.5 m/s
    starting     speed(t) < 0.5 m/s, then moves (launch from standstill)
    stopping     speed(t) >= 0.5 m/s and speed(t + W) < 0.5 m/s
    turn_left/right   |heading change over W| >= 30 deg   (intersection-type turn)
    curve_left/right  10 <= |heading change| < 30 deg     (road curve / gentle turn)
    lane_change_left/right  |heading change over 4 s| < 6 deg, lateral offset at t + 4 s > 2.5 m, speed > 5 m/s
    straight     everything else, split by speed change over W into
                 straight_cruise (|dv| < 1.5 m/s), straight_accel, straight_decel

    python examples/drive/PAI-AV/analyze_maneuvers.py --data_root <PAI-AV root> --out <csv>
"""

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

from starVLA.dataloader.pai_av_drive import ClipStore, EgoTrack, load_clip_table

US = 1e-6
STOP_V = 0.5


def label_track(track: EgoTrack, window: float = 3.0, lc_window: float = 4.0, step: float = 1.0):
    t_end = track.t_max - lc_window / US
    if t_end <= track.t_min:
        return []
    ts = np.arange(track.t_min, t_end, step / US)
    yaw = lambda t: np.interp(t, track.t, track.yaw)  # noqa: E731
    speed = lambda t: np.interp(t, track.t, track.speed)  # noqa: E731
    # body-frame forward velocity sign (speed is |v_xy|): project displacement over 0.2 s on heading
    out = []
    for t in ts:
        tw = t + window / US
        seg = (track.t >= t) & (track.t <= tw)
        v0, vw = speed(t), speed(tw)
        vmax = track.speed[seg].max() if seg.any() else v0
        p0, y0 = track.pose(t)
        p1, _ = track.pose(t + 0.2 / US)
        fwd = (p1 - p0) @ np.array([np.cos(y0), np.sin(y0)]) / 0.2
        dyaw = np.degrees(yaw(tw) - y0)
        dyaw_lc = np.degrees(yaw(t + lc_window / US) - y0)
        lat_lc = track.positions_ego(np.array([t + lc_window / US]), t)[0, 1]
        if fwd < -0.3:
            lab = "reverse"
        elif vmax < STOP_V:
            lab = "stopped"
        elif v0 < STOP_V:
            lab = "starting"
        elif vw < STOP_V:
            lab = "stopping"
        elif abs(dyaw) >= 30:
            lab = "turn_left" if dyaw > 0 else "turn_right"
        elif abs(dyaw) >= 10:
            lab = "curve_left" if dyaw > 0 else "curve_right"
        elif abs(dyaw_lc) < 6 and abs(lat_lc) > 2.5 and v0 > 5:
            lab = "lane_change_left" if lat_lc > 0 else "lane_change_right"
        else:
            dv = vw - v0
            lab = "straight_cruise" if abs(dv) < 1.5 else ("straight_accel" if dv > 0 else "straight_decel")
        out.append((lab, v0, dyaw))
    return out


def worker(args):
    data_root, rows = args
    store = ClipStore(data_root, max_open_zips=2)
    recs = []
    for clip_id, chunk, split in rows:
        try:
            track = EgoTrack(store.egomotion(clip_id, chunk))
        except Exception:
            continue
        for lab, v0, dyaw in label_track(track):
            recs.append((clip_id, split, lab, v0, dyaw))
    return recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--clip_fraction", type=float, default=1.0)
    args = p.parse_args()

    clips = load_clip_table(args.data_root)
    if args.clip_fraction < 1.0:
        clips = clips.sample(frac=args.clip_fraction, random_state=0)
    jobs = [(args.data_root, list(g[["clip_id", "chunk", "split"]].itertuples(index=False, name=None)))
            for _, g in clips.groupby("chunk")]
    with Pool(args.workers) as pool:
        parts = pool.map(worker, jobs, chunksize=4)
    df = pd.DataFrame([r for part in parts for r in part], columns=["clip_id", "split", "label", "speed", "dyaw_deg"])
    df.to_parquet(args.out)
    print(f"{df.clip_id.nunique()} clips, {len(df)} time points")


if __name__ == "__main__":
    main()
