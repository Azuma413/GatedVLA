"""Compare AnchorVLA codebooks for the accel / curvature action representation (val quantisation error).

Both codebooks have K anchors and are judged by what the policy can actually execute: an anchor is turned into
controls and integrated from the sample's own current speed v0, and the result is compared with the ground-truth
positions (AnchorVLA Eq. 3, mean point-wise L2 over t0 + 0.1 .. t0 + H·0.1 s).

  waypoint  k-means on train trajectories (metres); per val sample each anchor is encoded to controls with that
            sample's v0 (Alpamayo fit) and decoded again. ``waypoint_raw`` is the same codebook without the
            conversion (the error AnchorVLA would have with waypoint actions), for reference.
  control   k-means on train controls (mean/std normalised accel, curvature); per val sample each anchor is
            decoded with that sample's v0.

    python examples/drive/PAI-AV/compare_anchor_spaces.py --data_root <PAI-AV> --num_anchors 100
"""

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from starVLA.dataloader.pai_av_drive import ClipStore, DriveSampler, EgoTrack, classify_maneuver, load_clip_table
from starVLA.model.modules.action_space.unicycle import UnicycleCodec

US = 1e-6


def _worker(args):
    data_root, clips, horizon, t0s, seed = args
    rng = np.random.default_rng(seed)
    store = ClipStore(data_root)
    sampler = DriveSampler(action_horizon=horizon, action_mode="unicycle", state_mode="traj")
    out = {"xy": [], "ctl": [], "v0": [], "man": []}
    for row in clips.itertuples():
        try:
            track = EgoTrack(store.egomotion(row.clip_id, row.chunk))
        except Exception:
            continue
        lo, hi = sampler.valid_range(track)
        if hi <= lo:
            continue
        for t0 in rng.uniform(lo, hi, size=t0s):
            _, ctl, _ = sampler.state_action(track, t0)
            out["xy"].append(sampler.gt_xy(track, t0)[1:])
            out["ctl"].append(ctl)
            out["v0"].append(sampler.speed(track, t0))
            out["man"].append(classify_maneuver(track, t0))
    return {k: np.array(v) for k, v in out.items()}


def collect(data_root, split, n_clips, t0s, horizon, workers, seed):
    clips = load_clip_table(data_root, split).sample(frac=1.0, random_state=seed).head(n_clips).sort_values("chunk")
    shards = np.array_split(np.arange(len(clips)), workers)
    with Pool(workers) as pool:
        parts = pool.map(_worker, [(data_root, clips.iloc[s], horizon, t0s, seed + k) for k, s in enumerate(shards)])
    return {k: np.concatenate([p[k] for p in parts if len(p[k])]) for k in parts[0]}


def kmeans(x, k, iters=100, seed=0):
    """Lloyd with k-means++ init on flattened vectors."""
    flat = torch.as_tensor(x.reshape(len(x), -1), dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    centres = [flat[torch.randint(len(flat), (1,), generator=g)].squeeze(0)]
    d2 = ((flat - centres[0]) ** 2).sum(1)
    for _ in range(1, k):
        centres.append(flat[torch.multinomial(d2 / d2.sum(), 1, generator=g)].squeeze(0))
        d2 = torch.minimum(d2, ((flat - centres[-1]) ** 2).sum(1))
    c = torch.stack(centres)
    for _ in range(iters):
        assign = torch.cdist(flat, c).argmin(1)
        new = torch.stack([flat[assign == j].mean(0) if (assign == j).any() else c[j] for j in range(k)])
        if torch.allclose(new, c):
            break
        c = new
    return c.reshape(k, *x.shape[1:]).numpy()


def yaw_of(xy):
    """Heading along a trajectory (..., H, 2) starting at the origin; 0 while (almost) standing."""
    prev = np.concatenate([np.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], axis=-2)
    d = xy - prev
    yaw = np.arctan2(d[..., 1], d[..., 0])
    moving = np.linalg.norm(d, axis=-1) > 0.02
    yaw = np.where(moving, yaw, 0.0)
    return np.unwrap(yaw, axis=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--num_anchors", type=int, default=100)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--train_clips", type=int, default=6000)
    p.add_argument("--val_clips", type=int, default=1500)
    p.add_argument("--t0s_per_clip", type=int, default=8)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    tr = collect(args.data_root, "train", args.train_clips, args.t0s_per_clip, args.horizon, args.workers, 0)
    va = collect(args.data_root, "val", args.val_clips, args.t0s_per_clip, args.horizon, args.workers, 1)
    print(f"train {len(tr['xy'])}  val {len(va['xy'])}", flush=True)
    codec = UnicycleCodec(args.horizon)
    K = args.num_anchors

    wp = kmeans(tr["xy"], K)
    mean, std = tr["ctl"].mean((0, 1)), tr["ctl"].std((0, 1))
    ctl = kmeans((tr["ctl"] - mean) / std, K) * std + mean

    gt, v0 = torch.as_tensor(va["xy"]), torch.as_tensor(va["v0"])
    N = len(gt)

    def ade_of(decoded):  # (N, K, H, 2) -> (N, K)
        return torch.linalg.norm(decoded - gt[:, None], dim=-1).mean(-1)

    results = {}
    results["waypoint_raw"] = ade_of(torch.as_tensor(wp)[None].expand(N, -1, -1, -1))
    # control codebook: decode every anchor with each sample's v0
    dec = codec.decode(torch.as_tensor(ctl)[None].expand(N, -1, -1, -1).reshape(N * K, *ctl.shape[1:]),
                       v0[:, None].expand(-1, K).reshape(-1)).reshape(N, K, args.horizon, 2)
    results["control"] = ade_of(dec)
    # waypoint codebook through the action representation: encode with each sample's v0, decode again
    wp_yaw = yaw_of(wp)
    chunks = []
    for s in range(0, N, 500):
        n = min(500, N - s)
        enc = codec.encode(np.broadcast_to(wp, (n, K, *wp.shape[1:])), np.broadcast_to(wp_yaw, (n, K, wp.shape[1])),
                           np.repeat(va["v0"][s : s + n, None], K, 1))
        d = codec.decode(torch.as_tensor(enc).reshape(n * K, *enc.shape[2:]),
                         v0[s : s + n, None].expand(-1, K).reshape(-1)).reshape(n, K, args.horizon, 2)
        chunks.append(ade_of_slice(d, gt[s : s + n]))
        print(f"  waypoint->controls {s + n}/{N}", flush=True)
    results["waypoint"] = torch.cat(chunks)

    lat = np.abs(va["xy"][:, -1, 1])
    bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)]
    rows = []
    for name, d in results.items():
        best = d.min(1).values.numpy()
        row = {"codebook": name, "ADE": best.mean(), "ADE_p90": np.percentile(best, 90),
               "used": len(np.unique(d.argmin(1).numpy()))}
        for m in ("straight", "left", "right"):
            row[f"{m}"] = best[va["man"] == m].mean()
        for lo, hi in bins:
            sel = (lat >= lo) & (lat < hi)
            row[f"lat {lo}-{hi if hi < 1e8 else 'inf'} (n={sel.sum()})"] = best[sel].mean()
        for lo, hi in [(0, 5), (5, 15), (15, 25), (25, 1e9)]:
            sel = (va["v0"] >= lo) & (va["v0"] < hi)
            row[f"v {lo}-{hi if hi < 1e8 else 'inf'} (n={sel.sum()})"] = best[sel].mean()
        rows.append(row)
    pd.set_option("display.width", 300)
    print(f"\n== val quantisation ADE [m], K = {K} (executed: decoded from controls with the sample's v0)")
    print(pd.DataFrame(rows).set_index("codebook").T.round(3).to_string())


def ade_of_slice(decoded, gt):
    return torch.linalg.norm(decoded - gt[:, None], dim=-1).mean(-1)


if __name__ == "__main__":
    main()
