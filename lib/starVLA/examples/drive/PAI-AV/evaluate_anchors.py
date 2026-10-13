"""Evaluate an AnchorVLA trajectory codebook on held-out PAI-AV trajectories.

Reports, for train and val trajectories (egomotion only):
  - quantisation error to the nearest anchor (mean point-wise L2 = AnchorVLA Eq. 3, and final-point error),
    overall and per maneuver (left / right / straight / stop, by heading change over the horizon)
  - maneuver of each trajectory vs. maneuver of its nearest anchor (confusion matrix)
  - anchor usage (how many anchors are nearest for at least one val trajectory)
  - optionally a K sweep (re-running k-means on the train sample) of the val quantisation error

    python examples/drive/PAI-AV/evaluate_anchors.py --data_root <PAI-AV> --anchors <anchors.npz> --k_sweep 25,50,100,200,400
"""

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

from starVLA.dataloader.pai_av_drive import DriveSampler, load_clip_table

from prepare_paiav import _stats_worker, kmeans, mean_l2


def trajectories(data_root, split, n_clips, t0s, horizon, workers, seed):
    clips = load_clip_table(data_root, split)
    clips = clips.sample(n=min(n_clips, len(clips)), random_state=seed).sort_values("chunk")
    sampler = DriveSampler(action_horizon=horizon)
    shards = np.array_split(np.arange(len(clips)), workers)
    with Pool(workers) as pool:
        parts = pool.map(_stats_worker, [(data_root, clips.iloc[s], sampler, t0s, seed + k) for k, s in enumerate(shards)])
    return np.concatenate([a for _, a in parts]).reshape(-1, horizon, 2)


def maneuver(traj: np.ndarray, turn_deg: float = 15.0, stop_m: float = 1.0) -> np.ndarray:
    """left / right / straight / stop of trajectories (N, H, 2) from the final heading."""
    d = traj[:, -1] - traj[:, -4]
    heading = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    moving = np.linalg.norm(traj[:, -1], axis=-1) >= stop_m
    return np.where(~moving, "stop", np.where(heading > turn_deg, "left", np.where(heading < -turn_deg, "right", "straight")))


def kmeans_from(x: np.ndarray, init: np.ndarray, iters: int = 100) -> np.ndarray:
    """Lloyd iterations on flattened trajectories from given initial centres."""
    flat, centres = x.reshape(len(x), -1).astype(np.float64), init.reshape(len(init), -1).astype(np.float64)
    for _ in range(iters):
        assign = ((flat**2).sum(1)[:, None] - 2 * flat @ centres.T + (centres**2).sum(1)[None]).argmin(1)
        new = np.stack([flat[assign == j].mean(0) if (assign == j).any() else centres[j] for j in range(len(centres))])
        if np.allclose(new, centres):
            break
        centres = new
    return centres.reshape(init.shape).astype(np.float32)


def knn_outlier_mask(traj: np.ndarray, k: int = 10, quantile: float = 99.0) -> np.ndarray:
    """True for inliers: distance (Eq. 3) to the k-th nearest neighbour below the given percentile."""
    kth = np.empty(len(traj), dtype=np.float32)
    for i in range(0, len(traj), 1024):
        d = mean_l2(traj[i : i + 1024], traj)
        kth[i : i + 1024] = np.partition(d, k, axis=1)[:, k]  # index 0 is the point itself
    return kth <= np.percentile(kth, quantile)


def farthest_point_sampling(traj: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Greedy k-centre (VADv2-style vocabulary): repeatedly add the trajectory farthest (Eq. 3) from the set."""
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(traj)))]
    d = mean_l2(traj, traj[idx])[:, 0]
    for _ in range(1, k):
        idx.append(int(d.argmax()))
        d = np.minimum(d, mean_l2(traj, traj[idx[-1:]])[:, 0])
    return traj[idx].copy()


def build_codebook(method: str, fit: np.ndarray, k: int) -> np.ndarray:
    flat = fit.reshape(len(fit), -1).astype(np.float64)
    if method == "kmeans":
        return kmeans(fit, k, seed=0)
    if method == "kmeans_axis_std":  # scale x and y by their std over all points
        std = fit.reshape(-1, 2).std(0)
        return kmeans(fit / std, k, seed=0) * std
    if method == "kmeans_pca_white":  # whiten the principal components covering 99.9% of the variance
        mean = flat.mean(0)
        u, sv, vt = np.linalg.svd(flat - mean, full_matrices=False)
        var = sv**2
        m = int(np.searchsorted(np.cumsum(var) / var.sum(), 0.999) + 1)
        scale = np.sqrt(var[:m] / len(flat))
        z = ((flat - mean) @ vt[:m].T) / scale
        cz = kmeans(z[:, :, None], k, seed=0)[:, :, 0]
        return ((cz * scale) @ vt[:m] + mean).reshape(k, *fit.shape[1:]).astype(np.float32)
    inliers = fit[knn_outlier_mask(fit)]
    if method == "fps":
        return farthest_point_sampling(inliers, k)
    if method == "fps_kmeans":
        return kmeans_from(inliers, farthest_point_sampling(inliers, k))
    raise ValueError(method)


def report(name, traj, anchors):
    dist = mean_l2(traj, anchors)
    nearest = dist.argmin(1)
    ade = dist.min(1)
    fde = np.linalg.norm(traj[:, -1] - anchors[nearest][:, -1], axis=-1)
    cls_t, cls_a = maneuver(traj), maneuver(anchors)[nearest]
    rows = []
    for c in ["all", "straight", "left", "right", "stop"]:
        m = np.ones(len(traj), bool) if c == "all" else cls_t == c
        if m.any():
            rows.append({"maneuver": c, "n": int(m.sum()), "share": m.mean(),
                         "ADE_mean": ade[m].mean(), "ADE_p50": np.median(ade[m]), "ADE_p90": np.percentile(ade[m], 90),
                         "FDE_mean": fde[m].mean(), "FDE_p90": np.percentile(fde[m], 90),
                         "nearest_anchor_same_class": (cls_a[m] == cls_t[m]).mean()})
    print(f"\n== {name}: {len(traj)} trajectories, anchors used {len(np.unique(nearest))}/{len(anchors)}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.crosstab(pd.Series(cls_t, name="trajectory"), pd.Series(cls_a, name="nearest anchor")))
    return ade


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--n_clips", type=int, default=3000)
    p.add_argument("--t0s_per_clip", type=int, default=8)
    p.add_argument("--k_sweep", default="")
    p.add_argument("--train_clips_for_sweep", type=int, default=6000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--methods", default="", help="compare codebook constructions at K = len(anchors)")
    args = p.parse_args()

    anchors = np.load(args.anchors)["anchors"]
    horizon = anchors.shape[1]
    a_cls = maneuver(anchors)
    print("anchor classes:", {c: int((a_cls == c).sum()) for c in ["straight", "left", "right", "stop"]})

    val = trajectories(args.data_root, "val", args.n_clips, args.t0s_per_clip, horizon, args.workers, seed=1)
    if args.methods:
        fit = trajectories(args.data_root, "train", args.train_clips_for_sweep, args.t0s_per_clip, horizon, args.workers, seed=0)
        cls_v = maneuver(val)
        rows = []
        for method in args.methods.split(","):
            cb = build_codebook(method, fit, len(anchors))
            np.save(f"codebook_{method}.npy", cb)
            dist = mean_l2(val, cb)
            ade, near = dist.min(1), dist.argmin(1)
            fde = np.linalg.norm(val[:, -1] - cb[near][:, -1], axis=-1)
            same = maneuver(cb)[near] == cls_v
            row = {"method": method, "ADE": ade.mean(), "ADE_p90": np.percentile(ade, 90), "FDE": fde.mean(),
                   "same_class": same.mean(), "used": len(np.unique(near))}
            for c in ["straight", "left", "right", "stop"]:
                m = cls_v == c
                row[f"{c}_ADE"] = ade[m].mean()
                row[f"{c}_same"] = same[m].mean()
            row.update({f"n_{c}": int((maneuver(cb) == c).sum()) for c in ["straight", "left", "right", "stop"]})
            rows.append(row)
            print(row, flush=True)
        pd.set_option("display.width", 250)
        print("\n== codebook constructions (val, K = %d)\n" % len(anchors) + pd.DataFrame(rows).round(3).to_string(index=False))
        return
    train = trajectories(args.data_root, "train", args.n_clips, args.t0s_per_clip, horizon, args.workers, seed=2)
    report("train (held-out t0s / clips not used for k-means)", train, anchors)
    report("val", val, anchors)

    if args.k_sweep:
        fit = trajectories(args.data_root, "train", args.train_clips_for_sweep, args.t0s_per_clip, horizon, args.workers, seed=0)
        cls_v = maneuver(val)
        rows = []
        for k in [int(x) for x in args.k_sweep.split(",")]:
            cb = kmeans(fit, k, seed=0)
            ade = mean_l2(val, cb).min(1)
            row = {"K": k, "val_ADE_mean": ade.mean(), "val_ADE_p90": np.percentile(ade, 90)}
            for c in ["straight", "left", "right"]:
                row[f"{c}_ADE"] = ade[cls_v == c].mean()
            row.update({f"n_{c}_anchors": int((maneuver(cb) == c).sum()) for c in ["left", "right"]})
            rows.append(row)
            print(row, flush=True)
        print("\n== K sweep (val)\n" + pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
