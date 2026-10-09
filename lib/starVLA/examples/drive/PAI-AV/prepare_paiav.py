"""Prepare PAI-AV metadata for QwenDrivePI training (egomotion only; no video decoding except timestamps).

Writes into --out_dir:
    norm_stats.json        per-dimension q01/q99 of state (10) and action (2) over sampled train t0s
    eval_samples_val.json  fixed open-loop eval samples from the val split, balanced left/right/straight
    anchors.npz            (--num_anchors > 0) AnchorVLA trajectory-pattern codebook: k-means centres of the
                           sampled train trajectories (metres) and the per-step std of the residual to the
                           nearest anchor

    python examples/drive/PAI-AV/prepare_paiav.py \
        --data_root /lus/flare/projects/RobustViT/hiratsuka/datasets/PhysicalAI-Autonomous-Vehicles \
        --out_dir ../../outputs/qwendrive_pi/meta
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from starVLA.dataloader.pai_av_drive import (
    ACTION_NAMES,
    STATE_NAMES,
    DriveSampler,
    build_eval_index,
    load_clip_table,
)


def _stats_worker(args):
    data_root, clips, sampler, t0s, seed = args
    from starVLA.dataloader.pai_av_drive import ClipStore, EgoTrack

    rng = np.random.default_rng(seed)
    store = ClipStore(data_root)
    states, actions = [], []
    for row in clips.itertuples():
        try:
            track = EgoTrack(store.egomotion(row.clip_id, row.chunk))
        except Exception:
            continue
        lo, hi = sampler.valid_range(track)
        if hi <= lo:
            continue
        for t0 in rng.uniform(lo, hi, size=t0s):
            s, a, _ = sampler.state_action(track, t0)
            states.append(s)
            actions.append(a)
    return np.concatenate(states), np.concatenate(actions)


def mean_l2(traj: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """AnchorVLA Eq. 3: mean point-wise L2 distance, (N, H, 2) x (K, H, 2) -> (N, K)."""
    out = np.empty((len(traj), len(anchors)), dtype=np.float32)
    for i in range(0, len(traj), 4096):
        out[i : i + 4096] = np.linalg.norm(traj[i : i + 4096, None] - anchors[None], axis=-1).mean(-1)
    return out


def kmeans(x: np.ndarray, k: int, iters: int = 100, seed: int = 0) -> np.ndarray:
    """Plain k-means (k-means++ init) on flattened trajectories."""
    rng = np.random.default_rng(seed)
    flat = x.reshape(len(x), -1).astype(np.float64)
    centres = [flat[rng.integers(len(flat))]]
    d2 = ((flat - centres[0]) ** 2).sum(1)
    for _ in range(1, k):
        centres.append(flat[rng.choice(len(flat), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((flat - centres[-1]) ** 2).sum(1))
    centres = np.stack(centres)
    for _ in range(iters):
        dist = (flat**2).sum(1)[:, None] - 2 * flat @ centres.T + (centres**2).sum(1)[None]
        assign = dist.argmin(1)
        new = np.stack([flat[assign == j].mean(0) if (assign == j).any() else centres[j] for j in range(k)])
        if np.allclose(new, centres):
            break
        centres = new
    return centres.reshape(k, *x.shape[1:]).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--stats_clips", type=int, default=6000)
    p.add_argument("--t0s_per_clip", type=int, default=8)
    p.add_argument("--eval_per_class", type=int, default=80)
    p.add_argument("--eval_split", default="val")
    p.add_argument("--state_horizon", type=int, default=5)
    p.add_argument("--action_horizon", type=int, default=10)
    p.add_argument("--target_distance", type=float, default=25.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--num_anchors", type=int, default=0, help="AnchorVLA codebook size K (0: skip)")
    p.add_argument("--min_residual_std", type=float, default=0.05, help="floor (m) of the residual std")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sampler = DriveSampler(
        state_horizon=args.state_horizon, action_horizon=args.action_horizon, target_distance=args.target_distance
    )

    train = load_clip_table(args.data_root, "train")
    clips = train.sample(n=min(args.stats_clips, len(train)), random_state=args.seed).sort_values("chunk")
    shards = np.array_split(np.arange(len(clips)), args.workers)
    jobs = [(args.data_root, clips.iloc[s], sampler, args.t0s_per_clip, args.seed + k) for k, s in enumerate(shards)]
    with Pool(args.workers) as pool:
        parts = pool.map(_stats_worker, jobs)
    states = np.concatenate([s for s, _ in parts]).reshape(-1, len(STATE_NAMES))
    actions = np.concatenate([a for _, a in parts]).reshape(-1, len(ACTION_NAMES))
    pct = lambda x, q: np.percentile(x, q, axis=0).tolist()  # noqa: E731
    stats = {
        "state": {"q01": pct(states, 1), "q99": pct(states, 99), "names": STATE_NAMES},
        "action": {"q01": pct(actions, 1), "q99": pct(actions, 99), "names": ACTION_NAMES},
        "num_samples": int(len(states) // args.state_horizon),
        "sampler": vars(sampler),
    }
    (out / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    print(pd.DataFrame({"q01": stats["state"]["q01"], "q99": stats["state"]["q99"]}, index=STATE_NAMES))
    print(pd.DataFrame({"q01": stats["action"]["q01"], "q99": stats["action"]["q99"]}, index=ACTION_NAMES))

    if args.num_anchors > 0:
        traj = actions.reshape(-1, args.action_horizon, len(ACTION_NAMES))
        anchors = kmeans(traj, args.num_anchors, seed=args.seed)
        nearest = mean_l2(traj, anchors).argmin(1)
        residual_std = np.maximum((traj - anchors[nearest]).std(0), args.min_residual_std)
        np.savez(
            out / "anchors.npz",
            anchors=anchors,
            residual_std=residual_std.astype(np.float32),
            action_q01=np.asarray(stats["action"]["q01"], np.float32),
            action_q99=np.asarray(stats["action"]["q99"], np.float32),
        )
        counts = np.bincount(nearest, minlength=args.num_anchors)
        print(f"anchors: {anchors.shape}, cluster sizes min/median/max {counts.min()}/{int(np.median(counts))}/"
              f"{counts.max()}, mean residual |r| {np.abs(traj - anchors[nearest]).mean():.3f} m")

    val = load_clip_table(args.data_root, args.eval_split)
    samples = build_eval_index(args.data_root, val, sampler, per_class=args.eval_per_class, seed=args.seed)
    (out / f"eval_samples_{args.eval_split}.json").write_text(json.dumps(samples, indent=1))
    print(f"eval samples: {len(samples)}", pd.Series([s["maneuver"] for s in samples]).value_counts().to_dict())


if __name__ == "__main__":
    main()
