"""Prepare PAI-AV metadata for QwenDrivePI training (egomotion only; no video decoding except timestamps).

Writes into --out_dir:
    norm_stats.json        per-dimension q01/q99 of state (10) and action (2) over sampled train t0s
    eval_samples_val.json  fixed open-loop eval samples from the val split, balanced left/right/straight

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
    p.add_argument("--state_mode", default="drivemoe", choices=["drivemoe", "traj"])
    p.add_argument("--action_mode", default="waypoints", choices=["waypoints", "unicycle"])
    p.add_argument("--no_state_target_point", action="store_true", help="traj state without the target point")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sampler = DriveSampler(
        state_horizon=args.state_horizon, action_horizon=args.action_horizon, target_distance=args.target_distance,
        state_mode=args.state_mode, action_mode=args.action_mode, state_target_point=not args.no_state_target_point,
    )

    train = load_clip_table(args.data_root, "train")
    clips = train.sample(n=min(args.stats_clips, len(train)), random_state=args.seed).sort_values("chunk")
    shards = np.array_split(np.arange(len(clips)), args.workers)
    jobs = [(args.data_root, clips.iloc[s], sampler, args.t0s_per_clip, args.seed + k) for k, s in enumerate(shards)]
    with Pool(args.workers) as pool:
        parts = pool.map(_stats_worker, jobs)
    state_names, action_names = sampler.state_names, sampler.action_names
    states = np.concatenate([s for s, _ in parts]).reshape(-1, len(state_names))
    actions = np.concatenate([a for _, a in parts]).reshape(-1, len(action_names))
    pct = lambda x, q: np.percentile(x, q, axis=0).tolist()  # noqa: E731
    stats = {
        "state": {"q01": pct(states, 1), "q99": pct(states, 99), "names": state_names},
        "action": {"q01": pct(actions, 1), "q99": pct(actions, 99), "names": action_names},
        "num_samples": int(len(states) // args.state_horizon),
        "sampler": vars(sampler),
    }
    if sampler.action_mode == "unicycle":  # Alpamayo normalises accel / curvature by mean and std
        stats["action"].update(norm="mean_std", mean=actions.mean(0).tolist(), std=actions.std(0).tolist())
    (out / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    print(pd.DataFrame({"q01": stats["state"]["q01"], "q99": stats["state"]["q99"]}, index=state_names))
    print(pd.DataFrame({k: v for k, v in stats["action"].items() if k in ("q01", "q99", "mean", "std")}, index=action_names))

    val = load_clip_table(args.data_root, args.eval_split)
    samples = build_eval_index(args.data_root, val, sampler, per_class=args.eval_per_class, seed=args.seed)
    (out / f"eval_samples_{args.eval_split}.json").write_text(json.dumps(samples, indent=1))
    print(f"eval samples: {len(samples)}", pd.Series([s["maneuver"] for s in samples]).value_counts().to_dict())


if __name__ == "__main__":
    main()
