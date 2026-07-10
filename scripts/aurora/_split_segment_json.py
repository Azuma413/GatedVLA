#!/usr/bin/env python
"""Split a keyframe segment JSON (meta-action -> list of segments) into N shards by
clip_id, for parallel CoC labeling (one shard per tile-local vLLM server). Segments
of a clip stay together (video-load locality); clips are round-robin assigned so
shards are balanced. Structure (meta-action keys) is preserved in each shard.

  python _split_segment_json.py --input seg.json --num-shards 12 --out-dir shards/
    -> shards/segments_shard_00.json .. segments_shard_11.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    clips = sorted({seg["clip_id"] for segs in data.values() for seg in segs})
    shard_of = {c: i % args.num_shards for i, c in enumerate(clips)}

    shards = [defaultdict(list) for _ in range(args.num_shards)]
    n_seg = 0
    for meta_action, segs in data.items():
        for seg in segs:
            shards[shard_of[seg["clip_id"]]][meta_action].append(seg)
            n_seg += 1

    for i, shard in enumerate(shards):
        # Keep every meta-action key present (empty lists ok) for schema stability.
        full = {k: shard.get(k, []) for k in data}
        p = out / f"segments_shard_{i:02d}.json"
        p.write_text(json.dumps(full))
        print(f"  shard {i:02d}: {sum(len(v) for v in full.values())} segments -> {p}")

    print(f"total: {len(clips)} clips / {n_seg} segments over {args.num_shards} shards")


if __name__ == "__main__":
    main()
