#!/usr/bin/env python3
"""Find lead routes that still lack a COMPLETE collected clip.

A clip dir is named  <Town>_Rep-1_<batch>_route<idx>_<MM_DD_HH_MM_SS> .  The
<batch>/<idx> part only resolves to an original route id (route_000xxx) via the
per-campaign batch XMLs under logs/production_collection/<campaign>/batches/,
and the batching changed across campaigns (CHUNK 24 -> 12 -> 10), so the same
<batch>_route<idx> means different routes in different campaigns.  We therefore
resolve each clip by the campaign that was live at the clip's timestamp.

"Complete" = has all 4 cameras + ego_state + sync arrow files.  (lidar is
intentionally absent in part of the dataset and is not required.)

Prints, to --out, the route XML paths (one per line) from the target set that
have no complete clip.  Use that file to drive build_route_batches.py.
"""
import argparse, os, re, glob, datetime, sys, json
import xml.etree.ElementTree as ET

REPO = os.environ.get("REPO_DIR", "/lus/eagle/projects/RobustViT/hiratsuka/VLA4AD_Research")
CAMP_DIR = os.path.join(REPO, "logs/production_collection")
CLIP_DIR = os.path.join(REPO, "data/carla_alpamayo_4c_py123d/logs/carla_train")

REQUIRED = ["camera.ftcam_f0.arrow", "camera.ftcam_l0.arrow", "camera.ftcam_r0.arrow",
            "camera.ftcam_tele_f0.arrow",
            "ego_state_se3.arrow", "sync.arrow"]
CLIP_RE = re.compile(r'^(?P<town>Town[0-9A-Za-z]+)_Rep-?-?\d+_(?P<batch>.+)_route(?P<idx>\d+)_(?P<ts>\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$')


def load_campaigns():
    camps = []
    for name in sorted(os.listdir(CAMP_DIR)):
        m = re.match(r'(\d{8})_(\d{6})', name)
        if not m:
            continue
        dt = datetime.datetime.strptime(m.group(0)[:15], "%Y%m%d_%H%M%S")
        bmap = {}
        for xml in glob.glob(os.path.join(CAMP_DIR, name, "batches", "*.xml")):
            bn = os.path.splitext(os.path.basename(xml))[0]
            try:
                bmap[bn] = [r.get("id") for r in ET.parse(xml).getroot().iter("route")]
            except Exception:
                pass
        camps.append((dt, name, bmap))
    camps.sort()
    return camps


def route_id_from_json(json_path):
    """Return route id (e.g. 'route_000924') from a clip's sibling summary json,
    or None if the json is absent/unreadable or lacks a route_id."""
    try:
        with open(json_path) as f:
            raw = json.load(f).get("route_id", "")
    except (OSError, ValueError):
        return None
    m = re.search(r'route_\d+', raw)
    return m.group(0) if m else None


def collected_complete_ids(camps, year):
    def camp_for(ts):
        chosen = None
        for c in camps:
            if c[0] <= ts:
                chosen = c
            else:
                break
        return chosen

    ids = set()
    for d in os.listdir(CLIP_DIR):
        p = os.path.join(CLIP_DIR, d)
        if not os.path.isdir(p):
            continue
        if not all(os.path.isfile(os.path.join(p, f)) for f in REQUIRED):
            continue
        # Authoritative source: the clip's sibling <name>.json records route_id.
        # Prefer it over the campaign-batch reverse lookup, which resolves to the
        # wrong route for clips collected after the last campaign snapshot present
        # in logs/production_collection (e.g. the carla_train_sub back-fill).
        rid = route_id_from_json(p + ".json")
        if rid:
            ids.add(rid)
            continue
        m = CLIP_RE.match(d)
        if not m:
            continue
        batch, idx = m.group("batch"), int(m.group("idx"))
        if re.match(r'route_\d+$', batch):
            ids.add(batch)
            continue
        ts = datetime.datetime.strptime(f"{year}_" + m.group("ts"), "%Y_%m_%d_%H_%M_%S")
        c = camp_for(ts)
        rid = None
        if c and c[2].get(batch) and idx < len(c[2][batch]):
            rid = c[2][batch][idx]
        if rid is None:  # fallback: any campaign holding this batch+idx
            for _, _, bmap in reversed(camps):
                if bmap.get(batch) and idx < len(bmap[batch]):
                    rid = bmap[batch][idx]
                    break
        if rid:
            ids.add(rid)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    help="text file of target route XML paths (one per line)")
    ap.add_argument("--out", required=True, help="write missing route XML paths here")
    ap.add_argument("--year", type=int, default=datetime.date.today().year,
                    help="year for clip timestamps (clip names omit year)")
    args = ap.parse_args()

    target = {}
    with open(args.target) as f:
        for line in f:
            line = line.strip()
            if line:
                target[os.path.splitext(os.path.basename(line))[0]] = line

    camps = load_campaigns()
    have = collected_complete_ids(camps, args.year)
    missing = sorted(i for i in target if i not in have)
    with open(args.out, "w") as f:
        for i in missing:
            f.write(target[i] + "\n")

    print(f"target routes:    {len(target)}", file=sys.stderr)
    print(f"complete clips:   {len(have & set(target))}", file=sys.stderr)
    print(f"MISSING routes:   {len(missing)}  -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
