#!/usr/bin/env python3
"""Group single-route XMLs into batched route files (one <routes> with many
<route> children), grouped by town.

Running one CARLA server per batch (and letting the leaderboard evaluator loop
over the routes in the batch) lets each server process many routes WITHOUT a
restart. Grouping by town also means the client only load_world()s once per
batch (the 2nd+ routes keep the already-loaded town), which keeps the
TrafficManager valid -- switching towns mid-batch is avoided.

Usage:
  build_route_batches.py --routes-file list.txt --out-dir batches/ [--chunk N]
  build_route_batches.py --routes-dir dir/    --out-dir batches/ [--chunk N]

Writes batches/<town>_<idx>.xml and prints one batch path per line (a manifest
suitable for ROUTES_FILE of collect_py123d_parallel.sh).
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def route_town(xml_path: Path) -> str:
    route = ET.parse(xml_path).getroot().find("route")
    if route is None or "town" not in route.attrib:
        raise SystemExit(f"No <route town=...> in {xml_path}")
    return route.attrib["town"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes-file", type=Path, help="text file, one route XML path per line")
    ap.add_argument("--routes-dir", type=Path, help="directory of *.xml routes")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--chunk", type=int, default=0,
                    help="max routes per batch (0 = all routes of a town in one batch)")
    args = ap.parse_args()

    paths: list[Path] = []
    if args.routes_file:
        for line in args.routes_file.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                paths.append(Path(line))
    elif args.routes_dir:
        paths = sorted(args.routes_dir.glob("*.xml"))
    else:
        raise SystemExit("Set --routes-file or --routes-dir")
    if not paths:
        raise SystemExit("No routes found")

    by_town: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        by_town[route_town(p)].append(p)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[Path] = []
    for town in sorted(by_town):
        routes = by_town[town]
        chunk = args.chunk if args.chunk > 0 else len(routes)
        for i in range(0, len(routes), chunk):
            group = routes[i:i + chunk]
            root = ET.Element("routes")
            for rp in group:
                root.append(ET.parse(rp).getroot().find("route"))
            out = args.out_dir / f"{town}_{i // chunk:03d}.xml"
            ET.ElementTree(root).write(out, encoding="UTF-8", xml_declaration=True)
            manifest.append(out)

    for m in manifest:
        print(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
