#!/usr/bin/env python3
"""Filter a route manifest by basic XML validity.

The CARLA leaderboard route parser treats XML waypoints as sparse keypoints and
interpolates between adjacent points. Routes with fewer than two keypoints cannot
produce a valid trajectory, so they should not be sent to collection.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def count_waypoints(route: ET.Element) -> int:
    return len(route.findall("./waypoints/position"))


def route_info(xml_path: Path) -> tuple[str, str, int]:
    root = ET.parse(xml_path).getroot()
    route = root.find("route")
    if route is None:
        raise ValueError("missing <route>")
    return route.get("id", ""), route.get("town", ""), count_waypoints(route)


def read_manifest(path: Path) -> list[Path]:
    routes: list[Path] = []
    for line in path.read_text().splitlines():
        item = line.split("#", 1)[0].strip()
        if item:
            routes.append(Path(item))
    return routes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes-file", required=True, type=Path)
    parser.add_argument("--valid-out", required=True, type=Path)
    parser.add_argument("--invalid-out", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--min-waypoints", type=int, default=2)
    args = parser.parse_args()

    valid: list[str] = []
    invalid: list[str] = []
    buckets: Counter[str] = Counter()

    for path in read_manifest(args.routes_file):
        try:
            route_id, town, n_waypoints = route_info(path)
            bucket = "valid" if n_waypoints >= args.min_waypoints else "invalid"
            line = f"{path}\t{route_id}\t{town}\twaypoints={n_waypoints}"
        except Exception as exc:
            bucket = "invalid"
            line = f"{path}\tparse_error={exc}"

        buckets[bucket] += 1
        if bucket == "valid":
            valid.append(str(path))
        else:
            invalid.append(line)

    args.valid_out.parent.mkdir(parents=True, exist_ok=True)
    args.invalid_out.parent.mkdir(parents=True, exist_ok=True)
    args.valid_out.write_text("\n".join(valid) + ("\n" if valid else ""))
    args.invalid_out.write_text("\n".join(invalid) + ("\n" if invalid else ""))

    summary = [
        f"routes_total={buckets['valid'] + buckets['invalid']}",
        f"valid_min_waypoints_{args.min_waypoints}={buckets['valid']}",
        f"invalid_lt_min_waypoints_or_parse_error={buckets['invalid']}",
    ]
    summary_text = "\n".join(summary) + "\n"
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(summary_text)
    sys.stderr.write(summary_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
