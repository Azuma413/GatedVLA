#!/usr/bin/env python3
"""Split a route manifest into standard and heavy route manifests.

This is a pre-flight filter for local collection. CARLA/Py123D runtime is not
well predicted by XML validity alone: some towns and long sparse routes advance
very slowly with the full sensor suite. This tool keeps those routes out of the
normal queue so they can be collected separately with different settings.
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def read_manifest(path: Path) -> list[Path]:
    routes: list[Path] = []
    for line in path.read_text().splitlines():
        item = line.split("#", 1)[0].strip()
        if item:
            routes.append(Path(item))
    return routes


def route_info(xml_path: Path) -> tuple[str, str, int, float]:
    root = ET.parse(xml_path).getroot()
    route = root.find("route")
    if route is None:
        raise ValueError("missing <route>")

    positions = route.findall("./waypoints/position")
    points = [
        (float(pos.get("x", "0")), float(pos.get("y", "0")))
        for pos in positions
    ]
    distance_m = 0.0
    for prev, cur in zip(points, points[1:]):
        distance_m += math.hypot(cur[0] - prev[0], cur[1] - prev[1])

    return (
        route.get("id", xml_path.stem),
        route.get("town", ""),
        len(positions),
        distance_m,
    )


def parse_towns(value: str) -> set[str]:
    return {town.strip() for town in value.split(",") if town.strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes-file", required=True, type=Path)
    parser.add_argument("--standard-out", required=True, type=Path)
    parser.add_argument("--heavy-out", required=True, type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument(
        "--heavy-towns",
        default="Town12,Town13,Town15",
        help="Comma-separated town names always classified as heavy.",
    )
    parser.add_argument(
        "--heavy-route-ids",
        default="",
        help="Comma-separated route ids always classified as heavy.",
    )
    parser.add_argument("--max-standard-distance-m", type=float, default=250.0)
    parser.add_argument("--max-standard-waypoints", type=int, default=120)
    args = parser.parse_args()

    heavy_towns = parse_towns(args.heavy_towns)
    heavy_route_ids = parse_towns(args.heavy_route_ids)
    standard: list[str] = []
    heavy: list[str] = []
    report: list[str] = [
        "bucket\tpath\troute_id\ttown\twaypoints\tdistance_m\treason"
    ]
    buckets: Counter[str] = Counter()

    for path in read_manifest(args.routes_file):
        try:
            route_id, town, n_waypoints, distance_m = route_info(path)
            reasons: list[str] = []
            if town in heavy_towns:
                reasons.append(f"heavy_town:{town}")
            if route_id in heavy_route_ids:
                reasons.append(f"heavy_route_id:{route_id}")
            if distance_m > args.max_standard_distance_m:
                reasons.append(f"distance>{args.max_standard_distance_m:g}m")
            if n_waypoints > args.max_standard_waypoints:
                reasons.append(f"waypoints>{args.max_standard_waypoints}")
            bucket = "heavy" if reasons else "standard"
            reason = ",".join(reasons) if reasons else "standard"
            line = (
                f"{bucket}\t{path}\t{route_id}\t{town}\t{n_waypoints}\t"
                f"{distance_m:.1f}\t{reason}"
            )
        except Exception as exc:
            bucket = "heavy"
            line = f"heavy\t{path}\t\t\t\t\tparse_error:{exc}"

        buckets[bucket] += 1
        report.append(line)
        if bucket == "heavy":
            heavy.append(str(path))
        else:
            standard.append(str(path))

    args.standard_out.parent.mkdir(parents=True, exist_ok=True)
    args.heavy_out.parent.mkdir(parents=True, exist_ok=True)
    args.standard_out.write_text(
        "\n".join(standard) + ("\n" if standard else "")
    )
    args.heavy_out.write_text("\n".join(heavy) + ("\n" if heavy else ""))

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text("\n".join(report) + "\n")

    summary = [
        f"routes_total={buckets['standard'] + buckets['heavy']}",
        f"standard={buckets['standard']}",
        f"heavy={buckets['heavy']}",
        f"heavy_towns={','.join(sorted(heavy_towns))}",
        f"heavy_route_ids={','.join(sorted(heavy_route_ids))}",
        f"max_standard_distance_m={args.max_standard_distance_m:g}",
        f"max_standard_waypoints={args.max_standard_waypoints}",
    ]
    summary_text = "\n".join(summary) + "\n"
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(summary_text)
    sys.stderr.write(summary_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
