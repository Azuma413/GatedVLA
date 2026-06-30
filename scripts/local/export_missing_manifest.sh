#!/usr/bin/env bash
# RUN ON POLARIS (where the 1TB of collected clips lives).
#
# Produces a COMPACT manifest of routes that still lack a complete clip, so the
# on-prem server can collect them WITHOUT a copy of the multi-TB dataset.
# find_missing_routes.py needs to scan the full clip tree, which only exists on
# Polaris; this script freezes its result into one small text file to transfer.
#
# Output (under OUT_DIR, default logs/missing_manifest/<timestamp>/):
#   missing_routes.txt   route XML paths still uncollected (one per line, ~50KB)
#   missing_summary.txt  human-readable per-town / per-scenario counts
#
# The route XML files themselves are git-tracked, so the on-prem clone already
# has them — only missing_routes.txt needs to be transferred.
#
# Usage (on Polaris):
#   bash scripts/local/export_missing_manifest.sh
#   TARGET=logs/production_collection/<campaign>/routes_filtered.txt \
#     bash scripts/local/export_missing_manifest.sh

set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "${REPO_DIR}"
export REPO_DIR

# Full route set targeted by the production campaign (latest campaign by default).
TARGET=${TARGET:-$(ls -dt logs/production_collection/*/routes_filtered.txt 2>/dev/null | head -1 || true)}
OUT_DIR=${OUT_DIR:-logs/missing_manifest/$(date +%Y%m%d_%H%M%S)}

# Python: prefer lead.sif on Polaris, else the venv / system python.
PYTHON=${PYTHON:-}
if [ -z "${PYTHON}" ]; then
    if [ -x "${REPO_DIR}/.venv/bin/python3" ]; then
        PYTHON="${REPO_DIR}/.venv/bin/python3"
    else
        PYTHON=$(command -v python3)
    fi
fi

[ -n "${TARGET}" ] && [ -f "${TARGET}" ] || {
    echo "ERROR: TARGET route list not found: '${TARGET}'" >&2
    echo "  Set TARGET=<path to a production_collection/*/routes_filtered.txt>" >&2
    exit 2
}

mkdir -p "${OUT_DIR}"
MISSING="${OUT_DIR}/missing_routes.txt"
SUMMARY="${OUT_DIR}/missing_summary.txt"

echo "=== Export missing-route manifest (run on Polaris) ==="
echo "TARGET = ${TARGET}"
echo "OUT    = ${OUT_DIR}"
echo ""

# 1. Determine routes lacking a complete clip (scans the full clip tree).
"${PYTHON}" scripts/tools/find_missing_routes.py \
    --target "${TARGET}" \
    --out "${MISSING}"

n_missing=$(grep -c '[^[:space:]]' "${MISSING}" || true)

# 2. Compact human-readable summary (per town / per scenario).
"${PYTHON}" - "${MISSING}" "${TARGET}" > "${SUMMARY}" <<'PY'
import sys, xml.etree.ElementTree as ET
from collections import defaultdict

missing_file, target_file = sys.argv[1], sys.argv[2]

def town_of(path):
    try:
        r = ET.parse(path).getroot().find('.//route')
        return r.get('town', 'unknown') if r is not None else 'unknown'
    except Exception:
        return 'parse_error'

def scenario_of(path):
    # data/data_routes/lead/<scenario>/route_xxx.xml
    parts = path.replace('data/data_routes/lead/', '').split('/')
    return parts[0] if len(parts) > 1 else 'unknown'

missing = [l.strip() for l in open(missing_file) if l.strip()]
n_target = sum(1 for l in open(target_file) if l.strip())

by_town = defaultdict(int)
by_scn  = defaultdict(int)
for p in missing:
    by_town[town_of(p)] += 1
    by_scn[scenario_of(p)] += 1

print(f"target routes : {n_target}")
print(f"missing routes: {len(missing)}  ({100*len(missing)/n_target:.1f}%)")
print()
print("Missing per town:")
for t, c in sorted(by_town.items(), key=lambda x: -x[1]):
    print(f"  {t:<10} {c}")
print()
print("Missing per scenario:")
for s, c in sorted(by_scn.items(), key=lambda x: -x[1]):
    print(f"  {s:<40} {c}")
PY

cat "${SUMMARY}"
echo ""
echo "=== Manifest written ==="
echo "  ${MISSING}  ($(du -h "${MISSING}" | cut -f1))"
echo "  ${SUMMARY}"
echo ""
echo "Transfer ONLY this small file to the on-prem server (route XMLs are git-tracked):"
echo "  rsync -avz ${MISSING} <ONPREM_HOST>:/path/to/VLA4AD_Research/logs/missing_manifest/"
echo ""
echo "Then on-prem, collect against it (no 1TB dataset needed):"
echo "  MISSING_LIST=logs/missing_manifest/missing_routes.txt \\"
echo "    LAUNCH=1 bash scripts/local/collect_missing_local.sh"
