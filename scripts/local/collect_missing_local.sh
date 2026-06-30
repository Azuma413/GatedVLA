#!/usr/bin/env bash
# Orchestrate on-prem data collection for missing routes.
#
# Steps:
#   1. determine the missing-route list:
#        - MISSING_LIST set  → use that frozen list directly (NO dataset needed).
#                              This is the on-prem path: the list is produced on
#                              Polaris by scripts/local/export_missing_manifest.sh
#                              and only the ~50KB text file is transferred.
#        - MISSING_LIST unset → run find_missing_routes.py, which SCANS the full
#                              multi-TB clip tree (Polaris-only fallback).
#   2. build_route_batches.py  → group per-town into batch XMLs
#   3. collect_local_parallel.sh → run N_WORKERS parallel CARLA+client pairs
#   4. Print summary; repeat-resume is safe (already-collected routes are skipped
#      per batch via the leaderboard checkpoint_endpoint.json + --resume).
#
# DRY-RUN by default.  Pass LAUNCH=1 to actually start collection.
#
# Usage on the on-prem server (route XMLs are git-tracked, dataset NOT needed):
#   MISSING_LIST=logs/missing_manifest/missing_routes.txt \
#     bash scripts/local/collect_missing_local.sh                 # dry-run
#   MISSING_LIST=logs/missing_manifest/missing_routes.txt \
#     LAUNCH=1 bash scripts/local/collect_missing_local.sh        # run
#
# Resume (re-run with the SAME WORK to reuse batches; done routes are skipped):
#   WORK=logs/local_collection/<ts> MISSING_LIST=... LAUNCH=1 \
#     bash scripts/local/collect_missing_local.sh
#
# Run on fewer GPUs (e.g. only GPU 0):
#   N_WORKERS=1 MISSING_LIST=... LAUNCH=1 bash scripts/local/collect_missing_local.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Pre-computed missing-route list (frozen on Polaris). When set, find_missing_routes
# is skipped entirely and this list is used as-is — no multi-TB dataset required.
MISSING_LIST=${MISSING_LIST:-}

# Fallback (Polaris-only): target route list scanned by find_missing_routes.
# Latest production campaign's filtered list by default.
TARGET=${TARGET:-$(ls -dt logs/production_collection/*/routes_filtered.txt 2>/dev/null | head -1 || true)}

# Routes per CARLA session.
# Keep at 2: CARLA reliably crashes entering the 3rd route of a session
# (same constraint as resume_missing_collection.sh on Polaris).
CHUNK=${CHUNK:-2}

N_WORKERS=${N_WORKERS:-3}          # parallel CARLA+client pairs (= GPUs used)
N_GPUS=${N_GPUS:-${N_WORKERS}}
BASE_PORT=${BASE_PORT:-2000}
OUTPUT_ROOT=${OUTPUT_ROOT:-data}
QUALITY_LEVEL=${QUALITY_LEVEL:-Epic}
MAX_RETRIES=${MAX_RETRIES:-2}
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-7200}
CARLA_READY_TIMEOUT=${CARLA_READY_TIMEOUT:-180}
LAUNCH=${LAUNCH:-0}

WORK=${WORK:-logs/local_collection/$(date +%Y%m%d_%H%M%S)}

# ---------------------------------------------------------------------------
# Resolve helper paths
# ---------------------------------------------------------------------------
FIND_MISSING="${REPO_DIR}/scripts/tools/find_missing_routes.py"
BUILD_BATCHES="${REPO_DIR}/scripts/tools/build_route_batches.py"
COLLECT_SH="${REPO_DIR}/scripts/local/collect_local_parallel.sh"

# Python: prefer repo venv.
VENV_PYTHON=${VENV_PYTHON:-${REPO_DIR}/.venv/bin/python3}
if [ ! -x "${VENV_PYTHON}" ]; then
    VENV_PYTHON=$(command -v python3)
fi

# CARLA binary: try common locations.
if [ -z "${CARLA_BIN:-}" ]; then
    for _candidate in \
        "${REPO_DIR}/3rd_party/CARLA_0915/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" \
        "${REPO_DIR}/3rd_party/CARLA_0.9.15/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" \
        "/opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"; do
        if [ -x "${_candidate}" ]; then
            CARLA_BIN="${_candidate}"
            break
        fi
    done
fi
CARLA_BIN=${CARLA_BIN:-}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
[ -f "${BUILD_BATCHES}" ] || { echo "ERROR: ${BUILD_BATCHES} not found" >&2; exit 2; }
[ -f "${COLLECT_SH}" ]    || { echo "ERROR: ${COLLECT_SH} not found" >&2; exit 2; }
[ -x "${VENV_PYTHON}" ]   || { echo "ERROR: Python not found: ${VENV_PYTHON}" >&2; exit 2; }

# Either a frozen MISSING_LIST (on-prem) or a TARGET to scan (Polaris fallback).
if [ -n "${MISSING_LIST}" ]; then
    [ -f "${MISSING_LIST}" ] || { echo "ERROR: MISSING_LIST not found: ${MISSING_LIST}" >&2; exit 2; }
else
    [ -f "${FIND_MISSING}" ] || { echo "ERROR: ${FIND_MISSING} not found" >&2; exit 2; }
    if [ -z "${TARGET}" ] || [ ! -f "${TARGET}" ]; then
        echo "ERROR: neither MISSING_LIST nor a valid TARGET is set." >&2
        echo "  On-prem: set MISSING_LIST=<frozen list from export_missing_manifest.sh>." >&2
        echo "  Polaris: set TARGET=<path to a routes_filtered.txt> (scans full dataset)." >&2
        exit 2
    fi
fi

mkdir -p "${WORK}/batches"

echo "=== On-prem missing-route collection ==="
if [ -n "${MISSING_LIST}" ]; then
    echo "MISSING_LIST  = ${MISSING_LIST}  (frozen; dataset NOT scanned)"
else
    echo "TARGET        = ${TARGET}  (scanning full dataset via find_missing_routes)"
fi
echo "CHUNK         = ${CHUNK} routes/batch"
echo "N_WORKERS     = ${N_WORKERS} (GPUs: 0..$((N_GPUS-1)))"
echo "QUALITY_LEVEL = ${QUALITY_LEVEL}"
echo "OUTPUT_ROOT   = ${OUTPUT_ROOT}"
echo "WORK          = ${WORK}"
echo "LAUNCH        = ${LAUNCH}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Resolve the missing-route list
# ---------------------------------------------------------------------------
MISSING="${WORK}/routes_filtered.txt"
if [ -n "${MISSING_LIST}" ]; then
    # On-prem: use the frozen list as-is (no dataset scan).
    cp "${MISSING_LIST}" "${MISSING}"
else
    # Polaris fallback: scan the full clip tree to find what is still missing.
    REPO_DIR="${REPO_DIR}" "${VENV_PYTHON}" "${FIND_MISSING}" \
        --target "${TARGET}" \
        --out "${MISSING}"
fi

n_missing=$(grep -c '[^[:space:]]' "${MISSING}" || true)
if [ "${n_missing}" -eq 0 ]; then
    echo ">> Nothing missing — list is empty."
    exit 0
fi
echo "Missing routes: ${n_missing}"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Build batch XMLs (grouped per town, CHUNK routes each)
# ---------------------------------------------------------------------------
"${VENV_PYTHON}" "${BUILD_BATCHES}" \
    --routes-file "${MISSING}" \
    --out-dir "${WORK}/batches" \
    --chunk "${CHUNK}" \
    > "${WORK}/batch_manifest.txt"

n_batches=$(grep -c '[^[:space:]]' "${WORK}/batch_manifest.txt" || true)

# Estimate total time
"${VENV_PYTHON}" - "${MISSING}" "${N_WORKERS}" "${QUALITY_LEVEL}" <<'PY'
import sys, xml.etree.ElementTree as ET, math
from collections import defaultdict

target_file  = sys.argv[1]
n_workers    = int(sys.argv[2])
quality      = sys.argv[3]

# Empirical avg minutes/route per town (Polaris A100 Epic quality)
avg_min = {
    "Town15": 12.6, "Town12": 24.1, "Town13": 19.5, "Town10HD": 18.0,
    "Town06": 10.5, "Town07":  8.8, "Town03":  8.4, "Town05": 10.3, "Town04": 9.3,
}
default_avg = 12.0
# A6000 is estimated ~2x faster than A100 for CARLA Epic rendering
speedup = 2.0 if quality == "Epic" else 3.5

town_counts = defaultdict(int)
with open(target_file) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            route = ET.parse(line).getroot().find('.//route')
            town = route.get('town', 'unknown') if route is not None else 'unknown'
            town_counts[town] += 1
        except Exception:
            pass

total_min = 0
print("Time estimate per town (A6000, sequential):")
for town, count in sorted(town_counts.items(), key=lambda x: -x[1]):
    mins = avg_min.get(town, default_avg) / speedup * count
    total_min += mins
    print(f"  {town:<10}: {count:4d} routes × {avg_min.get(town, default_avg)/speedup:.1f} min = {mins/60:.1f} h")

print(f"\nTotal sequential : {total_min/60:.1f} h ({total_min/60/24:.1f} days)")
print(f"With {n_workers} workers  : {total_min/60/n_workers:.1f} h ({total_min/60/n_workers/24:.2f} days)")
PY

echo ""
echo "Batches: ${n_batches}  (${CHUNK} routes/batch)"
echo "Batch manifest: ${WORK}/batch_manifest.txt"
echo ""

if [ "${LAUNCH}" != "1" ]; then
    _ml_hint=""
    [ -n "${MISSING_LIST}" ] && _ml_hint="MISSING_LIST=${MISSING_LIST} "
    echo ">> DRY-RUN. Review the plan above."
    echo ">> Re-run with LAUNCH=1 to start collection:"
    echo ">>   ${_ml_hint}LAUNCH=1 bash scripts/local/collect_missing_local.sh"
    echo ">>   ${_ml_hint}WORK=${WORK} LAUNCH=1 bash scripts/local/collect_missing_local.sh  # resume same work dir"
    exit 0
fi

# Verify CARLA binary before launching
if [ -z "${CARLA_BIN}" ] || [ ! -x "${CARLA_BIN}" ]; then
    echo "ERROR: CARLA binary not found. Set CARLA_BIN=<path to CarlaUE4-Linux-Shipping>" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Step 3: Launch parallel collection
# ---------------------------------------------------------------------------
echo "Starting collection... (Ctrl-C to abort cleanly)"
echo "Logs: ${WORK}/"
echo ""

ROUTES_FILE="${WORK}/batch_manifest.txt" \
N_WORKERS="${N_WORKERS}" \
N_GPUS="${N_GPUS}" \
BASE_PORT="${BASE_PORT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
QUALITY_LEVEL="${QUALITY_LEVEL}" \
MAX_RETRIES="${MAX_RETRIES}" \
CLIENT_TIMEOUT="${CLIENT_TIMEOUT}" \
CARLA_READY_TIMEOUT="${CARLA_READY_TIMEOUT}" \
CARLA_BIN="${CARLA_BIN}" \
VENV_PYTHON="${VENV_PYTHON}" \
JOB_ID="$(basename "${WORK}")" \
bash "${COLLECT_SH}"

echo ""
echo "=== Done. Run find_missing_routes.py again to check completion: ==="
echo "  REPO_DIR=${REPO_DIR} ${VENV_PYTHON} ${FIND_MISSING} \\"
echo "    --target ${TARGET} --out /tmp/still_missing.txt"
