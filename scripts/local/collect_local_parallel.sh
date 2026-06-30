#!/usr/bin/env bash
# On-prem parallel CARLA data collection.  No PBS, no mpiexec, no Apptainer.
# Runs N_WORKERS parallel CARLA+client pairs on one machine using multiple GPUs.
#
# DESIGN
#   - N_WORKERS background workers share a dynamic queue (file-based flock counter).
#   - Each worker owns one GPU and one fixed RPC port slot.
#   - Each batch XML is one CARLA session: CARLA starts, client runs, CARLA dies.
#     Keep CHUNK=2 (default in collect_missing_local.sh) to avoid the known
#     CARLA-crashes-after-2-routes issue (same as resume_missing_collection.sh).
#   - All workers write to one SUMMARY_JSONL via flock.
#
# Usage:
#   ROUTES_FILE=<manifest> bash scripts/local/collect_local_parallel.sh
#
# Key env vars (all optional except ROUTES_FILE):
#   ROUTES_FILE           text file of batch XML paths, one per line
#   N_WORKERS             parallel workers (default: 3)
#   N_GPUS                GPUs on this machine (default: N_WORKERS)
#   BASE_PORT             first RPC port; worker k uses BASE_PORT+k*10 (default: 2000)
#   OUTPUT_ROOT           output base dir relative to REPO_DIR (default: data)
#   QUALITY_LEVEL         CARLA quality: Epic|Low (default: Epic)
#   MAX_RETRIES           per-batch client retries before marking failed (default: 2)
#   CARLA_READY_TIMEOUT   seconds to wait for CARLA to accept connections (default: 180)
#   CLIENT_TIMEOUT        seconds for entire batch before timeout (default: 7200)
#   CARLA_BIN             path to CarlaUE4-Linux-Shipping binary
#   VENV_PYTHON           path to Python interpreter with LEAD installed
#   JOB_ID                log dir label (default: timestamp)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_WORKERS=${N_WORKERS:-3}
N_GPUS=${N_GPUS:-${N_WORKERS}}
BASE_PORT=${BASE_PORT:-2000}
OUTPUT_ROOT=${OUTPUT_ROOT:-data}
QUALITY_LEVEL=${QUALITY_LEVEL:-Epic}
MAX_RETRIES=${MAX_RETRIES:-2}
CARLA_READY_TIMEOUT=${CARLA_READY_TIMEOUT:-180}
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-7200}
ROUTES_FILE=${ROUTES_FILE:-}
JOB_ID=${JOB_ID:-$(date +%Y%m%d_%H%M%S)}
JOB_ROOT="${REPO_DIR}/logs/local_collection/${JOB_ID}"
SUMMARY_JSONL="${JOB_ROOT}/summary.jsonl"
BATCH_COUNTER="${JOB_ROOT}/.next_batch_idx"

# CARLA binary: try common locations in order.
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

# Python: prefer the repo venv, then system python3.
VENV_PYTHON=${VENV_PYTHON:-${REPO_DIR}/.venv/bin/python3}
if [ ! -x "${VENV_PYTHON}" ]; then
    VENV_PYTHON=$(command -v python3)
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

die() { log "ERROR: $*" >&2; exit 2; }

require_exec() { [ -x "$1" ] || die "Not found or not executable: $1"; }

# Atomically fetch-and-increment the shared batch counter.
# Prints the index that this caller owns (0-based).
next_batch_idx() {
    (
        flock -x 200
        local idx
        idx=$(cat "${BATCH_COUNTER}" 2>/dev/null || echo "0")
        echo $(( idx + 1 )) > "${BATCH_COUNTER}"
        printf '%s' "${idx}"
    ) 200>"${BATCH_COUNTER}.lock"
}

# Append one JSON line to the shared summary under a lock.
write_summary() {
    local json="$1"
    (
        flock -x 201
        printf '%s\n' "${json}" >> "${SUMMARY_JSONL}"
    ) 201>"${SUMMARY_JSONL}.lock"
}

# Wait until TCP host:port is open, or timeout.
wait_for_port() {
    local host=$1 port=$2 timeout_s=$3
    "${VENV_PYTHON}" - "${host}" "${port}" "${timeout_s}" <<'PY'
import socket, sys, time
host, port, deadline = sys.argv[1], int(sys.argv[2]), time.time() + int(sys.argv[3])
while time.time() < deadline:
    try:
        socket.create_connection((host, port), timeout=3).close()
        sys.exit(0)
    except OSError:
        time.sleep(3)
print(f"Timed out waiting for {host}:{port}", file=sys.stderr)
sys.exit(1)
PY
}

# Kill all CarlaUE4 processes listening on a given port.
kill_carla_port() {
    local port=$1
    # Kill by PID if we have it.
    if [ -n "${_CARLA_PID:-}" ]; then
        kill -9 "${_CARLA_PID}" 2>/dev/null || true
    fi
    # Belt-and-suspenders: also kill by port pattern.
    pkill -9 -f "carla-rpc-port=${port}" 2>/dev/null || true
    # Give the OS a moment to release the port.
    sleep 2
}

# ---------------------------------------------------------------------------
# per-batch processing (runs inside a single worker's context)
# ---------------------------------------------------------------------------
process_batch() {
    local worker_id=$1 batch_path=$2 attempt_global=$3
    local gpu=$(( worker_id % N_GPUS ))
    local rpc_port=$(( BASE_PORT + worker_id * 10 ))
    local tm_port=$(( rpc_port + 500 ))
    local server_log="${JOB_ROOT}/server_$(printf '%02d' "${worker_id}").log"
    local client_log="${JOB_ROOT}/client_$(printf '%02d' "${worker_id}").log"
    local carla_saved="${JOB_ROOT}/carla_saved_$(printf '%02d' "${worker_id}")"

    mkdir -p "${carla_saved}"

    # ---- Start CARLA ----
    log "[w${worker_id}] Starting CARLA GPU=${gpu} port=${rpc_port} batch=${batch_path}"
    _CARLA_PID=""
    CARLA_LOG_DIR="${carla_saved}/Logs" \
    "${CARLA_BIN}" CarlaUE4 \
        -RenderOffScreen \
        -quality-level="${QUALITY_LEVEL}" \
        -carla-rpc-port="${rpc_port}" \
        -graphicsadapter="${gpu}" \
        >> "${server_log}" 2>&1 &
    _CARLA_PID=$!
    log "[w${worker_id}] CARLA pid=${_CARLA_PID}"

    # ---- Wait for CARLA ----
    local carla_ok=0
    if wait_for_port 127.0.0.1 "${rpc_port}" "${CARLA_READY_TIMEOUT}"; then
        carla_ok=1
        log "[w${worker_id}] CARLA ready on port ${rpc_port}"
    fi

    local batch_status=1
    if [ "${carla_ok}" -eq 1 ]; then
        # ---- Run client ----
        local attempt
        for attempt in $(seq 1 "${MAX_RETRIES}"); do
            log "[w${worker_id}] Client attempt ${attempt}/${MAX_RETRIES}"
            local client_status=0
            set +e
            LEAD_PROJECT_ROOT="${REPO_DIR}" \
            PY123D_DATA_ROOT="${REPO_DIR}/${OUTPUT_ROOT}/carla_alpamayo_4c_py123d" \
            LEAD_EXPERT_CONFIG="target_dataset=7 py123d_data_format=true use_radars=false lidar_stack_size=2 save_only_non_ground_lidar=false save_lidar_only_inside_bev=false" \
            timeout "${CLIENT_TIMEOUT}" \
            "${VENV_PYTHON}" -u -m lead \
                --expert \
                --py123d \
                --host 127.0.0.1 \
                --port "${rpc_port}" \
                --traffic-manager-port "${tm_port}" \
                --timeout 600 \
                --resume \
                --output-dir "${REPO_DIR}/${OUTPUT_ROOT}/outputs/$(basename "${batch_path}" .xml)" \
                --routes "${REPO_DIR}/${batch_path}" \
                >> "${client_log}" 2>&1
            client_status=$?
            set -e

            if [ "${client_status}" -eq 0 ]; then
                log "[w${worker_id}] Batch OK: ${batch_path}"
                write_summary "{\"worker\":${worker_id},\"batch\":\"${batch_path}\",\"status\":\"ok\",\"attempt\":${attempt}}"
                batch_status=0
                break
            fi

            local reason="client_exit_${client_status}"
            if grep -Eqi "Signal 11|Segmentation fault|segfault" "${server_log}" 2>/dev/null; then
                reason="signal_11"
            elif [ "${client_status}" -eq 124 ]; then
                reason="timeout"
            fi
            log "[w${worker_id}] Attempt ${attempt} failed: reason=${reason}" >&2
            write_summary "{\"worker\":${worker_id},\"batch\":\"${batch_path}\",\"status\":\"failed\",\"attempt\":${attempt},\"reason\":\"${reason}\"}"

            # Signal 11 in the server is unrecoverable for this batch.
            [ "${reason}" = "signal_11" ] && break
        done
    else
        log "[w${worker_id}] CARLA did not start within ${CARLA_READY_TIMEOUT}s" >&2
        write_summary "{\"worker\":${worker_id},\"batch\":\"${batch_path}\",\"status\":\"failed\",\"attempt\":${attempt_global},\"reason\":\"carla_start_timeout\"}"
    fi

    # ---- Kill CARLA ----
    kill_carla_port "${rpc_port}"
    _CARLA_PID=""

    if [ "${batch_status}" -ne 0 ]; then
        printf '%s\n' "${batch_path}" >> "${JOB_ROOT}/failed_batches.txt"
    fi
    return "${batch_status}"
}

# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------
worker_loop() {
    local worker_id=$1
    local worker_log="${JOB_ROOT}/worker_$(printf '%02d' "${worker_id}").log"
    # Redirect all output to a per-worker log; also tee to stdout for progress.
    exec > >(tee -a "${worker_log}") 2>&1

    log "[w${worker_id}] Started on GPU $(( worker_id % N_GPUS ))"

    local batch_count=0
    while true; do
        local batch_idx
        batch_idx=$(next_batch_idx)

        # Read line (1-indexed) from the routes manifest.
        local batch_path
        batch_path=$(sed -n "$(( batch_idx + 1 ))p" "${ROUTES_FILE}" 2>/dev/null || true)

        # Strip comments and whitespace.
        batch_path="${batch_path%%#*}"
        batch_path="${batch_path#"${batch_path%%[![:space:]]*}"}"
        batch_path="${batch_path%"${batch_path##*[![:space:]]}"}"

        if [ -z "${batch_path}" ]; then
            log "[w${worker_id}] Queue exhausted after ${batch_count} batches. Done."
            break
        fi

        log "[w${worker_id}] Batch ${batch_idx}: ${batch_path}"
        process_batch "${worker_id}" "${batch_path}" "${batch_idx}" || \
            log "[w${worker_id}] Batch ${batch_idx} failed (continuing to next)" >&2
        batch_count=$(( batch_count + 1 ))
    done

    log "[w${worker_id}] Worker finished (${batch_count} batches processed)."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    [ -n "${ROUTES_FILE}" ] || die "ROUTES_FILE is not set.
  Usage: ROUTES_FILE=<manifest> bash $0"
    [ -f "${ROUTES_FILE}" ] || die "ROUTES_FILE not found: ${ROUTES_FILE}"
    [ -n "${CARLA_BIN}" ] || die "CARLA binary not found. Set CARLA_BIN to the CarlaUE4-Linux-Shipping path."
    require_exec "${CARLA_BIN}"
    require_exec "${VENV_PYTHON}"

    mkdir -p "${JOB_ROOT}"
    echo 0 > "${BATCH_COUNTER}"
    : > "${SUMMARY_JSONL}"

    local n_batches
    n_batches=$(grep -c '[^[:space:]]' "${ROUTES_FILE}" || true)

    log "=== On-prem parallel collection ==="
    log "REPO_DIR      = ${REPO_DIR}"
    log "CARLA_BIN     = ${CARLA_BIN}"
    log "VENV_PYTHON   = ${VENV_PYTHON}"
    log "N_WORKERS     = ${N_WORKERS} / N_GPUS = ${N_GPUS}"
    log "BASE_PORT     = ${BASE_PORT}"
    log "QUALITY_LEVEL = ${QUALITY_LEVEL}"
    log "Batches total = ${n_batches}"
    log "JOB_ROOT      = ${JOB_ROOT}"
    log ""

    # Install SIGINT/SIGTERM handler to kill stray CARLAs on Ctrl-C.
    cleanup_all() {
        log "Caught signal — killing any remaining CARLA processes..."
        pkill -9 -f 'CarlaUE4-Linux-Shipping' 2>/dev/null || true
    }
    trap cleanup_all INT TERM

    # Launch all workers as background jobs.
    local pids=()
    for i in $(seq 0 $(( N_WORKERS - 1 ))); do
        worker_loop "${i}" &
        pids+=($!)
        log "Launched worker ${i} (pid=${pids[-1]})"
    done

    # Wait for all workers; collect worst exit code.
    local worst=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || worst=$?
    done

    log ""
    log "=== Collection finished ==="
    log "Summary (${SUMMARY_JSONL}):"
    if [ -f "${SUMMARY_JSONL}" ]; then
        "${VENV_PYTHON}" -c "
import json, collections
lines = [json.loads(l) for l in open('${SUMMARY_JSONL}') if l.strip()]
c = collections.Counter(d.get('status','?') for d in lines)
for status, count in sorted(c.items()):
    print(f'  {status}: {count}')
print(f'  total: {len(lines)}')
" 2>/dev/null || cat "${SUMMARY_JSONL}"
    fi

    if [ -f "${JOB_ROOT}/failed_batches.txt" ]; then
        local n_failed
        n_failed=$(grep -c '[^[:space:]]' "${JOB_ROOT}/failed_batches.txt" || true)
        log "WARNING: ${n_failed} batch(es) failed. See ${JOB_ROOT}/failed_batches.txt"
    fi

    return "${worst}"
}

main
