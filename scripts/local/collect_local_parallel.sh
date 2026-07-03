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
#   GPU_IDS               comma-separated physical GPU IDs to use (default: 0..N_GPUS-1)
#   BASE_PORT             first RPC port; worker k uses BASE_PORT+k*10 (default: 2000)
#   OUTPUT_ROOT           output base dir relative to REPO_DIR (default: data)
#   QUALITY_LEVEL         CARLA quality: Epic|Low (default: Epic)
#   MAX_RETRIES           per-batch client retries before marking failed (default: 2)
#   RETRY_RESTART_DELAY   seconds to wait after killing CARLA before retry (default: 5)
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
GPU_IDS=${GPU_IDS:-}
BASE_PORT=${BASE_PORT:-2000}
OUTPUT_ROOT=${OUTPUT_ROOT:-data}
QUALITY_LEVEL=${QUALITY_LEVEL:-Epic}
MAX_RETRIES=${MAX_RETRIES:-2}
RETRY_RESTART_DELAY=${RETRY_RESTART_DELAY:-5}
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

resolve_gpu_id() {
    local worker_id=$1
    if [ -n "${GPU_IDS}" ]; then
        IFS=',' read -ra _gpu_ids <<< "${GPU_IDS}"
        local n_ids=${#_gpu_ids[@]}
        [ "${n_ids}" -gt 0 ] || die "GPU_IDS is set but empty"
        printf '%s' "${_gpu_ids[$(( worker_id % n_ids ))]}"
    else
        printf '%s' "$(( worker_id % N_GPUS ))"
    fi
}

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

wait_for_port_closed() {
    local host=$1 port=$2 timeout_s=$3
    "${VENV_PYTHON}" - "${host}" "${port}" "${timeout_s}" <<'PY'
import socket, sys, time
host, port, deadline = sys.argv[1], int(sys.argv[2]), time.time() + int(sys.argv[3])
while time.time() < deadline:
    try:
        socket.create_connection((host, port), timeout=1).close()
        time.sleep(1)
    except OSError:
        sys.exit(0)
print(f"Timed out waiting for {host}:{port} to close", file=sys.stderr)
sys.exit(1)
PY
}

# Kill all CarlaUE4 processes listening on a given port.
kill_carla_port() {
    local port=$1
    # Kill by PID if we have it.
    if [ -n "${_CARLA_PID:-}" ]; then
        kill -TERM "${_CARLA_PID}" 2>/dev/null || true
        sleep 2
        kill -9 "${_CARLA_PID}" 2>/dev/null || true
    fi
    # Belt-and-suspenders: also kill by port pattern.
    pkill -TERM -f "carla-rpc-port=${port}" 2>/dev/null || true
    sleep 2
    pkill -9 -f "carla-rpc-port=${port}" 2>/dev/null || true
    wait_for_port_closed 127.0.0.1 "${port}" 15 || true
}

kill_client_ports() {
    local rpc_port=$1 tm_port=$2
    pkill -TERM -f -- "--traffic-manager-port ${tm_port}" 2>/dev/null || true
    pkill -TERM -f -- "--traffic-manager-port=${tm_port}" 2>/dev/null || true
    pkill -TERM -f -- "--port ${rpc_port} .*--traffic-manager-port ${tm_port}" 2>/dev/null || true
    pkill -TERM -f -- "leaderboard_evaluator_local.py .*--port ${rpc_port} .*--traffic-manager-port ${tm_port}" 2>/dev/null || true
    sleep 2
    pkill -9 -f -- "--traffic-manager-port ${tm_port}" 2>/dev/null || true
    pkill -9 -f -- "--traffic-manager-port=${tm_port}" 2>/dev/null || true
    pkill -9 -f -- "--port ${rpc_port} .*--traffic-manager-port ${tm_port}" 2>/dev/null || true
    pkill -9 -f -- "leaderboard_evaluator_local.py .*--port ${rpc_port} .*--traffic-manager-port ${tm_port}" 2>/dev/null || true
    wait_for_port_closed 127.0.0.1 "${tm_port}" 15 || true
    sleep 1
}

checkpoint_ok() {
    local checkpoint=$1
    "${VENV_PYTHON}" - "${checkpoint}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    print(f"checkpoint_missing:{path}")
    sys.exit(1)
try:
    data = json.loads(path.read_text())
except Exception as exc:
    print(f"checkpoint_parse_error:{exc}")
    sys.exit(1)
cp = data.get("_checkpoint", {})
progress = cp.get("progress", [0, 0])
records = cp.get("records", [])
bad = []
success_statuses = {"completed", "perfect"}
for rec in records:
    status = str(rec.get("status", "")).strip()
    status_l = status.lower()
    if status and not any(status_l.startswith(ok) for ok in success_statuses):
        bad.append(f"{rec.get('route_id', rec.get('index'))}:{status}")
entry_status = str(data.get("entry_status", ""))
if entry_status.lower() in {"crashed", "rejected"}:
    bad.append(f"entry_status:{entry_status}")
if len(progress) >= 2 and progress[0] != progress[1]:
    bad.append(f"progress:{progress[0]}/{progress[1]}")
if bad:
    print("checkpoint_failed:" + ";".join(bad))
    sys.exit(1)
print("checkpoint_ok")
PY
}

classify_client_log() {
    local log_file=$1 start_byte=$2
    "${VENV_PYTHON}" - "${log_file}" "${start_byte}" <<'PY'
import re, sys
from pathlib import Path
path = Path(sys.argv[1])
start = int(sys.argv[2])
try:
    with path.open('rb') as f:
        f.seek(start)
        text = f.read().decode('utf-8', 'ignore')
except FileNotFoundError:
    sys.exit(1)
patterns = [
    ("traffic_manager_bind", r"traffic manager.*bind error|failed to create because of bind error"),
    ("agent_index_error", r"IndexError: list index out of range"),
    ("sensor_no_data", r"SensorReceivedNoData|A sensor took too long to send their data"),
    ("actor_lost", r"Actor\(id=.*\) not found"),
    ("simulation_crashed", r"Failed - Simulation crashed|Simulation crashed"),
]
for name, pattern in patterns:
    if re.search(pattern, text, re.I):
        print(name)
        sys.exit(0)
sys.exit(1)
PY
}

json_escape() {
    "${VENV_PYTHON}" - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1])[1:-1])
PY
}


# ---------------------------------------------------------------------------
# per-batch processing (runs inside a single worker's context)
# ---------------------------------------------------------------------------
process_batch() {
    local worker_id=$1 batch_path=$2 attempt_global=$3
    local gpu
    gpu=$(resolve_gpu_id "${worker_id}")
    local rpc_port=$(( BASE_PORT + worker_id * 10 ))
    local tm_port=$(( rpc_port + 500 ))
    local server_log="${JOB_ROOT}/server_$(printf '%02d' "${worker_id}").log"
    local client_log="${JOB_ROOT}/client_$(printf '%02d' "${worker_id}").log"
    local carla_saved="${JOB_ROOT}/carla_saved_$(printf '%02d' "${worker_id}")"

    mkdir -p "${carla_saved}"

    local batch_status=1
    local attempt
    for attempt in $(seq 1 "${MAX_RETRIES}"); do
        # A failed leaderboard run can leave actors/world state behind. Start a
        # fresh CARLA server for every retry so attempt 2 is a real retry.
        kill_client_ports "${rpc_port}" "${tm_port}"
        kill_carla_port "${rpc_port}"
        log "[w${worker_id}] Starting CARLA GPU=${gpu} port=${rpc_port} batch=${batch_path} attempt=${attempt}/${MAX_RETRIES}"
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

        local carla_ok=0
        if wait_for_port 127.0.0.1 "${rpc_port}" "${CARLA_READY_TIMEOUT}"; then
            carla_ok=1
            log "[w${worker_id}] CARLA ready on port ${rpc_port}"
        fi

        local reason=""
        if [ "${carla_ok}" -ne 1 ]; then
            reason="carla_start_timeout"
            log "[w${worker_id}] CARLA did not start within ${CARLA_READY_TIMEOUT}s" >&2
        else
            log "[w${worker_id}] Client attempt ${attempt}/${MAX_RETRIES}"
            local client_status=0
            local client_log_start=0
            local fatal_reason=""
            local checkpoint="${REPO_DIR}/${OUTPUT_ROOT}/outputs/${JOB_ID}/$(basename "${batch_path}" .xml)/checkpoint_endpoint.json"
            client_log_start=$(wc -c < "${client_log}" 2>/dev/null || echo 0)
            set +e
            LEAD_PROJECT_ROOT="${REPO_DIR}" \
            PY123D_DATA_ROOT="${REPO_DIR}/${OUTPUT_ROOT}/carla_alpamayo_4c_py123d" \
            LEAD_EXPERT_CONFIG="target_dataset=7 py123d_data_format=true use_radars=false lidar_stack_size=2 save_only_non_ground_lidar=false save_lidar_only_inside_bev=false ${EXTRA_LEAD_EXPERT_CONFIG:-}" \
            timeout "${CLIENT_TIMEOUT}" \
            "${VENV_PYTHON}" -u -m lead \
                --expert \
                --py123d \
                --host 127.0.0.1 \
                --port "${rpc_port}" \
                --traffic-manager-port "${tm_port}" \
                --timeout 600 \
                --resume \
                --output-dir "${REPO_DIR}/${OUTPUT_ROOT}/outputs/${JOB_ID}/$(basename "${batch_path}" .xml)" \
                --routes "${REPO_DIR}/${batch_path}" \
                >> "${client_log}" 2>&1 &
            local client_pid=$!
            while kill -0 "${client_pid}" 2>/dev/null; do
                local client_stat=""
                client_stat=$(ps -p "${client_pid}" -o stat= 2>/dev/null || true)
                if [ -z "${client_stat}" ] || [[ "${client_stat}" == Z* ]]; then
                    break
                fi
                fatal_reason=$(classify_client_log "${client_log}" "${client_log_start}" 2>/dev/null)
                if [ -n "${fatal_reason}" ]; then
                    reason="${fatal_reason}"
                    log "[w${worker_id}] Fatal client log pattern detected: ${reason}" >&2
                    kill "${client_pid}" 2>/dev/null || true
                    sleep 2
                    kill -9 "${client_pid}" 2>/dev/null || true
                    wait "${client_pid}" 2>/dev/null
                    client_status=125
                    break
                fi
                sleep 5
            done
            if [ -z "${reason}" ]; then
                wait "${client_pid}"
                client_status=$?
            fi
            set -e

            if [ "${client_status}" -eq 0 ]; then
                local checkpoint_msg=""
                if checkpoint_msg=$(checkpoint_ok "${checkpoint}" 2>&1); then
                    log "[w${worker_id}] Batch OK: ${batch_path}"
                    write_summary "{\"worker\":${worker_id},\"batch\":\"${batch_path}\",\"status\":\"ok\",\"attempt\":${attempt}}"
                    batch_status=0
                    kill_client_ports "${rpc_port}" "${tm_port}"
                    kill_carla_port "${rpc_port}"
                    _CARLA_PID=""
                    break
                fi
                reason="${checkpoint_msg}"
                log "[w${worker_id}] Checkpoint indicates failed route: ${reason}" >&2
            fi

            if [ -z "${reason}" ]; then
                reason="client_exit_${client_status}"
                fatal_reason=$(classify_client_log "${client_log}" "${client_log_start}" 2>/dev/null || true)
                if [ -n "${fatal_reason}" ]; then
                    reason="${fatal_reason}"
                elif grep -Eqi "Signal 11|Segmentation fault|segfault" "${server_log}" 2>/dev/null; then
                    reason="signal_11"
                elif [ "${client_status}" -eq 124 ]; then
                    reason="timeout"
                fi
            fi
            log "[w${worker_id}] Attempt ${attempt} failed: reason=${reason}" >&2
        fi

        reason_json=$(json_escape "${reason}")
        write_summary "{\"worker\":${worker_id},\"batch\":\"${batch_path}\",\"status\":\"failed\",\"attempt\":${attempt},\"reason\":\"${reason_json}\"}"
        kill_client_ports "${rpc_port}" "${tm_port}"
        kill_carla_port "${rpc_port}"
        _CARLA_PID=""

        # Signal 11 in the server is unlikely to be recovered by retrying.
        [ "${reason}" = "signal_11" ] && break

        if [ "${attempt}" -lt "${MAX_RETRIES}" ]; then
            log "[w${worker_id}] Waiting ${RETRY_RESTART_DELAY}s before retry"
            sleep "${RETRY_RESTART_DELAY}"
        fi
    done

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

    log "[w${worker_id}] Started on GPU $(resolve_gpu_id "${worker_id}")"

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
    : > "${JOB_ROOT}/failed_batches.txt"

    local n_batches
    n_batches=$(grep -c '[^[:space:]]' "${ROUTES_FILE}" || true)

    log "=== On-prem parallel collection ==="
    log "REPO_DIR      = ${REPO_DIR}"
    log "CARLA_BIN     = ${CARLA_BIN}"
    log "VENV_PYTHON   = ${VENV_PYTHON}"
    log "N_WORKERS     = ${N_WORKERS} / N_GPUS = ${N_GPUS}"
    log "GPU_IDS       = ${GPU_IDS:-0..$((N_GPUS-1))}"
    log "BASE_PORT     = ${BASE_PORT}"
    log "QUALITY_LEVEL = ${QUALITY_LEVEL}"
    log "MAX_RETRIES   = ${MAX_RETRIES}"
    log "RETRY_DELAY   = ${RETRY_RESTART_DELAY}"
    log "Batches total = ${n_batches}"
    log "JOB_ROOT      = ${JOB_ROOT}"
    log ""

    # Install SIGINT/SIGTERM handler to kill stray CARLAs on Ctrl-C.
    cleanup_all() {
        log "Caught signal — killing any remaining CARLA/client processes..."
        pkill -9 -f 'CarlaUE4-Linux-Shipping' 2>/dev/null || true
        pkill -9 -f '/workspace/VLA4AD_Research/.venv/bin/python3 -u -m lead' 2>/dev/null || true
        pkill -9 -f 'leaderboard_evaluator_local.py' 2>/dev/null || true
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
