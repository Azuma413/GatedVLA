#!/usr/bin/env bash
# Parallel CARLA data collection: N server+client pairs run simultaneously.
#
# Usage (called from collect_py123d_parallel.pbs or directly):
#   ROUTES_FILE=/path/to/routes.txt N_PAIRS=4 bash collect_py123d_parallel.sh
#
# ROUTES_FILE: one route XML path per line (relative or absolute).
#              Lines starting with '#' and blank lines are ignored.
# N_PAIRS:     number of parallel server+client pairs.  Defaults to the number
#              of non-blank, non-comment lines in ROUTES_FILE.
#
# Each pair occupies two adjacent MPI ranks (2k = server, 2k+1 = client) on
# separate nodes.  mpiexec is called internally; do not wrap this script with
# mpiexec yourself.
set -euo pipefail

REPO_DIR=${REPO_DIR:-/eagle/RobustViT/hiratsuka/VLA4AD_Research}
IMAGE_DIR=${IMAGE_DIR:-${REPO_DIR}/containers}
CARLA_SIF=${CARLA_SIF:-${IMAGE_DIR}/carla.sif}
LEAD_SIF=${LEAD_SIF:-${IMAGE_DIR}/lead.sif}
ROUTES_FILE=${ROUTES_FILE:-}
ROUTES_DIR=${ROUTES_DIR:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-data}
BASE_PORT=${BASE_PORT:-20000}
MAX_RETRIES=${MAX_RETRIES:-2}
CARLA_READY_TIMEOUT=${CARLA_READY_TIMEOUT:-300}
CLIENT_READY_TIMEOUT=${CLIENT_READY_TIMEOUT:-300}
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-3600}
CARLA_PYTHONAPI_HOST=${CARLA_PYTHONAPI_HOST:-${REPO_DIR}/.carla_pythonapi}
CARLA_SERVER_MODE=${CARLA_SERVER_MODE:-host-vulkan}
JOB_ROOT=${JOB_ROOT:-${REPO_DIR}/logs/polaris_collection/${PBS_JOBID:-manual_parallel}}
SERVER_READY_DIR=${SERVER_READY_DIR:-${JOB_ROOT}/server_ready}
SUMMARY_JSONL=${SUMMARY_JSONL:-${JOB_ROOT}/summary.jsonl}
ROUTES_MANIFEST=${ROUTES_MANIFEST:-${JOB_ROOT}/routes_manifest.txt}

source "${REPO_DIR}/scripts/polaris/carla_apptainer_common.sh"

# Avoid filling the filesystem with multi-GB CARLA core dumps on a crash.
ulimit -c 0 2>/dev/null || true

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

rank_id() {
    printf '%s' "${PMI_RANK:-${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${ALPS_APP_PE:-0}}}}"
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "Missing file: $1" >&2
        exit 2
    fi
}

extract_carla_pythonapi() {
    if [ -f "${CARLA_PYTHONAPI_HOST}/agents/navigation/global_route_planner.py" ]; then
        return
    fi
    echo "Extracting CARLA PythonAPI agents to ${CARLA_PYTHONAPI_HOST}"
    mkdir -p "${CARLA_PYTHONAPI_HOST}"
    apptainer exec "${CARLA_SIF}" bash -c \
        'cd /opt/carla/PythonAPI/carla && tar -cf - agents' \
        | tar -C "${CARLA_PYTHONAPI_HOST}" -xf -
}

read_route_town() {
    # Arg is a path relative to REPO_DIR; resolve it inside the container.
    local route_rel=$1
    apptainer exec \
        -B "${REPO_DIR}:/workspace/VLA4AD_Research" \
        "${LEAD_SIF}" \
        lead-python - "/workspace/VLA4AD_Research/${route_rel}" <<'PY'
import sys
import xml.etree.ElementTree as ET
route_path = sys.argv[1]
root = ET.parse(route_path).getroot()
route = root.find("route")
if route is None or "town" not in route.attrib:
    raise SystemExit(f"Could not read route town from {route_path}")
print(route.attrib["town"])
PY
}

wait_for_port() {
    local host=$1
    local port=$2
    local timeout_s=$3
    python3 - "${host}" "${port}" "${timeout_s}" <<'PY'
import socket, sys, time
host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.time() + int(sys.argv[3])
while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(3)
        try:
            sock.connect((host, port))
            print(f"Connected to {host}:{port}")
            raise SystemExit(0)
        except OSError:
            time.sleep(2)
raise SystemExit(f"Timed out waiting for {host}:{port}")
PY
}

run_carla_connect_probe() {
    local lead_sif=$1
    local port=$2
    apptainer exec --nv --writable-tmpfs \
        -B "${REPO_DIR}:/workspace/VLA4AD_Research" \
        --env LEAD_PROJECT_ROOT=/workspace/VLA4AD_Research \
        "${lead_sif}" \
        lead-python - "${port}" <<'PY'
import sys, carla
port = int(sys.argv[1])
client = carla.Client("127.0.0.1", port)
client.set_timeout(10.0)
world = client.get_world()
print("world", world.get_map().name)
PY
}

# ---------------------------------------------------------------------------
# Route preparation
# ---------------------------------------------------------------------------

# Reads ROUTES_FILE / ROUTES_DIR into a cleaned list of route paths and
# writes ROUTES_MANIFEST.  Sets N_PAIRS based on the list length when
# N_PAIRS is not explicitly provided.
prepare_routes() {
    local raw_routes=()

    if [ -n "${ROUTES_FILE}" ]; then
        require_file "${ROUTES_FILE}"
        while IFS= read -r line || [ -n "${line}" ]; do
            line="${line%%#*}"
            line="${line#"${line%%[![:space:]]*}"}"
            line="${line%"${line##*[![:space:]]}"}"
            [ -z "${line}" ] && continue
            raw_routes+=("${line}")
        done < "${ROUTES_FILE}"
    elif [ -n "${ROUTES_DIR}" ]; then
        while IFS= read -r f; do
            raw_routes+=("${f}")
        done < <(find "${ROUTES_DIR}" -maxdepth 1 -name "*.xml" | sort)
    else
        echo "ERROR: Set ROUTES_FILE or ROUTES_DIR" >&2
        exit 2
    fi

    local n_routes=${#raw_routes[@]}
    if [ "${n_routes}" -eq 0 ]; then
        echo "ERROR: No routes found in ROUTES_FILE/ROUTES_DIR" >&2
        exit 2
    fi

    : "${N_PAIRS:=${n_routes}}"
    if [ "${N_PAIRS}" -gt "${n_routes}" ]; then
        echo "WARNING: N_PAIRS=${N_PAIRS} > n_routes=${n_routes}; reducing to ${n_routes}" >&2
        N_PAIRS=${n_routes}
    fi
    export N_PAIRS

    # Create variant routes and write manifest.
    local routes_dir="${JOB_ROOT}/routes"
    mkdir -p "${routes_dir}"
    : >"${ROUTES_MANIFEST}"

    local idx=0
    while [ "${idx}" -lt "${N_PAIRS}" ]; do
        local src="${raw_routes[${idx}]}"
        # Store paths RELATIVE to REPO_DIR. Workers reference them inside the
        # container as /workspace/VLA4AD_Research/<rel> (the host /eagle path is
        # not visible inside the container).
        src="${src#"${REPO_DIR}/"}"
        src="${src#/}"

        printf '%s\n' "${src}" >> "${ROUTES_MANIFEST}"
        idx=$((idx + 1))
    done

    echo "Routes manifest written to ${ROUTES_MANIFEST} (${N_PAIRS} scenarios)"
}

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

run_server() {
    local scenario_id=$1
    local route_for_job=$2

    local carla_port tm_port route_town_name carla_log carla_exit_log carla_saved_dir
    local -a carla_bind_args carla_env_args graphics_bind_args graphics_env_args

    carla_port=$((BASE_PORT + scenario_id * 10))
    tm_port=$((carla_port + 500))
    route_town_name=$(read_route_town "${route_for_job}")
    carla_log="${JOB_ROOT}/server_$(printf '%02d' "${scenario_id}").log"
    carla_exit_log="${JOB_ROOT}/server_$(printf '%02d' "${scenario_id}").exit.log"
    carla_saved_dir="${JOB_ROOT}/carla_saved_$(printf '%02d' "${scenario_id}")"
    local server_env="${SERVER_READY_DIR}/scenario_$(printf '%02d' "${scenario_id}").env"

    mkdir -p "${carla_saved_dir}" "${SERVER_READY_DIR}"

    carla_bind_args=(
        --bind "${carla_saved_dir}:/opt/carla/CarlaUE4/Saved"
        --bind "${carla_saved_dir}:/home/carla/CarlaUE4/Saved"
    )
    carla_env_args=(
        --env "CARLA_RPC_PORT=${carla_port}"
        --env "CARLA_PORT=${carla_port}"
        --env CARLA_LOG_DIR=/opt/carla/CarlaUE4/Saved/Logs
        --env NVIDIA_DRIVER_CAPABILITIES=all
    )
    if [ -n "${CARLA_GRAPHICS_ADAPTER:-}" ]; then
        carla_env_args+=(--env "CARLA_GRAPHICS_ADAPTER=${CARLA_GRAPHICS_ADAPTER}")
    fi
    case "${CARLA_SERVER_MODE}" in
        host-vulkan)
            mapfile -t graphics_bind_args < <(carla_graphics_bind_args)
            mapfile -t graphics_env_args < <(carla_graphics_env_args)
            carla_bind_args+=("${graphics_bind_args[@]}")
            carla_env_args+=("${graphics_env_args[@]}")
            ;;
        *)
            echo "Unapproved CARLA_SERVER_MODE=${CARLA_SERVER_MODE}; use host-vulkan." >&2
            exit 2
            ;;
    esac

    echo "scenario=${scenario_id} server_host=$(hostname -f) route=${route_for_job} town=${route_town_name} port=${carla_port}"

    cleanup() {
        # CARLA/UE4 IGNORES SIGTERM, so the old `kill` + `wait` blocked forever:
        # the rank never exited, mpiexec never finished, and the whole job sat idle
        # holding all its nodes until walltime (hours of wasted node-hours after the
        # actual collection had already finished in minutes). Force-kill with
        # SIGKILL and nuke any surviving CarlaUE4 on this node (cross-node => one
        # CARLA per node); do NOT block on wait.
        [ -n "${CARLA_PID:-}" ] && kill -9 "${CARLA_PID}" 2>/dev/null || true
        pkill -9 -f 'CarlaUE4-Linux-Shipping' 2>/dev/null || true
    }
    trap cleanup EXIT

    apptainer exec --nv --writable-tmpfs \
        --contain \
        "${carla_bind_args[@]}" \
        "${carla_env_args[@]}" \
        "${CARLA_SIF}" bash -c '
        # Keep this launch minimal: extra flags (-vulkan/-nosound/-unattended/
        # -NoSplash) make the client-side TrafficManager SIGSEGV on complex maps
        # (Town03+). This stock-style launch is verified to let
        # get_trafficmanager succeed. See collect_py123d_short_route.sh.
        exec /opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping \
            CarlaUE4 \
            -RenderOffScreen \
            -quality-level=Epic \
            -carla-rpc-port="${CARLA_PORT}" \
            -graphicsadapter="${CARLA_GRAPHICS_ADAPTER:-0}"
    ' >"${carla_log}" 2>&1 &
    CARLA_PID=$!
    echo "Started CARLA pid=${CARLA_PID}"

    local waited=0
    while [ "${waited}" -lt "${CARLA_READY_TIMEOUT}" ]; do
        if ! kill -0 "${CARLA_PID}" 2>/dev/null; then
            echo "CARLA exited before ready" >&2
            wait "${CARLA_PID}" >"${carla_exit_log}" 2>&1 || echo "CARLA exit status=$?" >&2
            tail -200 "${carla_log}" >&2 || true
            return 3
        fi
        if run_carla_connect_probe "${LEAD_SIF}" "${carla_port}"; then
            {
                printf 'host=%q\n'        "$(hostname -f)"
                printf 'SERVER_HOST=%q\n' "$(hostname -f)"
                printf 'CARLA_PORT=%q\n'  "${carla_port}"
                printf 'TM_PORT=%q\n'     "${tm_port}"
                printf 'ROUTE_FOR_JOB=%q\n' "${route_for_job}"
                printf 'SCENARIO_ID=%q\n' "${scenario_id}"
            } >"${server_env}.tmp"
            mv "${server_env}.tmp" "${server_env}"
            echo "Wrote ${server_env}"
            break
        fi
        sleep 10
        waited=$((waited + 10))
        echo "Waiting for CARLA... ${waited}/${CARLA_READY_TIMEOUT}s (scenario=${scenario_id})"
    done

    if [ ! -f "${server_env}" ]; then
        echo "CARLA did not become ready within ${CARLA_READY_TIMEOUT}s" >&2
        tail -200 "${carla_log}" >&2 || true
        return 4
    fi

    local waited=0
    while [ "${waited}" -lt "${CLIENT_TIMEOUT}" ]; do
        if [ -f "${JOB_ROOT}/scenario_$(printf '%02d' "${scenario_id}").client.done" ]; then
            echo "Client for scenario ${scenario_id} completed"
            return 0
        fi
        if [ -f "${JOB_ROOT}/scenario_$(printf '%02d' "${scenario_id}").client.failed" ]; then
            echo "Client for scenario ${scenario_id} failed" >&2
            return 5
        fi
        if ! kill -0 "${CARLA_PID}" 2>/dev/null; then
            echo "CARLA exited while client was running" >&2
            tail -200 "${carla_log}" >&2 || true
            return 6
        fi
        sleep 10
        waited=$((waited + 10))
    done

    echo "Timed out waiting for client (scenario=${scenario_id})" >&2
    return 7
}

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

run_client() {
    local scenario_id=$1
    local route_for_job=$2
    local server_env="${SERVER_READY_DIR}/scenario_$(printf '%02d' "${scenario_id}").env"
    local client_done="${JOB_ROOT}/scenario_$(printf '%02d' "${scenario_id}").client.done"
    local client_failed="${JOB_ROOT}/scenario_$(printf '%02d' "${scenario_id}").client.failed"

    local waited=0
    while [ "${waited}" -lt "${CLIENT_READY_TIMEOUT}" ] && [ ! -f "${server_env}" ]; do
        sleep 5
        waited=$((waited + 5))
        echo "Waiting for ${server_env}... ${waited}/${CLIENT_READY_TIMEOUT}s"
    done
    if [ ! -f "${server_env}" ]; then
        echo "Server did not publish ${server_env}" >&2
        touch "${client_failed}"
        return 8
    fi

    # shellcheck disable=SC1090
    source "${server_env}"
    echo "Client connecting to ${SERVER_HOST}:${CARLA_PORT}, tm=${TM_PORT}"

    if ! wait_for_port "${SERVER_HOST}" "${CARLA_PORT}" "${CLIENT_READY_TIMEOUT}"; then
        touch "${client_failed}"
        return 9
    fi

    mkdir -p "${OUTPUT_ROOT}/carla_alpamayo_4c_py123d"
    local attempt status reason
    for attempt in $(seq 1 "${MAX_RETRIES}"); do
        echo "Client attempt ${attempt}/${MAX_RETRIES} (scenario=${scenario_id})"
        set +e
        apptainer exec --nv --writable-tmpfs \
            -B "${REPO_DIR}:/workspace/VLA4AD_Research" \
            --env LEAD_PROJECT_ROOT=/workspace/VLA4AD_Research \
            --env PY123D_DATA_ROOT=/workspace/VLA4AD_Research/${OUTPUT_ROOT}/carla_alpamayo_4c_py123d \
            --env LEAD_EXPERT_CONFIG="target_dataset=7 py123d_data_format=true use_radars=false lidar_stack_size=2 save_only_non_ground_lidar=false save_lidar_only_inside_bev=false" \
            --env PYTHONPATH=/workspace/VLA4AD_Research:/workspace/VLA4AD_Research/.carla_pythonapi \
            "${LEAD_SIF}" \
            lead-python -u -m lead \
                --expert \
                --py123d \
                --host "${SERVER_HOST}" \
                --port "${CARLA_PORT}" \
                --traffic-manager-port "${TM_PORT}" \
                --timeout 600 \
                --resume \
                --output-dir "/workspace/VLA4AD_Research/${OUTPUT_ROOT}/outputs/$(basename "${ROUTE_FOR_JOB}" .xml)" \
                --routes "/workspace/VLA4AD_Research/${ROUTE_FOR_JOB}"
        status=$?
        set -e

        if [ "${status}" -eq 0 ]; then
            touch "${client_done}"
            printf '{"scenario_id":%s,"route":"%s","status":"ok","attempt":%s}\n' \
                "${scenario_id}" "${route_for_job}" "${attempt}" >>"${SUMMARY_JSONL}"
            echo "Collection finished (scenario=${scenario_id})"
            return 0
        fi

        reason="client_exit_${status}"
        local server_log="${JOB_ROOT}/server_$(printf '%02d' "${scenario_id}").log"
        if [ -f "${server_log}" ] && grep -Eqi "Signal 11|segmentation fault|Segmentation fault" "${server_log}"; then
            reason="signal_11"
        elif grep -Eqi "timeout|Timed out" \
                "${JOB_ROOT}/client_$(printf '%02d' "${scenario_id}").log" 2>/dev/null; then
            reason="timeout"
        fi
        printf '{"scenario_id":%s,"route":"%s","status":"failed","attempt":%s,"reason":"%s"}\n' \
            "${scenario_id}" "${route_for_job}" "${attempt}" "${reason}" >>"${SUMMARY_JSONL}"
        echo "Collection failed status=${status} reason=${reason} (scenario=${scenario_id})" >&2
        [ "${reason}" = "signal_11" ] && break
    done

    printf '%s\n' "${route_for_job}" >>"${JOB_ROOT}/failed_routes.txt"
    touch "${client_failed}"
    return "${status:-1}"
}

# ---------------------------------------------------------------------------
# Worker (called by mpiexec)
# ---------------------------------------------------------------------------

run_worker() {
    local rank
    rank=$(rank_id)
    local scenario_id=$((rank / 2))

    # Per-scenario log file (even rank = server, odd rank = client)
    if [ "$((rank % 2))" = "0" ]; then
        exec > >(tee -a "${JOB_ROOT}/server_$(printf '%02d' "${scenario_id}").log") 2>&1
    else
        exec > >(tee -a "${JOB_ROOT}/client_$(printf '%02d' "${scenario_id}").log") 2>&1
    fi

    echo "rank=${rank} scenario=${scenario_id} host=$(hostname -f)"

    # Read route from manifest (1-indexed)
    local route_for_job
    route_for_job=$(sed -n "$((scenario_id + 1))p" "${ROUTES_MANIFEST}")
    if [ -z "${route_for_job}" ]; then
        echo "No route for scenario ${scenario_id} (manifest has fewer lines)" >&2
        exit 0   # not an error: idle rank must not abort the mpiexec job
    fi

    # Per-rank error isolation: a server/client failure is logged but the rank
    # ALWAYS exits 0, so mpiexec never SIGTERMs the sibling pairs (one flaky
    # node/route no longer kills the whole allocation). Failures are still
    # recorded in summary.jsonl / failed_routes.txt by run_server/run_client.
    if [ "$((rank % 2))" = "0" ]; then
        ( run_server "${scenario_id}" "${route_for_job}" ) \
            || echo "scenario ${scenario_id} server rank failed (isolated, rc=$?)"
    else
        ( run_client "${scenario_id}" "${route_for_job}" ) \
            || echo "scenario ${scenario_id} client rank failed (isolated, rc=$?)"
    fi
    exit 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

run_main() {
    cd "${REPO_DIR}"
    mkdir -p logs "${JOB_ROOT}" "${SERVER_READY_DIR}"
    require_file "${CARLA_SIF}"
    require_file "${LEAD_SIF}"

    echo "REPO_DIR=${REPO_DIR}"
    echo "CARLA_SIF=${CARLA_SIF}"
    echo "LEAD_SIF=${LEAD_SIF}"
    echo "CARLA_SERVER_MODE=${CARLA_SERVER_MODE}"
    echo "BASE_PORT=${BASE_PORT}"
    echo "JOB_ROOT=${JOB_ROOT}"
    echo "SUMMARY_JSONL=${SUMMARY_JSONL}"
    [ -n "${PBS_NODEFILE:-}" ] && sort -u "${PBS_NODEFILE}"

    prepare_routes
    echo "N_PAIRS=${N_PAIRS}"

    # Single extraction before parallel workers start to avoid race.
    extract_carla_pythonapi
    prepare_host_graphics_overlay

    : >"${SUMMARY_JSONL}"
    # PPN = MPI ranks per node.
    #   PPN=1 (default): each server and each client gets its own node; clients
    #          connect to a server on a DIFFERENT node (cross-node, needs
    #          2*N_PAIRS nodes).
    #   PPN=2: ranks 2k (server) and 2k+1 (client) of pair k land on the same
    #          node; the client connects to its co-located server. One pair per
    #          node, needs N_PAIRS nodes. Useful for the 2-node debug queue.
    mpiexec -n $((N_PAIRS * 2)) -ppn "${PPN:-1}" bash "$0" --worker

    echo "All scenarios finished."
    echo "Summary:"
    cat "${SUMMARY_JSONL}"

    local failed
    failed=$(python3 -c "
import json, sys
data = [json.loads(l) for l in open('${SUMMARY_JSONL}') if l.strip()]
n = sum(1 for d in data if d.get('status') != 'ok')
print(n)
" 2>/dev/null || echo 0)
    if [ "${failed}" -ne 0 ]; then
        echo "${failed} scenario(s) failed; see ${JOB_ROOT}/failed_routes.txt" >&2
        return 1
    fi
}

if [ "${1:-}" = "--worker" ]; then
    run_worker
else
    run_main
fi
