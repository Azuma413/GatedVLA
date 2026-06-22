#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/eagle/RobustViT/hiratsuka/VLA4AD_Research}
IMAGE_DIR=${IMAGE_DIR:-${REPO_DIR}/containers}
CARLA_SIF=${CARLA_SIF:-${IMAGE_DIR}/carla.sif}
LEAD_SIF=${LEAD_SIF:-${IMAGE_DIR}/lead.sif}
ROUTE=${ROUTE:-data/data_routes/lead/noScenarios/route_000391.xml}
ROUTE_FOR_JOB=${ROUTE_FOR_JOB:-${ROUTE}}
OUTPUT_ROOT=${OUTPUT_ROOT:-data}
SCENARIO_ID=${SCENARIO_ID:-0}
BASE_PORT=${BASE_PORT:-20000}
MAX_RETRIES=${MAX_RETRIES:-2}
CARLA_READY_TIMEOUT=${CARLA_READY_TIMEOUT:-300}
CLIENT_READY_TIMEOUT=${CLIENT_READY_TIMEOUT:-300}
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-3600}
CARLA_PYTHONAPI_HOST=${CARLA_PYTHONAPI_HOST:-${REPO_DIR}/.carla_pythonapi}
CARLA_SERVER_MODE=${CARLA_SERVER_MODE:-host-vulkan}
JOB_ROOT=${JOB_ROOT:-${REPO_DIR}/logs/polaris_collection/${PBS_JOBID:-manual}}
SERVER_READY_DIR=${SERVER_READY_DIR:-${JOB_ROOT}/server_ready}
SERVER_ENV=${SERVER_ENV:-${SERVER_READY_DIR}/scenario_$(printf '%02d' "${SCENARIO_ID}").env}
SUMMARY_JSONL=${SUMMARY_JSONL:-${JOB_ROOT}/summary.jsonl}

source "${REPO_DIR}/scripts/polaris/carla_apptainer_common.sh"

# Avoid filling the filesystem with multi-GB CARLA core dumps on a crash.
ulimit -c 0 2>/dev/null || true

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

route_town() {
    apptainer exec \
        -B "${REPO_DIR}:/workspace/VLA4AD_Research" \
        "${LEAD_SIF}" \
        lead-python - "${ROUTE_FOR_JOB}" <<'PY'
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
import socket
import sys
import time

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
import sys
import carla

port = int(sys.argv[1])
client = carla.Client("127.0.0.1", port)
client.set_timeout(10.0)
world = client.get_world()
print("world", world.get_map().name)
PY
}

run_server() {
    cd "${REPO_DIR}"
    require_file "${CARLA_SIF}"
    require_file "${LEAD_SIF}"
    extract_carla_pythonapi

    local carla_port tm_port route_town_name carla_log carla_exit_log carla_saved_dir
    local -a carla_bind_args carla_env_args graphics_bind_args graphics_env_args software_env_args

    carla_port=${CARLA_PORT:-$((BASE_PORT + SCENARIO_ID * 10))}
    tm_port=${TM_PORT:-$((carla_port + 500))}
    route_town_name=$(route_town)
    carla_log="${JOB_ROOT}/server_$(printf '%02d' "${SCENARIO_ID}").log"
    carla_exit_log="${JOB_ROOT}/server_$(printf '%02d' "${SCENARIO_ID}").exit.log"
    carla_saved_dir="${JOB_ROOT}/carla_saved"
    mkdir -p "${carla_saved_dir}" "${SERVER_READY_DIR}"
    prepare_host_graphics_overlay

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
            echo "Unapproved CARLA_SERVER_MODE=${CARLA_SERVER_MODE}; run carla_load_world_matrix first and only collect with host-vulkan." >&2
            exit 2
            ;;
    esac

    echo "server_host=$(hostname -f)"
    echo "route=${ROUTE}"
    echo "route_for_job=${ROUTE_FOR_JOB}"
    echo "route_town=${route_town_name}"
    echo "carla_port=${carla_port}"
    echo "tm_port=${tm_port}"
    echo "carla_server_mode=${CARLA_SERVER_MODE}"

    cleanup() {
        if [ -n "${CARLA_PID:-}" ] && kill -0 "${CARLA_PID}" 2>/dev/null; then
            kill "${CARLA_PID}" 2>/dev/null || true
            wait "${CARLA_PID}" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT

    apptainer exec --nv --writable-tmpfs \
        --contain \
        "${carla_bind_args[@]}" \
        "${carla_env_args[@]}" \
        "${CARLA_SIF}" bash -c '
        case "'"${CARLA_SERVER_MODE}"'" in
        host-vulkan)
            # IMPORTANT: keep this launch minimal. Extra flags such as -vulkan /
            # -nosound / -unattended / -NoSplash were found to make the
            # client-side TrafficManager (get_trafficmanager) SIGSEGV on complex
            # maps (Town03+). The stock-style launch below
            # (-RenderOffScreen -carla-rpc-port -quality-level -graphicsadapter)
            # is verified to let get_trafficmanager succeed on Town03. Do not add
            # flags here without re-verifying TM on a complex map.
            exec /opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping \
                CarlaUE4 \
                -RenderOffScreen \
                -quality-level=Epic \
                -carla-rpc-port="${CARLA_PORT}" \
                -graphicsadapter="${CARLA_GRAPHICS_ADAPTER:-0}"
            ;;
        esac
    ' >"${carla_log}" 2>&1 &
    CARLA_PID=$!

    echo "Started CARLA pid=${CARLA_PID}; log=${carla_log}"

    local waited=0
    while [ "${waited}" -lt "${CARLA_READY_TIMEOUT}" ]; do
        if ! kill -0 "${CARLA_PID}" 2>/dev/null; then
            echo "CARLA process exited before becoming ready" >&2
            wait "${CARLA_PID}" >"${carla_exit_log}" 2>&1 || echo "CARLA exit status=$?" >&2
            tail -200 "${carla_log}" >&2 || true
            exit 3
        fi
        if run_carla_connect_probe "${LEAD_SIF}" "${carla_port}"; then
            {
                printf 'host=%q\n' "$(hostname -f)"
                printf 'rpc_port=%q\n' "${carla_port}"
                printf 'tm_port=%q\n' "${tm_port}"
                printf 'gpu_id=%q\n' "${CARLA_GRAPHICS_ADAPTER:-0}"
                printf 'route=%q\n' "${ROUTE_FOR_JOB}"
                printf 'SERVER_HOST=%q\n' "$(hostname -f)"
                printf 'CARLA_PORT=%q\n' "${carla_port}"
                printf 'TM_PORT=%q\n' "${tm_port}"
                printf 'ROUTE_TOWN=%q\n' "${route_town_name}"
                printf 'SCENARIO_ID=%q\n' "${SCENARIO_ID}"
            } >"${SERVER_ENV}.tmp"
            mv "${SERVER_ENV}.tmp" "${SERVER_ENV}"
            echo "Wrote ${SERVER_ENV}"
            break
        fi
        sleep 10
        waited=$((waited + 10))
        echo "Waiting for CARLA... ${waited}/${CARLA_READY_TIMEOUT}s"
    done

    if [ ! -f "${SERVER_ENV}" ]; then
        echo "CARLA did not become ready within ${CARLA_READY_TIMEOUT}s" >&2
        tail -200 "${carla_log}" >&2 || true
        exit 4
    fi

    waited=0
    while [ "${waited}" -lt "${CLIENT_TIMEOUT}" ]; do
        if [ -f "${JOB_ROOT}/client.done" ]; then
            echo "Client completed"
            return 0
        fi
        if [ -f "${JOB_ROOT}/client.failed" ]; then
            echo "Client failed" >&2
            return 5
        fi
        if ! kill -0 "${CARLA_PID}" 2>/dev/null; then
            echo "CARLA exited while client was running" >&2
            wait "${CARLA_PID}" >"${carla_exit_log}" 2>&1 || true
            tail -200 "${carla_log}" >&2 || true
            return 6
        fi
        sleep 10
        waited=$((waited + 10))
    done

    echo "Timed out waiting for client" >&2
    return 7
}

run_client() {
    cd "${REPO_DIR}"
    require_file "${LEAD_SIF}"

    local waited=0
    while [ "${waited}" -lt "${CLIENT_READY_TIMEOUT}" ] && [ ! -f "${SERVER_ENV}" ]; do
        sleep 5
        waited=$((waited + 5))
        echo "Waiting for ${SERVER_ENV}... ${waited}/${CLIENT_READY_TIMEOUT}s"
    done
    if [ ! -f "${SERVER_ENV}" ]; then
        echo "Server did not publish ${SERVER_ENV}" >&2
        touch "${JOB_ROOT}/client.failed"
        return 8
    fi

    # shellcheck disable=SC1090
    source "${SERVER_ENV}"
    echo "Client connecting to ${SERVER_HOST}:${CARLA_PORT}, traffic-manager=${TM_PORT}"
    if ! wait_for_port "${SERVER_HOST}" "${CARLA_PORT}" "${CLIENT_READY_TIMEOUT}"; then
        touch "${JOB_ROOT}/client.failed"
        return 9
    fi

    mkdir -p "${OUTPUT_ROOT}/carla_alpamayo_4c_py123d"
    local attempt status reason
    for attempt in $(seq 1 "${MAX_RETRIES}"); do
        echo "Client attempt ${attempt}/${MAX_RETRIES}"
        set +e
        apptainer exec --nv --writable-tmpfs \
            -B "${REPO_DIR}:/workspace/VLA4AD_Research" \
            --env LEAD_PROJECT_ROOT=/workspace/VLA4AD_Research \
            --env PY123D_DATA_ROOT=/workspace/VLA4AD_Research/${OUTPUT_ROOT}/carla_alpamayo_4c_py123d \
            --env LEAD_EXPERT_CONFIG="target_dataset=7 py123d_data_format=true use_radars=false lidar_stack_size=2 save_only_non_ground_lidar=false save_lidar_only_inside_bev=false" \
            --env PYTHONPATH=/workspace/VLA4AD_Research:/workspace/VLA4AD_Research/.carla_pythonapi \
            --env PYTHONFAULTHANDLER=1 \
            "${LEAD_SIF}" \
            lead-python -u -m lead \
                --expert \
                --py123d \
                --host "${SERVER_HOST}" \
                --port "${CARLA_PORT}" \
                --traffic-manager-port "${TM_PORT}" \
                --timeout 600 \
                --output-dir "/workspace/VLA4AD_Research/${OUTPUT_ROOT}/outputs" \
                --routes "/workspace/VLA4AD_Research/${ROUTE_FOR_JOB}"
        status=$?
        set -e

        if [ "${status}" -eq 0 ]; then
            touch "${JOB_ROOT}/client.done"
            printf '{"scenario_id":%s,"route":"%s","status":"ok","attempt":%s}\n' \
                "${SCENARIO_ID}" "${ROUTE_FOR_JOB}" "${attempt}" >>"${SUMMARY_JSONL}"
            echo "Short route collection finished"
            find "${OUTPUT_ROOT}" -maxdepth 5 -type f | sort | head -80
            return 0
        fi

        reason="client_exit_${status}"
        if [ -f "${JOB_ROOT}/server_$(printf '%02d' "${SCENARIO_ID}").log" ] && grep -Eqi "Signal 11|segmentation fault|Segmentation fault" "${JOB_ROOT}/server_$(printf '%02d' "${SCENARIO_ID}").log"; then
            reason="signal_11"
        elif grep -Eqi "timeout|Timed out" "${JOB_ROOT}/client_$(printf '%02d' "${SCENARIO_ID}").log" 2>/dev/null; then
            reason="timeout"
        fi
        printf '{"scenario_id":%s,"route":"%s","status":"failed","attempt":%s,"reason":"%s"}\n' \
            "${SCENARIO_ID}" "${ROUTE_FOR_JOB}" "${attempt}" "${reason}" >>"${SUMMARY_JSONL}"
        echo "Short route collection failed with status=${status} reason=${reason}" >&2
        if [ "${reason}" = "signal_11" ]; then
            break
        fi
    done

    printf '%s\n' "${ROUTE_FOR_JOB}" >>"${JOB_ROOT}/failed_routes.txt"
    touch "${JOB_ROOT}/client.failed"
    return "${status}"
}

run_worker() {
    local rank
    rank=$(rank_id)
    mkdir -p "${JOB_ROOT}"
    if [ "${rank}" = "0" ]; then
        exec > >(tee -a "${JOB_ROOT}/server_$(printf '%02d' "${SCENARIO_ID}").rank.log") 2>&1
    else
        exec > >(tee -a "${JOB_ROOT}/client_$(printf '%02d' "${SCENARIO_ID}").log") 2>&1
    fi

    echo "rank=${rank}"
    echo "host=$(hostname -f)"
    echo "job_root=${JOB_ROOT}"
    if [ "${rank}" = "0" ]; then
        run_server
    elif [ "${rank}" = "1" ]; then
        run_client
    else
        echo "Unexpected rank ${rank}; expected 0 or 1" >&2
        exit 2
    fi
}

run_main() {
    cd "${REPO_DIR}"
    mkdir -p logs "${JOB_ROOT}" "${SERVER_READY_DIR}"
    require_file "${CARLA_SIF}"
    require_file "${LEAD_SIF}"

    echo "REPO_DIR=${REPO_DIR}"
    echo "CARLA_SIF=${CARLA_SIF}"
    echo "LEAD_SIF=${LEAD_SIF}"
    echo "ROUTE=${ROUTE}"
    echo "ROUTE_FOR_JOB=${ROUTE_FOR_JOB}"
    echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
    echo "JOB_ROOT=${JOB_ROOT}"
    echo "SERVER_ENV=${SERVER_ENV}"
    echo "SUMMARY_JSONL=${SUMMARY_JSONL}"
    echo "BASE_PORT=${BASE_PORT}"
    echo "SCENARIO_ID=${SCENARIO_ID}"
    echo "PBS_NODEFILE=${PBS_NODEFILE:-}"
    if [ -n "${PBS_NODEFILE:-}" ]; then
        sort -u "${PBS_NODEFILE}"
    fi

    mpiexec -n 2 -ppn 1 bash "$0" --worker
}

if [ "${1:-}" = "--worker" ]; then
    run_worker
else
    run_main
fi
