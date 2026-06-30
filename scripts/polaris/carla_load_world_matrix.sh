#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/eagle/RobustViT/hiratsuka/VLA4AD_Research}
IMAGE_DIR=${IMAGE_DIR:-${REPO_DIR}/containers}
CARLA_SIF=${CARLA_SIF:-${IMAGE_DIR}/carla.sif}
LEAD_SIF=${LEAD_SIF:-${IMAGE_DIR}/lead.sif}
CARLA_SERVER_MODE=${CARLA_SERVER_MODE:-host-vulkan}
BASE_PORT=${BASE_PORT:-26000}
TOWN_TIMEOUT=${TOWN_TIMEOUT:-420}
CARLA_READY_TIMEOUT=${CARLA_READY_TIMEOUT:-180}
JOB_ID=${PBS_JOBID:-manual}
JOB_ROOT=${JOB_ROOT:-${REPO_DIR}/logs/polaris_collection/${JOB_ID}}
SUMMARY_JSONL=${SUMMARY_JSONL:-${JOB_ROOT}/summary.jsonl}
TOWNS=${TOWNS:-"Town01 Town01_Opt Town02 Town02_Opt Town03 Town03_Opt Town04 Town04_Opt Town05 Town05_Opt Town06 Town06_Opt Town07 Town07_Opt Town10HD Town10HD_Opt Town15"}
export REPO_DIR
export CARLA_SIF
export LEAD_SIF

source "${REPO_DIR}/scripts/polaris/carla_apptainer_common.sh"

require_file() {
    if [ ! -f "$1" ]; then
        echo "Missing file: $1" >&2
        exit 2
    fi
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
        sock.settimeout(2)
        try:
            sock.connect((host, port))
            raise SystemExit(0)
        except OSError:
            time.sleep(2)
raise SystemExit(1)
PY
}

json_escape() {
    python3 - "$1" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1]))
PY
}

append_summary() {
    local town=$1
    local status=$2
    local reason=$3
    local port=$4
    local elapsed_s=$5
    local log_file=$6
    local map_name=${7:-}
    local rgb_frame=${8:-}

    printf '{"town":%s,"status":%s,"reason":%s,"host":%s,"rpc_port":%s,"elapsed_s":%s,"log":%s,"map_name":%s,"rgb_frame":%s}\n' \
        "$(json_escape "${town}")" \
        "$(json_escape "${status}")" \
        "$(json_escape "${reason}")" \
        "$(json_escape "$(hostname -f)")" \
        "${port}" \
        "${elapsed_s}" \
        "$(json_escape "${log_file}")" \
        "$(json_escape "${map_name}")" \
        "$(json_escape "${rgb_frame}")" >>"${SUMMARY_JSONL}"
}

classify_failure() {
    local log_file=$1
    local probe_log=$2
    if grep -Eqi "Signal 11|segmentation fault|Segmentation fault" "${log_file}" "${probe_log}" 2>/dev/null; then
        printf '%s\n' "signal_11"
    elif grep -Eqi "no RGB image received|sensor no frame" "${probe_log}" 2>/dev/null; then
        printf '%s\n' "sensor_no_frame"
    elif grep -Eqi "timeout|Timed out|TIMEOUT" "${probe_log}" 2>/dev/null; then
        printf '%s\n' "timeout"
    else
        printf '%s\n' "probe_failed"
    fi
}

run_probe() {
    local host=$1
    local port=$2
    local town=$3

    apptainer exec --nv --writable-tmpfs \
        -B "${REPO_DIR}:/workspace/VLA4AD_Research" \
        --env LEAD_PROJECT_ROOT=/workspace/VLA4AD_Research \
        "${LEAD_SIF}" \
        lead-python - "${host}" "${port}" "${town}" <<'PY'
import json
import queue
import sys
import time

import carla

host = sys.argv[1]
port = int(sys.argv[2])
town = sys.argv[3]

client = carla.Client(host, port)
client.set_timeout(120.0)
started = time.time()
world = client.load_world(town)
map_name = world.get_map().name

settings = world.get_settings()
old = {
    "synchronous_mode": settings.synchronous_mode,
    "fixed_delta_seconds": settings.fixed_delta_seconds,
    "no_rendering_mode": settings.no_rendering_mode,
}
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
settings.no_rendering_mode = False
world.apply_settings(settings)

actors = []
rgb_frame = None
try:
    vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    vehicle = None
    for spawn in world.get_map().get_spawn_points()[:50]:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn)
        if vehicle is not None:
            break
    if vehicle is None:
        raise RuntimeError("failed to spawn vehicle")
    actors.append(vehicle)

    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "320")
    cam_bp.set_attribute("image_size_y", "240")
    cam_bp.set_attribute("fov", "90")
    camera = world.spawn_actor(
        cam_bp,
        carla.Transform(carla.Location(x=1.5, z=2.0)),
        attach_to=vehicle,
    )
    actors.append(camera)

    images = queue.Queue()
    camera.listen(images.put)
    for _ in range(50):
        world.tick()
        try:
            image = images.get(timeout=2.0)
        except queue.Empty:
            continue
        rgb_frame = image.frame
        break
    if rgb_frame is None:
        raise RuntimeError("sensor no frame: no RGB image received")

    for _ in range(30):
        world.tick()
finally:
    for actor in reversed(actors):
        actor.destroy()
    settings = world.get_settings()
    settings.synchronous_mode = old["synchronous_mode"]
    settings.fixed_delta_seconds = old["fixed_delta_seconds"]
    settings.no_rendering_mode = old["no_rendering_mode"]
    world.apply_settings(settings)

print(json.dumps({
    "town": town,
    "map_name": map_name,
    "rgb_frame": rgb_frame,
    "elapsed_s": round(time.time() - started, 3),
}, sort_keys=True))
PY
}
export -f run_probe

start_carla() {
    local port=$1
    local saved_dir=$2
    local log_file=$3
    local -a bind_args env_args graphics_bind_args graphics_env_args software_env_args

    bind_args=(
        --bind "${saved_dir}:/opt/carla/CarlaUE4/Saved"
        --bind "${saved_dir}:/home/carla/CarlaUE4/Saved"
    )
    env_args=(
        --env "CARLA_RPC_PORT=${port}"
        --env "CARLA_PORT=${port}"
        --env CARLA_LOG_DIR=/opt/carla/CarlaUE4/Saved/Logs
        --env NVIDIA_DRIVER_CAPABILITIES=all
        --env "CARLA_EXTRA_FLAGS=${CARLA_EXTRA_FLAGS:-}"
    )

    case "${CARLA_SERVER_MODE}" in
        host-vulkan)
            mapfile -t graphics_bind_args < <(carla_graphics_bind_args)
            mapfile -t graphics_env_args < <(carla_graphics_env_args)
            bind_args+=("${graphics_bind_args[@]}")
            env_args+=("${graphics_env_args[@]}")
            ;;
        software-vulkan)
            mapfile -t software_env_args < <(carla_software_env_args)
            env_args+=("${software_env_args[@]}")
            ;;
        host-opengl|vulkan|opengl|nullrhi)
            ;;
        *)
            echo "Unknown or unapproved CARLA_SERVER_MODE=${CARLA_SERVER_MODE}" >&2
            echo "Allowed: host-vulkan, software-vulkan, host-opengl, vulkan, opengl, nullrhi" >&2
            exit 2
            ;;
    esac

    apptainer exec --nv --writable-tmpfs \
        --contain \
        "${bind_args[@]}" \
        "${env_args[@]}" \
        "${CARLA_SIF}" bash -c '
        case "'"${CARLA_SERVER_MODE}"'" in
        nullrhi) renderer=(-nullrhi) ;;
        host-opengl|opengl) renderer=(-opengl -RenderOffScreen) ;;
        *) renderer=(-vulkan -RenderOffScreen) ;;
        esac
        exec /opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping \
            CarlaUE4 \
            "${renderer[@]}" \
            -nosound \
            -quality-level=Low \
            -carla-rpc-port="${CARLA_PORT}" \
            -graphicsadapter="${CARLA_GRAPHICS_ADAPTER:-0}" \
            -unattended \
            -NoSplash \
            -log \
            -stdout \
            -FullStdOutLogOutput \
            ${CARLA_EXTRA_FLAGS}
    ' >"${log_file}" 2>&1 &
    printf '%s\n' "$!"
}

run_one_town() {
    local idx=$1
    local town=$2
    local port=$((BASE_PORT + idx * 10))
    local saved_dir="${JOB_ROOT}/carla_saved/${town}"
    local server_log="${JOB_ROOT}/server_${idx}_${town}.log"
    local probe_log="${JOB_ROOT}/probe_${idx}_${town}.log"
    local started ended elapsed reason map_name rgb_frame

    mkdir -p "${saved_dir}"
    started=$(date +%s)
    echo "=== ${town} port=${port} ==="
    CARLA_PID=$(start_carla "${port}" "${saved_dir}" "${server_log}")
    trap 'kill "${CARLA_PID}" 2>/dev/null || true' RETURN

    if ! wait_for_port 127.0.0.1 "${port}" "${CARLA_READY_TIMEOUT}"; then
        ended=$(date +%s)
        kill "${CARLA_PID}" 2>/dev/null || true
        append_summary "${town}" "failed" "server_ready_timeout" "${port}" "$((ended - started))" "${server_log}"
        return 1
    fi

    set +e
    timeout "${TOWN_TIMEOUT}" bash -c 'run_probe "$@"' _ 127.0.0.1 "${port}" "${town}" >"${probe_log}" 2>&1
    local probe_status=$?
    set -e

    ended=$(date +%s)
    elapsed=$((ended - started))
    if [ "${probe_status}" -eq 0 ]; then
        map_name=$(python3 - "${probe_log}" <<'PY'
import json, sys
for line in open(sys.argv[1], encoding="utf-8", errors="ignore"):
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
        data = json.loads(line)
        print(data.get("map_name", ""))
        break
PY
)
        rgb_frame=$(python3 - "${probe_log}" <<'PY'
import json, sys
for line in open(sys.argv[1], encoding="utf-8", errors="ignore"):
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
        data = json.loads(line)
        print(data.get("rgb_frame", ""))
        break
PY
)
        append_summary "${town}" "ok" "ok" "${port}" "${elapsed}" "${server_log}" "${map_name}" "${rgb_frame}"
        echo "OK ${town} map=${map_name} rgb_frame=${rgb_frame}"
    else
        if ! kill -0 "${CARLA_PID}" 2>/dev/null; then
            reason=$(classify_failure "${server_log}" "${probe_log}")
        elif [ "${probe_status}" -eq 124 ]; then
            reason="timeout"
        else
            reason=$(classify_failure "${server_log}" "${probe_log}")
        fi
        append_summary "${town}" "failed" "${reason}" "${port}" "${elapsed}" "${server_log}"
        echo "FAILED ${town} reason=${reason}; see ${probe_log} and ${server_log}" >&2
    fi

    kill "${CARLA_PID}" 2>/dev/null || true
    wait "${CARLA_PID}" 2>/dev/null || true
    trap - RETURN
}

main() {
    cd "${REPO_DIR}"
    require_file "${CARLA_SIF}"
    require_file "${LEAD_SIF}"
    mkdir -p "${JOB_ROOT}/carla_saved"
    : >"${SUMMARY_JSONL}"
    prepare_host_graphics_overlay

    echo "CARLA_SIF=${CARLA_SIF}"
    echo "LEAD_SIF=${LEAD_SIF}"
    echo "CARLA_SERVER_MODE=${CARLA_SERVER_MODE}"
    echo "JOB_ROOT=${JOB_ROOT}"
    echo "SUMMARY_JSONL=${SUMMARY_JSONL}"
    echo "TOWNS=${TOWNS}"

    local idx=0
    local failed=0
    for town in ${TOWNS}; do
        if ! run_one_town "${idx}" "${town}"; then
            failed=$((failed + 1))
        fi
        idx=$((idx + 1))
    done

    echo "Summary:"
    cat "${SUMMARY_JSONL}"
    if [ "${failed}" -ne 0 ]; then
        echo "${failed} town(s) failed" >&2
        return 1
    fi
}

main "$@"
