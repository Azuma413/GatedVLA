#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/eagle/RobustViT/hiratsuka/VLA4AD_Research}
IMAGE_DIR=${IMAGE_DIR:-${REPO_DIR}/containers}
CARLA_SIF=${CARLA_SIF:-${IMAGE_DIR}/carla.sif}
LEAD_SIF=${LEAD_SIF:-${IMAGE_DIR}/lead.sif}
CARLA_READY_TIMEOUT=${CARLA_READY_TIMEOUT:-90}

source "${REPO_DIR}/scripts/polaris/carla_apptainer_common.sh"

cd "${REPO_DIR}"
mkdir -p logs

require_file() {
    if [ ! -f "$1" ]; then
        echo "Missing file: $1" >&2
        exit 2
    fi
}

require_file "${CARLA_SIF}"
require_file "${LEAD_SIF}"

run_client_probe() {
    local port=$1
    run_carla_rgb_probe "${LEAD_SIF}" "${port}" "${REPO_DIR}"
}

wait_for_carla() {
    local pid=$1
    local port=$2
    local waited=0
    while [ "${waited}" -lt "${CARLA_READY_TIMEOUT}" ]; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            return 10
        fi
        if run_client_probe "${port}"; then
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
        echo "Waiting for CARLA on ${port}: ${waited}/${CARLA_READY_TIMEOUT}s"
    done
    return 11
}

run_mode() {
    local name=$1
    shift
    local port streaming_port log_dir log_file pid status carla_status
    local -a bind_args env_args graphics_bind_args graphics_env_args software_env_args
    port=$(pick_port)
    streaming_port=$(pick_port)
    env_args=(
        --env "CARLA_RPC_PORT=${port}"
        --env "CARLA_PORT=${port}"
        --env "CARLA_STREAMING_PORT=${streaming_port}"
        --env CARLA_LOG_DIR=/opt/carla/CarlaUE4/Saved/Logs
        --env NVIDIA_DRIVER_CAPABILITIES=all
    )
    log_dir="${REPO_DIR}/logs/carla_render_${name}_${PBS_JOBID:-manual}"
    log_file="${REPO_DIR}/logs/carla_render_${name}_${PBS_JOBID:-manual}.log"
    rm -rf "${log_dir}"
    mkdir -p "${log_dir}"

    bind_args=(
        --bind "${log_dir}:/opt/carla/CarlaUE4/Saved"
        --bind "${log_dir}:/home/carla/CarlaUE4/Saved"
    )
    if [[ "${name}" == host_graphics_* ]]; then
        mapfile -t graphics_bind_args < <(carla_graphics_bind_args)
        mapfile -t graphics_env_args < <(carla_graphics_env_args)
        bind_args+=("${graphics_bind_args[@]}")
        env_args+=("${graphics_env_args[@]}")
    fi
    if [[ "${name}" == software_* ]]; then
        mapfile -t software_env_args < <(carla_software_env_args)
        env_args+=("${software_env_args[@]}")
    fi

    echo
    echo "===== ${name} ====="
    echo "port=${port} streaming_port=${streaming_port}"
    timeout --signal=TERM 120s \
    apptainer exec --nv --writable-tmpfs \
        --contain \
        "${bind_args[@]}" \
        "${env_args[@]}" \
        "${CARLA_SIF}" \
        "$@" >"${log_file}" 2>&1 &
    pid=$!

    set +e
    wait_for_carla "${pid}" "${port}"
    status=$?
    if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" >/dev/null 2>&1
        carla_status=$?
    else
        wait "${pid}" >/dev/null 2>&1
        carla_status=$?
    fi
    set -e

    echo "status=${status}"
    echo "carla_exit_status=${carla_status}"
    echo "--- ${log_file} tail ---"
    tail -120 "${log_file}" || true
    echo "--- Saved/Logs tail ---"
    find "${log_dir}" -maxdepth 1 -type f -print -exec tail -80 {} \; || true
    if [ "${status}" -eq 0 ]; then
        echo "PASS ${name}"
        return 0
    fi
    echo "FAIL ${name}"
    return 1
}

echo "host=$(hostname)"
nvidia-smi || true
prepare_host_graphics_overlay
apptainer exec --nv "${CARLA_SIF}" bash -c 'ldconfig -p | grep -E "vulkan|GLX|EGL|nvidia" | head -80 || true'
apptainer exec --nv \
    $(carla_graphics_bind_args) \
    $(carla_graphics_env_args) \
    "${CARLA_SIF}" bash -c 'ldconfig -p | grep -E "vulkan|GLX|EGL|nvidia" | head -80 || true'

overall=1
run_mode direct_render bash -c 'exec /opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="${CARLA_PORT}" -carla-streaming-port="${CARLA_STREAMING_PORT}" -unattended -NoSplash -log -stdout -FullStdOutLogOutput' && overall=0
run_mode direct_vulkan bash -c 'exec /opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -vulkan -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="${CARLA_PORT}" -carla-streaming-port="${CARLA_STREAMING_PORT}" -unattended -NoSplash -log -stdout -FullStdOutLogOutput' && overall=0
run_mode wrapper bash -c 'exec /opt/bin/carla-server' && overall=0
run_mode host_graphics_direct_render bash -c 'exec /opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="${CARLA_PORT}" -carla-streaming-port="${CARLA_STREAMING_PORT}" -unattended -NoSplash -log -stdout -FullStdOutLogOutput' && overall=0
run_mode host_graphics_direct_vulkan bash -c 'exec /opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -vulkan -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="${CARLA_PORT}" -carla-streaming-port="${CARLA_STREAMING_PORT}" -unattended -NoSplash -log -stdout -FullStdOutLogOutput' && overall=0
run_mode host_graphics_wrapper bash -c 'exec /opt/bin/carla-server' && overall=0
run_mode software_direct_vulkan bash -c 'exec /opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -vulkan -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="${CARLA_PORT}" -carla-streaming-port="${CARLA_STREAMING_PORT}" -unattended -NoSplash -log -stdout -FullStdOutLogOutput' && overall=0
run_mode software_wrapper bash -c 'exec /opt/bin/carla-server -vulkan' && overall=0

exit "${overall}"
