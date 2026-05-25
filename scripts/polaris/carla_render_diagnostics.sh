#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/eagle/RobustViT/hiratsuka/VLA4AD_Research}
IMAGE_DIR=${IMAGE_DIR:-${REPO_DIR}/containers}
CARLA_SIF=${CARLA_SIF:-${IMAGE_DIR}/carla.sif}
LEAD_SIF=${LEAD_SIF:-${IMAGE_DIR}/lead.sif}
CARLA_READY_TIMEOUT=${CARLA_READY_TIMEOUT:-90}
HOST_NVIDIA_LIB_DIR=${HOST_NVIDIA_LIB_DIR:-${REPO_DIR}/.apptainer/nvidia-libs}
HOST_NVIDIA_JSON_DIR=${HOST_NVIDIA_JSON_DIR:-${REPO_DIR}/.apptainer/nvidia-json}

cd "${REPO_DIR}"
mkdir -p logs

pick_port() {
    python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
PY
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "Missing file: $1" >&2
        exit 2
    fi
}

require_file "${CARLA_SIF}"
require_file "${LEAD_SIF}"

prepare_host_graphics_overlay() {
    mkdir -p "${HOST_NVIDIA_LIB_DIR}" "${HOST_NVIDIA_JSON_DIR}/vulkan/icd.d" "${HOST_NVIDIA_JSON_DIR}/glvnd/egl_vendor.d"
    find "${HOST_NVIDIA_LIB_DIR}" -type l -delete
    for lib in /usr/lib64/libcuda.so* /usr/lib64/libGLX_nvidia.so* /usr/lib64/libEGL_nvidia.so* /usr/lib64/libnvidia*.so*; do
        [ -e "${lib}" ] || continue
        ln -sfn "/host/usr/lib64/$(basename "${lib}")" "${HOST_NVIDIA_LIB_DIR}/$(basename "${lib}")"
    done
    cat >"${HOST_NVIDIA_JSON_DIR}/vulkan/icd.d/nvidia_icd.x86_64.json" <<'JSON'
{
    "file_format_version" : "1.0.1",
    "ICD": {
        "library_path": "/nvidia-libs/libGLX_nvidia.so.0",
        "api_version" : "1.4.303"
    }
}
JSON
    cat >"${HOST_NVIDIA_JSON_DIR}/glvnd/egl_vendor.d/10_nvidia.json" <<'JSON'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "/nvidia-libs/libEGL_nvidia.so.0"
    }
}
JSON
}

run_client_probe() {
    local port=$1
    apptainer exec --nv --writable-tmpfs \
        -B "${REPO_DIR}:/workspace/VLA4AD_Research" \
        --env LEAD_PROJECT_ROOT=/workspace/VLA4AD_Research \
        "${LEAD_SIF}" \
        lead-python - "$port" <<'PY'
import queue
import sys
import carla

port = int(sys.argv[1])
client = carla.Client("127.0.0.1", port)
client.set_timeout(10.0)
world = client.get_world()
print("world", world.get_map().name)

settings = world.get_settings()
old_settings = settings
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

actors = []
try:
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    spawn = world.get_map().get_spawn_points()[0]
    vehicle = world.try_spawn_actor(bp, spawn)
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
    for _ in range(20):
        world.tick()
        try:
            image = images.get(timeout=2.0)
            print("rgb", image.width, image.height, image.frame)
            break
        except queue.Empty:
            pass
    else:
        raise RuntimeError("no RGB image received")
finally:
    for actor in reversed(actors):
        actor.destroy()
    world.apply_settings(old_settings)
PY
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
    local port streaming_port log_dir log_file pid status
    local -a bind_args env_args
    env_args=()
    port=$(pick_port)
    streaming_port=$(pick_port)
    log_dir="${REPO_DIR}/logs/carla_render_${name}_${PBS_JOBID:-manual}"
    log_file="${REPO_DIR}/logs/carla_render_${name}_${PBS_JOBID:-manual}.log"
    rm -rf "${log_dir}"
    mkdir -p "${log_dir}"

    bind_args=(-B "${log_dir}:/opt/carla/CarlaUE4/Saved/Logs")
    if [[ "${name}" == host_graphics_* ]]; then
        bind_args+=(
            -B /usr/lib64:/host/usr/lib64:ro
            -B "${HOST_NVIDIA_LIB_DIR}:/nvidia-libs:ro"
            -B "${HOST_NVIDIA_JSON_DIR}/vulkan/icd.d:/nvidia-json/vulkan/icd.d:ro"
            -B "${HOST_NVIDIA_JSON_DIR}/glvnd/egl_vendor.d:/nvidia-json/glvnd/egl_vendor.d:ro"
        )
        env_args+=(
            --env LD_LIBRARY_PATH=/nvidia-libs:/usr/local/lib:/usr/lib/x86_64-linux-gnu
            --env VK_ICD_FILENAMES=/nvidia-json/vulkan/icd.d/nvidia_icd.x86_64.json
            --env __EGL_VENDOR_LIBRARY_FILENAMES=/nvidia-json/glvnd/egl_vendor.d/10_nvidia.json
            --env __GLX_VENDOR_LIBRARY_NAME=nvidia
        )
    fi

    echo
    echo "===== ${name} ====="
    echo "port=${port} streaming_port=${streaming_port}"
    CARLA_PORT="${port}" \
    CARLA_STREAMING_PORT="${streaming_port}" \
    DISPLAY= \
    NVIDIA_DRIVER_CAPABILITIES=all \
    timeout --signal=TERM 120s \
    apptainer exec --nv --writable-tmpfs \
        "${bind_args[@]}" \
        "${env_args[@]}" \
        "${CARLA_SIF}" \
        "$@" >"${log_file}" 2>&1 &
    pid=$!

    set +e
    wait_for_carla "${pid}" "${port}"
    status=$?
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" >/dev/null 2>&1 || true
    set -e

    echo "status=${status}"
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
apptainer exec --nv "${CARLA_SIF}" bash -lc 'ldconfig -p | grep -E "vulkan|GLX|EGL|nvidia" | head -80 || true'
apptainer exec --nv \
    -B /usr/lib64:/host/usr/lib64:ro \
    -B "${HOST_NVIDIA_LIB_DIR}:/nvidia-libs:ro" \
    -B "${HOST_NVIDIA_JSON_DIR}/vulkan/icd.d:/nvidia-json/vulkan/icd.d:ro" \
    -B "${HOST_NVIDIA_JSON_DIR}/glvnd/egl_vendor.d:/nvidia-json/glvnd/egl_vendor.d:ro" \
    --env LD_LIBRARY_PATH=/nvidia-libs:/usr/local/lib:/usr/lib/x86_64-linux-gnu \
    --env VK_ICD_FILENAMES=/nvidia-json/vulkan/icd.d/nvidia_icd.x86_64.json \
    --env __EGL_VENDOR_LIBRARY_FILENAMES=/nvidia-json/glvnd/egl_vendor.d/10_nvidia.json \
    "${CARLA_SIF}" bash -lc 'ldconfig -p | grep -E "vulkan|GLX|EGL|nvidia" | head -80 || true'

overall=1
run_mode direct_render bash -lc 'exec /opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="${CARLA_PORT}" -carla-streaming-port="${CARLA_STREAMING_PORT}" -unattended -NoSplash -log -stdout -FullStdOutLogOutput' && overall=0
run_mode direct_vulkan bash -lc 'exec /opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -vulkan -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="${CARLA_PORT}" -carla-streaming-port="${CARLA_STREAMING_PORT}" -unattended -NoSplash -log -stdout -FullStdOutLogOutput' && overall=0
run_mode wrapper bash -lc 'exec /opt/bin/carla-server' && overall=0
run_mode host_graphics_direct_render bash -lc 'exec /opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="${CARLA_PORT}" -carla-streaming-port="${CARLA_STREAMING_PORT}" -unattended -NoSplash -log -stdout -FullStdOutLogOutput' && overall=0
run_mode host_graphics_direct_vulkan bash -lc 'exec /opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -vulkan -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="${CARLA_PORT}" -carla-streaming-port="${CARLA_STREAMING_PORT}" -unattended -NoSplash -log -stdout -FullStdOutLogOutput' && overall=0
run_mode host_graphics_wrapper bash -lc 'exec /opt/bin/carla-server' && overall=0

exit "${overall}"
