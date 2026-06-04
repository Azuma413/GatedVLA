#!/usr/bin/env bash

set -euo pipefail

REPO_DIR=${REPO_DIR:-/eagle/RobustViT/hiratsuka/VLA4AD_Research}
HOST_NVIDIA_LIB_DIR=${HOST_NVIDIA_LIB_DIR:-${REPO_DIR}/.apptainer/nvidia-libs}
HOST_NVIDIA_JSON_DIR=${HOST_NVIDIA_JSON_DIR:-${REPO_DIR}/.apptainer/nvidia-json}

prepare_host_graphics_overlay() {
    mkdir -p \
        "${HOST_NVIDIA_LIB_DIR}" \
        "${HOST_NVIDIA_JSON_DIR}/vulkan/icd.d" \
        "${HOST_NVIDIA_JSON_DIR}/vulkan/implicit_layer.d" \
        "${HOST_NVIDIA_JSON_DIR}/glvnd/egl_vendor.d"

    find "${HOST_NVIDIA_LIB_DIR}" -type l -delete
    for lib in /usr/lib64/libcuda.so* /usr/lib64/libGLX_nvidia.so* /usr/lib64/libEGL_nvidia.so* /usr/lib64/libnvidia*.so*; do
        [ -e "${lib}" ] || continue
        ln -sfn "/host/usr/lib64/$(basename "${lib}")" "${HOST_NVIDIA_LIB_DIR}/$(basename "${lib}")"
    done

    if [ -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ]; then
        cp /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json "${HOST_NVIDIA_JSON_DIR}/vulkan/icd.d/nvidia_icd.x86_64.json"
    else
        cat >"${HOST_NVIDIA_JSON_DIR}/vulkan/icd.d/nvidia_icd.x86_64.json" <<'JSON'
{
    "file_format_version" : "1.0.1",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.277"
    }
}
JSON
    fi

    if [ -f /usr/share/vulkan/implicit_layer.d/nvidia_layers.json ]; then
        cp /usr/share/vulkan/implicit_layer.d/nvidia_layers.json "${HOST_NVIDIA_JSON_DIR}/vulkan/implicit_layer.d/nvidia_layers.json"
    fi

    cat >"${HOST_NVIDIA_JSON_DIR}/glvnd/egl_vendor.d/10_nvidia.json" <<'JSON'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
JSON
}

carla_graphics_bind_args() {
    printf '%s\n' \
        -B /usr/lib64:/host/usr/lib64:ro \
        -B "${HOST_NVIDIA_LIB_DIR}:/nvidia-libs:ro" \
        -B "${HOST_NVIDIA_JSON_DIR}/vulkan/icd.d:/nvidia-json/vulkan/icd.d:ro" \
        -B "${HOST_NVIDIA_JSON_DIR}/vulkan/implicit_layer.d:/nvidia-json/vulkan/implicit_layer.d:ro" \
        -B "${HOST_NVIDIA_JSON_DIR}/glvnd/egl_vendor.d:/nvidia-json/glvnd/egl_vendor.d:ro"
}

carla_graphics_env_args() {
    printf '%s\n' \
        --env LD_LIBRARY_PATH=/nvidia-libs:/usr/local/lib:/usr/lib/x86_64-linux-gnu \
        --env VK_ICD_FILENAMES=/nvidia-json/vulkan/icd.d/nvidia_icd.x86_64.json \
        --env VK_LAYER_PATH=/nvidia-json/vulkan/implicit_layer.d \
        --env __EGL_VENDOR_LIBRARY_FILENAMES=/nvidia-json/glvnd/egl_vendor.d/10_nvidia.json \
        --env __GLX_VENDOR_LIBRARY_NAME=nvidia \
        --env NVIDIA_DRIVER_CAPABILITIES=all
}

carla_software_env_args() {
    printf '%s\n' \
        --env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
        --env LIBGL_ALWAYS_SOFTWARE=1 \
        --env MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
        --env LP_NUM_THREADS="${CARLA_SOFTWARE_THREADS:-16}"
}

pick_port() {
    python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
PY
}

run_carla_rgb_probe() {
    local lead_sif=$1
    local port=$2
    local bind_root=${3:-${REPO_DIR}}

    apptainer exec --nv --writable-tmpfs \
        -B "${bind_root}:/workspace/VLA4AD_Research" \
        --env LEAD_PROJECT_ROOT=/workspace/VLA4AD_Research \
        "${lead_sif}" \
        lead-python - "${port}" <<'PY'
import queue
import sys
import copy
import carla

port = int(sys.argv[1])
client = carla.Client("127.0.0.1", port)
client.set_timeout(10.0)
world = client.get_world()
print("world", world.get_map().name)

settings = world.get_settings()
old_settings = copy.copy(settings)
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
settings.no_rendering_mode = False
world.apply_settings(settings)

actors = []
try:
    vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    vehicle = None
    for spawn in spawn_points[:20]:
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
    for _ in range(40):
        world.tick()
        try:
            image = images.get(timeout=2.0)
        except queue.Empty:
            continue
        print("rgb", image.width, image.height, image.frame)
        break
    else:
        raise RuntimeError("no RGB image received")
finally:
    for actor in reversed(actors):
        actor.destroy()
    world.apply_settings(old_settings)
PY
}
