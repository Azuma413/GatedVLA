#!/usr/bin/env bash

set -euo pipefail

REPO_DIR=${REPO_DIR:-/eagle/RobustViT/hiratsuka/VLA4AD_Research}
HOST_NVIDIA_LIB_DIR=${HOST_NVIDIA_LIB_DIR:-${REPO_DIR}/.apptainer/nvidia-libs}
HOST_NVIDIA_JSON_DIR=${HOST_NVIDIA_JSON_DIR:-${REPO_DIR}/.apptainer/nvidia-json}

prepare_host_graphics_overlay() {
    mkdir -p \
        "${HOST_NVIDIA_LIB_DIR}" \
        "${HOST_NVIDIA_JSON_DIR}/vulkan/icd.d"

    find "${HOST_NVIDIA_LIB_DIR}" -type l -delete
    for lib in /usr/lib64/libcuda.so* /usr/lib64/libGLX_nvidia.so* /usr/lib64/libEGL_nvidia.so* /usr/lib64/libnvidia*.so*; do
        [ -e "${lib}" ] || continue
        ln -sfn "/host/usr/lib64/$(basename "${lib}")" "${HOST_NVIDIA_LIB_DIR}/$(basename "${lib}")"
    done

    if [ -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ]; then
        cp /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json "${HOST_NVIDIA_JSON_DIR}/vulkan/icd.d/nvidia_icd.json"
    else
        cat >"${HOST_NVIDIA_JSON_DIR}/vulkan/icd.d/nvidia_icd.json" <<'JSON'
{
    "file_format_version" : "1.0.1",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.277"
    }
}
JSON
    fi
}

carla_graphics_bind_args() {
    printf '%s\n' \
        --bind /usr/lib64/libGLX_nvidia.so.0:/usr/lib64/libGLX_nvidia.so.0 \
        --bind /usr/lib64/libEGL_nvidia.so.0:/usr/lib64/libEGL_nvidia.so.0 \
        --bind /usr/lib64/libnvidia-glcore.so.570.124.06:/usr/lib64/libnvidia-glcore.so.570.124.06 \
        --bind "${HOST_NVIDIA_JSON_DIR}/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json"
}

carla_graphics_env_args() {
    printf '%s\n' \
        --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
        --env CARLA_RENDERER=vulkan \
        --env CARLA_STDOUT_LOG=1 \
        --env CARLA_LOG_DIR=/opt/carla/CarlaUE4/Saved/Logs \
        --env NVIDIA_DRIVER_CAPABILITIES=all
}

carla_software_env_args() {
    printf '%s\n' \
        --env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
        --env LIBGL_ALWAYS_SOFTWARE=1 \
        --env MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
        --env LP_NUM_THREADS="${CARLA_SOFTWARE_THREADS:-16}"
}

# Write a DefaultEngine.ini copy whose default maps point at <town>, so the
# CARLA server boots directly on that map. The packaged CARLA 0.9.15 server
# ignores the command-line map argument (verified on Polaris: passing
# /Game/Carla/Maps/Town03_Opt or Town03_Opt still boots ServerDefaultMap,
# Town10HD_Opt). Booting on the correct map avoids an in-client load_world(),
# which on a remote client corrupts the TrafficManager and SIGSEGVs on the
# next get_trafficmanager()/apply_settings() call.
#
# Usage: prepare_boot_map_ini <carla_sif> <town> <out_ini>
prepare_boot_map_ini() {
    local carla_sif=$1
    local town=$2
    local out_ini=$3
    local map_obj="/Game/Carla/Maps/${town}.${town}"

    mkdir -p "$(dirname "${out_ini}")"
    apptainer exec "${carla_sif}" cat /opt/carla/CarlaUE4/Config/DefaultEngine.ini \
        > "${out_ini}.base"
    sed -E \
        -e "s#^(GameDefaultMap=).*#\1${map_obj}#" \
        -e "s#^(ServerDefaultMap=).*#\1${map_obj}#" \
        -e "s#^(EditorStartupMap=).*#\1${map_obj}#" \
        "${out_ini}.base" > "${out_ini}"
    rm -f "${out_ini}.base"
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
import carla

port = int(sys.argv[1])
client = carla.Client("127.0.0.1", port)
client.set_timeout(10.0)
world = client.get_world()
print("world", world.get_map().name)

settings = world.get_settings()
old_no_rendering_mode = settings.no_rendering_mode
old_synchronous_mode = settings.synchronous_mode
old_fixed_delta_seconds = settings.fixed_delta_seconds
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
    settings = world.get_settings()
    settings.no_rendering_mode = old_no_rendering_mode
    settings.synchronous_mode = old_synchronous_mode
    settings.fixed_delta_seconds = old_fixed_delta_seconds
    world.apply_settings(settings)
PY
}
