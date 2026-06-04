#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/eagle/RobustViT/hiratsuka/VLA4AD_Research}
IMAGE_DIR=${IMAGE_DIR:-${REPO_DIR}/containers}
CARLA_SIF=${CARLA_SIF:-${IMAGE_DIR}/carla.sif}
LEAD_SIF=${LEAD_SIF:-${IMAGE_DIR}/lead.sif}
ROUTE=${ROUTE:-data/data_routes/lead/noScenarios/route_000391.xml}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/polaris_smoke}
CARLA_READY_TIMEOUT=${CARLA_READY_TIMEOUT:-300}
CARLA_PYTHONAPI_HOST=${CARLA_PYTHONAPI_HOST:-${REPO_DIR}/.carla_pythonapi}
CARLA_SERVER_MODE=${CARLA_SERVER_MODE:-host-vulkan}

source "${REPO_DIR}/scripts/polaris/carla_apptainer_common.sh"

cd "${REPO_DIR}"
mkdir -p logs "${OUTPUT_ROOT}/carla_alpamayo_4c_py123d" "${OUTPUT_ROOT}/results/noScenarios" "${OUTPUT_ROOT}/data/noScenarios"

if [ ! -f "${CARLA_SIF}" ]; then
    echo "Missing CARLA image: ${CARLA_SIF}" >&2
    echo "Run: qsub scripts/polaris/build_apptainer.pbs" >&2
    exit 2
fi
if [ ! -f "${LEAD_SIF}" ]; then
    echo "Missing LEAD image: ${LEAD_SIF}" >&2
    echo "Run: qsub scripts/polaris/build_apptainer.pbs" >&2
    exit 2
fi

if [ ! -f "${CARLA_PYTHONAPI_HOST}/agents/navigation/global_route_planner.py" ]; then
    echo "Extracting CARLA PythonAPI agents to ${CARLA_PYTHONAPI_HOST}"
    mkdir -p "${CARLA_PYTHONAPI_HOST}"
    apptainer exec "${CARLA_SIF}" bash -c \
        'cd /opt/carla/PythonAPI/carla && tar -cf - agents' \
        | tar -C "${CARLA_PYTHONAPI_HOST}" -xf -
fi

CARLA_PORT=${CARLA_PORT:-$(pick_port)}
CARLA_STREAMING_PORT=${CARLA_STREAMING_PORT:-$(pick_port)}
TM_PORT=${TM_PORT:-$(pick_port)}
ROUTE_NUMBER=$(basename "${ROUTE}" .xml)
CHECKPOINT_ENDPOINT="${OUTPUT_ROOT}/results/noScenarios/${ROUTE_NUMBER}_result.json"
SAVE_PATH="${OUTPUT_ROOT}/data/noScenarios"
CARLA_LOG="${REPO_DIR}/logs/carla_smoke_${PBS_JOBID:-manual}.log"
CARLA_EXIT_LOG="${REPO_DIR}/logs/carla_smoke_${PBS_JOBID:-manual}.exit.log"
CARLA_SAVED_LOG_DIR="${REPO_DIR}/logs/carla_saved_${PBS_JOBID:-manual}"
mkdir -p "${CARLA_SAVED_LOG_DIR}"
prepare_host_graphics_overlay

CARLA_BIND_ARGS=(-B "${CARLA_SAVED_LOG_DIR}:/opt/carla/CarlaUE4/Saved/Logs")
CARLA_ENV_ARGS=(
    --env "CARLA_PORT=${CARLA_PORT}"
    --env "CARLA_STREAMING_PORT=${CARLA_STREAMING_PORT}"
    --env CARLA_LOG_DIR=/tmp/carla-logs
    --env NVIDIA_DRIVER_CAPABILITIES=all
    --env DISPLAY=
)
case "${CARLA_SERVER_MODE}" in
    host|host-vulkan|host-render)
        mapfile -t CARLA_GRAPHICS_BIND_ARGS < <(carla_graphics_bind_args)
        mapfile -t CARLA_GRAPHICS_ENV_ARGS < <(carla_graphics_env_args)
        CARLA_BIND_ARGS+=("${CARLA_GRAPHICS_BIND_ARGS[@]}")
        CARLA_ENV_ARGS+=("${CARLA_GRAPHICS_ENV_ARGS[@]}")
        ;;
    software|software-vulkan)
        mapfile -t CARLA_SOFTWARE_ENV_ARGS < <(carla_software_env_args)
        CARLA_ENV_ARGS+=("${CARLA_SOFTWARE_ENV_ARGS[@]}")
        ;;
    render|vulkan|nullrhi)
        ;;
    *)
        echo "Unknown CARLA_SERVER_MODE=${CARLA_SERVER_MODE}" >&2
        echo "Expected one of: host-vulkan, host-render, host, software-vulkan, software, vulkan, render, nullrhi" >&2
        exit 2
        ;;
esac

echo "REPO_DIR=${REPO_DIR}"
echo "CARLA_SIF=${CARLA_SIF}"
echo "LEAD_SIF=${LEAD_SIF}"
echo "ROUTE=${ROUTE}"
echo "CARLA_PORT=${CARLA_PORT}"
echo "CARLA_STREAMING_PORT=${CARLA_STREAMING_PORT}"
echo "TM_PORT=${TM_PORT}"
echo "CARLA_SERVER_MODE=${CARLA_SERVER_MODE}"

cleanup() {
    if [ -n "${CARLA_PID:-}" ] && kill -0 "${CARLA_PID}" 2>/dev/null; then
        kill "${CARLA_PID}" 2>/dev/null || true
        wait "${CARLA_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

apptainer exec --nv --writable-tmpfs \
    "${CARLA_BIND_ARGS[@]}" \
    "${CARLA_ENV_ARGS[@]}" \
    "${CARLA_SIF}" bash -c '
    case "'"${CARLA_SERVER_MODE}"'" in
    nullrhi)
        exec /opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping \
            CarlaUE4 \
            -nullrhi \
            -nosound \
            -carla-rpc-port="${CARLA_PORT}" \
            -carla-streaming-port="${CARLA_STREAMING_PORT}" \
            -unattended \
            -NoSplash \
            -log \
            -stdout \
            -FullStdOutLogOutput
        ;;
    host-vulkan|vulkan)
        exec /opt/bin/carla-server -vulkan
        ;;
    software|software-vulkan)
        exec /opt/bin/carla-server -vulkan
        ;;
    host|host-render|render)
        exec /opt/bin/carla-server
        ;;
    esac
' >"${CARLA_LOG}" 2>&1 &
CARLA_PID=$!

echo "Started CARLA pid=${CARLA_PID}; log=${CARLA_LOG}"

CARLA_READY=0
SECONDS_WAITED=0
while [ "${SECONDS_WAITED}" -lt "${CARLA_READY_TIMEOUT}" ]; do
    if ! kill -0 "${CARLA_PID}" 2>/dev/null; then
        echo "CARLA process exited before becoming ready" >&2
        wait "${CARLA_PID}" >"${CARLA_EXIT_LOG}" 2>&1 || echo "CARLA exit status=$?" >&2
        tail -200 "${CARLA_LOG}" >&2 || true
        tail -50 "${CARLA_EXIT_LOG}" >&2 || true
        find "${CARLA_SAVED_LOG_DIR}" -type f -maxdepth 1 -print -exec tail -120 {} \; >&2 || true
        exit 3
    fi
    if run_carla_rgb_probe "${LEAD_SIF}" "${CARLA_PORT}" "${REPO_DIR}"
    then
        CARLA_READY=1
        break
    fi
    sleep 10
    SECONDS_WAITED=$((SECONDS_WAITED + 10))
    echo "Waiting for CARLA... ${SECONDS_WAITED}/${CARLA_READY_TIMEOUT}s"
done

if [ "${CARLA_READY}" -ne 1 ]; then
    echo "CARLA did not become ready within ${CARLA_READY_TIMEOUT}s" >&2
    tail -200 "${CARLA_LOG}" >&2 || true
    find "${CARLA_SAVED_LOG_DIR}" -type f -maxdepth 1 -print -exec tail -120 {} \; >&2 || true
    exit 4
fi

apptainer exec --nv --writable-tmpfs \
    -B "${REPO_DIR}:/workspace/VLA4AD_Research" \
    --env LEAD_PROJECT_ROOT=/workspace/VLA4AD_Research \
    --env PY123D_DATA_ROOT=/workspace/VLA4AD_Research/${OUTPUT_ROOT}/carla_alpamayo_4c_py123d \
    --env LEAD_EXPERT_CONFIG="target_dataset=7 py123d_data_format=true use_radars=false lidar_stack_size=2 save_only_non_ground_lidar=false save_lidar_only_inside_bev=false" \
    --env SCENARIO_RUNNER_ROOT=/workspace/VLA4AD_Research/3rd_party/scenario_runner_autopilot \
    --env LEADERBOARD_ROOT=/workspace/VLA4AD_Research/3rd_party/leaderboard_autopilot \
    --env PYTHONPATH=/workspace/VLA4AD_Research:/workspace/VLA4AD_Research/.carla_pythonapi:/workspace/VLA4AD_Research/3rd_party/leaderboard_autopilot:/workspace/VLA4AD_Research/3rd_party/scenario_runner_autopilot \
    --env DEBUG_CHALLENGE=0 \
    --env DATAGEN=1 \
    --env SAVE_PATH=/workspace/VLA4AD_Research/${SAVE_PATH} \
    --env CHECKPOINT_ENDPOINT=/workspace/VLA4AD_Research/${CHECKPOINT_ENDPOINT} \
    --env TEAM_CONFIG=/workspace/VLA4AD_Research/${ROUTE} \
    "${LEAD_SIF}" \
    lead-python -u 3rd_party/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py \
        --port="${CARLA_PORT}" \
        --traffic-manager-port="${TM_PORT}" \
        --traffic-manager-seed=0 \
        --routes="/workspace/VLA4AD_Research/${ROUTE}" \
        --repetitions=1 \
        --track=MAP \
        --checkpoint="/workspace/VLA4AD_Research/${CHECKPOINT_ENDPOINT}" \
        --agent="/workspace/VLA4AD_Research/lead/expert/expert_py123d.py" \
        --agent-config="/workspace/VLA4AD_Research/${ROUTE}" \
        --debug=0 \
        --resume=1 \
        --timeout=600

echo "Smoke test finished"
echo "Result: ${CHECKPOINT_ENDPOINT}"
find "${OUTPUT_ROOT}" -maxdepth 4 -type f | sort | head -50
