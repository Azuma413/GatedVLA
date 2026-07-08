#!/bin/bash
# Evaluate VLA-JEPA in Bench2Drive closed-loop using LEAD's agent/controller.
#
# Prerequisites (on the on-prem NVIDIA server):
#   1. CARLA 0.9.15 running (host/port below).
#   2. VLA-JEPA policy server running: scripts/eval/vlajepa_server.sh
#   3. This script runs in the LEAD runtime (lead.sif / lead env). The LEAD agent
#      needs the `websockets` + `msgpack` packages to talk to the server:
#        pip install websockets msgpack
#
# LEAD owns waypoint integration + PID control; VLA-JEPA only predicts ego deltas.
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "${REPO_DIR}"

export LEAD_PROJECT_ROOT=${LEAD_PROJECT_ROOT:-${REPO_DIR}}

ROUTE=${ROUTE:-data/benchmark_routes/bench2drive/23687.xml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/vlajepa_b2d_eval}
CARLA_HOST=${CARLA_HOST:-127.0.0.1}
CARLA_PORT=${CARLA_PORT:-2000}
TM_PORT=${TM_PORT:-8000}
TIMEOUT=${TIMEOUT:-600}

# VLA-JEPA server connection + inference cadence (consumed by vlajepa_agent.py).
export VLAJEPA_SERVER_HOST=${VLAJEPA_SERVER_HOST:-127.0.0.1}
export VLAJEPA_SERVER_PORT=${VLAJEPA_SERVER_PORT:-10093}
export VLAJEPA_INFERENCE_INTERVAL=${VLAJEPA_INFERENCE_INTERVAL:-2}

# 4-camera input video (front_wide|cross_left|front_tele|cross_right横連結) like Alpamayo.
export LEAD_CLOSED_LOOP_CONFIG=${LEAD_CLOSED_LOOP_CONFIG:-'produce_input_video=true produce_frame_frequency=1'}

python -m lead \
    --checkpoint vlajepa \
    --vlajepa \
    --bench2drive \
    --routes "${ROUTE}" \
    --output-dir "${OUTPUT_ROOT}" \
    --host "${CARLA_HOST}" \
    --port "${CARLA_PORT}" \
    --traffic-manager-port "${TM_PORT}" \
    --timeout "${TIMEOUT}"
