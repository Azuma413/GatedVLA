#!/bin/bash
# Launch the VLA-JEPA driving policy server (reuses VLA-JEPA's own
# deployment/model_server/server_policy.py, no driving-specific server code).
#
# Run this in a VLA-JEPA runtime (its conda env or a CUDA vlajepa container) on
# the on-prem inference server. The LEAD Bench2Drive agent connects to it over a
# websocket (see scripts/eval/eval_vlajepa_bench2drive.sh).
#
# The checkpoint must be a trainer output .pt whose run dir contains config.yaml
# and dataset_statistics.json, e.g.:
#   outputs/vlajepa_carla/carla_ft/final_model/pytorch_model.pt
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}
VLAJEPA_DIR="${REPO_DIR}/3rd_party/VLA-JEPA"
CKPT_PATH=${CKPT_PATH:-${REPO_DIR}/outputs/vlajepa_carla/carla_ft/final_model/pytorch_model.pt}
PORT=${PORT:-10093}
CUDA=${CUDA:-0}

# Offline HF processor/tokenizer (Qwen3-VL) if pre-downloaded.
export HF_HOME=${HF_HOME:-${REPO_DIR}/outputs/hf_home}

cd "${VLAJEPA_DIR}"
export PYTHONPATH="${VLAJEPA_DIR}:${PYTHONPATH:-}"

echo "Starting VLA-JEPA policy server: ckpt=${CKPT_PATH} port=${PORT} cuda=${CUDA}"
exec python -m deployment.model_server.server_policy \
    --ckpt_path "${CKPT_PATH}" \
    --port "${PORT}" \
    --cuda "${CUDA}"
