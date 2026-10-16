#!/usr/bin/env bash
# The PAI track with the challenge preset (+e2e_challenge=dev), the QwenDrivePI driver running on the host
# from lib/alpasim/qwendrive_pi_venv (lib/alpasim/tools/setup_qwendrive_pi_venv.sh) instead of the image.

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# Declare input and output directories relative to py123d_garage
_bundle_dir="$(realpath -s "${BUNDLE_DIR:-outputs/checkpoints/qwendrive_pi/qdpi_paiav_12ep_h25}")"
_out_dir="$(realpath -m "${OUTPUT_DIR:-outputs/evaluation/alpasim/qwendrive_pi/$(date +%F_%H-%M-%S)}")"

# Declare the port for the driver to listen on, defaulting to 6789 if not set.
export ALPASIM_DRIVER_PORT="${ALPASIM_DRIVER_PORT:-6789}"

# Start the driver in the background, offline like the image.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 lib/alpasim/qwendrive_pi_venv/bin/python -m py123d_garage.evaluation.alpasim.evaluate \
	hydra.run.dir="${_out_dir}/driver" \
	policy_config.evaluation_checkpoint_file="${_bundle_dir}/model.safetensors" \
	policy_config.evaluation_sensor_rig_file="${_bundle_dir}/sensor_rig_0.yaml" \
	port="${ALPASIM_DRIVER_PORT}" &

# Stop the driver when the wizard is done.
_driver_pid=$!
trap 'kill "${_driver_pid}" 2>/dev/null || true' EXIT

# Run the evaluation wizard, which will connect to the driver and run the simulation.
uv run --project lib/alpasim/alpasim alpasim_wizard +e2e_challenge=dev wizard.log_dir="${_out_dir}" "$@"
