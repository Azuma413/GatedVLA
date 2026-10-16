#!/bin/bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# The QwenDrivePI driver needs transformers 5 (Qwen3.5), the AlpaSim venv pins transformers 4, so the driver
# gets its own venv with the packages of lib/alpasim/tools/submission_qwendrive_pi.Dockerfile.
_venv="lib/alpasim/qwendrive_pi_venv"
_wheels="$(mktemp -d)"
trap 'rm -rf "${_wheels}"' EXIT

uv venv --python 3.12 "${_venv}"
uv pip install --python "${_venv}/bin/python" -r pyproject.toml --extra alpasim --extra qwendrive-pi
(cd lib/alpasim/alpasim/src/grpc && uv build --wheel --out-dir "${_wheels}")
uv pip install --python "${_venv}/bin/python" --find-links "${_wheels}" alpasim-grpc
uv pip install --python "${_venv}/bin/python" --no-deps -e . -e lib/starVLA
"${_venv}/bin/python" -c "import py123d_garage.policy.qwendrive_pi.qwendrive_pi_policy, starVLA.model.framework.VLM4A.QwenDrivePI"
