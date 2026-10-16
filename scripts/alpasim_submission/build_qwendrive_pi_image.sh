#!/usr/bin/env bash

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Declare the QwenDrivePI bundle of scripts/pretrained/qwendrive_pi.sh and the image tag
_bundle_dir="$(realpath -s "${BUNDLE_DIR:-outputs/checkpoints/qwendrive_pi/qdpi_paiav_12ep_h25}")"
_run_dir="$(dirname "$(realpath "${_bundle_dir}/model.safetensors")")"
_tag="${TAG:-qwendrive-pi}"

# Build the alpasim gRPC wheel the image installs
_wheels="$(mktemp -d)"
trap 'rm -rf "${_wheels}"' EXIT
(cd lib/alpasim/alpasim/src/grpc && uv build --wheel --out-dir "${_wheels}")

# Build the image py123d-garage-alpasim:TAG from this checkout, starVLA, the starVLA run files and the bundle
docker buildx build \
	-f lib/alpasim/tools/submission_qwendrive_pi.Dockerfile \
	--build-context starvla=lib/starVLA \
	--build-context run="${_run_dir}" \
	--build-context bundle="${_bundle_dir}" \
	--build-context wheels="${_wheels}" \
	--build-arg GIT_HASH="$(git rev-parse --short HEAD)" \
	--load \
	-t "py123d-garage-alpasim:${_tag}" \
	.
