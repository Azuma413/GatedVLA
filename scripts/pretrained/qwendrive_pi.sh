#!/usr/bin/env bash

set -euo pipefail

# Go to the root of the repository, so that relative paths work.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Declare the starVLA QwenDrivePI checkpoint (step directory with model.safetensors, config.yaml and
# norm_stats.json) and the bundle directory the garage evaluates.
_run_dir="${RUN_DIR:-outputs/qwendrive_pi/qdpi_paiav_12ep_h25}"
_step_dir="$(realpath "${STEP_DIR:-$(find "${_run_dir}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'step_*' | sort | tail -1)}")"
_out_dir="${OUT_DIR:-outputs/checkpoints/qwendrive_pi/$(basename "${_run_dir}")}"
mkdir -p "${_out_dir}"

# Link the starVLA run files under the names the QwenDrivePI policy reads
ln -sfn "${_step_dir}/model.safetensors" "${_out_dir}/model.safetensors"
ln -sfn "${_step_dir}/config.yaml" "${_out_dir}/starvla_config.yaml"
ln -sfn "${_step_dir}/norm_stats.json" "${_out_dir}/norm_stats.json"

# AnchorVLA runs read their anchor codebook from framework.anchor.path at build time, not from the checkpoint;
# copy the local copy of that file (ANCHORS, or <meta dir of anchor.path> next to the run directory) as a
# plain file, since the image build copies the bundle without following links out of it.
_anchor_path="$(uv run --project lib/alpasim/alpasim python -c 'import sys; from omegaconf import OmegaConf; a = OmegaConf.load(sys.argv[1]).framework.get("anchor", {}); print(a.get("path") or "" if a.get("enabled", False) else "")' "${_step_dir}/config.yaml")"
rm -f "${_out_dir}/anchors.npz"
if [[ -n "${_anchor_path}" ]]; then
	_anchors="${ANCHORS:-$(dirname "${_run_dir}")/$(basename "$(dirname "${_anchor_path}")")/$(basename "${_anchor_path}")}"
	[[ -f "${_anchors}" ]] || {
		echo "the anchor codebook ${_anchors} of ${_anchor_path} is missing; set ANCHORS" >&2
		exit 1
	}
	cp "${_anchors}" "${_out_dir}/anchors.npz"
fi

# Download the config, tokenizer and processor files of the base VLM; the checkpoint holds its weights
_base_vlm="$(uv run --project lib/alpasim/alpasim python -c 'import sys; from omegaconf import OmegaConf; print(OmegaConf.load(sys.argv[1]).framework.qwenvl.base_vlm)' "${_step_dir}/config.yaml")"
uvx --from "huggingface_hub>=0.34" hf download "${_base_vlm}" \
	config.json generation_config.json preprocessor_config.json chat_template.jinja chat_template.json \
	tokenizer.json tokenizer_config.json vocab.json merges.txt \
	--local-dir "${_out_dir}/base_vlm/$(basename "${_base_vlm}")"
rm -rf "${_out_dir}/base_vlm/$(basename "${_base_vlm}")/.cache"

# Write the policy config
cat >"${_out_dir}/config.yaml" <<EOF
policy_config:
  target: py123d_garage.policy.qwendrive_pi.qwendrive_pi_policy:QwenDrivePIPolicy
  qwendrive_pi_config: {}
EOF

# Write the sensor rig: the PAI front wide camera the model was trained on
cat >"${_out_dir}/sensor_rig_0.yaml" <<EOF
modalities:
  camera.ftcam_f0:
    camera_model: ftheta
    camera_name: camera_front_wide_120fov
    camera_id: 12
    intrinsics:
    - 961.27124
    - 753.274353
    - 0.0
    - 931.7612940673102
    - 32.987987112086635
    - -72.82956582342041
    - 22.720089068156213
    - 0.0
    - 0.0
    - 0.0010741659390590083
    - -4.5748863088807144e-08
    - 1.0254764446258745e-10
    - -3.2366948806555124e-14
    - 0.0
    width: 1920
    height: 1080
    camera_to_imu_se3:
    - 1.801815152168274
    - -0.042615413665771484
    - 1.2806285619735718
    - -0.49909161198285273
    - 0.4954226054863875
    - -0.5023466779326323
    - 0.5031021963691722
EOF
echo "QwenDrivePI bundle of ${_step_dir} in ${_out_dir}"
