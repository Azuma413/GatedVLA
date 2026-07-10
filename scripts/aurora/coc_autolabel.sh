#!/usr/bin/env bash
# CoC autolabeling stages, run INSIDE containers/coc_autolabeler.sif on Aurora.
#   Step 1  meta-action  (CPU, trajdata pai loader on the PAI dataset)
#   Step 2  keyframe      (CPU)
#   Step 2b video extract (CPU, camera_front_wide_120fov zip -> mp4)
#   Step 3  CoC labels    (VLM: REMOTE Qwen3-VL served by vLLM on the Aurora XPU host)
#
# Aurora has no NVIDIA GPU, so unlike the Polaris flow (local vLLM in-container),
# Step 3 runs in *remote* mode: VLMCoCRemoteAgent posts to an OpenAI-compatible
# vLLM server the PBS wrapper launched natively on the XPU host (127.0.0.1:50000,
# port fixed by chat_utils_local_server.py).
#
# The autolabeler needs >=17.5s per clip; shorter clips are dropped by trajdata.
set -euo pipefail

WS=/workspace/VLA4AD_Research
COC_ROOT=${COC_ROOT:-${WS}/3rd_party/alpamayo-coc-autolabeler}
export COC_ROOT
export PYTHONPATH="${COC_ROOT}/src:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
# Stage 4 only talks to the local vLLM server; never via the ALCF proxy.
export no_proxy="localhost,127.0.0.1,${no_proxy:-}"
export NO_PROXY="${no_proxy}"
export http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY=""

STAGE=${STAGE:-all}                 # all | meta | keyframe | video | coc | prep
MODEL_NAME=${MODEL_NAME:-qwen3_vl_2b}
NUM_WORKERS=${NUM_WORKERS:-4}
SERVER_IP=${SERVER_IP:-127.0.0.1}
CLIP_LIST=${CLIP_LIST:-}            # optional text file of clip ids (one per line)
# Fixed CoC output dir for resumable runs. When set, the segment generator skips
# segments that already have a valid cot_*.yaml, so a walltime-limited full run can
# be re-submitted to continue. Empty => a new timestamped experiments/ dir each run.
RESUME_DIR=${RESUME_DIR:-}

PAI_DIR=${PAI_DIR:-${WS}/data/carla_alpamayo_pai}
WORK=${WORK:-${WS}/outputs/coc_autolabel_aurora}
CACHE_DIR=${CACHE_DIR:-${WORK}/trajdata_cache}
META_DIR=${META_DIR:-${WORK}/meta_action}
KF_DIR=${KF_DIR:-${WORK}/keyframes}
VID_ROOT=${VID_ROOT:-${WORK}/extracted_videos}
EXP_NAME=${EXP_NAME:-${MODEL_NAME}_carla_aurora}

# Overridable so parallel workers can each take a per-shard segment json.
SEGMENT_JSON="${SEGMENT_JSON:-${KF_DIR}/segments_relative_timestamp_sampled.json}"
META_FINAL="${META_DIR}/final_outputs"
VIDEO_ZIP_DIR="${PAI_DIR}/camera/camera_front_wide_120fov"

mkdir -p "${WORK}" "${CACHE_DIR}" "${META_DIR}" "${KF_DIR}" "${VID_ROOT}"
cd "${WORK}"

run_meta() {
  echo "[coc] Step 1: meta-action (trajdata pai)"
  local extra=()
  [ -n "${CLIP_LIST}" ] && extra+=(--scene_list "${CLIP_LIST}")
  python -m meta_action.data_labeling \
    --dataset_name pai \
    --meta_action_names all_ego \
    --data_dir "${PAI_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --save_dir "${META_DIR}" \
    --num_workers "${NUM_WORKERS}" \
    "${extra[@]}"
  echo "[coc] meta-action final_outputs:"; ls -la "${META_FINAL}" || true
}

run_keyframe() {
  echo "[coc] Step 2: keyframe selection"
  python -m coc_labeling.keyframe_auto_select \
    --meta_action_dir "${META_FINAL}" \
    --output_dir "${KF_DIR}"
  echo "[coc] keyframe json:"; ls -la "${SEGMENT_JSON}" || true
}

run_video() {
  echo "[coc] Step 2b: extract front-wide videos for keyframe clips"
  python "${COC_ROOT}/scripts/extract_pai_videos_from_index.py" \
    --index-file "${SEGMENT_JSON}" \
    --video-zip-dir "${VIDEO_ZIP_DIR}" \
    --output-video-root "${VID_ROOT}"
  echo "[coc] extracted videos:"; ls -la "${VID_ROOT}/camera" 2>/dev/null | head || true
}

run_coc() {
  echo "[coc] Step 3: CoC labeling with REMOTE ${MODEL_NAME} at ${SERVER_IP}:${COC_LABELING_MODEL_SERVER_PORT:-50000}"
  export COC_LABELING_MODEL_SERVER_IP="${SERVER_IP}"
  local resume_arg="resume_exp_dir=null"
  [ -n "${RESUME_DIR}" ] && { mkdir -p "${RESUME_DIR}"; resume_arg="resume_exp_dir=${RESUME_DIR}"; }
  # Unique hydra run dir per worker so parallel shards (same WORK) don't collide.
  local run_tag; run_tag="$(basename "${SEGMENT_JSON}" .json)_$$"
  python -m coc_labeling.data_labeling \
    --config-name=base_config_vlm_rel_ts \
    model_name="${MODEL_NAME}" \
    vlm_agent.agent_name=VLMCoCRemoteAgent \
    server_ip="${SERVER_IP}" \
    "${resume_arg}" \
    exp_name="${EXP_NAME}" \
    data.data_dir="${PAI_DIR}" \
    data.cache_dir="${CACHE_DIR}" \
    data.meta_action_dir="${META_FINAL}" \
    data.video_dir="${VID_ROOT}" \
    data.segment_config_path="${SEGMENT_JSON}" \
    data_loader.keyframe.meta_action_filter=null \
    data_loader.video.save_segment_videos=false \
    hydra.run.dir="${WORK}/hydra_runs/${run_tag}"
  echo "[coc] CoC experiment outputs under: ${WORK}/experiments"
  find "${WORK}/experiments" -name 'cot_*.yaml' 2>/dev/null | head || true
}

case "${STAGE}" in
  meta)     run_meta ;;
  keyframe) run_keyframe ;;
  video)    run_video ;;
  coc)      run_coc ;;
  prep)     run_meta; run_keyframe; run_video ;;
  all)      run_meta; run_keyframe; run_video; run_coc ;;
  *) echo "Unknown STAGE=${STAGE} (use all|meta|keyframe|video|coc|prep)" >&2; exit 2 ;;
esac

echo "=== CoC autolabel ${STAGE} done ==="
