#!/usr/bin/env bash
# End-to-end CoC autolabeling verification on the existing CARLA->PAI clips.
#
# Runs the NVlabs alpamayo-coc-autolabeler pipeline inside containers/coc_autolabeler.sif:
#   Step 1  meta-action  (CPU, trajdata pai loader on data/carla_alpamayo_pai)
#   Step 2  keyframe      (CPU)
#   Step 2b video extract (CPU, camera_front_wide_120fov zip -> mp4)
#   Step 3  CoC labels    (VLM: model_name=qwen3.5_27b_fp8 on 4x A100)
#
# NOTE: the autolabeler needs >=17.5s (1s history + 16.5s future-min) per clip;
# short CARLA clips below that are silently dropped by trajdata. Most of the 6
# verification clips are ~17s and only the longer ones survive. This run verifies
# the pipeline end-to-end on whatever survives; large-scale runs need >=18-20s clips.
set -euo pipefail

WS=/workspace/VLA4AD_Research
COC_ROOT=${COC_ROOT:-${WS}/3rd_party/alpamayo-coc-autolabeler}
export COC_ROOT
export PYTHONPATH="${COC_ROOT}/src:${PYTHONPATH:-}"
# Keep BLAS/trajdata worker threads modest.
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

STAGE=${STAGE:-all}                 # all | meta | keyframe | video | coc
MODEL_NAME=${MODEL_NAME:-qwen3.5_27b_fp8}
NUM_WORKERS=${NUM_WORKERS:-4}

PAI_DIR=${PAI_DIR:-${WS}/data/carla_alpamayo_pai}
WORK=${WORK:-${WS}/outputs/coc_autolabel}
CACHE_DIR=${CACHE_DIR:-${WORK}/trajdata_cache}
META_DIR=${META_DIR:-${WORK}/meta_action}
KF_DIR=${KF_DIR:-${WORK}/keyframes}
VID_ROOT=${VID_ROOT:-${WORK}/extracted_videos}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-${WS}/outputs/coc_hf_cache}
EXP_NAME=${EXP_NAME:-${MODEL_NAME}_carla}

SEGMENT_JSON="${KF_DIR}/segments_relative_timestamp_sampled.json"
META_FINAL="${META_DIR}/final_outputs"
VIDEO_ZIP_DIR="${PAI_DIR}/camera/camera_front_wide_120fov"

mkdir -p "${WORK}" "${CACHE_DIR}" "${META_DIR}" "${KF_DIR}" "${VID_ROOT}" "${MODEL_CACHE_DIR}"
# Pipeline outputs (experiments/, hydra run dirs) land under WORK.
cd "${WORK}"

run_meta() {
  echo "[coc] Step 1: meta-action (trajdata pai)"
  python -m meta_action.data_labeling \
    --dataset_name pai \
    --meta_action_names all_ego \
    --data_dir "${PAI_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --save_dir "${META_DIR}" \
    --num_workers "${NUM_WORKERS}"
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
  echo "[coc] Step 3: CoC labeling with ${MODEL_NAME}"
  # Restrict GPUs for small smoke models (vLLM tensor_parallel_size = #visible GPUs).
  # e.g. COC_VISIBLE_DEVICES=0 keeps a 2B model on a single GPU on a 4-GPU node.
  if [ -n "${COC_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${COC_VISIBLE_DEVICES}"
    echo "[coc] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  fi
  export MODEL_CACHE_DIR HF_HOME="${MODEL_CACHE_DIR}"
  python -m coc_labeling.data_labeling \
    --config-name=base_config_vlm_rel_ts \
    model_name="${MODEL_NAME}" \
    resume_exp_dir=null \
    exp_name="${EXP_NAME}" \
    data.data_dir="${PAI_DIR}" \
    data.cache_dir="${CACHE_DIR}" \
    data.meta_action_dir="${META_FINAL}" \
    data.video_dir="${VID_ROOT}" \
    data.segment_config_path="${SEGMENT_JSON}" \
    data_loader.keyframe.meta_action_filter=null \
    data_loader.video.save_segment_videos=false
  echo "[coc] CoC experiment outputs under: ${WORK}/experiments"
  find "${WORK}/experiments" -name 'cot_*.yaml' 2>/dev/null | head || true
}

case "${STAGE}" in
  meta)     run_meta ;;
  keyframe) run_keyframe ;;
  video)    run_video ;;
  coc)      run_coc ;;
  prep)     run_meta; run_keyframe; run_video ;;   # CPU-only compatibility check
  all)      run_meta; run_keyframe; run_video; run_coc ;;
  *) echo "Unknown STAGE=${STAGE} (use all|meta|keyframe|video|coc|prep)" >&2; exit 2 ;;
esac

echo "=== CoC autolabel ${STAGE} done ==="
