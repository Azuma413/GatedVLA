#!/bin/bash
# Download the gated nvidia/Alpamayo-1.5-10B checkpoint and convert it to the
# Alpamayo-1 (A1) format that the SFT recipe loads from disk. Run on an Aurora
# login node (needs the ALCF proxy + HF_TOKEN from .env). CPU-only and fast:
# `convert_checkpoint.py to-a1` just symlinks the weight shards and rewrites
# config.json (model_type / architectures / hydra _target_ paths). No XPU.
#
#   bash scripts/aurora/prepare_ckpt_alpamayo15.sh
# Produces:
#   outputs/checkpoints/Alpamayo-1.5-10B            (raw HF download, A1.5 format)
#   outputs/checkpoints/Alpamayo-1.5-10B-A1-format  (SFT-ready, A1 format)
set -eo pipefail

REPO_DIR=${REPO_DIR:-/lus/flare/projects/RobustViT/hiratsuka/VLA4AD_Research}
RECIPES=${REPO_DIR}/3rd_party/alpamayo-recipes
RAW=${RAW:-${REPO_DIR}/outputs/checkpoints/Alpamayo-1.5-10B}
A1=${A1:-${REPO_DIR}/outputs/checkpoints/Alpamayo-1.5-10B-A1-format}

module use /soft/modulefiles
module load frameworks

cd "${REPO_DIR}"
if [ -f "${REPO_DIR}/.env" ]; then set -a; . "${REPO_DIR}/.env"; set +a; fi
export https_proxy=${https_proxy:-http://proxy.alcf.anl.gov:3128}
export http_proxy=${http_proxy:-http://proxy.alcf.anl.gov:3128}
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN

if [ ! -f "${RAW}/config.json" ]; then
  echo "=== downloading nvidia/Alpamayo-1.5-10B -> ${RAW} ==="
  hf download nvidia/Alpamayo-1.5-10B --local-dir "${RAW}"
else
  echo "=== ${RAW} already present, skipping download ==="
fi

echo "=== converting to A1 format -> ${A1} ==="
PYTHONPATH="${REPO_DIR}/3rd_party/alpamayo/src:${RECIPES}/src:${RECIPES}/recipes:${PYTHONPATH:-}" \
  python "${RECIPES}/scripts/convert_checkpoint.py" to-a1 --input "${RAW}" --output "${A1}"

echo "=== done ==="
ls -la "${A1}"
