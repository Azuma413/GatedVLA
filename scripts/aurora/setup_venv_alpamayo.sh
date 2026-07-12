#!/bin/bash
# Create a venv layered on Aurora's `frameworks` module (torch 2.10+xpu,
# transformers 4.57, accelerate, deepspeed) that adds only the Alpamayo-R1 SFT
# deps missing from frameworks. Run once on a login node (needs the ALCF proxy
# for pip). The generic CUDA/IPEX container cannot see Aurora's XPUs, so Alpamayo
# SFT runs natively like VLA-JEPA (see scripts/aurora/setup_venv_vlajepa.sh).
#
# Do NOT reinstall torch/torchvision/transformers/accelerate/deepspeed/numpy —
# those come from the module; reinstalling breaks XPU. flash-attn / vllm /
# cosmos-rl (CUDA-only, RL/eval path) are intentionally omitted: SFT uses sdpa.
# NB: no `set -u` — Aurora's lmod init references unbound vars.
set -eo pipefail

REPO_DIR=${REPO_DIR:-/lus/flare/projects/RobustViT/hiratsuka/VLA4AD_Research}
VENV=${VENV:-${REPO_DIR}/outputs/alpamayo_venv}

module use /soft/modulefiles
module load frameworks

export https_proxy=${https_proxy:-http://proxy.alcf.anl.gov:3128}
export http_proxy=${http_proxy:-http://proxy.alcf.anl.gov:3128}

if [ ! -d "${VENV}" ]; then
  python -m venv --system-site-packages "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade pip
# Only deps missing from frameworks. torch/torchvision/numpy/pandas/scipy/
# transformers/accelerate/deepspeed/safetensors come from the module.
python -m pip install --no-cache-dir \
    "hydra-core>=1.3.2" \
    "hydra-colorlog>=1.2.0" \
    "omegaconf>=2.3.0" \
    "einops>=0.8.1" \
    "physical-ai-av>=0.2.0" \
    "torchmetrics==1.8.2" \
    "wandb>=0.25.0" \
    "rich>=14.3.3" \
    "mediapy>=1.2.4" \
    "av>=16.0.1" \
    "opencv-python-headless>=4.10.0"

# hydra/physical-ai-av can pull a CUDA torch as a transitive dep, which would
# shadow the frameworks XPU torch and silently fall back to CPU. Force-remove any
# torch/torchvision/nvidia-cu12 wheels that landed so the module's torch 2.10+xpu
# (via --system-site-packages) is used.
python -m pip uninstall -y torch torchvision 2>/dev/null || true
NV=$(python -m pip list 2>/dev/null | awk '/^nvidia-c|^nvidia-nccl|^nvidia-nvshmem|^nvidia-nvjitlink|^nvidia-nvtx|^cuda-bindings|^cuda-pathfinder/{print $1}')
[ -n "${NV}" ] && python -m pip uninstall -y ${NV} >/dev/null 2>&1 || true

PYTHONPATH="${REPO_DIR}/3rd_party/alpamayo/src:${REPO_DIR}/3rd_party/alpamayo" python - <<'PY'
import torch, transformers, accelerate, deepspeed, hydra, omegaconf, einops
import physical_ai_av, torchmetrics, wandb, pandas, scipy, safetensors
print("torch", torch.__version__, "xpu?", torch.xpu.is_available(), "tiles", torch.xpu.device_count())
print("transformers", transformers.__version__, "accelerate", accelerate.__version__, "deepspeed", deepspeed.__version__)
import alpamayo_r1
from finetune.sft import train_hf  # noqa: F401  (import-time smoke: no flash_attn/vllm/cosmos pulled)
print("alpamayo_r1 + finetune.sft.train_hf import OK")
print("venv layering OK")
PY
echo "venv ready at ${VENV}"
