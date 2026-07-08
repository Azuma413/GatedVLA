#!/bin/bash
# Create a venv layered on Aurora's `frameworks` module (torch 2.10+xpu) that adds
# only the VLA-JEPA deps missing from frameworks. Run once on a login node (needs
# the ALCF proxy for pip). torch/torchvision/transformers/accelerate/deepspeed/numpy
# are kept from the frameworks module (do NOT reinstall them — that breaks XPU).
# NB: no `set -u` — Aurora's lmod init references unbound vars (ZSH_EVAL_CONTEXT).
set -eo pipefail

REPO_DIR=${REPO_DIR:-/lus/flare/projects/RobustViT/hiratsuka/VLA4AD_Research}
VENV=${VENV:-${REPO_DIR}/outputs/vlajepa_venv}

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
# Only the deps missing from frameworks/2025.3.1. No torch/torchvision/numpy/
# transformers/accelerate/deepspeed here — those come from the module.
python -m pip install --no-cache-dir \
    diffusers \
    timm \
    "eva-decord==0.6.1" \
    numpydantic \
    av \
    qwen-vl-utils \
    tensorboard \
    "albumentations==1.4.18"

# timm/diffusers pull a CUDA torch as a dependency, which would shadow the
# frameworks XPU torch and silently fall back to CPU. Force-remove any torch /
# torchvision / nvidia-cu12 wheels that landed in the venv so the module's
# torch 2.10+xpu (from --system-site-packages) is used.
python -m pip uninstall -y torch torchvision 2>/dev/null || true
NV=$(python -m pip list 2>/dev/null | awk '/^nvidia-c|^nvidia-nccl|^nvidia-nvshmem|^nvidia-nvjitlink|^nvidia-nvtx|^cuda-bindings|^cuda-pathfinder/{print $1}')
[ -n "${NV}" ] && python -m pip uninstall -y ${NV} >/dev/null 2>&1 || true

python - <<'PY'
import torch, transformers, accelerate, deepspeed, diffusers, timm, decord, numpydantic, av, tensorboard
print("torch", torch.__version__, "transformers", transformers.__version__)
print("diffusers", diffusers.__version__, "timm", timm.__version__, "decord ok, av", av.__version__)
print("venv layering OK")
PY
echo "venv ready at ${VENV}"
