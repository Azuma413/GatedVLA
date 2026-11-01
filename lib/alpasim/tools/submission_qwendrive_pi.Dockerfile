# The challenge submission image of a starVLA QwenDrivePI checkpoint; submission.Dockerfile plus starVLA.
#
# Build contexts (scripts/alpasim_submission/build_qwendrive_pi_image.sh):
#   .        this checkout (filtered by .dockerignore)
#   starvla  lib/starVLA
#   run      the starVLA step directory: model.safetensors, config.yaml, norm_stats.json
#   bundle   the garage bundle of scripts/pretrained/qwendrive_pi.sh: config.yaml, sensor_rig_0.yaml, base_vlm/
#   aeb      the emergency-braking detector, a transformers RT-DETR checkpoint; empty to bake none
#
# Challenge constraints this image follows: no network at run time (every file is baked in, HF offline),
# read-only root filesystem (caches and HOME in /tmp), at most 16 GiB VRAM (about 11 GiB in bf16).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCH_HOME=/tmp/torch

# gcc and the Python headers: Triton compiles the launchers of flash-linear-attention's kernels at warmup.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev python3-pip gcc libc6-dev git libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

COPY pyproject.toml /opt/py123d-garage/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python -r /opt/py123d-garage/pyproject.toml --extra alpasim --extra qwendrive-pi

COPY --from=wheels . /tmp/wheels
RUN uv pip install --python /opt/venv/bin/python --find-links /tmp/wheels alpasim-grpc && rm -rf /tmp/wheels

# The checkpoint before the code, so code changes rebuild only the small layers
COPY --from=run model.safetensors /opt/checkpoint/model.safetensors
COPY --from=run config.yaml /opt/checkpoint/starvla_config.yaml
COPY --from=run norm_stats.json /opt/checkpoint/norm_stats.json
COPY --from=bundle base_vlm /opt/checkpoint/base_vlm
# config.yaml always, anchors.npz only for AnchorVLA bundles (the glob may match nothing).
COPY --from=bundle config.yaml anchors.np[z] /opt/checkpoint/
COPY --from=bundle sensor_rig_0.yaml /opt/checkpoint/sensor_rig.yaml
# Enabled by the override aeb_config.detector_dir=/opt/aeb.
COPY --from=aeb . /opt/aeb

COPY --from=starvla pyproject.toml README.md LICENSE /opt/starvla/
COPY --from=starvla starVLA /opt/starvla/starVLA
COPY --from=starvla deployment /opt/starvla/deployment
RUN uv pip install --python /opt/venv/bin/python --no-deps /opt/starvla

COPY . /opt/py123d-garage
RUN uv pip install --python /opt/venv/bin/python --no-deps /opt/py123d-garage

ARG GIT_HASH=unknown
# Driver overrides baked into the submission (the challenge starts the image without arguments), e.g.
# "temporal_ensemble_plans=4 +policy_config.qwendrive_pi_config.num_action_samples=8".
ARG DRIVER_OVERRIDES=""
ENV PY123D_GARAGE_CONFIG="\
hydra.run.dir=/tmp/alpasim_driver \
policy_config.evaluation_checkpoint_file=/opt/checkpoint/model.safetensors \
policy_config.evaluation_sensor_rig_file=/opt/checkpoint/sensor_rig.yaml \
git_hash=${GIT_HASH} ${DRIVER_OVERRIDES}"

ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib TRITON_CACHE_DIR=/tmp/triton OMP_NUM_THREADS=8
ENTRYPOINT ["sh", "-c", "exec python -m py123d_garage.evaluation.alpasim.evaluate ${PY123D_GARAGE_CONFIG} host=\"${ALPASIM_DRIVER_HOST:-0.0.0.0}\" port=\"${ALPASIM_DRIVER_PORT:-6789}\" \"$@\"", "sh"]
