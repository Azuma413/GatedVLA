#!/bin/bash
#PBS -l select=1:system=polaris
#PBS -q debug
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -l filesystems=home:eagle
#PBS -A RobustViT
#PBS -o logs/carla_server_polaris.out
#PBS -e logs/carla_server_polaris.err

set -euo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "${PBS_O_WORKDIR}"
else
    cd ~/gitrepo/simlingo_multiview
fi

ml use /soft/modulefiles
ml spack-pe-base
ml apptainer

export BASE_SCRATCH_DIR=/local/scratch
export APPTAINER_TMPDIR="${BASE_SCRATCH_DIR}/apptainer-tmpdir"
mkdir -p "${APPTAINER_TMPDIR}"
export APPTAINER_CACHEDIR="${BASE_SCRATCH_DIR}/apptainer-cachedir"
mkdir -p "${APPTAINER_CACHEDIR}"

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

CARLA_SAVED_BASE="${CARLA_SAVED_BASE:-${PWD}/logs/carla_saved}"
CARLA_INSTANCE_ID="${PBS_JOBID:-manual-$$}"
CARLA_SAVED_DIR="${CARLA_SAVED_BASE}/${CARLA_INSTANCE_ID}"
mkdir -p "${CARLA_SAVED_DIR}"

export APPTAINERENV_VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export APPTAINERENV_CARLA_STDOUT_LOG=1
export APPTAINERENV_CARLA_LOG_DIR=/opt/carla/CarlaUE4/Saved/Logs

SERVER_ARGS=()
HAS_PORT_ARG=0
HAS_GRAPHICS_ARG=0
for arg in "$@"; do
    case "${arg}" in
        -carla-rpc-port|-carla-rpc-port=*)
            HAS_PORT_ARG=1
            ;;
        -graphicsadapter|-graphicsadapter=*)
            HAS_GRAPHICS_ARG=1
            ;;
    esac
done

if [ "${HAS_PORT_ARG}" = "0" ]; then
    SERVER_ARGS+=("-carla-rpc-port=${CARLA_RPC_PORT:-20000}")
fi

if [ "${HAS_GRAPHICS_ARG}" = "0" ] && [ -n "${CARLA_GRAPHICS_ADAPTER:-}" ]; then
    SERVER_ARGS+=("-graphicsadapter=${CARLA_GRAPHICS_ADAPTER}")
fi

echo "==== diagnostics ===="
date -u
hostname
echo "PWD=${PWD}"
echo "CARLA_SAVED_DIR=${CARLA_SAVED_DIR}"
echo "SERVER_ARGS=${SERVER_ARGS[*]:-<none>}"
nvidia-smi || true
echo "====================="

apptainer exec --nv --contain --writable-tmpfs \
    --bind /usr/lib64/libGLX_nvidia.so.0:/usr/lib64/libGLX_nvidia.so.0 \
    --bind /usr/lib64/libEGL_nvidia.so.0:/usr/lib64/libEGL_nvidia.so.0 \
    --bind /usr/lib64/libnvidia-glcore.so.570.124.06:/usr/lib64/libnvidia-glcore.so.570.124.06 \
    --bind /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json:/etc/vulkan/icd.d/nvidia_icd.json \
    --bind "${CARLA_SAVED_DIR}:/opt/carla/CarlaUE4/Saved" \
    --bind "${CARLA_SAVED_DIR}:/home/carla/CarlaUE4/Saved" \
    ./carla_server.sif carla-server "${SERVER_ARGS[@]}" "$@"
