#!/usr/bin/env bash
# Resume LEAD CARLA collection: collect only the routes that still lack a
# COMPLETE clip, shaping jobs to land in Polaris's fast `medium` exec queue.
#
# Why medium: on Polaris `prod` is a ROUTING queue -> small/medium/large by
# nodect+walltime. `medium` (25-99 nodes, walltime<=6h, max 10 queued) actually
# schedules; plain `prod`/`preemptable` sit forever. Submit to `-q prod` and let
# it route to medium by keeping each job 26-98 nodes and walltime<=6h.
#
# DRY-RUN by default (prints plan, writes batches). Re-run with LAUNCH=1 to qsub.
#
# Usage:
#   bash scripts/polaris/resume_missing_collection.sh                 # plan only
#   LAUNCH=1 bash scripts/polaris/resume_missing_collection.sh        # submit
set -euo pipefail

REPO_DIR=${REPO_DIR:-/lus/eagle/projects/RobustViT/hiratsuka/VLA4AD_Research}
cd "${REPO_DIR}"
export REPO_DIR

# Target = the production route set (latest campaign's filtered list, i.e. lead
# routes in the working-town allowlist). Override TARGET to use another list.
TARGET=${TARGET:-$(ls -dt logs/production_collection/*/routes_filtered.txt 2>/dev/null | head -1)}
# chunk=2 (NOT 10+): the CARLA server reliably SIGABRTs entering the 3rd route of
# a session (see memory carla-crash-after-2-routes). Each pair does exactly one
# batch and dies with its CARLA, so chunk=2 = ~100% yield; chunk=10 wasted 80%.
CHUNK=${CHUNK:-2}            # routes/batch; both finish (~45min) before the crash
JOB_PAIRS=${JOB_PAIRS:-13}   # pairs/job -> 26 nodes (medium min is 25)
MAX_JOBS=${MAX_JOBS:-10}     # medium max_queued per project
WALLTIME=${WALLTIME:-02:00:00}  # 2-route batch ~45min; short walltime backfills faster
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-21600}
# 1200s (not 600): under parallel-FS contention some CARLA servers need >10min
# to boot big maps; 600s timed out ~5/13 batches per job on 2026-06-28.
CARLA_READY_TIMEOUT=${CARLA_READY_TIMEOUT:-1200}
LAUNCH=${LAUNCH:-0}
WORK=${WORK:-logs/production_collection/$(date +%Y%m%d_%H%M%S)_resume_missing}

[ -n "${TARGET}" ] && [ -f "${TARGET}" ] || { echo "ERROR: TARGET list not found: '${TARGET}'" >&2; exit 2; }
mkdir -p "${WORK}/batches"
echo "=== Resume-missing plan ==="
echo "TARGET=${TARGET}  CHUNK=${CHUNK}  JOB_PAIRS=${JOB_PAIRS}  WALLTIME=${WALLTIME}"
echo "WORK=${WORK}"

ml use /soft/modulefiles >/dev/null 2>&1 || true; ml python >/dev/null 2>&1 || true

# 1. detect routes lacking a complete clip
MISSING="${WORK}/routes_filtered.txt"
python3 scripts/tools/find_missing_routes.py --target "${TARGET}" --out "${MISSING}"
n_missing=$(grep -c . "${MISSING}" || true)
[ "${n_missing}" -eq 0 ] && { echo ">> Nothing missing — dataset is complete."; exit 0; }

# 2. batch per-town
python3 scripts/tools/build_route_batches.py --routes-file "${MISSING}" \
    --out-dir "${WORK}/batches" --chunk "${CHUNK}" > "${WORK}/batch_manifest.txt"
n_batches=$(grep -c . "${WORK}/batch_manifest.txt")
n_jobs=$(( (n_batches + JOB_PAIRS - 1) / JOB_PAIRS ))
echo "missing routes: ${n_missing}   batches: ${n_batches}   jobs: ${n_jobs} (cap ${MAX_JOBS})"
if [ "${n_jobs}" -gt "${MAX_JOBS}" ]; then
    echo "NOTE: ${n_jobs} > ${MAX_JOBS} jobs; only first ${MAX_JOBS} submitted this wave. Re-run to continue."
    n_jobs=${MAX_JOBS}
fi

if [ "${LAUNCH}" != "1" ]; then
    echo ">> DRY-RUN. Re-run with LAUNCH=1 to submit ${n_jobs} jobs to medium."
    exit 0
fi

# 3. submit independent jobs to the medium exec queue (via prod routing)
: >"${WORK}/submitted_jobs.txt"
job_idx=0; start=1
while [ "${start}" -le "${n_batches}" ] && [ "${job_idx}" -lt "${MAX_JOBS}" ]; do
    end=$(( start + JOB_PAIRS - 1 )); [ "${end}" -gt "${n_batches}" ] && end=${n_batches}
    pairs=$(( end - start + 1 ))
    slice="${WORK}/job_$(printf '%03d' "${job_idx}").txt"
    sed -n "${start},${end}p" "${WORK}/batch_manifest.txt" >"${slice}"
    nodes=$(( pairs * 2 )); [ "${nodes}" -lt 26 ] && nodes=26   # medium min 25
    # shellcheck disable=SC2086
    job=$(qsub -q prod -l select=${nodes}:ngpus=4 -l walltime="${WALLTIME}" \
        -o "logs/polaris_collect_missing_$(printf '%03d' "${job_idx}").out" \
        -v ROUTES_FILE="${slice}",N_PAIRS="${pairs}",OUTPUT_ROOT=data,BASE_PORT=$((20000 + job_idx * 1000)),CLIENT_TIMEOUT="${CLIENT_TIMEOUT}",CARLA_READY_TIMEOUT="${CARLA_READY_TIMEOUT}" \
        scripts/polaris/collect_py123d_parallel.pbs)
    echo "job ${job_idx}: batches=${pairs} nodes=${nodes} -> ${job}"
    printf '%s\t%s\n' "${job%%.*}" "${slice}" >>"${WORK}/submitted_jobs.txt"
    job_idx=$(( job_idx + 1 )); start=$(( end + 1 ))
done
echo "=== submitted ${job_idx} jobs to medium. Monitor: qstat -u \$USER ==="
