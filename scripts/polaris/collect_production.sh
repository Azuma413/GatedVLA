#!/usr/bin/env bash
# Production CARLA data-collection orchestrator.
#
# Collects a large route set (default: the scenario-rich `lead` routes) in one
# campaign by:
#   1. filtering route XMLs to a working-town allowlist (TOWNS),
#   2. grouping them per-town into batched XMLs (build_route_batches.py) so each
#      CARLA server processes many same-town routes without a restart,
#   3. submitting parallel collection (collect_py123d_parallel.{pbs,sh}) at large
#      scale, splitting batches into waves of MAX_PAIRS chained prod jobs.
#
# DRY-RUN by default: prints the plan and writes the batch manifest, but does NOT
# submit anything. Re-run with LAUNCH=1 to actually submit the prod jobs.
#
# Resume: re-running with the same WORK reuses batches. A wave that dies can be
# resubmitted; already-collected logs under OUT_ROOT/carla_alpamayo_4c_py123d are
# left in place (the per-route expert evaluator skips finished routes).
#
# Usage:
#   # 1) inspect the plan (no submission):
#   TOWNS="Town03 Town04 Town05 Town06 Town07 Town10HD Town15" \
#     bash scripts/polaris/collect_production.sh
#
#   # 2) launch the full campaign at scale:
#   LAUNCH=1 TOWNS="...working towns..." CHUNK=40 MAX_PAIRS=64 \
#     bash scripts/polaris/collect_production.sh
set -euo pipefail

REPO_DIR=${REPO_DIR:-/eagle/RobustViT/hiratsuka/VLA4AD_Research}
cd "${REPO_DIR}"

ROUTE_SRC=${ROUTE_SRC:-data/data_routes/lead}     # dir scanned recursively for *.xml
# Working-town allowlist. Default = towns verified to load_world()+TM on the
# current carla.sif. Expand once the load_world matrix confirms more towns
# (e.g. add "Town04 Town05 Town10HD Town12 Town13" after Phase 0 passes).
TOWNS=${TOWNS:-"Town03 Town06 Town07 Town15"}
CHUNK=${CHUNK:-40}                                # routes per batch (per CARLA server)
OUT_ROOT=${OUT_ROOT:-data}
# Batches are spread over MANY INDEPENDENT jobs of JOB_PAIRS pairs each, NOT one
# giant mpiexec job. A monolithic 86-pair/172-node job is fragile: when any ONE
# of its ranks dies (a single flaky node), mpiexec SIGTERMs the whole job and all
# 172 nodes of work is lost (observed 2026-06-26: job 7222276 died at 2:49 from
# "rank 0 died from signal 9"). Independent jobs isolate node failures to one
# small job, schedule far faster, and resume by just resubmitting that job.
JOB_PAIRS=${JOB_PAIRS:-8}                          # pairs per independent job (=> 2*JOB_PAIRS nodes; keep >=5 for prod's 10-node min)
WALLTIME=${WALLTIME:-24:00:00}
# Server waits this long for a client (= one whole batch of CHUNK routes) to
# finish before killing CARLA. A 40-route batch runs for hours, so the worker
# default (3600s=1h) would abort it; cover the full walltime instead.
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-86400}
# Big maps (Town13/Town15) boot in ~1-2min standalone; allow margin under
# parallel filesystem contention before declaring the server dead.
CARLA_READY_TIMEOUT=${CARLA_READY_TIMEOUT:-600}
QUEUE=${QUEUE:-prod}
LAUNCH=${LAUNCH:-0}
WORK=${WORK:-logs/production_collection/$(date +%Y%m%d_%H%M%S)}

mkdir -p "${WORK}"
FILTERED="${WORK}/routes_filtered.txt"
BATCH_DIR="${WORK}/batches"
BATCH_MANIFEST="${WORK}/batch_manifest.txt"

echo "=== Production collection plan ==="
echo "ROUTE_SRC=${ROUTE_SRC}"
echo "TOWNS=${TOWNS}"
echo "CHUNK=${CHUNK}  JOB_PAIRS=${JOB_PAIRS}  QUEUE=${QUEUE}  WALLTIME=${WALLTIME}"
echo "WORK=${WORK}"
echo

# --- 1. Filter routes to the town allowlist ---------------------------------
: >"${FILTERED}"
declare -A town_count
total=0
while IFS= read -r xml; do
    town=$(grep -m1 -oE 'town="[^"]+"' "${xml}" 2>/dev/null | sed -E 's/town="([^"]+)"/\1/') || true
    [ -z "${town}" ] && continue
    for t in ${TOWNS}; do
        if [ "${town}" = "${t}" ]; then
            printf '%s\n' "${xml}" >>"${FILTERED}"
            town_count[${town}]=$(( ${town_count[${town}]:-0} + 1 ))
            total=$((total + 1))
            break
        fi
    done
done < <(find "${ROUTE_SRC}" -name '*.xml' | sort)

if [ "${total}" -eq 0 ]; then
    echo "ERROR: no routes matched TOWNS='${TOWNS}' under ${ROUTE_SRC}" >&2
    exit 2
fi

echo "Routes selected per town:"
for t in $(printf '%s\n' "${!town_count[@]}" | sort); do
    printf '  %-10s %d\n' "${t}" "${town_count[${t}]}"
done
echo "  TOTAL      ${total}"
echo

# --- 2. Group into per-town batched XMLs ------------------------------------
python3 scripts/tools/build_route_batches.py \
    --routes-file "${FILTERED}" \
    --out-dir "${BATCH_DIR}" \
    --chunk "${CHUNK}" >"${BATCH_MANIFEST}"

n_batches=$(grep -c . "${BATCH_MANIFEST}")
n_jobs=$(( (n_batches + JOB_PAIRS - 1) / JOB_PAIRS ))
echo "Batches: ${n_batches}  (chunk=${CHUNK} routes/batch)"
echo "Jobs:    ${n_jobs} independent jobs of <= ${JOB_PAIRS} pairs (=> <= $((JOB_PAIRS*2)) nodes each, ${QUEUE})"
echo "Peak nodes if all run at once: ~$((n_jobs * JOB_PAIRS * 2))"
echo "Batch manifest: ${BATCH_MANIFEST}"
echo

if [ "${LAUNCH}" != "1" ]; then
    echo ">> DRY-RUN. Review the plan above, then re-run with LAUNCH=1 to submit."
    echo ">> Example: LAUNCH=1 TOWNS=\"${TOWNS}\" CHUNK=${CHUNK} JOB_PAIRS=${JOB_PAIRS} bash scripts/polaris/collect_production.sh"
    exit 0
fi

# --- 3. Submit many INDEPENDENT small jobs (no inter-job dependency) ---------
# Each job processes a slice of the batch manifest. A node failure in one job
# only loses that job's <= JOB_PAIRS batches; the rest keep running.
job_idx=0
start=1
: >"${WORK}/submitted_jobs.txt"
while [ "${start}" -le "${n_batches}" ]; do
    end=$(( start + JOB_PAIRS - 1 ))
    [ "${end}" -gt "${n_batches}" ] && end=${n_batches}
    job_pairs=$(( end - start + 1 ))
    slice="${WORK}/job_$(printf '%03d' "${job_idx}").txt"
    sed -n "${start},${end}p" "${BATCH_MANIFEST}" >"${slice}"
    nodes=$(( job_pairs * 2 ))

    # shellcheck disable=SC2086
    job=$(qsub -q "${QUEUE}" -l select=${nodes}:ngpus=4 -l walltime="${WALLTIME}" \
        -v ROUTES_FILE="${slice}",N_PAIRS="${job_pairs}",OUTPUT_ROOT="${OUT_ROOT}",BASE_PORT=$((20000 + job_idx * 1000)),CLIENT_TIMEOUT="${CLIENT_TIMEOUT}",CARLA_READY_TIMEOUT="${CARLA_READY_TIMEOUT}" \
        scripts/polaris/collect_py123d_parallel.pbs)
    echo "job ${job_idx}: pairs=${job_pairs} nodes=${nodes} -> ${job}"
    printf '%s\t%s\n' "${job%%.*}" "${slice}" >>"${WORK}/submitted_jobs.txt"
    job_idx=$(( job_idx + 1 ))
    start=$(( end + 1 ))
done

echo
echo "=== Submitted ${n_jobs} independent jobs. Monitor with: qstat -u \$USER ==="
echo "Job->slice map: ${WORK}/submitted_jobs.txt"
echo "Per-job logs: logs/polaris_collection/<jobid>/  (summary.jsonl, failed_routes.txt)"
echo "Collected data: ${OUT_ROOT}/carla_alpamayo_4c_py123d/"
