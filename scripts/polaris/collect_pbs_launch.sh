#!/usr/bin/env bash
# Submit N_JOBS independent collection jobs that cooperate on one shared campaign
# (shared route list + claim queue). Scale = N_JOBS * NODES_PER_JOB nodes running
# concurrently; robustness = a node failure only kills its own job, the rest keep
# draining the queue, and resubmitting re-claims stale routes.
#
#   N_JOBS=8 bash scripts/polaris/collect_pbs_launch.sh           # 8 x 16 nodes
#   N_JOBS=4 NODES_PER_JOB=24 WALLTIME=12:00:00 bash scripts/polaris/collect_pbs_launch.sh
set -euo pipefail
REPO_DIR=${REPO_DIR:-/eagle/RobustViT/hiratsuka/VLA4AD_Research}
cd "${REPO_DIR}"

N_JOBS=${N_JOBS:-8}
NODES_PER_JOB=${NODES_PER_JOB:-16}   # >=10 for prod
WALLTIME=${WALLTIME:-6:00:00}
QUEUE=${QUEUE:-prod}
CAMPAIGN=${CAMPAIGN:-lead}
TOWNS=${TOWNS:-"Town03 Town04 Town05 Town06 Town07 Town10HD Town12 Town13 Town15"}
ROUTE_SRC=${ROUTE_SRC:-data/data_routes/lead}
OUT_ROOT=${OUT_ROOT:-data}

echo "Submitting ${N_JOBS} jobs x ${NODES_PER_JOB} nodes (campaign=${CAMPAIGN}, queue=${QUEUE}, walltime=${WALLTIME})"
echo "Peak nodes: $((N_JOBS * NODES_PER_JOB))"
for j in $(seq 1 "${N_JOBS}"); do
    job=$(qsub -q "${QUEUE}" -l select="${NODES_PER_JOB}":ngpus=4 -l walltime="${WALLTIME}" \
        -v CAMPAIGN="${CAMPAIGN}",TOWNS="${TOWNS}",ROUTE_SRC="${ROUTE_SRC}",OUT_ROOT="${OUT_ROOT}" \
        scripts/polaris/collect_pbs.pbs)
    echo "  job ${j}/${N_JOBS} -> ${job}"
done
echo
echo "Monitor:  qstat -u \$USER"
echo "Progress: wc -l logs/collection_campaign/${CAMPAIGN}/summary.jsonl ; grep -c '\"status\":\"ok\"' logs/collection_campaign/${CAMPAIGN}/summary.jsonl"
echo "Data:     ${OUT_ROOT}/carla_alpamayo_4c_py123d/"
