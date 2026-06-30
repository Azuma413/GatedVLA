#!/usr/bin/env bash
# Polaris/PBS port of LEAD's slurm/data_collection/collect_data.py.
#
# Robustness model copied from the original (which is why it survives node
# flakiness, unlike the old mpiexec-coupled collect_py123d_parallel.sh):
#   * one route is processed independently on one node, with the CARLA server and
#     the leaderboard evaluator CO-LOCATED (client connects to 127.0.0.1);
#   * routes are pulled from a shared ATOMIC claim queue (mkdir), so ranks never
#     step on each other and load auto-balances;
#   * each route writes its OWN per-route checkpoint -> no shared-output race;
#   * a route failure is caught and logged; the rank ALWAYS continues and exits 0,
#     so one bad route/node never aborts the whole allocation (set -e is NOT used);
#   * resume across re-runs: a route whose checkpoint is already complete is
#     skipped; failed/unfinished routes are retried (MAX_ATTEMPTS within a run,
#     and again on resubmission since claims live only inside this PBS job).
#
# Run via collect_pbs.pbs (which calls this as the PBS job and as the per-rank
# worker through mpiexec -ppn 1).
set -uo pipefail   # deliberately NOT -e

REPO_DIR=${REPO_DIR:-/eagle/RobustViT/hiratsuka/VLA4AD_Research}
IMAGE_DIR=${IMAGE_DIR:-${REPO_DIR}/containers}
CARLA_SIF=${CARLA_SIF:-${IMAGE_DIR}/carla.sif}
LEAD_SIF=${LEAD_SIF:-${IMAGE_DIR}/lead.sif}
WS=/workspace/VLA4AD_Research

ROUTE_SRC=${ROUTE_SRC:-data/data_routes/lead}                 # scanned recursively
TOWNS=${TOWNS:-"Town03 Town04 Town05 Town06 Town07 Town10HD Town12 Town13 Town15"}
OUT_ROOT=${OUT_ROOT:-data}
DATASET=${DATASET:-carla_alpamayo_4c_py123d}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-2}
CARLA_READY_TIMEOUT=${CARLA_READY_TIMEOUT:-600}               # per-route CARLA boot wait
# Hard cap (s) on one route's evaluator. lead scenario routes are long and the
# expert_py123d agent renders many sensors (~1s/step), so a route can take
# 20-40 min; cap generously so legitimate routes complete instead of being killed.
ROUTE_TIMEOUT=${ROUTE_TIMEOUT:-2700}
BASE_PORT=${BASE_PORT:-20000}

# SHARED campaign dir: many independent PBS jobs cooperate through one route list
# + claim queue + summary here, so node death in one job is isolated and the
# others keep draining the queue. Per-PBS JOB_ROOT only holds this job's logs.
CAMPAIGN=${CAMPAIGN:-lead}
CAMPAIGN_DIR=${CAMPAIGN_DIR:-${REPO_DIR}/logs/collection_campaign/${CAMPAIGN}}
ROUTE_LIST=${ROUTE_LIST:-${CAMPAIGN_DIR}/routes_all.txt}
CLAIMS_DIR=${CLAIMS_DIR:-${CAMPAIGN_DIR}/claims}
SUMMARY=${SUMMARY:-${CAMPAIGN_DIR}/summary.jsonl}
JOB_ROOT=${JOB_ROOT:-${CAMPAIGN_DIR}/jobs/${PBS_JOBID:-manual_pbs}}
# A claim older than this means its owner (a dead node/job) will never finish, so
# the route can be safely re-claimed. Must exceed one route's worst-case time.
STALE_SECS=${STALE_SECS:-$(( CARLA_READY_TIMEOUT + ROUTE_TIMEOUT + 600 ))}
CARLA_PYTHONAPI_HOST=${CARLA_PYTHONAPI_HOST:-${REPO_DIR}/.carla_pythonapi}

source "${REPO_DIR}/scripts/polaris/carla_apptainer_common.sh"
ulimit -c 0 2>/dev/null || true

rank_id() { printf '%s' "${PMI_RANK:-${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${ALPS_APP_PE:-0}}}}"; }

route_num()  { basename "$1" .xml; }
route_scen() { local d; d=$(basename "$(dirname "$1")"); [ -n "$d" ] && printf '%s' "$d" || printf 'noScenarios'; }
route_town() { grep -m1 -oE 'town="[^"]+"' "$1" 2>/dev/null | sed -E 's/town="([^"]+)"/\1/'; }

# Port of collect_data.py is_job_done(): a route is done iff its leaderboard
# checkpoint exists, progress is complete, and no record is Failed/empty-score.
route_done() {
    local ckpt=$1
    [ -f "${ckpt}" ] || return 1
    python3 - "${ckpt}" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    cp = d["_checkpoint"]; pr = cp.get("progress", [])
    if len(pr) < 2 or pr[0] < pr[1]: sys.exit(1)
    for r in cp.get("records", []):
        if r.get("scores", {}).get("score_route", 0) <= 1e-11: sys.exit(1)
        if r.get("status", "").startswith(("Failed", "Started")): sys.exit(1)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

extract_carla_pythonapi() {
    [ -f "${CARLA_PYTHONAPI_HOST}/agents/navigation/global_route_planner.py" ] && return
    mkdir -p "${CARLA_PYTHONAPI_HOST}"
    apptainer exec "${CARLA_SIF}" bash -c 'cd /opt/carla/PythonAPI/carla && tar -cf - agents' \
        | tar -C "${CARLA_PYTHONAPI_HOST}" -xf -
}

carla_probe() {
    local port=$1
    apptainer exec --nv --writable-tmpfs -B "${REPO_DIR}:${WS}" \
        --env LEAD_PROJECT_ROOT="${WS}" "${LEAD_SIF}" \
        lead-python - "${port}" <<'PY' 2>/dev/null
import sys, carla
c = carla.Client("127.0.0.1", int(sys.argv[1])); c.set_timeout(10.0)
print(c.get_world().get_map().name)
PY
}

# Boot a CARLA server on THIS node (minimal flags: extra flags crash the TM on
# complex maps). Echoes the PID.
start_carla() {
    local port=$1 saved=$2 log=$3
    local -a gb ge
    mkdir -p "${saved}"
    mapfile -t gb < <(carla_graphics_bind_args)
    mapfile -t ge < <(carla_graphics_env_args)
    apptainer exec --nv --writable-tmpfs --contain \
        --bind "${saved}:/opt/carla/CarlaUE4/Saved" \
        --bind "${saved}:/home/carla/CarlaUE4/Saved" \
        "${gb[@]}" "${ge[@]}" \
        --env CARLA_RPC_PORT="${port}" --env NVIDIA_DRIVER_CAPABILITIES=all \
        "${CARLA_SIF}" bash -c '
            exec /opt/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 \
                -RenderOffScreen -quality-level=Epic \
                -carla-rpc-port="'"${port}"'" -graphicsadapter=0
        ' >"${log}" 2>&1 &
    printf '%s\n' "$!"
}

# Stop the CARLA server started by start_carla. CARLA/UE4 IGNORES SIGTERM, so a
# plain `kill` leaves it alive: it keeps holding the fixed rpc port, the next
# route on this node then fails to boot its own server, and a survivor lingers at
# job end so the rank/mpiexec/PBS job never cleanly terminates. Force-kill with
# SIGKILL and also nuke any CARLA still bound to this rpc port (ppn=1 => one CARLA
# per node, but match on the port to stay precise). Do NOT block on `wait`: the
# apptainer child was backgrounded inside a command-substitution subshell, so it
# is not a child of this shell and `wait` cannot reap it anyway.
stop_carla() {
    local pid=$1 port=$2
    [ -n "${pid}" ] && kill -9 "${pid}" 2>/dev/null || true
    pkill -9 -f "CarlaUE4-Linux-Shipping.*carla-rpc-port=${port}" 2>/dev/null || true
}

log_summary() {  # route status reason attempt
    printf '{"route":"%s","status":"%s","reason":"%s","attempt":%s,"rank":%s}\n' \
        "$1" "$2" "$3" "$4" "$(rank_id)" >>"${SUMMARY}"
}

process_route() {
    local route=$1
    local num scen town rpc tm outdir ckpt saved clog attempt status reason cpid ok
    num=$(route_num "${route}"); scen=$(route_scen "${route}"); town=$(route_town "${route}")
    rpc=${BASE_PORT}; tm=$((BASE_PORT + 500))
    outdir="${OUT_ROOT}/outputs/${scen}/${num}"
    ckpt="${REPO_DIR}/${outdir}/checkpoint_endpoint.json"
    saved="${JOB_ROOT}/carla_saved_$(rank_id)"
    clog="${JOB_ROOT}/carla_$(rank_id).log"

    if route_done "${ckpt}"; then
        echo "[rank $(rank_id)] SKIP done: ${route}"; log_summary "${route}" ok already_done 0; return 0
    fi
    mkdir -p "${REPO_DIR}/${outdir}"

    for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
        echo "[rank $(rank_id)] route=${num} town=${town} scen=${scen} attempt=${attempt}"
        cpid=$(start_carla "${rpc}" "${saved}" "${clog}")
        ok=""
        for _ in $(seq 1 $((CARLA_READY_TIMEOUT / 10))); do
            kill -0 "${cpid}" 2>/dev/null || break
            if carla_probe "${rpc}" >/dev/null 2>&1; then ok=1; break; fi
            sleep 10
        done
        if [ -z "${ok}" ]; then
            stop_carla "${cpid}" "${rpc}"
            reason="carla_not_ready"; status="failed"
            grep -Eqi 'Signal 11|Segmentation fault' "${clog}" && reason="signal_11"
            log_summary "${route}" "${status}" "${reason}" "${attempt}"
            continue
        fi

        status=0
        timeout "${ROUTE_TIMEOUT}" apptainer exec --nv --writable-tmpfs \
            -B "${REPO_DIR}:${WS}" \
            --env LEAD_PROJECT_ROOT="${WS}" \
            --env PY123D_DATA_ROOT="${WS}/${OUT_ROOT}/${DATASET}" \
            --env LEAD_EXPERT_CONFIG="target_dataset=7 py123d_data_format=true use_radars=false lidar_stack_size=2 save_only_non_ground_lidar=false save_lidar_only_inside_bev=false" \
            --env PYTHONPATH="${WS}:${WS}/.carla_pythonapi" \
            "${LEAD_SIF}" \
            lead-python -u -m lead --expert --py123d \
                --host 127.0.0.1 --port "${rpc}" --traffic-manager-port "${tm}" \
                --timeout 600 \
                --output-dir "${WS}/${outdir}" \
                --routes "${WS}/${route}" || status=$?

        stop_carla "${cpid}" "${rpc}"

        if route_done "${ckpt}"; then
            log_summary "${route}" ok ok "${attempt}"; echo "[rank $(rank_id)] OK ${num}"; return 0
        fi
        reason="incomplete"; [ "${status}" = "124" ] && reason="route_timeout"
        grep -Eqi 'Signal 11|Segmentation fault' "${clog}" && reason="signal_11"
        log_summary "${route}" failed "${reason}" "${attempt}"
    done
    return 0   # never propagate failure: keep the rank (and the allocation) alive
}

try_claim() {
    # Echo 0 (claimed) on success. mkdir is atomic; a claim older than STALE_SECS
    # belongs to a dead owner and is stolen (rmdir+mkdir, racy-but-rare).
    local num=$1 dir="${CLAIMS_DIR}/$1" age
    if mkdir "${dir}" 2>/dev/null; then echo 0; return; fi
    age=$(( $(date +%s) - $(stat -c %Y "${dir}" 2>/dev/null || date +%s) ))
    if [ "${age}" -gt "${STALE_SECS}" ]; then
        rmdir "${dir}" 2>/dev/null && mkdir "${dir}" 2>/dev/null && { echo 0; return; }
    fi
    echo 1
}

claim_next() {
    # Claim the next not-done, unclaimed route. Echoes its path or "".
    local n total start idx route num ckpt
    total=$(wc -l <"${ROUTE_LIST}")
    [ "${total}" -eq 0 ] && { printf ''; return 0; }
    start=$(( (RANDOM + $(rank_id) * 7919) % total ))   # spread ranks across the list
    for n in $(seq 0 $((total - 1))); do
        idx=$(( (start + n) % total + 1 ))
        route=$(sed -n "${idx}p" "${ROUTE_LIST}")
        [ -z "${route}" ] && continue
        num=$(route_num "${route}")
        ckpt="${REPO_DIR}/${OUT_ROOT}/outputs/$(route_scen "${route}")/${num}/checkpoint_endpoint.json"
        route_done "${ckpt}" && continue                # already collected -> skip (resume)
        if [ "$(try_claim "${num}")" = "0" ]; then printf '%s\n' "${route}"; return 0; fi
    done
    printf '' ; return 0
}

worker() {
    echo "[rank $(rank_id)] worker on $(hostname -f)"
    local route
    while true; do
        route=$(claim_next)
        [ -z "${route}" ] && { echo "[rank $(rank_id)] queue empty; done"; break; }
        # Subshell isolation: even if a route's processing aborts (timeout 124,
        # CARLA crash, an unexpected exit), it can NEVER take down this worker or,
        # via mpiexec, the whole allocation. The rank always stays alive.
        ( process_route "${route}" ) || echo "[rank $(rank_id)] isolated failure on ${route} (rc=$?)"
    done
    exit 0   # a worker rank must always exit 0 so mpiexec never aborts siblings
}

build_route_list() {
    # Idempotent: only the first cooperating job builds the shared list (guarded by
    # an atomic mkdir lock); the rest reuse it.
    if [ -s "${ROUTE_LIST}" ]; then echo "Reusing route list: $(wc -l <"${ROUTE_LIST}") routes"; return; fi
    if ! mkdir "${CAMPAIGN_DIR}/.build_lock" 2>/dev/null; then
        echo "Another job is building the route list; waiting..."
        while [ ! -s "${ROUTE_LIST}" ]; do sleep 5; done
        echo "Route list ready: $(wc -l <"${ROUTE_LIST}") routes"; return
    fi
    local tmp="${ROUTE_LIST}.tmp" xml town t
    : >"${tmp}"
    while IFS= read -r xml; do
        town=$(route_town "${xml}")
        for t in ${TOWNS}; do [ "${town}" = "${t}" ] && { printf '%s\n' "${xml}" >>"${tmp}"; break; }; done
    done < <(find "${ROUTE_SRC}" -name '*.xml' | sort)
    sort -R "${tmp}" -o "${tmp}" 2>/dev/null || true   # mix town order across ranks
    mv "${tmp}" "${ROUTE_LIST}"
    echo "Built route list: $(wc -l <"${ROUTE_LIST}") routes (towns: ${TOWNS})"
}

main() {
    cd "${REPO_DIR}"
    mkdir -p "${CAMPAIGN_DIR}" "${JOB_ROOT}" "${CLAIMS_DIR}" "${OUT_ROOT}/${DATASET}"
    touch "${SUMMARY}"   # shared across cooperating jobs; do NOT truncate
    [ -f "${CARLA_SIF}" ] || { echo "missing ${CARLA_SIF}" >&2; exit 2; }
    [ -f "${LEAD_SIF}" ]  || { echo "missing ${LEAD_SIF}" >&2; exit 2; }
    build_route_list
    extract_carla_pythonapi
    prepare_host_graphics_overlay
    local nodes
    nodes=$(sort -u "${PBS_NODEFILE}" | wc -l)
    echo "Launching ${nodes} per-node workers (one route at a time each)"
    mpiexec -n "${nodes}" -ppn 1 bash "$0" --worker
    echo "=== collection pass complete ==="
    echo "ok:        $(grep -c '"status":"ok"' "${SUMMARY}" 2>/dev/null)"
    echo "failed:    $(grep -c '"status":"failed"' "${SUMMARY}" 2>/dev/null)"
    grep -oE '"reason":"[^"]*"' "${SUMMARY}" 2>/dev/null | sort | uniq -c
}

if [ "${1:-}" = "--worker" ]; then worker; else main; fi
