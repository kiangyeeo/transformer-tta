#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
EXPERIMENT_ROOT="${PROJECT_DIR}/experiment_logs/shot_otta_office_lbi_joint_sweep_20260818"
PLAN="${EXPERIMENT_ROOT}/plans/machine3/plan.json"
RUNS_ROOT="${EXPERIMENT_ROOT}/runs"
LOG_ROOT="${EXPERIMENT_ROOT}/launcher_logs/machine3"
COMMAND_HISTORY="${EXPERIMENT_ROOT}/command_history/machine3/commands.sh"
LOCK_DIR="${EXPERIMENT_ROOT}/locks/machine3.lock"
PREPARE_READY="${EXPERIMENT_ROOT}/phase_records/machine3/PREPARE_READY.json"
PLAN_SHA256="4da45bfc7762c801d9fa2e62e349a02f4ec7f021315b4842cc4bce67f3067984"
LOCK_ACQUIRED=0

release_lock() {
    if [[ "${LOCK_ACQUIRED}" == "1" ]] && [[ -f "${LOCK_DIR}/pid" ]]; then
        if [[ "$(<"${LOCK_DIR}/pid")" == "$$" ]]; then
            rm -f "${LOCK_DIR}/hostname" "${LOCK_DIR}/pid" "${LOCK_DIR}/start_time"
            rmdir "${LOCK_DIR}" 2>/dev/null || true
        fi
    fi
}

trap 'exit 130' INT
trap 'exit 143' TERM
trap release_lock EXIT

[[ -d "${PROJECT_DIR}" ]]
cd "${PROJECT_DIR}"
[[ "$(pwd -P)" == "${PROJECT_DIR}" ]]
[[ -f "${PLAN}" ]]
[[ -f "${PREPARE_READY}" ]]

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda not found" >&2
    exit 1
}

conda run --no-capture-output -n SHOT_TTA python - "${PREPARE_READY}" "${PLAN_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if record.get("plan_sha256") != sys.argv[2]:
    raise SystemExit("PREPARE_READY plan_sha256 does not match the frozen plan SHA256")
if record.get("experiment_count") != 132:
    raise SystemExit("PREPARE_READY experiment_count is not 132")
PY

conda run --no-capture-output -n SHOT_TTA python tools/validate_office_lbi_joint_sweep_machine3_plan.py \
    "${PLAN}" --expected-sha256 "${PLAN_SHA256}"

mkdir -p "$(dirname "${LOCK_DIR}")"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    owner_pid="unknown"
    if [[ -f "${LOCK_DIR}/pid" ]]; then
        owner_pid="$(<"${LOCK_DIR}/pid")"
    fi
    if [[ "${owner_pid}" != "unknown" ]] && kill -0 "${owner_pid}" 2>/dev/null; then
        echo "machine3 lock is held by live PID ${owner_pid}; refusing double submission" >&2
    else
        echo "machine3 lock exists (PID ${owner_pid}); refusing automatic stale-lock removal" >&2
    fi
    exit 1
fi
LOCK_ACQUIRED=1
printf '%s\n' "$(hostname)" > "${LOCK_DIR}/hostname"
printf '%s\n' "$$" > "${LOCK_DIR}/pid"
printf '%s\n' "$(date --iso-8601=seconds)" > "${LOCK_DIR}/start_time"

mkdir -p "${LOG_ROOT}/run" "$(dirname "${COMMAND_HISTORY}")"

echo "Launching exact machine3 plan: ${PLAN}"
echo "Canonical runs root: ${RUNS_ROOT}"
echo "GPU allocation: 0,1,2,3,4,5,6,7; max-workers=16; workers-per-gpu=2"

conda run --no-capture-output -n SHOT_TTA python tools/run_experiments_multi_gpu.py "${PLAN}" \
    --runs-root "${RUNS_ROOT}" \
    --logs-root "${LOG_ROOT}/run" \
    --workdir "${PROJECT_DIR}" \
    --gpus 0,1,2,3,4,5,6,7 \
    --max-workers 16 \
    --workers-per-gpu 2 \
    --resume-partial-runs \
    --command-history "${COMMAND_HISTORY}" \
    2>&1 | tee "${LOG_ROOT}/launcher_foreground.log"

echo "Machine3 plan returned; no FINALIZE or additional search was launched."
