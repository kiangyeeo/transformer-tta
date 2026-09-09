#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
FORMAL_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_baselines_seed2026_formal_20260827"
PLAN_DIR="${FORMAL_ROOT}/plans/node2_da"
PLAN_PATH="${PLAN_DIR}/plan.json"
MATRIX_PATH="${PLAN_DIR}/matrix.yaml"
PREPARE_RECORD="${FORMAL_ROOT}/phase_records/node2_da/PREPARE.md"
RUNS_ROOT="${FORMAL_ROOT}/runs"
LOGS_ROOT="${FORMAL_ROOT}/launcher_logs/node2_da"
COMMAND_HISTORY="${FORMAL_ROOT}/command_history/node2_da_commands.sh"
LOCK_PATH="${FORMAL_ROOT}/locks/node2_da.lock"

mkdir -p "$(dirname "${LOCK_PATH}")"
if [[ ! -f "${PLAN_PATH}" || ! -f "${MATRIX_PATH}" || ! -f "${PREPARE_RECORD}" ]]; then
  echo "ERROR: Node 2 formal plan, matrix, or PREPARE record is missing." >&2
  exit 1
fi
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: Node 2 launcher lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi
cleanup() {
  local status=$?
  rmdir "${LOCK_PATH}" 2>/dev/null || true
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

# Static plan verification only; this does not query or reserve a GPU.
conda run --no-capture-output -n SHOT_TTA python "${PLAN_DIR}/build_plan.py" --verify-only

mkdir -p "${LOGS_ROOT}" "$(dirname "${COMMAND_HISTORY}")"
conda run --no-capture-output -n SHOT_TTA python "${PROJECT_ROOT}/tools/run_experiments_multi_gpu.py" \
  "${PLAN_PATH}" \
  --runs-root "${RUNS_ROOT}" \
  --logs-root "${LOGS_ROOT}" \
  --workdir "${PROJECT_ROOT}" \
  --gpus 0,1,2,3 \
  --max-workers 4 \
  --workers-per-gpu 1 \
  --resume \
  --resume-partial-runs \
  --command-history "${COMMAND_HISTORY}"
