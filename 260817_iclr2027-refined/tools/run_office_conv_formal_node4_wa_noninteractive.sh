#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
FORMAL_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_baselines_seed2026_formal_20260827"
PLAN_DIR="${FORMAL_ROOT}/plans/node4_wa"
PLAN_PATH="${PLAN_DIR}/plan.json"
MATRIX_PATH="${PLAN_DIR}/matrix.yaml"
PREPARE_RECORD="${FORMAL_ROOT}/phase_records/node4_wa/PREPARE.md"
RUNS_ROOT="${FORMAL_ROOT}/runs"
LOGS_ROOT="${FORMAL_ROOT}/launcher_logs/node4_wa"
COMMAND_HISTORY="${FORMAL_ROOT}/command_history/node4_wa_commands.sh"
LOCK_PATH="${FORMAL_ROOT}/locks/node4_wa.lock"

mkdir -p "$(dirname "${LOCK_PATH}")"
if [[ ! -f "${PLAN_PATH}" || ! -f "${MATRIX_PATH}" || ! -f "${PREPARE_RECORD}" ]]; then
  echo "ERROR: Node 4 formal plan, matrix, or PREPARE record is missing." >&2
  exit 1
fi
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: Node 4 launcher lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi
cleanup() {
  local status=$?
  rmdir "${LOCK_PATH}" 2>/dev/null || true
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

# Pure static validation only; this does not query or reserve a GPU.
conda run --no-capture-output -n SHOT_TTA python \
  "${PLAN_DIR}/generate_plan.py" \
  --matrix "${MATRIX_PATH}" \
  --base-config "${PROJECT_ROOT}/configs/otta_conv_lbi_protocol_20260826_v1.yaml" \
  --output-dir "${PLAN_DIR}" \
  --verify-only

mkdir -p "${LOGS_ROOT}" "$(dirname "${COMMAND_HISTORY}")"
conda run --no-capture-output -n SHOT_TTA python \
  "${PROJECT_ROOT}/tools/run_experiments_multi_gpu.py" \
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
