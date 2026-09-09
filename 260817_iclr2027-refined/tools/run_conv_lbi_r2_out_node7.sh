#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
CONTROL_ROOT="${PROJECT_ROOT}/experiment_logs/conv_lbi_r2_out_32gpu_20260828"
CELL_ID="omega_0p1__lr_0p01"
PLAN_PATH="${CONTROL_ROOT}/plans/node7_omega_0p1__lr_0p01.jsonl"
PREPARE_RECORD="${CONTROL_ROOT}/node_records/node7_PREPARE.md"
PREPARE_TOOL="${PROJECT_ROOT}/tools/prepare_conv_lbi_r2_out_node7.py"
RUNS_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/runs/${CELL_ID}"
LOGS_ROOT="${CONTROL_ROOT}/launcher_logs/node7_${CELL_ID}"
COMMAND_HISTORY="${CONTROL_ROOT}/command_history/node7_${CELL_ID}_commands.sh"
LOCK_PATH="${CONTROL_ROOT}/locks/node7_${CELL_ID}.lock"

if [[ ! -f "${PLAN_PATH}" || ! -f "${PREPARE_RECORD}" || ! -f "${PREPARE_TOOL}" ]]; then
  echo "ERROR: Node 7 R2 plan, PREPARE record, or verifier is missing." >&2
  exit 1
fi

mkdir -p "$(dirname "${LOCK_PATH}")"
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: Node 7 R2 launcher lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi

cleanup() {
  local status=$?
  rmdir "${LOCK_PATH}" 2>/dev/null || true
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

cd "${PROJECT_ROOT}"

# CPU-only canonical-plan validation; this does not query or reserve a GPU.
conda run --no-capture-output -n SHOT_TTA \
  python "${PREPARE_TOOL}" --verify-existing

mkdir -p "${LOGS_ROOT}" "$(dirname "${COMMAND_HISTORY}")"
conda run --no-capture-output -n SHOT_TTA \
  python "${PROJECT_ROOT}/tools/run_experiments_multi_gpu.py" \
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
