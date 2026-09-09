#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
R2_ROOT="${PROJECT_ROOT}/experiment_logs/conv_lbi_r2_out_32gpu_20260828"
CELL_ID="omega_0p025__lr_0p0025"
PLAN_PATH="${R2_ROOT}/plans/node2_${CELL_ID}.jsonl"
PREPARE_TOOL="${R2_ROOT}/plans/prepare_node2_${CELL_ID}.py"
PREPARE_RECORD="${R2_ROOT}/node_records/node2_PREPARE.md"
RUNS_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/runs/${CELL_ID}"
LOGS_ROOT="${R2_ROOT}/launcher_logs/node2_${CELL_ID}"
COMMAND_HISTORY="${R2_ROOT}/command_history/node2_${CELL_ID}_commands.sh"
LOCK_PATH="${R2_ROOT}/locks/node2_${CELL_ID}.lock"

if [[ ! -f "${PLAN_PATH}" || ! -f "${PREPARE_TOOL}" || ! -f "${PREPARE_RECORD}" ]]; then
  echo "ERROR: Node 2 plan or PREPARE artifacts are missing; rerun MODE=PREPARE." >&2
  exit 1
fi

mkdir -p "$(dirname "${LOCK_PATH}")"
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

# CPU-only exact on-disk plan validation; it does not query GPUs or train.
conda run --no-capture-output -n SHOT_TTA python "${PREPARE_TOOL}" --verify-only

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
