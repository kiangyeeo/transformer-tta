#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
BRANCH_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_lbi_r0f_filter_reachability_seed2026_20260827"
PLAN_PATH="${PROJECT_ROOT}/experiment_logs/conv_lbi_32gpu_burst_20260827/plans/node2_filter_r0f.jsonl"
PREPARE_RECORD="${PROJECT_ROOT}/experiment_logs/conv_lbi_32gpu_burst_20260827/node_records/node2_filter_r0f_PREPARE.md"
RUNS_ROOT="${BRANCH_ROOT}/runs"
LOGS_ROOT="${BRANCH_ROOT}/launcher_logs/node2_filter_r0f"
COMMAND_HISTORY="${BRANCH_ROOT}/command_history/node2_filter_r0f_commands.sh"
LOCK_PATH="${BRANCH_ROOT}/locks/node2_filter_r0f.lock"

if [[ ! -f "${PLAN_PATH}" || ! -f "${PREPARE_RECORD}" ]]; then
  echo "ERROR: Node 2 R0F plan or PREPARE record is missing." >&2
  exit 1
fi
mkdir -p "$(dirname "${LOCK_PATH}")"
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: Node 2 R0F launcher lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi
cleanup() {
  local status=$?
  rmdir "${LOCK_PATH}" 2>/dev/null || true
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

# CPU-only static validation; it does not query or reserve a GPU.
conda run --no-capture-output -n SHOT_TTA python "${PROJECT_ROOT}/tools/prepare_office_conv_lbi_node2_filter_r0f.py" \
  --plan "${PLAN_PATH}" --verify-only

mkdir -p "${LOGS_ROOT}" "$(dirname "${COMMAND_HISTORY}")"
conda run --no-capture-output -n SHOT_TTA python "${PROJECT_ROOT}/tools/run_experiments_multi_gpu.py" \
  "${PLAN_PATH}" --runs-root "${RUNS_ROOT}" --logs-root "${LOGS_ROOT}" --workdir "${PROJECT_ROOT}" \
  --gpus 0,1,2,3 --max-workers 4 --workers-per-gpu 1 --resume --resume-partial-runs \
  --command-history "${COMMAND_HISTORY}"
