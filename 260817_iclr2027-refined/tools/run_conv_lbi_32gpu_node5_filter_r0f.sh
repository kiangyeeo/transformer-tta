#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
SEARCH_ROOT="${PROJECT_ROOT}/experiment_logs/conv_lbi_32gpu_burst_20260827"
BRANCH_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_lbi_r0f_filter_reachability_seed2026_20260827"
PLAN_PATH="${SEARCH_ROOT}/plans/node5_filter_r0f.runner.json"
PREPARE_RECORD="${SEARCH_ROOT}/node_records/node5_filter_r0f_PREPARE.md"
RUNS_ROOT="${BRANCH_ROOT}/runs"
LOGS_ROOT="${BRANCH_ROOT}/launcher_logs/node5_filter_r0f"
COMMAND_HISTORY="${BRANCH_ROOT}/command_history/node5_filter_r0f_commands.sh"
LOCK_PATH="${BRANCH_ROOT}/locks/node5_filter_r0f.lock"
GPUS="0,1,2,3"

if [[ ! -f "${PLAN_PATH}" || ! -f "${PREPARE_RECORD}" ]]; then
  echo "ERROR: Node 5 plan or PREPARE record is missing; run MODE=PREPARE first." >&2
  exit 1
fi

mkdir -p "$(dirname "${LOCK_PATH}")"
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: Node 5 launcher lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi
cleanup() {
  local status=$?
  rmdir "${LOCK_PATH}" 2>/dev/null || true
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

# CPU-only preflight: validates exact JSONL/runner-plan parity and frozen inputs.
conda run --no-capture-output -n SHOT_TTA python \
  "${PROJECT_ROOT}/tools/prepare_conv_lbi_32gpu_node5_filter_r0f.py" \
  --verify-only

mkdir -p "${LOGS_ROOT}" "$(dirname "${COMMAND_HISTORY}")"
conda run --no-capture-output -n SHOT_TTA python \
  "${PROJECT_ROOT}/tools/run_experiments_multi_gpu.py" \
  "${PLAN_PATH}" \
  --runs-root "${RUNS_ROOT}" \
  --logs-root "${LOGS_ROOT}" \
  --workdir "${PROJECT_ROOT}" \
  --gpus "${GPUS}" \
  --max-workers 4 \
  --workers-per-gpu 1 \
  --resume \
  --resume-partial-runs \
  --command-history "${COMMAND_HISTORY}"
