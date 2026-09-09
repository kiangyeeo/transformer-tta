#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
BURST_ROOT="${PROJECT_ROOT}/experiment_logs/conv_lbi_32gpu_burst_20260827"
BRANCH_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_lbi_r0f_filter_reachability_seed2026_20260827"
PLAN_PATH="${BURST_ROOT}/plans/node3_filter_r0f.json"
PLAN_JSONL="${BURST_ROOT}/plans/node3_filter_r0f.jsonl"
PREPARE_TOOL="${BURST_ROOT}/plans/prepare_node3_filter_r0f.py"
PREPARE_RECORD="${BURST_ROOT}/node_records/node3_filter_r0f_PREPARE.md"
RUNS_ROOT="${BRANCH_ROOT}/runs"
LOGS_ROOT="${BURST_ROOT}/launcher_logs/node3_filter_r0f"
COMMAND_HISTORY="${BURST_ROOT}/command_history/node3_filter_r0f_commands.sh"
LOCK_PATH="${BURST_ROOT}/locks/node3_filter_r0f.lock"

if [[ ! -f "${PLAN_PATH}" || ! -f "${PLAN_JSONL}" || ! -f "${PREPARE_TOOL}" || ! -f "${PREPARE_RECORD}" ]]; then
  echo "ERROR: Node 3 plan or PREPARE artifacts are missing; rerun MODE=PREPARE." >&2
  exit 1
fi

mkdir -p "$(dirname "${LOCK_PATH}")"
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: Node 3 launcher lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi
cleanup() {
  local status=$?
  rmdir "${LOCK_PATH}" 2>/dev/null || true
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

# CPU-only preflight: verifies frozen provenance, plan identities, and JSONL.
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
