#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
FORMAL_ROOT="${PROJECT_ROOT}/experiment_logs/visda_conv_baselines_seed2026_formal_20260827"
PLAN_PATH="${FORMAL_ROOT}/plans/node6_out/plan.json"
MATRIX_PATH="${FORMAL_ROOT}/plans/node6_out/matrix.yaml"
PREPARE_RECORD="${FORMAL_ROOT}/phase_records/node6_out/PREPARE.md"
RUNS_ROOT="${FORMAL_ROOT}/runs"
LOGS_ROOT="${FORMAL_ROOT}/launcher_logs/node6_out"
COMMAND_HISTORY="${FORMAL_ROOT}/command_history/node6_out_commands.sh"
LOCK_PATH="${FORMAL_ROOT}/locks/node6_out.lock"

mkdir -p "$(dirname "${LOCK_PATH}")"
if [[ ! -f "${PLAN_PATH}" || ! -f "${MATRIX_PATH}" || ! -f "${PREPARE_RECORD}" ]]; then
  echo "ERROR: Node 6 formal plan, matrix, or PREPARE record is missing." >&2
  exit 1
fi
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: Node 6 launcher lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi
cleanup() {
  local status=$?
  rmdir "${LOCK_PATH}" 2>/dev/null || true
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

# CPU-only formal-plan preflight; this does not query or reserve a GPU.
conda run --no-capture-output -n SHOT_TTA python \
  "${PROJECT_ROOT}/tools/validate_visda_conv_formal_node6_out_preflight.py" \
  "${PLAN_PATH}"

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
