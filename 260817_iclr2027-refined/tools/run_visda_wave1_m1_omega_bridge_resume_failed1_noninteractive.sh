#!/usr/bin/env bash
set -euo pipefail

# Resume the sole M1 bridge identity that failed after two completed batches.
# This script is for a different, one-GPU machine, whose only local device is
# GPU 0.  It therefore receives a fresh CUDA context by construction.
PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
ENV_NAME="SHOT_TTA"
SEARCH_ROOT="${PROJECT_DIR}/experiment_logs/visda_fc_lbi_seed2026_search_20260819"
PLAN_PATH="${SEARCH_ROOT}/plans/wave1_m1_omega_bridge_resume_failed1/plan.json"
RUNS_ROOT="${SEARCH_ROOT}/runs"
LOGS_ROOT="${SEARCH_ROOT}/launcher_logs/wave1_m1_omega_bridge_resume_failed1"
COMMAND_HISTORY="${SEARCH_ROOT}/command_history/wave1_m1_omega_bridge_resume_failed1_commands.sh"
GPU_ID="${GPU_ID:-0}"

cd "${PROJECT_DIR}"
mkdir -p "${LOGS_ROOT}" "$(dirname "${COMMAND_HISTORY}")"

exec conda run --no-capture-output -n "${ENV_NAME}" \
  python tools/run_experiments_multi_gpu.py \
  "${PLAN_PATH}" \
  --runs-root "${RUNS_ROOT}" \
  --logs-root "${LOGS_ROOT}" \
  --workdir "${PROJECT_DIR}" \
  --gpus "${GPU_ID}" \
  --max-workers 1 \
  --workers-per-gpu 1 \
  --resume \
  --resume-partial-runs \
  --command-history "${COMMAND_HISTORY}"
