#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
SEARCH_ROOT="${PROJECT_DIR}/experiment_logs/visda_fc_lbi_seed2026_search_20260819"
BASELINE_PLAN="${SEARCH_ROOT}/plans/wave1_m1_baselines/plan.json"
BRIDGE_PLAN="${SEARCH_ROOT}/plans/wave1_m1_omega_bridge_budget001/plan.json"
RUNS_ROOT="${SEARCH_ROOT}/runs"
BASELINE_LOG_ROOT="${SEARCH_ROOT}/launcher_logs/wave1_m1_baselines"
BRIDGE_LOG_ROOT="${SEARCH_ROOT}/launcher_logs/wave1_m1_omega_bridge_budget001"
BASELINE_COMMAND_HISTORY="${SEARCH_ROOT}/command_history/wave1_m1_baselines.sh"
BRIDGE_COMMAND_HISTORY="${SEARCH_ROOT}/command_history/wave1_m1_omega_bridge_budget001.sh"

cd "${PROJECT_DIR}"

mkdir -p \
  "${BASELINE_LOG_ROOT}" \
  "${BRIDGE_LOG_ROOT}" \
  "$(dirname "${BASELINE_COMMAND_HISTORY}")"

conda run --no-capture-output -n SHOT_TTA \
  python tools/run_experiments_multi_gpu.py "${BASELINE_PLAN}" \
  --runs-root "${RUNS_ROOT}" \
  --logs-root "${BASELINE_LOG_ROOT}" \
  --workdir "${PROJECT_DIR}" \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-workers 8 \
  --workers-per-gpu 1 \
  --resume \
  --command-history "${BASELINE_COMMAND_HISTORY}"

conda run --no-capture-output -n SHOT_TTA \
  python tools/run_experiments_multi_gpu.py "${BRIDGE_PLAN}" \
  --runs-root "${RUNS_ROOT}" \
  --logs-root "${BRIDGE_LOG_ROOT}" \
  --workdir "${PROJECT_DIR}" \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-workers 8 \
  --workers-per-gpu 1 \
  --resume \
  --resume-partial-runs \
  --command-history "${BRIDGE_COMMAND_HISTORY}"
