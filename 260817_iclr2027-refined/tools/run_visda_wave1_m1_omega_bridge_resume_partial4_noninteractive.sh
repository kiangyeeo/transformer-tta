#!/usr/bin/env bash
set -euo pipefail

# Resume the four M1 bridge identities whose matching stream checkpoints stop
# at batches 193, 195, 197, and 201 of the 217-batch VisDA-C target stream.
PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
ENV_NAME="SHOT_TTA"
SEARCH_ROOT="${PROJECT_DIR}/experiment_logs/visda_fc_lbi_seed2026_search_20260819"
PLAN_PATH="${SEARCH_ROOT}/plans/wave1_m1_omega_bridge_resume_partial4/plan.json"
RUNS_ROOT="${SEARCH_ROOT}/runs"
LOGS_ROOT="${SEARCH_ROOT}/launcher_logs/wave1_m1_omega_bridge_resume_partial4"
COMMAND_HISTORY="${SEARCH_ROOT}/command_history/wave1_m1_omega_bridge_resume_partial4_commands.sh"

# This script is for the separate four-GPU machine.  GPU identifiers are local
# to that machine, so its default is all four locally visible devices.
GPU_IDS="${GPU_IDS:-0,1,2,3}"

cd "${PROJECT_DIR}"
mkdir -p "${LOGS_ROOT}" "$(dirname "${COMMAND_HISTORY}")"

exec conda run --no-capture-output -n "${ENV_NAME}" \
  python tools/run_experiments_multi_gpu.py \
  "${PLAN_PATH}" \
  --runs-root "${RUNS_ROOT}" \
  --logs-root "${LOGS_ROOT}" \
  --workdir "${PROJECT_DIR}" \
  --gpus "${GPU_IDS}" \
  --max-workers 4 \
  --workers-per-gpu 1 \
  --resume \
  --resume-partial-runs \
  --command-history "${COMMAND_HISTORY}"
