#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
EXPERIMENT_ROOT="${PROJECT_DIR}/experiment_logs/shot_otta_office_seed2026_stage1_20260818"
BASELINE_PLAN="${EXPERIMENT_ROOT}/plans/machine3/baseline/plan.json"
STAGE1_PLAN="${EXPERIMENT_ROOT}/plans/machine3/stage1/plan.json"
RUNS_ROOT="${EXPERIMENT_ROOT}/runs"
BASELINE_LOG_ROOT="${EXPERIMENT_ROOT}/launcher_logs/machine3/baseline"
STAGE1_LOG_ROOT="${EXPERIMENT_ROOT}/launcher_logs/machine3/stage1"
BASELINE_COMMAND_HISTORY="${EXPERIMENT_ROOT}/command_history/machine3/baseline.sh"
STAGE1_COMMAND_HISTORY="${EXPERIMENT_ROOT}/command_history/machine3/stage1.sh"
BASELINE_LAUNCHER_LOG="${BASELINE_LOG_ROOT}/launcher_foreground.log"
STAGE1_LAUNCHER_LOG="${STAGE1_LOG_ROOT}/launcher_foreground.log"
COMMAND_HISTORY_ROOT="${EXPERIMENT_ROOT}/command_history/machine3"

cd "${PROJECT_DIR}"

printf '%s\n' "Office Night Machine 3 PREPARE launcher"
printf '%s\n' "project_dir=${PROJECT_DIR}"
printf '%s\n' "experiment_root=${EXPERIMENT_ROOT}"
printf '%s\n' "machine=machine3"
printf '%s\n' "baseline_responsibility=Office W->A and W->D; 24 entries"
printf '%s\n' "stage1_responsibility=rho=0.002, K=1049; 8 anchors x 6 transfers = 48 entries"
printf '%s\n' "gpu_allocation=0,1,2,3,4,5,6,7"
printf '%s\n' "conda_environment=SHOT_TTA"

mkdir -p "${BASELINE_LOG_ROOT}" "${STAGE1_LOG_ROOT}" "${COMMAND_HISTORY_ROOT}"

conda run --no-capture-output -n SHOT_TTA \
  python tools/run_experiments_multi_gpu.py "${BASELINE_PLAN}" \
  --runs-root "${RUNS_ROOT}" \
  --logs-root "${BASELINE_LOG_ROOT}" \
  --workdir "${PROJECT_DIR}" \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-workers 8 \
  --workers-per-gpu 1 \
  --command-history "${BASELINE_COMMAND_HISTORY}" \
  >"${BASELINE_LAUNCHER_LOG}" 2>&1

conda run --no-capture-output -n SHOT_TTA \
  python tools/run_experiments_multi_gpu.py "${STAGE1_PLAN}" \
  --runs-root "${RUNS_ROOT}" \
  --logs-root "${STAGE1_LOG_ROOT}" \
  --workdir "${PROJECT_DIR}" \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-workers 16 \
  --workers-per-gpu 2 \
  --resume-partial-runs \
  --command-history "${STAGE1_COMMAND_HISTORY}" \
  >"${STAGE1_LAUNCHER_LOG}" 2>&1

printf '%s\n' "Baseline and Stage-1 training phases have returned."
printf '%s\n' "PREPARE launcher STOP"
