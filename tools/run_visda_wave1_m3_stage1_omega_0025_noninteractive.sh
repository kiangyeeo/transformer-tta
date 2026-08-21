#!/usr/bin/env bash
set -euo pipefail

# Keep this process attached: run_experiments_multi_gpu.py waits for every
# assigned job, skips completed matching identities, and resumes compatible
# module_lbi partial streams at completed-batch boundaries.
PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI"
PROJECT_DIR="$PROJECT_ROOT/260817_iclr2027-refined"
ENV_NAME="SHOT_TTA"
SEARCH_ROOT="$PROJECT_DIR/experiment_logs/visda_fc_lbi_seed2026_search_20260819"
PLAN_PATH="$SEARCH_ROOT/plans/wave1_m3_stage1_budget001_omega_0025/plan.json"
RUNS_ROOT="$SEARCH_ROOT/runs"
LOGS_ROOT="$SEARCH_ROOT/launcher_logs/wave1_m3_stage1_budget001_omega_0025"
COMMAND_HISTORY="$SEARCH_ROOT/command_history/wave1_m3_stage1_budget001_omega_0025_commands.sh"

cd "$PROJECT_DIR"
mkdir -p "$LOGS_ROOT" "$(dirname "$COMMAND_HISTORY")"

exec conda run --no-capture-output -n "$ENV_NAME" \
    python tools/run_experiments_multi_gpu.py "$PLAN_PATH" \
    --runs-root "$RUNS_ROOT" \
    --logs-root "$LOGS_ROOT" \
    --workdir "$PROJECT_DIR" \
    --gpus 0,1,2,3,4,5,6,7 \
    --max-workers 8 \
    --workers-per-gpu 1 \
    --resume \
    --resume-partial-runs \
    --command-history "$COMMAND_HISTORY"
