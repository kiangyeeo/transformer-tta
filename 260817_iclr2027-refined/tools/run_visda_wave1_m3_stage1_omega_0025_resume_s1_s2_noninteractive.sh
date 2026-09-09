#!/usr/bin/env bash
set -euo pipefail

# Resume only the missing S1/S2 identities from the original M3 plan.
# The status-aware runner skips completed S3-S8 and restores compatible
# module_lbi checkpoints at the next completed online-batch boundary.
PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI"
PROJECT_DIR="$PROJECT_ROOT/260817_iclr2027-refined"
ENV_NAME="SHOT_TTA"
SEARCH_ROOT="$PROJECT_DIR/experiment_logs/visda_fc_lbi_seed2026_search_20260819"
PLAN_PATH="$SEARCH_ROOT/plans/wave1_m3_stage1_budget001_omega_0025/plan.json"
RUNS_ROOT="$SEARCH_ROOT/runs"
LOGS_ROOT="$SEARCH_ROOT/launcher_logs/wave1_m3_stage1_budget001_omega_0025_resume_s1_s2"
COMMAND_HISTORY="$SEARCH_ROOT/command_history/wave1_m3_stage1_budget001_omega_0025_resume_s1_s2_commands.sh"

cd "$PROJECT_DIR"
mkdir -p "$LOGS_ROOT" "$(dirname "$COMMAND_HISTORY")"

exec conda run --no-capture-output -n "$ENV_NAME" \
    python tools/run_experiments_multi_gpu.py "$PLAN_PATH" \
    --runs-root "$RUNS_ROOT" \
    --logs-root "$LOGS_ROOT" \
    --workdir "$PROJECT_DIR" \
    --gpus 0,1 \
    --max-workers 2 \
    --workers-per-gpu 1 \
    --resume \
    --resume-partial-runs \
    --command-history "$COMMAND_HISTORY"
