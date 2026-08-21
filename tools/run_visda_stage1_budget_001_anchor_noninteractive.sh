#!/usr/bin/env bash
set -euo pipefail

# Non-interactive launcher for the VisDA-C Stage-1 budget-0.001 anchor grid.
# Keep the training process in this script so batch/non-interactive runners
# wait for the actual jobs instead of exiting after a detached launcher starts.
PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI"
ENV_NAME="SHOT_TTA"
SEARCH_ROOT="$PROJECT_ROOT/iclr2027/experiment_logs/visda_lbi_bs256_search_20260804T033931Z"
PLAN="$SEARCH_ROOT/plans/stage1_budget_001_anchor/plan.json"
RUNS_ROOT="$SEARCH_ROOT/results/runs"
LOGS_ROOT="$SEARCH_ROOT/launcher_logs/stage1_budget_001_anchor"
COMMAND_HISTORY="$SEARCH_ROOT/COMMAND_HISTORY.sh"

cd "$PROJECT_ROOT"
mkdir -p "$LOGS_ROOT"

echo "project root: $PROJECT_ROOT"
echo "conda environment: $ENV_NAME"
echo "plan: $PLAN"
echo "runs root: $RUNS_ROOT"
echo "logs root: $LOGS_ROOT"

exec conda run --no-capture-output -n "$ENV_NAME" \
    python iclr2027/tools/run_experiments_multi_gpu.py "$PLAN" \
    --runs-root "$RUNS_ROOT" \
    --logs-root "$LOGS_ROOT" \
    --workdir "$PROJECT_ROOT" \
    --gpus 0,1,2,3,4,5,6,7 \
    --max-workers 8 \
    --workers-per-gpu 1 \
    --command-history "$COMMAND_HISTORY"
