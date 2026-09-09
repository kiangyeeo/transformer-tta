#!/usr/bin/env bash
set -euo pipefail

# Foreground, non-interactive Machine-4 launcher for the exact eight-point
# VisDA-C Wave-2 cross-budget omega hedge grid C1-C8.
PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI"
PROJECT_DIR="$PROJECT_ROOT/260817_iclr2027-refined"
ENV_NAME="SHOT_TTA"
SEARCH_ROOT="$PROJECT_DIR/experiment_logs/visda_fc_lbi_seed2026_search_20260819"
PLAN_PATH="$SEARCH_ROOT/plans/wave2_m4_cross_budget_omega_hedge/plan.json"
RUNS_ROOT="$SEARCH_ROOT/runs"
LOGS_ROOT="$SEARCH_ROOT/launcher_logs/wave2_m4_cross_budget_omega_hedge"
COMMAND_HISTORY="$SEARCH_ROOT/command_history/wave2_m4_cross_budget_omega_hedge_commands.sh"

cd "$PROJECT_DIR"
mkdir -p "$LOGS_ROOT" "$(dirname "$COMMAND_HISTORY")"

echo "project directory: $PROJECT_DIR"
echo "conda environment: $ENV_NAME"
echo "plan: $PLAN_PATH"
echo "runs root: $RUNS_ROOT"
echo "launcher logs: $LOGS_ROOT"
echo "requested GPUs: 0,1,2,3,4,5,6,7"
echo "max workers: 8; workers per GPU: 1"

# Keep the invoking shell attached until all planned jobs finish; no tmux or
# detachment is used. The existing current-repo multi-GPU runner owns resume
# and partial stream checkpoint handling.
exec conda run --no-capture-output -n "$ENV_NAME" \
  python tools/run_experiments_multi_gpu.py \
  "$PLAN_PATH" \
  --runs-root "$RUNS_ROOT" \
  --logs-root "$LOGS_ROOT" \
  --workdir "$PROJECT_DIR" \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-workers 8 \
  --workers-per-gpu 1 \
  --resume \
  --resume-partial-runs \
  --command-history "$COMMAND_HISTORY"
