#!/usr/bin/env bash
set -euo pipefail

# Foreground, non-interactive two-GPU continuation for the incomplete Wave-2
# Machine-4 C5/C6 runs. Completed matching identities are skipped; compatible
# in-progress module_lbi streams resume from their completed-batch checkpoints.
PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI"
PROJECT_DIR="$PROJECT_ROOT/260817_iclr2027-refined"
ENV_NAME="SHOT_TTA"
SEARCH_ROOT="$PROJECT_DIR/experiment_logs/visda_fc_lbi_seed2026_search_20260819"
PLAN_PATH="$SEARCH_ROOT/plans/wave2_m4_cross_budget_omega_hedge/plan.json"
RUNS_ROOT="$SEARCH_ROOT/runs"
LOGS_ROOT="$SEARCH_ROOT/launcher_logs/wave2_m4_cross_budget_omega_hedge_resume_2gpu"
COMMAND_HISTORY="$SEARCH_ROOT/command_history/wave2_m4_cross_budget_omega_hedge_resume_2gpu_commands.sh"

cd "$PROJECT_DIR"
mkdir -p "$LOGS_ROOT" "$(dirname "$COMMAND_HISTORY")"

echo "project directory: $PROJECT_DIR"
echo "conda environment: $ENV_NAME"
echo "plan: $PLAN_PATH"
echo "requested GPUs: 0,1"
echo "max workers: 2; workers per GPU: 1"
echo "resume mode: completed identities are skipped; partial LBI streams resume"

exec conda run --no-capture-output -n "$ENV_NAME" \
  python tools/run_experiments_multi_gpu.py \
  "$PLAN_PATH" \
  --runs-root "$RUNS_ROOT" \
  --logs-root "$LOGS_ROOT" \
  --workdir "$PROJECT_DIR" \
  --gpus 0,1 \
  --max-workers 2 \
  --workers-per-gpu 1 \
  --resume \
  --resume-partial-runs \
  --command-history "$COMMAND_HISTORY"
