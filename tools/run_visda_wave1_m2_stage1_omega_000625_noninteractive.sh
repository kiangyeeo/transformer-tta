#!/usr/bin/env bash
set -euo pipefail

# Foreground, non-interactive Machine-2 launcher for the exact eight-point
# VisDA-C Wave-1 Stage-1 grid at budget=0.001 and omega=0.00625.
PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
ENV_NAME="SHOT_TTA"
SEARCH_ROOT="$PROJECT_DIR/experiment_logs/visda_fc_lbi_seed2026_search_20260819"
PLAN_PATH="$SEARCH_ROOT/plans/wave1_m2_stage1_budget001_omega_000625/plan.json"
RUNS_ROOT="$SEARCH_ROOT/runs"
LOGS_ROOT="$SEARCH_ROOT/launcher_logs/wave1_m2_stage1_budget001_omega_000625"
COMMAND_HISTORY="$SEARCH_ROOT/command_history/wave1_m2_stage1_budget001_omega_000625_commands.sh"

cd "$PROJECT_DIR"
mkdir -p "$LOGS_ROOT" "$(dirname "$COMMAND_HISTORY")"

echo "project directory: $PROJECT_DIR"
echo "conda environment: $ENV_NAME"
echo "plan: $PLAN_PATH"
echo "runs root: $RUNS_ROOT"
echo "launcher logs: $LOGS_ROOT"
echo "requested GPUs: 0,1,2,3,4,5,6,7"
echo "max workers: 8; workers per GPU: 1"

# exec deliberately keeps the invoking shell attached until the multi-GPU
# executor has finished all planned jobs; it does not use tmux or detachment.
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
