#!/usr/bin/env bash
set -euo pipefail

# Resume only Wave-2 M2 A1 on one physical GPU.  This dedicated one-row plan
# prevents this host from claiming the A2/A3 checkpoints assigned elsewhere.
PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
ENV_NAME="SHOT_TTA"
SEARCH_ROOT="$PROJECT_DIR/experiment_logs/visda_fc_lbi_seed2026_search_20260819"
PLAN_PATH="$SEARCH_ROOT/plans/wave2_m2_stage1_budget0005_resume_machine1/plan.json"
RUNS_ROOT="$SEARCH_ROOT/runs"
LOGS_ROOT="$SEARCH_ROOT/launcher_logs/wave2_m2_stage1_budget0005_resume_machine1"
COMMAND_HISTORY="$SEARCH_ROOT/command_history/wave2_m2_stage1_budget0005_resume_machine1_commands.sh"
GPU_ID="${GPU_ID:-0}"

cd "$PROJECT_DIR"
mkdir -p "$LOGS_ROOT" "$(dirname "$COMMAND_HISTORY")"

echo "resuming A1 only: alpha=0.050, nu=1.0"
echo "one-GPU allocation: GPU $GPU_ID; max-workers=1; workers-per-gpu=1"
echo "plan: $PLAN_PATH"

# The existing executor finds A1's matching in-progress stream checkpoint,
# passes --resume-run-dir to train.py, and remains attached until it exits.
exec conda run --no-capture-output -n "$ENV_NAME" \
  python tools/run_experiments_multi_gpu.py \
  "$PLAN_PATH" \
  --runs-root "$RUNS_ROOT" \
  --logs-root "$LOGS_ROOT" \
  --workdir "$PROJECT_DIR" \
  --gpus "$GPU_ID" \
  --max-workers 1 \
  --workers-per-gpu 1 \
  --resume \
  --resume-partial-runs \
  --command-history "$COMMAND_HISTORY"
