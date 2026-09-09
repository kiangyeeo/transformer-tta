#!/usr/bin/env bash
set -euo pipefail

# Resume only Wave-2 M2 A2 and A3 on two physical GPUs.  This dedicated
# two-row plan cannot claim the A1 checkpoint assigned to the one-GPU host.
PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
ENV_NAME="SHOT_TTA"
SEARCH_ROOT="$PROJECT_DIR/experiment_logs/visda_fc_lbi_seed2026_search_20260819"
PLAN_PATH="$SEARCH_ROOT/plans/wave2_m2_stage1_budget0005_resume_machine2/plan.json"
RUNS_ROOT="$SEARCH_ROOT/runs"
LOGS_ROOT="$SEARCH_ROOT/launcher_logs/wave2_m2_stage1_budget0005_resume_machine2"
COMMAND_HISTORY="$SEARCH_ROOT/command_history/wave2_m2_stage1_budget0005_resume_machine2_commands.sh"
GPU_IDS="${GPU_IDS:-0,1}"

cd "$PROJECT_DIR"
mkdir -p "$LOGS_ROOT" "$(dirname "$COMMAND_HISTORY")"

echo "resuming A2 and A3 only: A2=(alpha=0.050, nu=0.5); A3=(alpha=0.075, nu=1.0)"
echo "two-GPU allocation: GPUs $GPU_IDS; max-workers=2; workers-per-gpu=1"
echo "plan: $PLAN_PATH"

# The existing executor assigns one task per GPU, discovers each matching
# in-progress stream checkpoint, and remains attached until both tasks exit.
exec conda run --no-capture-output -n "$ENV_NAME" \
  python tools/run_experiments_multi_gpu.py \
  "$PLAN_PATH" \
  --runs-root "$RUNS_ROOT" \
  --logs-root "$LOGS_ROOT" \
  --workdir "$PROJECT_DIR" \
  --gpus "$GPU_IDS" \
  --max-workers 2 \
  --workers-per-gpu 1 \
  --resume \
  --resume-partial-runs \
  --command-history "$COMMAND_HISTORY"
