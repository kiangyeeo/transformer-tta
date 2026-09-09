#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="$PROJECT_DIR/experiment_logs/shot_otta_office_seed2026_stage1_20260818"
BASELINE_PLAN="$EXPERIMENT_ROOT/plans/baseline_machine1/plan.json"
STAGE1_PLAN="$EXPERIMENT_ROOT/plans/lbi_stage1_machine1_budget_0005/plan.json"
BASELINE_LOGS="$EXPERIMENT_ROOT/launcher_logs/baseline_machine1"
STAGE1_LOGS="$EXPERIMENT_ROOT/launcher_logs/lbi_stage1_machine1_budget_0005"
COMMAND_HISTORY="$EXPERIMENT_ROOT/command_history/machine1_commands.sh"
RUNS_ROOT="$EXPERIMENT_ROOT/runs"

mkdir -p "$(dirname "$COMMAND_HISTORY")" "$BASELINE_LOGS" "$STAGE1_LOGS" "$RUNS_ROOT"

echo "[machine1] environment=SHOT_TTA"
echo "[machine1] project_dir=$PROJECT_DIR"
echo "[machine1] experiment_root=$EXPERIMENT_ROOT"
echo "[machine1] gpu_allocation=0,1,2,3,4,5,6,7"
echo "[machine1] baseline_responsibility=A->D,A->W"
echo "[machine1] stage1_responsibility=rho=0.0005,K=262,anchors=8,transfers=6"

BASELINE_COMMAND=(
  conda run --no-capture-output -n SHOT_TTA
  python "$PROJECT_DIR/tools/run_experiments_multi_gpu.py"
  "$BASELINE_PLAN"
  --runs-root "$RUNS_ROOT"
  --logs-root "$BASELINE_LOGS"
  --workdir "$PROJECT_DIR"
  --gpus 0,1,2,3,4,5,6,7
  --max-workers 8
  --workers-per-gpu 1
  --command-history "$COMMAND_HISTORY"
)

printf '%q ' "${BASELINE_COMMAND[@]}" >> "$COMMAND_HISTORY"
printf '\n' >> "$COMMAND_HISTORY"
"${BASELINE_COMMAND[@]}"

STAGE1_COMMAND=(
  conda run --no-capture-output -n SHOT_TTA
  python "$PROJECT_DIR/tools/run_experiments_multi_gpu.py"
  "$STAGE1_PLAN"
  --runs-root "$RUNS_ROOT"
  --logs-root "$STAGE1_LOGS"
  --workdir "$PROJECT_DIR"
  --gpus 0,1,2,3,4,5,6,7
  --max-workers 16
  --workers-per-gpu 2
  --resume-partial-runs
  --command-history "$COMMAND_HISTORY"
)

printf '%q ' "${STAGE1_COMMAND[@]}" >> "$COMMAND_HISTORY"
printf '\n' >> "$COMMAND_HISTORY"
"${STAGE1_COMMAND[@]}"

echo "[machine1] baseline and Stage-1 training phases have returned"
