#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
EXPERIMENT_ROOT="$PROJECT_DIR/experiment_logs/shot_otta_office_seed2026_stage1_20260818"
BASELINE_PLAN="$EXPERIMENT_ROOT/plans/baseline_machine2/plan.json"
STAGE1_PLAN="$EXPERIMENT_ROOT/plans/lbi_stage1_machine2/plan.json"
RUNS_ROOT="$EXPERIMENT_ROOT/runs"
BASELINE_LOG_ROOT="$EXPERIMENT_ROOT/launcher_logs/baseline_machine2"
STAGE1_LOG_ROOT="$EXPERIMENT_ROOT/launcher_logs/lbi_stage1_machine2"
BASELINE_STDOUT="$BASELINE_LOG_ROOT/launcher_stdout.log"
STAGE1_STDOUT="$STAGE1_LOG_ROOT/launcher_stdout.log"
COMMAND_HISTORY="$EXPERIMENT_ROOT/phase_records/machine2/COMMAND_HISTORY.sh"

mkdir -p "$BASELINE_LOG_ROOT" "$STAGE1_LOG_ROOT" "$(dirname "$COMMAND_HISTORY")"

record_command() {
    {
        printf 'cd %q\n' "$PROJECT_DIR"
        printf '%q ' "$@"
        printf '\n'
    } >> "$COMMAND_HISTORY"
}

run_recorded() {
    record_command "$@"
    "$@"
}

printf 'environment=SHOT_TTA\n'
printf 'project_dir=%s\n' "$PROJECT_DIR"
printf 'experiment_root=%s\n' "$EXPERIMENT_ROOT"
printf 'machine_tag=machine2\n'
printf 'baseline_responsibility=D->A,D->W\n'
printf 'stage1_responsibility=rho=0.001,K=524,anchors=8,transfers=6\n'
printf 'training_policy=foreground;no_detach;no_finalize\n'

BASELINE_CMD=(
    conda run --no-capture-output -n SHOT_TTA
    python "$PROJECT_DIR/tools/run_experiments_multi_gpu.py"
    "$BASELINE_PLAN"
    --runs-root "$RUNS_ROOT"
    --logs-root "$BASELINE_LOG_ROOT"
    --workdir "$PROJECT_DIR"
    --gpus 0,1,2,3,4,5,6,7
    --max-workers 8
    --workers-per-gpu 1
    --command-history "$COMMAND_HISTORY"
)
run_recorded "${BASELINE_CMD[@]}" >> "$BASELINE_STDOUT" 2>&1

STAGE1_CMD=(
    conda run --no-capture-output -n SHOT_TTA
    python "$PROJECT_DIR/tools/run_experiments_multi_gpu.py"
    "$STAGE1_PLAN"
    --runs-root "$RUNS_ROOT"
    --logs-root "$STAGE1_LOG_ROOT"
    --workdir "$PROJECT_DIR"
    --gpus 0,1,2,3,4,5,6,7
    --max-workers 16
    --workers-per-gpu 2
    --resume-partial-runs
    --command-history "$COMMAND_HISTORY"
)
run_recorded "${STAGE1_CMD[@]}" >> "$STAGE1_STDOUT" 2>&1

printf 'training_phases_returned=baseline_and_stage1\n'
printf 'STOP\n'
