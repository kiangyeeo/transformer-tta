#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
SWEEP_ROOT="$PROJECT_DIR/experiment_logs/shot_otta_office_lbi_joint_sweep_20260818"
PLAN="$SWEEP_ROOT/plans/machine1/plan.json"
RUNS_ROOT="$SWEEP_ROOT/runs"
PHASE_RECORD="$SWEEP_ROOT/phase_records/machine1/PREPARE_READY.json"
LAUNCHER_LOGS="$SWEEP_ROOT/launcher_logs/machine1"
COMMAND_HISTORY="$SWEEP_ROOT/command_history/machine1_commands.sh"
LOCK="$SWEEP_ROOT/locks/machine1.lock"

if [[ "$(pwd -P)" != "$PROJECT_DIR" ]]; then
  echo "must run from $PROJECT_DIR" >&2
  exit 1
fi
if [[ ! -f "$PLAN" || ! -f "$PHASE_RECORD" ]]; then
  echo "PREPARE artifacts are missing" >&2
  exit 1
fi

command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda not found" >&2
  exit 1
}

mkdir -p "$(dirname "$LOCK")"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "machine1 lock already exists: $LOCK" >&2
  exit 1
fi
LOCK_ACQUIRED=1
cleanup_lock() {
  if [[ "${LOCK_ACQUIRED:-0}" == 1 ]]; then
    rm -f "$LOCK/hostname" "$LOCK/pid" "$LOCK/start_time_utc"
    rmdir "$LOCK" 2>/dev/null || true
  fi
}
on_interrupt() {
  cleanup_lock
  exit 130
}
on_terminate() {
  cleanup_lock
  exit 143
}
trap cleanup_lock EXIT
trap on_interrupt INT
trap on_terminate TERM
printf '%s\n' "$(hostname)" > "$LOCK/hostname"
printf '%s\n' "$$" > "$LOCK/pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$LOCK/start_time_utc"

conda run --no-capture-output -n SHOT_TTA python tools/validate_office_lbi_joint_plan.py \
  "$PLAN" --phase-record "$PHASE_RECORD"

mkdir -p "$LAUNCHER_LOGS" "$(dirname "$COMMAND_HISTORY")"
conda run --no-capture-output -n SHOT_TTA python tools/run_experiments_multi_gpu.py \
  "$PLAN" \
  --runs-root "$RUNS_ROOT" \
  --logs-root "$LAUNCHER_LOGS" \
  --workdir "$PROJECT_DIR" \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-workers 16 \
  --workers-per-gpu 2 \
  --resume-partial-runs \
  --command-history "$COMMAND_HISTORY"
