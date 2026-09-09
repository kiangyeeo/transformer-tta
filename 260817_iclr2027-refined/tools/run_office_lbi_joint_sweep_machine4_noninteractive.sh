#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
SWEEP_ROOT="$PROJECT_DIR/experiment_logs/shot_otta_office_lbi_joint_sweep_20260818"
PLAN_PATH="$SWEEP_ROOT/plans/machine4/plan.json"
RUNS_ROOT="$SWEEP_ROOT/runs"
PLAN_VALIDATOR="$PROJECT_DIR/tools/validate_office_lbi_joint_sweep_machine4_plan.py"
PLAN_SHA256="6909e445be1c30e56c2e05dab0c0e1398b8b0676963950c6dd93b52fdf268823"
LOCK_ROOT="$SWEEP_ROOT/locks"
LOCK_DIR="$LOCK_ROOT/machine4.lock"
LAUNCH_LOG_ROOT="$SWEEP_ROOT/launcher_logs/machine4"
COMMAND_HISTORY="$SWEEP_ROOT/command_history/machine4_commands.sh"

if [[ "$(pwd -P)" != "$PROJECT_DIR" ]]; then
  echo "machine4 launch must start from $PROJECT_DIR" >&2
  exit 1
fi

if [[ ! -f "$PLAN_PATH" ]]; then
  echo "missing exact PREPARE plan: $PLAN_PATH" >&2
  exit 1
fi

command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda not found" >&2
  exit 1
}

mkdir -p "$LOCK_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "machine4 lock already exists: $LOCK_DIR" >&2
  if [[ -f "$LOCK_DIR/owner" ]]; then
    sed -n '1,20p' "$LOCK_DIR/owner" >&2
  fi
  exit 1
fi

cleanup_lock() {
  local status=$?
  trap - EXIT INT TERM
  rm -f "$LOCK_DIR/owner"
  rmdir "$LOCK_DIR" 2>/dev/null || true
  exit "$status"
}
trap cleanup_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'pid=%s\n' "$$"
  printf 'start_time_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'plan=%s\n' "$PLAN_PATH"
  printf 'plan_sha256=%s\n' "$PLAN_SHA256"
} > "$LOCK_DIR/owner"

conda run --no-capture-output -n SHOT_TTA python "$PLAN_VALIDATOR" \
  --plan "$PLAN_PATH" \
  --runs-root "$RUNS_ROOT" \
  --expected-sha "$PLAN_SHA256"

conda run --no-capture-output -n SHOT_TTA python -c '
import torch
count = torch.cuda.device_count()
if not torch.cuda.is_available() or count != 8:
    raise SystemExit(f"SHOT_TTA must expose exactly 8 CUDA GPUs; available={torch.cuda.is_available()} count={count}")
print(f"SHOT_TTA CUDA preflight passed: {count} GPUs")
'

mkdir -p "$LAUNCH_LOG_ROOT" "$(dirname "$COMMAND_HISTORY")"
conda run --no-capture-output -n SHOT_TTA python "$PROJECT_DIR/tools/run_experiments_multi_gpu.py" \
  "$PLAN_PATH" \
  --runs-root "$RUNS_ROOT" \
  --logs-root "$LAUNCH_LOG_ROOT" \
  --workdir "$PROJECT_DIR" \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-workers 16 \
  --workers-per-gpu 2 \
  --resume-partial-runs \
  --command-history "$COMMAND_HISTORY"
