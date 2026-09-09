#!/usr/bin/env bash
set -euo pipefail

# Foreground, non-interactive executor for the eight new Wave-2 M1 rows.
# Do not run this script during MODE=PREPARE.
WORKSPACE_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI"
PROJECT_DIR="$WORKSPACE_ROOT/260817_iclr2027-refined"
ENV_NAME="SHOT_TTA"
SEARCH_ROOT="$PROJECT_DIR/experiment_logs/visda_fc_lbi_seed2026_search_20260819"
PLAN_PATH="$SEARCH_ROOT/plans/wave2_m1_budget001_final_joint/plan.json"
PLAN_SHA256="d6e093a3bcbe655d458a58a0930bebbff54fdb6601e4b684226553aff33e0713"
RUNS_ROOT="$SEARCH_ROOT/runs"
LOGS_ROOT="$SEARCH_ROOT/launcher_logs/wave2_m1_budget001_final_joint"
COMMAND_HISTORY="$SEARCH_ROOT/command_history/wave2_m1_budget001_final_joint_commands.sh"

cd "$WORKSPACE_ROOT"

if [[ ! -f "$PLAN_PATH" ]]; then
  echo "missing prepared plan: $PLAN_PATH" >&2
  exit 1
fi

if [[ "$(sha256sum "$PLAN_PATH" | awk '{print $1}')" != "$PLAN_SHA256" ]]; then
  echo "prepared plan SHA256 does not match: $PLAN_PATH" >&2
  exit 1
fi

command -v conda >/dev/null 2>&1 || {
  echo "conda not found; required environment is $ENV_NAME" >&2
  exit 1
}

mkdir -p "$LOGS_ROOT" "$(dirname "$COMMAND_HISTORY")"

# Stay attached until every plan identity has completed. The runner skips
# completed matching identities and resumes compatible partial LBI streams.
exec conda run --no-capture-output -n "$ENV_NAME" \
  python "$PROJECT_DIR/tools/run_experiments_multi_gpu.py" "$PLAN_PATH" \
  --runs-root "$RUNS_ROOT" \
  --logs-root "$LOGS_ROOT" \
  --workdir "$PROJECT_DIR" \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-workers 8 \
  --workers-per-gpu 1 \
  --resume \
  --resume-partial-runs \
  --command-history "$COMMAND_HISTORY"
