#!/usr/bin/env bash
set -euo pipefail

# Foreground, non-interactive single-GPU resume for the incomplete Wave-2 M3
# VisDA-C budget=0.002 B1 identity. The plan-status preflight skips the seven
# completed identities and resumes only the matching incomplete LBI stream.
WORKSPACE_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI"
PROJECT_DIR="$WORKSPACE_ROOT/260817_iclr2027-refined"
ENV_NAME="SHOT_TTA"
SEARCH_ROOT="$PROJECT_DIR/experiment_logs/visda_fc_lbi_seed2026_search_20260819"
PLAN_PATH="$SEARCH_ROOT/plans/wave2_m3_stage1_budget002/plan.json"
PLAN_SHA256="48a75a23e24bf09d53e1e052e22ddb19b5510dd44b1fe033b700978ea0cdf7c5"
RUNS_ROOT="$SEARCH_ROOT/runs"
LOGS_ROOT="$SEARCH_ROOT/launcher_logs/wave2_m3_stage1_budget002_resume_single_gpu"
COMMAND_HISTORY="$SEARCH_ROOT/command_history/wave2_m3_stage1_budget002_resume_single_gpu_commands.sh"

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

# The runner finds the matching completed-batch stream checkpoint and appends
# only the remaining work. It remains attached until the resumed identity
# completes or fails.
conda run --no-capture-output -n "$ENV_NAME" \
  python "$PROJECT_DIR/tools/run_experiments_multi_gpu.py" "$PLAN_PATH" \
  --runs-root "$RUNS_ROOT" \
  --logs-root "$LOGS_ROOT" \
  --workdir "$PROJECT_DIR" \
  --gpus 0 \
  --max-workers 1 \
  --workers-per-gpu 1 \
  --resume \
  --resume-partial-runs \
  --command-history "$COMMAND_HISTORY"
