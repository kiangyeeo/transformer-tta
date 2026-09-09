#!/usr/bin/env bash
set -euo pipefail

# Resume only the unfinished S1 identity from its compatible partial-stream
# checkpoint.  The shared 8-row plan is intentional: the executor skips its
# seven completed matching identities and resumes the sole missing S1 run.
PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
ENV_NAME="SHOT_TTA"
SEARCH_ROOT="$PROJECT_DIR/experiment_logs/visda_fc_lbi_seed2026_search_20260819"
PLAN_PATH="$SEARCH_ROOT/plans/wave1_m2_stage1_budget001_omega_000625/plan.json"
RUNS_ROOT="$SEARCH_ROOT/runs"
LOGS_ROOT="$SEARCH_ROOT/launcher_logs/wave1_m2_stage1_budget001_omega_000625_resume_s1_single_gpu"
COMMAND_HISTORY="$SEARCH_ROOT/command_history/wave1_m2_stage1_budget001_omega_000625_resume_s1_single_gpu_commands.sh"

# One physical GPU only.  Override as, for example, GPU_ID=1 bash <script>.
GPU_ID="${GPU_ID:-0}"
S1_KEY="shot__otta__VISDA-C__s0-t1__seed2026__module_lbi__budget0.001__d58fb6c758f9"
S1_SHA256="d58fb6c758f90c2f8137ed02a84d2375d55ff7ea2cb851aae2698e67dd8ce6fa"

cd "$PROJECT_DIR"
mkdir -p "$LOGS_ROOT" "$(dirname "$COMMAND_HISTORY")"

echo "resuming only unfinished plan identity: $S1_KEY"
echo "S1 scientific SHA256: $S1_SHA256"
echo "one-GPU allocation: GPU $GPU_ID; max-workers=1; workers-per-gpu=1"
echo "completed matching identities in the shared plan are skipped"

# exec keeps the caller attached until the resumed S1 process exits.  The
# executor discovers the furthest matching in-progress stream checkpoint and
# passes it to train.py as --resume-run-dir.
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
