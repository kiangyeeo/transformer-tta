#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
PREP_ROOT="${PROJECT_ROOT}/experiment_logs/conv_lbi_r2_out_32gpu_20260828"
CELL_ID="omega_0p1__lr_0p0025"
BRANCH_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/runs/${CELL_ID}"
PLAN_PATH="${PREP_ROOT}/plans/node5_${CELL_ID}.jsonl"
PREPARE_TOOL="${PROJECT_ROOT}/tools/prepare_conv_lbi_r2_out_node5.py"
PREPARE_RECORD="${PREP_ROOT}/node_records/node5_PREPARE.md"
RUNS_ROOT="${BRANCH_ROOT}"
LOGS_ROOT="${BRANCH_ROOT}/launcher_logs/node5_${CELL_ID}"
COMMAND_HISTORY="${BRANCH_ROOT}/command_history/node5_${CELL_ID}_commands.sh"
LOCK_PATH="${BRANCH_ROOT}/locks/node5_${CELL_ID}.lock"
LAUNCH_PLAN=""

if [[ ! -f "${PLAN_PATH}" || ! -f "${PREPARE_TOOL}" || ! -f "${PREPARE_RECORD}" ]]; then
  echo "ERROR: Node 5 plan, PREPARE verifier, or PREPARE record is missing." >&2
  exit 1
fi

mkdir -p "$(dirname "${LOCK_PATH}")"
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: Node 5 launcher lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi

cleanup() {
  local status=$?
  [[ -z "${LAUNCH_PLAN}" ]] || rm -f "${LAUNCH_PLAN}"
  rmdir "${LOCK_PATH}" 2>/dev/null || true
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

cd "${PROJECT_ROOT}"

# CPU-only validation of the immutable JSONL plan; this does not query GPUs.
conda run --no-capture-output -n SHOT_TTA \
  python "${PREPARE_TOOL}" --verify-existing

# The repository executor consumes a JSON object, so wrap the immutable JSONL
# in an ephemeral launcher input without changing the scientific plan.
LAUNCH_PLAN="$(mktemp /tmp/node5_omega_0p1__lr_0p0025_launch.XXXXXX.json)"
conda run --no-capture-output -n SHOT_TTA python - "${PLAN_PATH}" "${LAUNCH_PLAN}" "${RUNS_ROOT}" <<'PY'
import json
import pathlib
import sys

source, destination, runs_root = map(pathlib.Path, sys.argv[1:])
entries = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(entries) != 18:
    raise SystemExit(f"expected 18 JSONL rows, got {len(entries)}")
destination.write_text(json.dumps({
    "plan_schema_version": 1,
    "node": 5,
    "phase": "R2_out_omega_stage2_lr_joint_search",
    "cell_id": "omega_0p1__lr_0p0025",
    "experiment_count": len(entries),
    "runs_root": str(runs_root.resolve()),
    "experiments": entries,
}, indent=2) + "\n", encoding="utf-8")
PY

mkdir -p "${LOGS_ROOT}" "$(dirname "${COMMAND_HISTORY}")"
conda run --no-capture-output -n SHOT_TTA \
  python tools/run_experiments_multi_gpu.py \
  "${LAUNCH_PLAN}" \
  --runs-root "${RUNS_ROOT}" \
  --logs-root "${LOGS_ROOT}" \
  --workdir "${PROJECT_ROOT}" \
  --gpus 0,1,2,3 \
  --max-workers 4 \
  --workers-per-gpu 1 \
  --resume \
  --resume-partial-runs \
  --command-history "${COMMAND_HISTORY}"
