#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
BURST_ROOT="${PROJECT_ROOT}/experiment_logs/conv_lbi_32gpu_burst_20260827"
BRANCH_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_lbi_r1_out_stage1_validation_seed2026_20260827"
PLAN_PATH="${BURST_ROOT}/plans/node6_out_r1.jsonl"
PREPARE_RECORD="${BURST_ROOT}/node_records/node6_out_r1_PREPARE.md"
RUNS_ROOT="${BRANCH_ROOT}/runs"
LOGS_ROOT="${BRANCH_ROOT}/launcher_logs/node6_out_r1"
COMMAND_HISTORY="${BRANCH_ROOT}/command_history/node6_out_r1_commands.sh"
LOCK_PATH="${BRANCH_ROOT}/locks/node6_out_r1.lock"
LAUNCH_PLAN=""

if [[ ! -f "${PLAN_PATH}" || ! -f "${PREPARE_RECORD}" ]]; then
  echo "ERROR: Node 6 JSONL plan or PREPARE record is missing." >&2
  exit 1
fi

mkdir -p "$(dirname "${LOCK_PATH}")"
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: Node 6 launcher lock already exists: ${LOCK_PATH}" >&2
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

# CPU-only frozen-plan validation; it does not query or reserve a GPU.
conda run --no-capture-output -n SHOT_TTA \
  python tools/prepare_conv_lbi_32gpu_node6_out_r1.py --verify-existing

# The repository executor accepts a JSON object.  Keep the prescribed JSONL
# plan immutable and convert it only to an ephemeral launcher input.
LAUNCH_PLAN="$(mktemp /tmp/node6_out_r1_launch.XXXXXX.json)"
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
    "node": "node6_out_r1",
    "phase": "R1_out_stage1_validation",
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
