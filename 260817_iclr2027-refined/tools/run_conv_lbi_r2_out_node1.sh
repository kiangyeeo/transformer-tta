#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
PLAN_ROOT="${PROJECT_ROOT}/experiment_logs/conv_lbi_r2_out_32gpu_20260828"
PLAN_PATH="${PLAN_ROOT}/plans/node1_omega_0p00625__lr_0p01.jsonl"
PREPARE_TOOL="${PROJECT_ROOT}/tools/prepare_conv_lbi_r2_out_node1.py"
PREPARE_RECORD="${PLAN_ROOT}/node_records/node1_PREPARE.md"
RUNS_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828/runs/omega_0p00625__lr_0p01"
LOGS_ROOT="${RUNS_ROOT}/launcher_logs/node1_omega_0p00625__lr_0p01"
COMMAND_HISTORY="${RUNS_ROOT}/command_history/node1_omega_0p00625__lr_0p01_commands.sh"
LOCK_PATH="${RUNS_ROOT}/locks/node1_omega_0p00625__lr_0p01.lock"
LAUNCH_PLAN=""

if [[ ! -f "${PLAN_PATH}" || ! -f "${PREPARE_TOOL}" || ! -f "${PREPARE_RECORD}" ]]; then
  echo "ERROR: Node 1 plan, verifier, or PREPARE record is missing." >&2
  exit 1
fi

mkdir -p "$(dirname "${LOCK_PATH}")"
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: Node 1 launcher lock already exists: ${LOCK_PATH}" >&2
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

# CPU-only verification of the immutable exact plan; this does not query GPUs.
conda run --no-capture-output -n SHOT_TTA \
  python "${PREPARE_TOOL}" --verify-existing

# The repository executor consumes a JSON object; keep the prescribed JSONL immutable.
LAUNCH_PLAN="$(mktemp /tmp/node1_r2_out_launch.XXXXXX.json)"
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
    "node": "node1",
    "phase": "R2_out_joint_sweep",
    "cell_id": "omega_0p00625__lr_0p01",
    "experiment_count": len(entries),
    "runs_root": str(runs_root.resolve()),
    "experiments": entries,
}, indent=2) + "\n", encoding="utf-8")
PY

mkdir -p "${LOGS_ROOT}" "$(dirname "${COMMAND_HISTORY}")"
conda run --no-capture-output -n SHOT_TTA \
  python "${PROJECT_ROOT}/tools/run_experiments_multi_gpu.py" \
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
