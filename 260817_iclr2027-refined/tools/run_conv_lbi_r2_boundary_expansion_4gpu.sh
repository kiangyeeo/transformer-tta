#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
EXPANSION_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_lbi_r2_out_boundary_expansion_seed2026_20260828"
PREPARE_TOOL="${PROJECT_ROOT}/tools/prepare_conv_lbi_r2_boundary_expansion.py"
LOCK_PATH="${EXPANSION_ROOT}/.boundary_expansion_4gpu.lock"

declare -a PLAN_NAMES=(
  "E0_p0005_omega_0p30_lr_0p010"
  "E1_p0005_omega_0p10_lr_0p020"
  "E2_p0005_omega_0p30_lr_0p020"
  "E3_p001_omega_0p30_lr_0p0025"
)

if [[ ! -f "${PREPARE_TOOL}" || ! -f "${EXPANSION_ROOT}/phase_records/R2_BOUNDARY_EXPANSION/PREPARE.md" ]]; then
  echo "ERROR: PREPARE verifier or PREPARE record is missing." >&2
  exit 1
fi
for plan_name in "${PLAN_NAMES[@]}"; do
  if [[ ! -f "${EXPANSION_ROOT}/plans/${plan_name}.jsonl" ]]; then
    echo "ERROR: missing required plan: ${plan_name}.jsonl" >&2
    exit 1
  fi
done

if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: boundary-expansion launcher lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi
cleanup() {
  local status=$?
  rmdir "${LOCK_PATH}" 2>/dev/null || true
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

cd "${PROJECT_ROOT}"

# CPU-only canonical-plan validation; this does not query or reserve a GPU.
conda run --no-capture-output -n SHOT_TTA \
  python "${PREPARE_TOOL}" --verify-existing

launch_cell() {
  local cell_id=$1
  local gpu_id=$2
  local plan_name=$3
  local cell_root="${EXPANSION_ROOT}/runs/${plan_name}"
  local logs_root="${EXPANSION_ROOT}/launcher_logs/${cell_id}_${plan_name}"
  local command_history="${EXPANSION_ROOT}/command_history/${cell_id}_${plan_name}_commands.sh"
  mkdir -p "${logs_root}" "$(dirname "${command_history}")"
  conda run --no-capture-output -n SHOT_TTA \
    python "${PROJECT_ROOT}/tools/run_experiments_multi_gpu.py" \
    "${EXPANSION_ROOT}/plans/${plan_name}.jsonl" \
    --runs-root "${cell_root}" \
    --logs-root "${logs_root}" \
    --workdir "${PROJECT_ROOT}" \
    --gpus "${gpu_id}" \
    --max-workers 1 \
    --workers-per-gpu 1 \
    --resume \
    --resume-partial-runs \
    --command-history "${command_history}"
}

launch_cell E0 0 E0_p0005_omega_0p30_lr_0p010 &
pid_e0=$!
launch_cell E1 1 E1_p0005_omega_0p10_lr_0p020 &
pid_e1=$!
launch_cell E2 2 E2_p0005_omega_0p30_lr_0p020 &
pid_e2=$!
launch_cell E3 3 E3_p001_omega_0p30_lr_0p0025 &
pid_e3=$!

failed=0
for pid in "${pid_e0}" "${pid_e1}" "${pid_e2}" "${pid_e3}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
exit "${failed}"
