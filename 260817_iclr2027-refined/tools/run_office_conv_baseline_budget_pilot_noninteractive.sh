#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
PILOT_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_baseline_budget_pilot_seed2026_20260827"
PLAN_DIR="${PILOT_ROOT}/plans/p3a_office_da"
PLAN_PATH="${PLAN_DIR}/plan.json"
WAVE_MANIFEST="${PLAN_DIR}/wave_manifest.json"
PREPARE_RECORD="${PILOT_ROOT}/phase_records/p3a_office_da/PREPARE.md"
RUNS_ROOT="${PILOT_ROOT}/runs"
LOGS_ROOT="${PILOT_ROOT}/launcher_logs/p3a_office_da"
COMMAND_HISTORY="${PILOT_ROOT}/command_history/p3a_office_da_commands.sh"
LOCK_PATH="${PILOT_ROOT}/locks/p3a_office_da.lock"

mkdir -p "$(dirname "${LOCK_PATH}")"

if [[ ! -f "${PLAN_PATH}" || ! -f "${WAVE_MANIFEST}" || ! -f "${PREPARE_RECORD}" ]]; then
  echo "ERROR: P3a plan, wave manifest, or PREPARE record is missing; rerun PREPARE first." >&2
  exit 1
fi

if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: pilot launcher lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi
cleanup() {
  local status=$?
  rmdir "${LOCK_PATH}" 2>/dev/null || true
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

# This performs only plan validation; it intentionally does not probe GPUs.
conda run --no-capture-output -n SHOT_TTA python - "${PLAN_PATH}" "${WAVE_MANIFEST}" "${RUNS_ROOT}" <<'PY'
import json
import pathlib
import sys

plan_path = pathlib.Path(sys.argv[1]).resolve()
manifest_path = pathlib.Path(sys.argv[2]).resolve()
runs_root = pathlib.Path(sys.argv[3]).resolve()
plan = json.loads(plan_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
entries = plan.get("experiments", [])
expected_k = {
    "out_channel": {0.0005: 4, 0.001: 9, 0.002: 18, 0.005: 46},
    "filter_connection": {0.0005: 3276, 0.001: 6553, 0.002: 13107, 0.005: 32768},
}
expected_waves = [
    (0, "conv_module_dense", 1),
    (1, "conv_out_random", 4),
    (2, "conv_out_magnitude", 4),
    (3, "conv_out_saliency", 4),
    (4, "conv_filter_random", 4),
    (5, "conv_filter_magnitude", 4),
    (6, "conv_filter_saliency", 4),
]
if plan.get("experiment_count") != 25 or len(entries) != 25:
    raise SystemExit("P3a plan count must be exactly 25")
if len({entry.get("experiment_key") for entry in entries}) != 25:
    raise SystemExit("P3a plan keys are not unique")
if len({entry.get("experiment_config_sha256") for entry in entries}) != 25:
    raise SystemExit("P3a scientific identities are not unique")
if any(entry.get("variant", "").startswith("conv_") and entry["variant"].endswith("_lbi") for entry in entries):
    raise SystemExit("P3a must not contain conv_*_lbi")
for entry in entries:
    if entry.get("dataset") != "office" or (entry.get("source"), entry.get("target")) != (1, 0):
        raise SystemExit("P3a plan contains an unintended Office transfer or dataset")
    if entry.get("seed") != 2026:
        raise SystemExit("P3a seed drifted")
    if entry.get("variant") == "conv_module_dense":
        if entry.get("requested_budget") != 1.0 or entry.get("group_mode") is not None:
            raise SystemExit("dense anchor drifted")
    else:
        mode, rho = entry.get("group_mode"), entry.get("requested_budget")
        if mode not in expected_k or rho not in expected_k[mode]:
            raise SystemExit("sparse grouping or budget drifted")
        if entry.get("max_group_count") != expected_k[mode][rho]:
            raise SystemExit("planned integer group budget drifted")
        if entry.get("variant", "").endswith("_random"):
            if entry.get("selection_seed") != 2026 or entry.get("num_random_masks") != 3:
                raise SystemExit("Random frozen seed or child-mask count drifted")
    try:
        pathlib.Path(entry["expected_output_root"]).resolve().relative_to(runs_root)
    except ValueError as error:
        raise SystemExit("planned output escapes P3a runs root") from error
waves = manifest.get("waves", [])
if [(wave.get("wave"), wave.get("variant"), wave.get("experiment_count")) for wave in waves] != expected_waves:
    raise SystemExit("P3a wave manifest drifted")
for wave_number, variant, expected_count in expected_waves:
    wave_path = pathlib.Path(waves[wave_number]["plan_path"]).resolve()
    wave_plan = json.loads(wave_path.read_text(encoding="utf-8"))
    wave_entries = wave_plan.get("experiments", [])
    if wave_plan.get("experiment_count") != expected_count or len(wave_entries) != expected_count:
        raise SystemExit(f"wave {wave_number} count drifted")
    if any(entry.get("wave") != wave_number or entry.get("variant") != variant for entry in wave_entries):
        raise SystemExit(f"wave {wave_number} membership drifted")
PY

mkdir -p "${LOGS_ROOT}" "$(dirname "${COMMAND_HISTORY}")"
wave_plans=(
  "${PLAN_DIR}/wave_0_conv_module_dense.json"
  "${PLAN_DIR}/wave_1_conv_out_random.json"
  "${PLAN_DIR}/wave_2_conv_out_magnitude.json"
  "${PLAN_DIR}/wave_3_conv_out_saliency.json"
  "${PLAN_DIR}/wave_4_conv_filter_random.json"
  "${PLAN_DIR}/wave_5_conv_filter_magnitude.json"
  "${PLAN_DIR}/wave_6_conv_filter_saliency.json"
)

# Each foreground runner invocation is a wave barrier: the next wave is not
# submitted until every condition in the current wave has completed.
for wave_index in "${!wave_plans[@]}"; do
  conda run --no-capture-output -n SHOT_TTA python "${PROJECT_ROOT}/tools/run_experiments_multi_gpu.py" \
    "${wave_plans[wave_index]}" \
    --runs-root "${RUNS_ROOT}" \
    --logs-root "${LOGS_ROOT}/wave_${wave_index}" \
    --workdir "${PROJECT_ROOT}" \
    --gpus 0,1,2,3 \
    --max-workers 4 \
    --workers-per-gpu 1 \
    --resume \
    --resume-partial-runs \
    --command-history "${COMMAND_HISTORY}"
done
