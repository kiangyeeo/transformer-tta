#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
R0_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_lbi_r0_reachability_seed2026_20260827"
PLAN_DIR="${R0_ROOT}/plans/r0_da_reachability"
PLAN_PATH="${PLAN_DIR}/plan.json"
WAVE_MANIFEST="${PLAN_DIR}/wave_manifest.json"
PREPARE_RECORD="${R0_ROOT}/phase_records/R0/PREPARE.md"
RUNS_ROOT="${R0_ROOT}/runs"
LOGS_ROOT="${R0_ROOT}/launcher_logs/R0"
COMMAND_HISTORY="${R0_ROOT}/command_history/R0_commands.sh"
LOCK_PATH="${R0_ROOT}/locks/R0.lock"

if [[ ! -f "${PLAN_PATH}" || ! -f "${WAVE_MANIFEST}" || ! -f "${PREPARE_RECORD}" ]]; then
  echo "ERROR: R0 plan, wave manifest, or PREPARE record is missing; run PREPARE first." >&2
  exit 1
fi

mkdir -p "$(dirname "${LOCK_PATH}")"
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: R0 launcher lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi
cleanup() {
  local status=$?
  rmdir "${LOCK_PATH}" 2>/dev/null || true
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

# CPU-only fail-fast validation; no CUDA/GPU query is performed here.
conda run --no-capture-output -n SHOT_TTA python - "${PLAN_PATH}" "${WAVE_MANIFEST}" "${RUNS_ROOT}" <<'PY'
import json
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
runs_root = pathlib.Path(sys.argv[3]).resolve()
entries = plan.get("experiments", [])
modes = {
    "out_channel": ("conv_out_lbi", 0, 9216, {0.0005: 4, 0.001: 9, 0.002: 18}),
    "filter_connection": ("conv_filter_lbi", 1, 6553600, {0.0005: 3276, 0.001: 6553, 0.002: 13107}),
}
anchors = {"A0": (0.10, 0.50), "A1": (0.05, 0.50), "A2": (0.20, 0.50), "A3": (0.10, 0.25), "A4": (0.10, 1.00)}
budgets = {0.0005, 0.001, 0.002}
if plan.get("phase") != "R0" or plan.get("experiment_count") != 30 or len(entries) != 30:
    raise SystemExit("R0 plan count or phase must be exactly 30/R0")
if plan.get("protocol_revision") != "OTTA_CONV_LBI_PROTOCOL_20260826_v1":
    raise SystemExit("frozen Conv protocol revision drifted")
if plan.get("search_protocol_revision") != "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1":
    raise SystemExit("search protocol provenance drifted")
if plan.get("implementation_revision") != "iclr2027_refined_conv_20260826_v1":
    raise SystemExit("frozen Conv implementation revision drifted")
for field in ("experiment_key", "experiment_config_sha256", "expected_output_root"):
    values = [entry.get(field) for entry in entries]
    if len(values) != len(set(values)):
        raise SystemExit(f"R0 {field} values are not unique")
for entry in entries:
    mode = entry.get("group_mode")
    if mode not in modes:
        raise SystemExit("R0 contains an invalid group mode")
    variant, wave, total_groups, expected_k = modes[mode]
    if (entry.get("variant"), entry.get("wave")) != (variant, wave):
        raise SystemExit("R0 variant/wave drifted")
    if (entry.get("dataset"), entry.get("transfer"), entry.get("source"), entry.get("target"), entry.get("seed")) != ("office", "DA", 1, 0, 2026):
        raise SystemExit("R0 dataset/transfer/seed drifted")
    if (entry.get("source_name"), entry.get("target_name")) != ("dslr", "amazon"):
        raise SystemExit("R0 Office D->A domain names drifted")
    rho = entry.get("requested_budget")
    if rho not in budgets or entry.get("max_group_count") != expected_k[rho]:
        raise SystemExit("R0 formal budget/K_G drifted; pilot .005 is forbidden")
    if entry.get("total_group_count") != total_groups:
        raise SystemExit("R0 total group count drifted")
    anchor = entry.get("anchor")
    if anchor not in anchors or (entry.get("alpha"), entry.get("nu")) != anchors[anchor]:
        raise SystemExit("R0 anchor alpha/nu drifted")
    expected_lbi = {"alpha": entry["alpha"], "kappa": 1.0, "nu": entry["nu"], "omega": 0.00625, "stage1_max_steps": 3000, "budget_tolerance": 0.0001, "stage2_lr": 0.005, "stage2_steps": 1, "delta_nonzero_tolerance": 1.0e-12, "support_threshold": 1.0e-4}
    if entry.get("lbi") != expected_lbi:
        raise SystemExit("R0 LBI tuple or frozen constants drifted")
    if entry.get("search_protocol_revision") != "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1":
        raise SystemExit("R0 row search provenance drifted")
    try:
        pathlib.Path(entry["expected_output_root"]).resolve().relative_to(runs_root)
    except ValueError as error:
        raise SystemExit("R0 output root escapes R0 runs root") from error
if sum(entry.get("wave") == 0 for entry in entries) != 15 or sum(entry.get("wave") == 1 for entry in entries) != 15:
    raise SystemExit("R0 wave counts must be 15/15")
waves = manifest.get("waves", [])
if [(wave.get("wave"), wave.get("group_mode"), wave.get("experiment_count")) for wave in waves] != [(0, "out_channel", 15), (1, "filter_connection", 15)]:
    raise SystemExit("R0 wave manifest drifted")
for wave in waves:
    wave_plan = json.loads(pathlib.Path(wave["plan_path"]).read_text(encoding="utf-8"))
    wave_entries = wave_plan.get("experiments", [])
    if wave_plan.get("experiment_count") != 15 or len(wave_entries) != 15:
        raise SystemExit("R0 wave plan count drifted")
    if any(entry.get("wave") != wave["wave"] or entry.get("group_mode") != wave["group_mode"] for entry in wave_entries):
        raise SystemExit("R0 wave membership drifted")
PY

mkdir -p "${LOGS_ROOT}" "$(dirname "${COMMAND_HISTORY}")"
wave_plans=(
  "${PLAN_DIR}/wave_0_out_channel.json"
  "${PLAN_DIR}/wave_1_filter_connection.json"
)
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
