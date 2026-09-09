#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
FORMAL_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_baselines_seed2026_formal_20260827"
PLAN_DIR="${FORMAL_ROOT}/plans/node3_dw"
PLAN_PATH="${PLAN_DIR}/plan.json"
WAVE_MANIFEST="${PLAN_DIR}/wave_manifest.json"
PREPARE_RECORD="${FORMAL_ROOT}/phase_records/node3_dw/PREPARE.md"
RUNS_ROOT="${FORMAL_ROOT}/runs"
LOGS_ROOT="${FORMAL_ROOT}/launcher_logs/node3_dw"
COMMAND_HISTORY="${FORMAL_ROOT}/command_history/node3_dw_commands.sh"
LOCK_PATH="${FORMAL_ROOT}/locks/node3_dw.lock"

mkdir -p "$(dirname "${LOCK_PATH}")"
if [[ ! -f "${PLAN_PATH}" || ! -f "${WAVE_MANIFEST}" || ! -f "${PREPARE_RECORD}" ]]; then
  echo "ERROR: Node 3 D->W plan, wave manifest, or PREPARE record is missing; rerun PREPARE first." >&2
  exit 1
fi
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: Node 3 D->W launcher lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi
cleanup() {
  local status=$?
  rmdir "${LOCK_PATH}" 2>/dev/null || true
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

conda run --no-capture-output -n SHOT_TTA python - "${PLAN_PATH}" "${WAVE_MANIFEST}" "${RUNS_ROOT}" <<'PY'
import json
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
runs_root = pathlib.Path(sys.argv[3]).resolve()
entries = plan.get("experiments", [])
ratios = {0.0005, 0.001, 0.002}
expected_k = {"out_channel": {0.0005: 4, 0.001: 9, 0.002: 18},
              "filter_connection": {0.0005: 3276, 0.001: 6553, 0.002: 13107}}
expected_waves = [(0, "conv_module_dense", 1), (1, "conv_out_random", 3),
                  (2, "conv_out_magnitude", 3), (3, "conv_out_saliency", 3),
                  (4, "conv_filter_random", 3), (5, "conv_filter_magnitude", 3),
                  (6, "conv_filter_saliency", 3)]
if plan.get("experiment_count") != 19 or len(entries) != 19:
    raise SystemExit("Node 3 plan count must be exactly 19")
for field in ("experiment_key", "experiment_config_sha256", "expected_output_root"):
    if len({entry.get(field) for entry in entries}) != 19:
        raise SystemExit(f"Node 3 {field} values are not unique")
if any("_lbi" in entry.get("variant", "") for entry in entries):
    raise SystemExit("Node 3 must not contain conv_*_lbi")
if any(entry.get("dataset") != "office" or (entry.get("source"), entry.get("target")) != (1, 2)
       or entry.get("seed") != 2026 for entry in entries):
    raise SystemExit("Node 3 transfer or formal seed drifted")
for entry in entries:
    if entry.get("variant") == "conv_module_dense":
        if entry.get("requested_budget") != 1.0 or entry.get("group_mode") is not None:
            raise SystemExit("Node 3 dense anchor drifted")
    else:
        mode, rho = entry.get("group_mode"), entry.get("requested_budget")
        if mode not in expected_k or rho not in ratios or entry.get("max_group_count") != expected_k[mode][rho]:
            raise SystemExit("Node 3 sparse budget or K_G drifted")
        if entry.get("variant", "").endswith("_random") and not (
            entry.get("selection_seed") == 2026
            and entry.get("selection_seed_mode") == "same_as_run_seed"
            and entry.get("num_random_masks") == 3
            and entry.get("random_child_mask_indices") == [0, 1, 2]
        ):
            raise SystemExit("Node 3 Random three-mask semantics drifted")
    try:
        pathlib.Path(entry["expected_output_root"]).resolve().relative_to(runs_root)
    except ValueError as error:
        raise SystemExit("planned output escapes Node 3 runs root") from error
waves = manifest.get("waves", [])
if [(wave.get("wave"), wave.get("variant"), wave.get("experiment_count")) for wave in waves] != expected_waves:
    raise SystemExit("Node 3 wave manifest drifted")
for wave_number, variant, count in expected_waves:
    wave_plan = json.loads(pathlib.Path(waves[wave_number]["plan_path"]).read_text(encoding="utf-8"))
    wave_entries = wave_plan.get("experiments", [])
    if wave_plan.get("experiment_count") != count or len(wave_entries) != count:
        raise SystemExit(f"Node 3 wave {wave_number} count drifted")
    if any(entry.get("wave") != wave_number or entry.get("variant") != variant for entry in wave_entries):
        raise SystemExit(f"Node 3 wave {wave_number} membership drifted")
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
