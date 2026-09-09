#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
FORMAL_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_baselines_seed2026_formal_20260827"
PLAN_DIR="${FORMAL_ROOT}/plans/node1_aw"
PLAN_PATH="${PLAN_DIR}/plan.json"
WAVE_MANIFEST="${PLAN_DIR}/wave_manifest.json"
PREPARE_RECORD="${FORMAL_ROOT}/phase_records/node1_aw/PREPARE.md"
RUNS_ROOT="${FORMAL_ROOT}/runs"
LOGS_ROOT="${FORMAL_ROOT}/launcher_logs/node1_aw"
COMMAND_HISTORY="${FORMAL_ROOT}/command_history/node1_aw_commands.sh"
LOCK_PATH="${FORMAL_ROOT}/locks/node1_aw.lock"

mkdir -p "$(dirname "${LOCK_PATH}")"
if [[ ! -f "${PLAN_PATH}" || ! -f "${WAVE_MANIFEST}" || ! -f "${PREPARE_RECORD}" ]]; then
  echo "ERROR: formal Node 1 plan, manifest, or PREPARE record is missing." >&2
  exit 1
fi
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: formal Node 1 launcher lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi
cleanup() {
  local status=$?
  rmdir "${LOCK_PATH}" 2>/dev/null || true
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

# Validation is CPU-only and deliberately avoids probing any GPU.
conda run --no-capture-output -n SHOT_TTA python - "${PLAN_PATH}" "${WAVE_MANIFEST}" "${RUNS_ROOT}" <<'PY'
import json
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
runs_root = pathlib.Path(sys.argv[3]).resolve()
entries = plan.get("experiments", [])
formal_budgets = {0.0005, 0.001, 0.002}
expected_k = {
    "out_channel": {0.0005: 4, 0.001: 9, 0.002: 18},
    "filter_connection": {0.0005: 3276, 0.001: 6553, 0.002: 13107},
}
expected_waves = [(0, "conv_module_dense", 1), (1, "conv_out_random", 3),
                  (2, "conv_out_magnitude", 3), (3, "conv_out_saliency", 3),
                  (4, "conv_filter_random", 3), (5, "conv_filter_magnitude", 3),
                  (6, "conv_filter_saliency", 3)]
if plan.get("experiment_count") != 19 or len(entries) != 19:
    raise SystemExit("Node 1 must contain exactly 19 scientific conditions")
if len({e.get("experiment_key") for e in entries}) != 19 or len({e.get("experiment_config_sha256") for e in entries}) != 19:
    raise SystemExit("Node 1 identities are not unique")
if len({e.get("expected_output_root") for e in entries}) != 19:
    raise SystemExit("Node 1 output roots are not unique")
for entry in entries:
    if entry.get("dataset") != "office" or (entry.get("source"), entry.get("target")) != (0, 2) or entry.get("seed") != 2026:
        raise SystemExit("Node 1 transfer identity drifted from Office A->W")
    if "lbi" in entry.get("variant", "") or entry.get("requested_budget") == 0.005:
        raise SystemExit("Node 1 contains forbidden LBI or pilot budget")
    if entry.get("variant") != "conv_module_dense":
        if entry.get("requested_budget") not in formal_budgets or entry.get("max_group_count") != expected_k[entry.get("group_mode")][entry.get("requested_budget")]:
            raise SystemExit("Node 1 sparse budget or K_G drifted")
        if entry["variant"].endswith("_random") and (entry.get("selection_seed"), entry.get("num_random_masks")) != (2026, 3):
            raise SystemExit("Node 1 Random must have formal seed 2026 and exactly three child masks")
    try:
        pathlib.Path(entry["expected_output_root"]).resolve().relative_to(runs_root)
    except ValueError as error:
        raise SystemExit("Node 1 output escapes formal runs root") from error
if [(w.get("wave"), w.get("variant"), w.get("experiment_count")) for w in manifest.get("waves", [])] != expected_waves:
    raise SystemExit("Node 1 wave manifest drifted")
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
for wave_plan in "${wave_plans[@]}"; do
  conda run --no-capture-output -n SHOT_TTA python "${PROJECT_ROOT}/tools/run_experiments_multi_gpu.py" \
    "${wave_plan}" --runs-root "${RUNS_ROOT}" --logs-root "${LOGS_ROOT}/$(basename "${wave_plan}" .json)" \
    --workdir "${PROJECT_ROOT}" --gpus 0,1,2,3 --max-workers 4 --workers-per-gpu 1 \
    --resume --resume-partial-runs --command-history "${COMMAND_HISTORY}"
done
