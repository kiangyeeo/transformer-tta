#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
FINAL_ROOT="${PROJECT_ROOT}/experiment_logs/shot_otta_office_lbi_final_formal_20260818"
RUNS_ROOT="${FINAL_ROOT}/runs"
PLAN_PATH="${FINAL_ROOT}/plans/machine1/plan.json"
PREPARE_READY_PATH="${FINAL_ROOT}/phase_records/machine1/PREPARE_READY.json"
LAUNCHER_LOG_ROOT="${FINAL_ROOT}/launcher_logs/machine1"
COMMAND_HISTORY="${FINAL_ROOT}/command_history/machine1_commands.sh"
PROVENANCE_PATH="${FINAL_ROOT}/phase_records/machine1/runtime_provenance.json"
LOCK_PATH="${FINAL_ROOT}/locks/machine1.lock"
BASELINE_ROOT="${PROJECT_ROOT}/experiment_logs/shot_otta_office_seed2026_stage1_20260818"

mkdir -p "${LAUNCHER_LOG_ROOT}" "$(dirname "${COMMAND_HISTORY}")" \
  "$(dirname "${PROVENANCE_PATH}")" "$(dirname "${LOCK_PATH}")"

LOCK_HELD=0
cleanup() {
  local status=$?
  if [[ "${LOCK_HELD}" == "1" ]]; then
    rm -f "${LOCK_PATH}/owner.txt"
    rmdir "${LOCK_PATH}" 2>/dev/null || true
    LOCK_HELD=0
  fi
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: lock already exists or cannot be created: ${LOCK_PATH}" >&2
  exit 1
fi
LOCK_HELD=1
{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'pid=%s\n' "$$"
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${LOCK_PATH}/owner.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command is unavailable" >&2
  exit 1
fi

if [[ ! -f "${PLAN_PATH}" || ! -f "${PREPARE_READY_PATH}" ]]; then
  echo "ERROR: PREPARE plan or PREPARE_READY marker is missing" >&2
  exit 1
fi

conda run --no-capture-output -n SHOT_TTA python -c 'import sys; print(sys.version)' >/dev/null

ACTUAL_PLAN_SHA256="$(sha256sum "${PLAN_PATH}" | awk '{print $1}')"
EXPECTED_PLAN_SHA256="$(conda run --no-capture-output -n SHOT_TTA python -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["plan_sha256"])' \
  "${PREPARE_READY_PATH}")"
if [[ "${ACTUAL_PLAN_SHA256}" != "${EXPECTED_PLAN_SHA256}" ]]; then
  echo "ERROR: plan SHA256 does not match PREPARE_READY" >&2
  echo "expected=${EXPECTED_PLAN_SHA256} actual=${ACTUAL_PLAN_SHA256}" >&2
  exit 1
fi

conda run --no-capture-output -n SHOT_TTA python - "${PLAN_PATH}" "${RUNS_ROOT}" "${PROJECT_ROOT}" <<'PY'
import json
import math
import pathlib
import sys

plan_path = pathlib.Path(sys.argv[1]).resolve()
runs_root = pathlib.Path(sys.argv[2]).resolve()
project_root = pathlib.Path(sys.argv[3]).resolve()
plan = json.loads(plan_path.read_text(encoding="utf-8"))
entries = plan.get("experiments", [])
if plan.get("experiment_count") != 6 or len(entries) != 6:
    raise SystemExit("plan count is not exactly 6")
keys = [entry.get("experiment_key") for entry in entries]
if len(set(keys)) != 6:
    raise SystemExit("plan experiment keys are not unique")
transfers = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
observed = set()
for entry in entries:
    observed.add((entry.get("source"), entry.get("target")))
    if entry.get("dataset") != "office" or entry.get("variant") != "module_lbi":
        raise SystemExit("plan contains an unintended dataset or variant")
    if entry.get("seed") != 2026 or entry.get("requested_budget") != 0.0005:
        raise SystemExit("plan seed or budget drifted")
    expected_tuple = {
        "alpha": 0.10, "kappa": 1.0, "nu": 0.25, "omega": 0.30,
        "stage1_max_steps": 3000, "budget_tolerance": 1.0e-4,
        "stage2_lr": 0.020, "stage2_steps_requested": 1,
        "delta_nonzero_tolerance": 1.0e-12,
    }
    for field, expected in expected_tuple.items():
        if entry.get(field) != expected:
            raise SystemExit(f"plan LBI tuple drifted at {field}: {entry.get(field)!r}")
    scientific_lbi = entry.get("scientific_config", {}).get("lbi", {})
    if scientific_lbi.get("support_threshold") != 1.0e-4:
        raise SystemExit("plan support_threshold drifted")
    data = entry.get("scientific_config", {}).get("data", {})
    if data.get("batch_size") != 64 or data.get("workers") != 4:
        raise SystemExit("plan Office batch/workers drifted")
    if entry.get("effective_overrides", {}).get("save_model") is not False:
        raise SystemExit("plan save_model is not false")
    expected = entry.get("expected_output_root")
    if not isinstance(expected, str):
        raise SystemExit("plan expected_output_root is missing")
    expected_path = pathlib.Path(expected).resolve()
    try:
        expected_path.relative_to(runs_root)
        expected_path.relative_to(project_root)
    except ValueError as error:
        raise SystemExit(f"plan output path is outside the frozen roots: {expected}") from error
    if str(expected_path).startswith("/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/experiment_logs/"):
        raise SystemExit("plan output path points to the historical workspace root")
if observed != transfers:
    raise SystemExit(f"plan transfer set mismatch: {sorted(observed)}")
PY

conda run --no-capture-output -n SHOT_TTA python - "${PROVENANCE_PATH}" "${BASELINE_ROOT}" <<'PY'
import json
import pathlib
import socket
import sys

import torch

provenance_path = pathlib.Path(sys.argv[1])
baseline_root = pathlib.Path(sys.argv[2])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to start formal efficiency runs")
if torch.cuda.device_count() < 6:
    raise SystemExit(f"fewer than six addressable GPUs: {torch.cuda.device_count()}")

devices = []
for index in range(6):
    props = torch.cuda.get_device_properties(index)
    devices.append({
        "gpu_device_index": index,
        "gpu_name": torch.cuda.get_device_name(index),
        "gpu_total_memory_bytes": int(props.total_memory),
    })
models = {item["gpu_name"] for item in devices}
if len(models) != 1:
    raise SystemExit(f"GPUs 0..5 are not the same model: {sorted(models)}")

baseline_models = set()
if baseline_root.is_dir():
    for summary_path in baseline_root.rglob("summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            summary.get("dataset") == "office"
            and summary.get("seed") == 2026
            and summary.get("runtime_comparable") is True
            and summary.get("gpu_name")
        ):
            baseline_models.add(summary["gpu_name"])
if baseline_models and models != baseline_models:
    raise SystemExit(
        "GPU model differs from formal baseline efficiency hardware; "
        "PU/FO may remain scientifically usable, but runtime comparison "
        f"would not satisfy the frozen protocol (current={sorted(models)}, "
        f"baseline={sorted(baseline_models)})"
    )

payload = {
    "machine_tag": "machine1",
    "hostname": socket.gethostname(),
    "gpu_indices": [0, 1, 2, 3, 4, 5],
    "gpu_count_visible": int(torch.cuda.device_count()),
    "gpus": devices,
    "gpu_model_uniform": True,
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "baseline_gpu_models": sorted(baseline_models),
    "runtime_comparable": True,
    "runtime_comparable_reason": (
        "six dedicated GPUs, workers_per_gpu=1, uniform GPU model, and "
        "baseline model matched when baseline provenance was available"
    ),
}
provenance_path.parent.mkdir(parents=True, exist_ok=True)
provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

conda run --no-capture-output -n SHOT_TTA python "${PROJECT_ROOT}/tools/run_experiments_multi_gpu.py" \
  "${PLAN_PATH}" \
  --runs-root "${RUNS_ROOT}" \
  --logs-root "${LAUNCHER_LOG_ROOT}" \
  --workdir "${PROJECT_ROOT}" \
  --gpus 0,1,2,3,4,5 \
  --max-workers 6 \
  --workers-per-gpu 1 \
  --resume-partial-runs \
  --command-history "${COMMAND_HISTORY}"
