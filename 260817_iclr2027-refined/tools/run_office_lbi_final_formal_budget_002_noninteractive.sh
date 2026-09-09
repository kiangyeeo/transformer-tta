#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
EXPERIMENT_ROOT="${PROJECT_DIR}/experiment_logs/shot_otta_office_lbi_final_formal_20260818"
PLAN="${EXPERIMENT_ROOT}/plans/machine3/plan.json"
RUNS_ROOT="${EXPERIMENT_ROOT}/runs"
LOG_ROOT="${EXPERIMENT_ROOT}/launcher_logs/machine3"
COMMAND_HISTORY="${EXPERIMENT_ROOT}/command_history/machine3/commands.sh"
LOCK_DIR="${EXPERIMENT_ROOT}/locks/machine3.lock"
PREPARE_READY="${EXPERIMENT_ROOT}/phase_records/machine3/PREPARE_READY.json"
BASELINE_ROOT="${PROJECT_DIR}/experiment_logs/shot_otta_office_seed2026_stage1_20260818"
PLAN_SHA256="76c853d72318acfbda500fc270fa5cfd5dbd9b20f0441cbf701312805edc5116"
LOCK_ACQUIRED=0

release_lock() {
    if [[ "${LOCK_ACQUIRED}" == "1" ]] && [[ -f "${LOCK_DIR}/pid" ]]; then
        if [[ "$(<"${LOCK_DIR}/pid")" == "$$" ]]; then
            rm -f "${LOCK_DIR}/hostname" "${LOCK_DIR}/pid" "${LOCK_DIR}/start_time"
            rmdir "${LOCK_DIR}" 2>/dev/null || true
        fi
    fi
}

trap 'exit 130' INT
trap 'exit 143' TERM
trap release_lock EXIT

[[ -d "${PROJECT_DIR}" ]]
cd "${PROJECT_DIR}"
[[ "$(pwd -P)" == "${PROJECT_DIR}" ]]
[[ -f "${PLAN}" ]]
[[ -f "${PREPARE_READY}" ]]

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda command not found" >&2
    exit 1
}

conda run --no-capture-output -n SHOT_TTA python -c 'import sys; print(sys.version)' >/dev/null

mkdir -p "$(dirname "${LOCK_DIR}")"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    owner_pid="unknown"
    if [[ -f "${LOCK_DIR}/pid" ]]; then
        owner_pid="$(<"${LOCK_DIR}/pid")"
    fi
    if [[ "${owner_pid}" != "unknown" ]] && kill -0 "${owner_pid}" 2>/dev/null; then
        echo "machine3 final-formal lock is held by live PID ${owner_pid}; refusing double submission" >&2
    else
        echo "machine3 final-formal lock exists (PID ${owner_pid}); refusing automatic stale-lock removal" >&2
    fi
    exit 1
fi
LOCK_ACQUIRED=1
printf '%s\n' "$(hostname)" > "${LOCK_DIR}/hostname"
printf '%s\n' "$$" > "${LOCK_DIR}/pid"
printf '%s\n' "$(date --iso-8601=seconds)" > "${LOCK_DIR}/start_time"

mkdir -p "${LOG_ROOT}" "$(dirname "${COMMAND_HISTORY}")"

conda run --no-capture-output -n SHOT_TTA python - "${PREPARE_READY}" "${PLAN}" "${PLAN_SHA256}" "${RUNS_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

ready_path, plan_path, expected_sha, runs_root = map(Path, sys.argv[1:5])
ready = json.loads(ready_path.read_text(encoding="utf-8"))
raw = plan_path.read_bytes()
if hashlib.sha256(raw).hexdigest() != sys.argv[3]:
    raise SystemExit("plan SHA256 does not match PREPARE_READY")
if ready.get("plan_sha256") != sys.argv[3]:
    raise SystemExit("PREPARE_READY plan_sha256 mismatch")
if ready.get("experiment_count") != 6 or ready.get("unique_experiment_keys") != 6:
    raise SystemExit("PREPARE_READY does not describe exactly six unique experiments")

plan = json.loads(raw.decode("utf-8"))
entries = plan.get("experiments")
if plan.get("experiment_count") != 6 or not isinstance(entries, list) or len(entries) != 6:
    raise SystemExit("final-formal plan must contain exactly six entries")
if len({entry.get("experiment_key") for entry in entries}) != 6:
    raise SystemExit("final-formal plan experiment keys are not unique")
if {(entry.get("source"), entry.get("target")) for entry in entries} != {
    (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
}:
    raise SystemExit("final-formal plan does not contain exactly the six Office transfers")

runs_root = runs_root.resolve()
for index, entry in enumerate(entries):
    prefix = f"entry {index}"
    if (entry.get("dataset"), entry.get("variant"), entry.get("seed")) != ("office", "module_lbi", 2026):
        raise SystemExit(f"{prefix}: dataset/variant/seed mismatch")
    if entry.get("requested_budget") != 0.002:
        raise SystemExit(f"{prefix}: budget mismatch")
    if (entry.get("alpha"), entry.get("kappa"), entry.get("nu"), entry.get("omega"), entry.get("stage2_lr")) != (0.15, 1.0, 0.5, 0.3, 0.005):
        raise SystemExit(f"{prefix}: frozen LBI tuple mismatch")
    if (entry.get("stage1_max_steps"), entry.get("stage2_steps_requested")) != (3000, 1):
        raise SystemExit(f"{prefix}: frozen LBI step setting mismatch")
    if (entry.get("budget_tolerance"), entry.get("delta_nonzero_tolerance")) != (1.0e-4, 1.0e-12):
        raise SystemExit(f"{prefix}: frozen tolerance mismatch")
    scientific = entry.get("scientific_config", {})
    if scientific.get("model", {}).get("backbone") != "resnet50":
        raise SystemExit(f"{prefix}: backbone mismatch")
    if scientific.get("data", {}).get("batch_size") != 64 or scientific.get("data", {}).get("workers") != 4:
        raise SystemExit(f"{prefix}: batch/worker mismatch")
    if scientific.get("lbi", {}).get("support_threshold") != 1.0e-4:
        raise SystemExit(f"{prefix}: support threshold mismatch")
    if scientific.get("variant_policy", {}).get("candidate_scope") != "netB.bottleneck":
        raise SystemExit(f"{prefix}: candidate scope mismatch")
    if entry.get("lbi_resolved") is not True or not scientific.get("lbi"):
        raise SystemExit(f"{prefix}: unresolved LBI tuple")
    command = entry.get("command_args", [])
    if "--no-save-model" not in command or "--output-root" not in command:
        raise SystemExit(f"{prefix}: save_model/output-root command guard failed")
    command_root = Path(command[command.index("--output-root") + 1]).resolve()
    expected_root = Path(entry.get("expected_output_root", "")).resolve()
    if command_root != runs_root:
        raise SystemExit(f"{prefix}: command output root mismatch")
    if expected_root != runs_root and runs_root not in expected_root.parents:
        raise SystemExit(f"{prefix}: expected_output_root is outside canonical RUNS_ROOT")
    if str(expected_root).startswith("/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/experiment_logs/"):
        raise SystemExit(f"{prefix}: legacy output path detected")
print("runtime preflight: plan, hash, tuple, count, uniqueness, and paths passed")
PY

conda run --no-capture-output -n SHOT_TTA python - "${LOG_ROOT}/runtime_preflight.json" "${BASELINE_ROOT}" <<'PY'
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

output_path = Path(sys.argv[1])
baseline_root = Path(sys.argv[2])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to start formal efficiency runs")
if torch.cuda.device_count() < 6:
    raise SystemExit(f"fewer than six addressable GPUs: {torch.cuda.device_count()}")

hostname = socket.gethostname()
records = []
for index in range(6):
    properties = torch.cuda.get_device_properties(index)
    records.append({
        "hostname": hostname,
        "gpu_name": torch.cuda.get_device_name(index),
        "gpu_index": index,
        "gpu_total_memory_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    })

models = {record["gpu_name"] for record in records}
if len(models) != 1:
    raise SystemExit(f"GPUs 0..5 are not the same model: {sorted(models)}")

baseline_records = []
for summary_path in baseline_root.rglob("summary.json"):
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if isinstance(summary.get("gpu_name"), str):
        baseline_records.append(summary)
baseline_models = {record["gpu_name"] for record in baseline_records}
baseline_torch = {record.get("torch_version") for record in baseline_records if record.get("torch_version")}
baseline_cuda = {record.get("cuda_version") for record in baseline_records if record.get("cuda_version")}
if baseline_models and models != baseline_models:
    raise SystemExit(
        "GPU model differs from formal baseline; PU/FO may remain scientifically usable, "
        "but runtime_comparable=true would violate the frozen formal efficiency protocol: "
        f"current={sorted(models)}, baseline={sorted(baseline_models)}"
    )
if baseline_torch and {torch.__version__} != baseline_torch:
    raise SystemExit(f"PyTorch version differs from formal baseline: current={torch.__version__}, baseline={sorted(baseline_torch)}")
if baseline_cuda and {torch.version.cuda} != baseline_cuda:
    raise SystemExit(f"CUDA version differs from formal baseline: current={torch.version.cuda}, baseline={sorted(baseline_cuda)}")

payload = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "runtime_comparable": True,
    "baseline_root": str(baseline_root.resolve()),
    "baseline_gpu_models": sorted(baseline_models),
    "baseline_torch_versions": sorted(baseline_torch),
    "baseline_cuda_versions": sorted(baseline_cuda),
    "gpus": records,
}
output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

echo "Launching exact final-formal machine3 plan: ${PLAN}"
echo "Canonical runs root: ${RUNS_ROOT}"
echo "GPU allocation: 0,1,2,3,4,5; max-workers=6; workers-per-gpu=1; GPUs 6,7 unused"

conda run --no-capture-output -n SHOT_TTA python tools/run_experiments_multi_gpu.py "${PLAN}" \
    --runs-root "${RUNS_ROOT}" \
    --logs-root "${LOG_ROOT}/run" \
    --workdir "${PROJECT_DIR}" \
    --gpus 0,1,2,3,4,5 \
    --max-workers 6 \
    --workers-per-gpu 1 \
    --resume-partial-runs \
    --command-history "${COMMAND_HISTORY}" \
    2>&1 | tee "${LOG_ROOT}/launcher_foreground.log"

echo "Machine3 final-formal plan returned; no FINALIZE, baseline, or tuning command was launched."
