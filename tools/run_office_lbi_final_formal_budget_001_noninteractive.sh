#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
FINAL_ROOT="$PROJECT_DIR/experiment_logs/shot_otta_office_lbi_final_formal_20260818"
PLAN_PATH="$FINAL_ROOT/plans/machine2/plan.json"
PREPARE_READY="$FINAL_ROOT/phase_records/machine2/PREPARE_READY.json"
RUNS_ROOT="$FINAL_ROOT/runs"
LOGS_ROOT="$FINAL_ROOT/launcher_logs/machine2"
COMMAND_HISTORY="$FINAL_ROOT/command_history/machine2_commands.sh"
LOCK_DIR="$FINAL_ROOT/locks/machine2.lock"
HARDWARE_RECORD="$FINAL_ROOT/phase_records/machine2/RUNTIME_HARDWARE.json"

cd "$PROJECT_DIR"

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda not found" >&2
    exit 1
}

mkdir -p "$FINAL_ROOT/locks" "$LOGS_ROOT" "$(dirname "$COMMAND_HISTORY")" "$(dirname "$HARDWARE_RECORD")"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "machine2 lock already exists: $LOCK_DIR" >&2
    echo "Inspect lock.json and do not remove a live process lock automatically." >&2
    exit 1
fi

cleanup_lock() {
    rm -f "$LOCK_DIR/lock.json"
    rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

conda run --no-capture-output -n SHOT_TTA python - "$LOCK_DIR/lock.json" "$$" <<'PY'
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "machine_tag": "machine2",
            "hostname": socket.gethostname(),
            "pid": int(sys.argv[2]),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

conda run --no-capture-output -n SHOT_TTA python - "$PROJECT_DIR" "$PLAN_PATH" "$PREPARE_READY" "$RUNS_ROOT" "$HARDWARE_RECORD" <<'PY'
import hashlib
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

project_dir, plan_name, ready_name, runs_root_name, hardware_name = sys.argv[1:]
project = Path(project_dir).resolve()
plan_path = Path(plan_name).resolve()
ready_path = Path(ready_name).resolve()
runs_root = Path(runs_root_name).resolve()
hardware_path = Path(hardware_name).resolve()

if Path.cwd().resolve() != project:
    raise SystemExit(f"wrong project path: {Path.cwd().resolve()} != {project}")
if not plan_path.is_file():
    raise SystemExit(f"missing plan: {plan_path}")
if not ready_path.is_file():
    raise SystemExit(f"missing PREPARE_READY: {ready_path}")

plan = json.loads(plan_path.read_text(encoding="utf-8"))
ready = json.loads(ready_path.read_text(encoding="utf-8"))
actual_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
if ready.get("plan_sha256") != actual_sha:
    raise SystemExit("plan SHA256 does not match PREPARE_READY")
if ready.get("plan_path") != str(plan_path):
    raise SystemExit("PREPARE_READY plan_path mismatch")
if plan.get("experiment_count") != 6 or len(plan.get("experiments", [])) != 6:
    raise SystemExit("plan count is not exactly 6")

experiments = plan["experiments"]
keys = [entry.get("experiment_key") for entry in experiments]
if len(set(keys)) != 6:
    raise SystemExit("plan experiment_key values are not unique")
expected_transfers = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
observed_transfers = {(entry.get("source"), entry.get("target")) for entry in experiments}
if observed_transfers != expected_transfers:
    raise SystemExit(f"unexpected Office transfers: {sorted(observed_transfers)}")

for entry in experiments:
    if entry.get("variant") != "module_lbi":
        raise SystemExit("plan contains a non-module_lbi entry")
    if entry.get("seed") != 2026 or entry.get("requested_budget") != 0.001:
        raise SystemExit("plan contains an incorrect seed or budget")
    if entry.get("lbi_resolved") is not True:
        raise SystemExit("plan contains unresolved LBI tuple")
    expected_tuple = {
        "alpha": 0.10,
        "kappa": 1.0,
        "nu": 0.25,
        "omega": 0.30,
        "stage1_max_steps": 3000,
        "budget_tolerance": 1.0e-4,
        "stage2_lr": 0.020,
        "stage2_steps_requested": 1,
        "delta_nonzero_tolerance": 1.0e-12,
        "support_threshold": 1.0e-4,
    }
    for field, expected in expected_tuple.items():
        actual = entry.get(field)
        if field == "support_threshold":
            actual = entry.get("scientific_config", {}).get("lbi", {}).get(field)
        if actual != expected:
            raise SystemExit(f"frozen tuple mismatch in {field}: {actual!r}")
    expected = Path(entry.get("expected_output_root", "")).resolve()
    try:
        expected.relative_to(runs_root)
        expected.relative_to(project)
    except ValueError as error:
        raise SystemExit(f"expected_output_root escapes canonical project root: {expected}") from error
    if str(expected).startswith("/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/experiment_logs/"):
        raise SystemExit(f"expected_output_root points to historical root: {expected}")
    args = entry.get("command_args", [])
    try:
        output_root = args[args.index("--output-root") + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit("plan command has no valid --output-root") from error
    if output_root != str(runs_root):
        raise SystemExit("plan command output root does not match canonical RUNS_ROOT")
    if entry.get("effective_overrides", {}).get("save_model") is not False:
        raise SystemExit("plan save_model is not false")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable on the execution machine")
gpu_count = torch.cuda.device_count()
if gpu_count < 6:
    raise SystemExit(f"at least GPUs 0..5 are required; visible count={gpu_count}")

gpus = []
for index in range(6):
    props = torch.cuda.get_device_properties(index)
    gpus.append({
        "index": index,
        "model": props.name,
        "total_memory_mb": int(props.total_memory // (1024 * 1024)),
    })
models = {gpu["model"] for gpu in gpus}
if len(models) != 1:
    raise SystemExit(f"GPUs 0..5 are not the same model: {sorted(models)}")

baseline_root = project / "experiment_logs" / "shot_otta_office_seed2026_stage1_20260818"
baseline_models = set()
if baseline_root.is_dir():
    names = {"hardware_provenance.json", "runtime_hardware.json", "gpu_provenance.json", "hardware.json"}
    for path in baseline_root.rglob("*"):
        if path.name.lower() not in names:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.lower() in {"gpu_model", "gpu_name", "model"} and isinstance(item, str):
                        baseline_models.add(item)
                    elif isinstance(item, (dict, list)):
                        stack.append(item)
            elif isinstance(value, list):
                stack.extend(value)
if baseline_models and not baseline_models.intersection(models):
    raise SystemExit(
        "GPU model differs from formal baseline provenance: "
        f"current={sorted(models)}, baseline={sorted(baseline_models)}"
    )

record = {
    "machine_tag": "machine2",
    "hostname": socket.gethostname(),
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "gpu_indices": list(range(6)),
    "gpus": gpus,
    "pytorch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "visible_gpu_count": gpu_count,
    "baseline_provenance_found": bool(baseline_models),
    "baseline_gpu_models": sorted(baseline_models),
    "runtime_comparable": True,
}
hardware_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
PY

conda run --no-capture-output -n SHOT_TTA python tools/run_experiments_multi_gpu.py \
    "$PLAN_PATH" \
    --runs-root "$RUNS_ROOT" \
    --logs-root "$LOGS_ROOT" \
    --workdir "$PROJECT_DIR" \
    --gpus 0,1,2,3,4,5 \
    --max-workers 6 \
    --workers-per-gpu 1 \
    --resume-partial-runs \
    --command-history "$COMMAND_HISTORY"
