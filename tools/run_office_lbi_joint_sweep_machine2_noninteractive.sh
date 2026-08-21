#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
SWEEP_ROOT="$PROJECT_DIR/experiment_logs/shot_otta_office_lbi_joint_sweep_20260818"
PLAN_PATH="$SWEEP_ROOT/plans/machine2/plan.json"
PREPARE_READY="$SWEEP_ROOT/phase_records/machine2/PREPARE_READY.json"
RUNS_ROOT="$SWEEP_ROOT/runs"
LOGS_ROOT="$SWEEP_ROOT/launcher_logs/machine2"
COMMAND_HISTORY="$SWEEP_ROOT/command_history/machine2_commands.sh"
LOCK_DIR="$SWEEP_ROOT/locks/machine2.lock"

cd "$PROJECT_DIR"

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda not found" >&2
    exit 1
}

conda run --no-capture-output -n SHOT_TTA python - "$PROJECT_DIR" "$PLAN_PATH" "$PREPARE_READY" "$RUNS_ROOT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

project_dir, plan_name, ready_name, runs_root_name = sys.argv[1:]
project = Path(project_dir).resolve()
plan_path = Path(plan_name).resolve()
ready_path = Path(ready_name).resolve()
runs_root = Path(runs_root_name).resolve()

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
if plan.get("experiment_count") != 132 or len(plan.get("experiments", [])) != 132:
    raise SystemExit("plan count is not exactly 132")

experiments = plan["experiments"]
keys = [entry.get("experiment_key") for entry in experiments]
if len(set(keys)) != 132:
    raise SystemExit("plan experiment_key values are not unique")

pair_counts = {}
for entry in experiments:
    pair = (entry.get("alpha"), entry.get("kappa"), entry.get("nu"))
    pair_counts[pair] = pair_counts.get(pair, 0) + 1
if pair_counts != {(0.1, 1.0, 0.25): 66, (0.1, 1.5, 0.5): 66}:
    raise SystemExit(f"unexpected owned-pair counts: {pair_counts}")

for entry in experiments:
    expected = Path(entry.get("expected_output_root", "")).resolve()
    try:
        expected.relative_to(runs_root)
    except ValueError as error:
        raise SystemExit(f"expected_output_root escapes canonical runs root: {expected}") from error
    if entry.get("omega") == 0.20 and entry.get("stage2_lr") == 0.020:
        raise SystemExit("forbidden reused omega=0.20/stage2_lr=0.020 entry in new plan")
    args = entry.get("command_args", [])
    try:
        output_root = args[args.index("--output-root") + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit("plan command has no valid --output-root") from error
    if output_root != str(runs_root):
        raise SystemExit("plan command output root does not match canonical runs root")

print("machine2 plan preflight passed: 132 unique entries, pair counts 66+66")
PY

mkdir -p "$SWEEP_ROOT/locks" "$LOGS_ROOT" "$(dirname "$COMMAND_HISTORY")"
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
import os
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

conda run --no-capture-output -n SHOT_TTA python tools/run_experiments_multi_gpu.py \
    "$PLAN_PATH" \
    --runs-root "$RUNS_ROOT" \
    --logs-root "$LOGS_ROOT" \
    --workdir "$PROJECT_DIR" \
    --gpus 0,1,2,3,4,5,6,7 \
    --max-workers 16 \
    --workers-per-gpu 2 \
    --resume-partial-runs \
    --command-history "$COMMAND_HISTORY"
