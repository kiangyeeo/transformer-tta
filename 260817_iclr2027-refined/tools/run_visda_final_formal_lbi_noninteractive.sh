#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
FORMAL_ROOT="${PROJECT_DIR}/experiment_logs/visda_fc_lbi_formal_seed2026_20260825"
PLAN_PATH="${FORMAL_ROOT}/plans/visda_final_formal_lbi/plan.json"
RUNS_ROOT="${FORMAL_ROOT}/runs"
LOGS_ROOT="${FORMAL_ROOT}/launcher_logs/visda_final_formal_lbi"
COMMAND_HISTORY="${FORMAL_ROOT}/command_history/visda_final_formal_lbi/commands.sh"

[[ -d "${PROJECT_DIR}" ]]
[[ -f "${PLAN_PATH}" ]]
cd "${PROJECT_DIR}"
[[ "$(pwd -P)" == "${PROJECT_DIR}" ]]

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda command not found" >&2
    exit 1
}

# Keep the formal root self-contained and fail closed before a child process
# can use a non-final tuple, an extra identity, or a tuning output path.
conda run --no-capture-output -n SHOT_TTA python - "${PLAN_PATH}" "${RUNS_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1]).resolve()
runs_root = Path(sys.argv[2]).resolve()
plan = json.loads(plan_path.read_text(encoding="utf-8"))
entries = plan.get("experiments")
expected = {
    (0.0005, 0.125, 1.0, 0.5, 0.00625, 0.010): 262,
    (0.001, 0.10, 1.0, 0.5, 0.003125, 0.0025): 524,
    (0.002, 0.15, 1.0, 0.5, 0.0015625, 0.005): 1049,
}

if plan.get("experiment_count") != 3 or not isinstance(entries, list) or len(entries) != 3:
    raise SystemExit("formal VisDA plan must contain exactly three entries")
keys = [entry.get("experiment_key") for entry in entries]
shas = [entry.get("experiment_config_sha256") for entry in entries]
if len(set(keys)) != 3 or len(set(shas)) != 3:
    raise SystemExit("formal VisDA plan identities/config SHAs must be unique")

seen = set()
for index, entry in enumerate(entries):
    prefix = f"entry {index}"
    if (entry.get("dataset"), entry.get("source"), entry.get("target"), entry.get("seed"), entry.get("variant")) != ("VISDA-C", 0, 1, 2026, "module_lbi"):
        raise SystemExit(f"{prefix}: dataset/transfer/seed/variant mismatch")
    scientific = entry.get("scientific_config", {})
    if scientific.get("data", {}).get("batch_size") != 256 or scientific.get("data", {}).get("workers") != 4:
        raise SystemExit(f"{prefix}: batch_size/workers mismatch")
    if scientific.get("variant_policy", {}).get("candidate_scope") != "netB.bottleneck" or scientific.get("variant_policy", {}).get("bn_stats_frozen") is not True:
        raise SystemExit(f"{prefix}: controlled-FC candidate/BN policy mismatch")
    if (entry.get("stage1_max_steps"), entry.get("stage2_steps_requested"), scientific.get("lbi", {}).get("support_threshold")) != (3000, 1, 1.0e-4):
        raise SystemExit(f"{prefix}: frozen LBI constants mismatch")
    frozen = (entry.get("requested_budget"), entry.get("alpha"), entry.get("kappa"), entry.get("nu"), entry.get("omega"), entry.get("stage2_lr"))
    if frozen not in expected:
        raise SystemExit(f"{prefix}: unexpected or tuning tuple {frozen!r}")
    if entry.get("lbi_resolved") is not True:
        raise SystemExit(f"{prefix}: unresolved LBI tuple")
    if int(scientific.get("variant_policy", {}).get("candidate_param_count", 524544)) != 524544:
        raise SystemExit(f"{prefix}: FC candidate count mismatch")
    if int(entry.get("max_support_count", expected[frozen])) != expected[frozen]:
        raise SystemExit(f"{prefix}: strict integer K mismatch")
    command = entry.get("command_args", [])
    if "--output-root" not in command or "--no-save-model" not in command:
        raise SystemExit(f"{prefix}: command output/save-model guard failed")
    command_root = Path(command[command.index("--output-root") + 1]).resolve()
    expected_root = Path(entry.get("expected_output_root", "")).resolve()
    if command_root != runs_root or runs_root not in expected_root.parents:
        raise SystemExit(f"{prefix}: output path is outside the new formal root")
    seen.add(frozen)

if set(expected) != seen:
    raise SystemExit("formal VisDA plan does not contain exactly F1, F2, and F3")
print("formal VisDA preflight passed: exactly F1/F2/F3, unique identities/SHAs, frozen tuples, and new output root")
PY

mkdir -p "${LOGS_ROOT}" "$(dirname "${COMMAND_HISTORY}")"

exec conda run --no-capture-output -n SHOT_TTA python tools/run_experiments_multi_gpu.py \
    "${PLAN_PATH}" \
    --runs-root "${RUNS_ROOT}" \
    --logs-root "${LOGS_ROOT}" \
    --workdir "${PROJECT_DIR}" \
    --gpus 0,1,2 \
    --max-workers 3 \
    --workers-per-gpu 1 \
    --resume \
    --resume-partial-runs \
    --command-history "${COMMAND_HISTORY}"
