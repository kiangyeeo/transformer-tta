#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
BURST_ROOT="${PROJECT_ROOT}/experiment_logs/conv_lbi_32gpu_burst_20260827"
BRANCH_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_lbi_r0f_filter_reachability_seed2026_20260827"
PLAN_PATH="${BURST_ROOT}/plans/node4_filter_r0f.jsonl"
PREPARE_RECORD="${BURST_ROOT}/node_records/node4_filter_r0f_PREPARE.md"
RUNS_ROOT="${BRANCH_ROOT}/runs"
LOGS_ROOT="${BRANCH_ROOT}/launcher_logs/node4_filter_r0f"
COMMAND_HISTORY="${BRANCH_ROOT}/command_history/node4_filter_r0f_commands.sh"
LOCK_PATH="${BRANCH_ROOT}/locks/node4_filter_r0f.lock"

cd "${PROJECT_ROOT}"

if [[ ! -f "${PLAN_PATH}" || ! -f "${PREPARE_RECORD}" ]]; then
  echo "ERROR: Node 4 plan or PREPARE record is missing; run PREPARE first." >&2
  exit 1
fi
mkdir -p "$(dirname "${LOCK_PATH}")"
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: Node 4 launcher lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi
cleanup() {
  local status=$?
  rmdir "${LOCK_PATH}" 2>/dev/null || true
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

# CPU-only fail-fast plan validation; this does not query or reserve a GPU.
conda run --no-capture-output -n SHOT_TTA python - "${PLAN_PATH}" "${RUNS_ROOT}" <<'PY'
import json
import pathlib
import sys
from experiment_identity import build_experiment_identity
from shot_otta.artifacts import experiment_output_root
from shot_otta.config import DATASETS, apply_overrides, load_yaml, resolve_effective_config
from train import build_parser
from tools.run_experiments_multi_gpu import checkpoint_eligible
plan_path, runs_root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).resolve()
rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines() if line.strip()]
anchors = {"F0": (0.25, 0.5), "F1": (0.3, 0.5), "F2": (0.4, 0.5), "F3": (0.6, 0.5)}
tail = {"kappa": 1.0, "omega": 0.00625, "stage1_max_steps": 3000, "budget_tolerance": 0.0001, "stage2_lr": 0.005, "stage2_steps": 1, "delta_nonzero_tolerance": 1e-12, "support_threshold": 1e-4}
if len(rows) != 4 or {row.get("anchor") for row in rows} != set(anchors): raise SystemExit("exactly F0--F3 required")
if DATASETS["office"]["domains"] != ["amazon", "dslr", "webcam"]: raise SystemExit("Office domain mapping drifted")
for field in ("experiment_key", "experiment_config_sha256", "expected_output_root"):
    if len({row.get(field) for row in rows}) != 4: raise SystemExit(f"duplicate {field}")
for row in rows:
    if (row.get("phase"), row.get("dataset"), row.get("transfer"), row.get("source"), row.get("target"), row.get("seed")) != ("R0F", "office", "DA", 1, 0, 2026): raise SystemExit("transfer drifted")
    if (row.get("variant"), row.get("group_mode"), row.get("rho_G"), row.get("requested_budget"), row.get("K_G"), row.get("max_group_count"), row.get("total_group_count"), row.get("runtime_comparable")) != ("conv_filter_lbi", "filter_connection", 0.002, 0.002, 13107, 13107, 6553600, False): raise SystemExit("method/budget/runtime drifted")
    if (row.get("alpha"), row.get("nu")) != anchors[row["anchor"]]: raise SystemExit("anchor tuple drifted")
    if row.get("lbi") != {"alpha": row["alpha"], "nu": row["nu"], **tail}: raise SystemExit("LBI settings drifted")
    if row.get("requested_budget") == 0.005 or row.get("search_protocol_revision") != "OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2": raise SystemExit("prohibited budget or protocol drifted")
    if not checkpoint_eligible(row): raise SystemExit("completed-batch resume unavailable")
    try: pathlib.Path(row["expected_output_root"]).resolve().relative_to(runs_root)
    except ValueError as error: raise SystemExit("output root escapes branch") from error
    args = build_parser().parse_args(row["command_args"][2:])
    effective = resolve_effective_config(apply_overrides(load_yaml(args.config), args), pathlib.Path.cwd().parent)
    identity = build_experiment_identity(effective)
    if (identity["experiment_key"], identity["experiment_config_sha256"], experiment_output_root(effective)) != (row["experiment_key"], row["experiment_config_sha256"], row["expected_output_root"]): raise SystemExit("identity or output root drifted")
print("NODE4_R0F_PLAN_VALID: PASS")
PY

mkdir -p "${LOGS_ROOT}" "$(dirname "${COMMAND_HISTORY}")"
conda run --no-capture-output -n SHOT_TTA python "${PROJECT_ROOT}/tools/run_experiments_multi_gpu.py" \
  "${PLAN_PATH}" \
  --runs-root "${RUNS_ROOT}" \
  --logs-root "${LOGS_ROOT}" \
  --workdir "${PROJECT_ROOT}" \
  --gpus 0,1,2,3 \
  --max-workers 4 \
  --workers-per-gpu 1 \
  --resume \
  --resume-partial-runs \
  --command-history "${COMMAND_HISTORY}"
