#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined"
BRANCH_ROOT="${PROJECT_ROOT}/experiment_logs/office_conv_lbi_r0f_filter_reachability_seed2026_20260827"
PLAN_JSONL="${PROJECT_ROOT}/experiment_logs/conv_lbi_32gpu_burst_20260827/plans/node0_filter_r0f.jsonl"
RUNS_ROOT="${BRANCH_ROOT}/runs"
LOGS_ROOT="${BRANCH_ROOT}/launcher_logs/node0_filter_r0f"
COMMAND_HISTORY="${BRANCH_ROOT}/command_history/node0_filter_r0f_commands.sh"
LOCK_PATH="${BRANCH_ROOT}/locks/node0_filter_r0f.lock"

if [[ ! -f "${PLAN_JSONL}" ]]; then
  echo "ERROR: Node 0 R0F plan is missing: ${PLAN_JSONL}" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOCK_PATH}")"
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "ERROR: Node 0 R0F launcher lock already exists: ${LOCK_PATH}" >&2
  exit 1
fi
PLAN_JSON="$(mktemp /tmp/node0_filter_r0f_plan.XXXXXX.json)"
cleanup() {
  local status=$?
  rm -f "${PLAN_JSON}"
  rmdir "${LOCK_PATH}" 2>/dev/null || true
  trap - EXIT INT TERM HUP
  return "${status}"
}
trap cleanup EXIT INT TERM HUP

# Convert the immutable JSONL audit plan to the canonical runner schema and
# fail closed on all frozen scientific requirements before any GPU work.
conda run --no-capture-output -n SHOT_TTA python - "${PLAN_JSONL}" "${PLAN_JSON}" "${RUNS_ROOT}" <<'PY'
import copy
import json
import pathlib
import sys

project_root = pathlib.Path('/inspire/hdd/global_user/gaoyachen-253308310317/PJ/Split-LBI/260817_iclr2027-refined')
sys.path.insert(0, str(project_root))
from core.lbi.diagnostics import max_support_count
from experiment_identity import build_experiment_identity
from protocol_constants import CONV_IMPLEMENTATION_REVISION, CONV_PROTOCOL_REVISION, EFFICIENCY_PROTOCOL_REVISION, FORMAL_SEED, SOURCE_CHECKPOINT_REVISION
from shot_otta.artifacts import experiment_output_root
from shot_otta.config import DATASETS, load_yaml, resolve_effective_config

plan_jsonl, plan_json, runs_root = map(pathlib.Path, sys.argv[1:])
rows = [json.loads(line) for line in plan_jsonl.read_text(encoding='utf-8').splitlines() if line.strip()]
expected_anchors = {'F0': (0.25, 0.5), 'F1': (0.3, 0.5), 'F2': (0.4, 0.5), 'F3': (0.6, 0.5)}
protocols = [
    project_root / 'protocol/shot-otta_conv/OTTA_CONV_LBI_PROTOCOL_20260826_v1.md',
    project_root / 'protocol/shot-otta_conv/OTTA_CONV_BASELINE_FORMAL_20260827_v1.md',
    project_root / 'protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md',
    project_root / 'protocol/shot-otta_conv/OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md',
]
if not all(path.is_file() for path in protocols):
    raise SystemExit('all four frozen protocol files are required')
if DATASETS['office']['domains'] != ['amazon', 'dslr', 'webcam']:
    raise SystemExit('repository Office domain mapping drifted')
if len(rows) != 4 or {row.get('anchor') for row in rows} != set(expected_anchors):
    raise SystemExit('Node 0 must contain exactly F0-F3')
for field in ('experiment_key', 'experiment_config_sha256', 'expected_output_root'):
    if len({row.get(field) for row in rows}) != 4:
        raise SystemExit(f'Node 0 {field} values must be unique')
if any(row.get('requested_budget') == 0.005 or row.get('rho_G') == 0.005 for row in rows):
    raise SystemExit('prohibited .005 budget found')
base = load_yaml(project_root / 'configs/otta_conv_lbi_protocol_20260826_v1.yaml')
for row in rows:
    anchor = row['anchor']
    alpha, nu = expected_anchors[anchor]
    lbi = {'alpha': alpha, 'kappa': 1.0, 'nu': nu, 'omega': 0.00625, 'stage1_max_steps': 3000, 'budget_tolerance': 0.0001, 'stage2_lr': 0.005, 'stage2_steps': 1, 'delta_nonzero_tolerance': 1e-12, 'support_threshold': 1e-4}
    required = {'implementation_revision': CONV_IMPLEMENTATION_REVISION, 'protocol_revision': CONV_PROTOCOL_REVISION, 'search_protocol_revision': 'OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2', 'efficiency_protocol_revision': EFFICIENCY_PROTOCOL_REVISION, 'source_checkpoint_revision': SOURCE_CHECKPOINT_REVISION, 'method': 'shot', 'task': 'otta', 'phase': 'R0F', 'dataset': 'office', 'transfer': 'DA', 'source': 1, 'target': 0, 'source_name': 'dslr', 'target_name': 'amazon', 'seed': FORMAL_SEED, 'variant': 'conv_filter_lbi', 'group_mode': 'filter_connection', 'alpha': alpha, 'nu': nu, 'requested_budget': 0.0005, 'rho_G': 0.0005, 'total_group_count': 6553600, 'max_group_count': 3276, 'K_G': 3276, 'candidate_conv_scalar_count': 12845056, 'runtime_comparable': False}
    if any(row.get(key) != value for key, value in required.items()) or row.get('lbi') != lbi:
        raise SystemExit(f'frozen scientific setting drift for {anchor}')
    if 'out_channel' in json.dumps(row) or 'baseline' in json.dumps(row).lower() or 'visda' in json.dumps(row).lower():
        raise SystemExit(f'prohibited scope found for {anchor}')
    config = copy.deepcopy(base)
    config.update({'protocol_track': 'conv', 'method': 'shot', 'task': 'otta', 'formal_protocol': True, 'protocol_revision': CONV_PROTOCOL_REVISION, 'implementation_revision': CONV_IMPLEMENTATION_REVISION, 'search_protocol_revision': 'OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2', 'source_checkpoint_revision': SOURCE_CHECKPOINT_REVISION, 'seed': FORMAL_SEED, 'variant': 'conv_filter_lbi', 'requested_budget': 0.0005, 'group_mode': 'filter_connection', 'lbi': lbi, 'allow_unresolved_lbi': False})
    config['data'].update({'dataset': 'office', 'source': 1, 'target': 0})
    config['output'].update({'root': row['output_root'], 'save_model': False, 'run_name': None})
    config['device']['gpu_id'] = '0'
    effective = resolve_effective_config(config, str(project_root.parent))
    identity = build_experiment_identity(effective)
    if (row['experiment_key'], row['experiment_config_sha256'], row['expected_output_root']) != (identity['experiment_key'], identity['experiment_config_sha256'], experiment_output_root(effective)):
        raise SystemExit(f'identity or output root drift for {anchor}')
    try:
        pathlib.Path(row['expected_output_root']).resolve().relative_to(runs_root.resolve())
    except ValueError as error:
        raise SystemExit(f'output root escapes Node 0 runs root for {anchor}') from error
payload = {'plan_schema_version': 1, 'node': 'node0_filter_r0f', 'phase': 'R0F', 'protocol_revision': CONV_PROTOCOL_REVISION, 'search_protocol_revision': 'OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2', 'implementation_revision': CONV_IMPLEMENTATION_REVISION, 'efficiency_protocol_revision': EFFICIENCY_PROTOCOL_REVISION, 'source_checkpoint_revision': SOURCE_CHECKPOINT_REVISION, 'experiment_count': 4, 'runs_root': str(runs_root.resolve()), 'runtime_comparable': False, 'experiments': rows}
plan_json.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'valid': True, 'experiment_count': 4, 'unique_keys': 4, 'unique_scientific_sha256': 4, 'unique_output_roots': 4, 'runtime_comparable': False}))
PY

mkdir -p "${LOGS_ROOT}" "$(dirname "${COMMAND_HISTORY}")"
conda run --no-capture-output -n SHOT_TTA python "${PROJECT_ROOT}/tools/run_experiments_multi_gpu.py" \
  "${PLAN_JSON}" \
  --runs-root "${RUNS_ROOT}" \
  --logs-root "${LOGS_ROOT}" \
  --workdir "${PROJECT_ROOT}" \
  --gpus 0,1,2,3 \
  --max-workers 4 \
  --workers-per-gpu 1 \
  --resume \
  --resume-partial-runs \
  --command-history "${COMMAND_HISTORY}"
