#!/usr/bin/env bash
set -euo pipefail

# Run only the 21 remaining VisDA-C Stage-2 conditions for the selected
# alpha=0.200, kappa=1.0, nu=1.00 anchor. C01-C03 at rho=.002 are already
# complete and are intentionally absent. This launcher never auto-resumes.

PROJECT_DIR="/home/nas3/biod/wangkangyi/transformer-tta"
PYTHON="/home/nas3/biod/wangkangyi/envs/lbi/bin/python"
CONFIG="$PROJECT_DIR/transformer/group_lbi/config.yaml"
STAGE2_ROOT="/home/nas3/biod/wangkangyi/results/transformer_otta_group_lbi_tuning/stage2"
ANCHOR_TAG="kappa1_alpha-0.200_nu-1.00"
RESULT_ROOT="$STAGE2_ROOT/runs/visda-c"
STAMP="$(date -u +%Y%m%dT%H%M%S.%6NZ)"
LAUNCH_ROOT="$STAGE2_ROOT/launchers/${ANCHOR_TAG}_visda_remaining21/launcher_${STAMP}"
DRY_RUN="${DRY_RUN:-0}"

export HF_HOME="/home/nas3/biod/wangkangyi/hf-cache"
export TORCH_HOME="/home/nas3/biod/wangkangyi/hf-cache/torch"
export PIP_CACHE_DIR="/home/nas3/biod/wangkangyi/pip-cache"
export CONDA_PKGS_DIRS="/home/nas3/biod/wangkangyi/conda-pkgs"
export TMPDIR="/home/nas3/biod/wangkangyi/tmp/transformer_group_lbi_kappa1_stage2"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd "$PROJECT_DIR"
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 1; }
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 1; }

"$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
checks = {
    "formal_seed": config.get("formal_seed") == 2026,
    "stage2_steps": config.get("stage2_optimization", {}).get("steps") == 1,
    "budget_0.001_K": config.get("selection", {}).get("integer_budgets", [None, None, None])[1] == 6,
    "budget_0.002_K": config.get("selection", {}).get("integer_budgets", [None, None, None])[2] == 13,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("Frozen config preflight failed: " + ", ".join(failed))
PY

mkdir -p "$LAUNCH_ROOT/logs" "$STAGE2_ROOT/launchers/${ANCHOR_TAG}_visda_remaining21" "$TMPDIR"
ln -sfn "$LAUNCH_ROOT" "$STAGE2_ROOT/launchers/${ANCHOR_TAG}_visda_remaining21/latest"
exec > >(tee -a "$LAUNCH_ROOT/launcher.log") 2>&1

COMBO_IDS=(C01 C02 C03 C04 C05 C06 C07 C08 C09 C10 C11 C12)
declare -A OMEGA STAGE2_LR
OMEGA[C01]=0.05; STAGE2_LR[C01]=0.005
OMEGA[C02]=0.05; STAGE2_LR[C02]=0.010
OMEGA[C03]=0.05; STAGE2_LR[C03]=0.020
OMEGA[C04]=0.10; STAGE2_LR[C04]=0.005
OMEGA[C05]=0.10; STAGE2_LR[C05]=0.010
OMEGA[C06]=0.10; STAGE2_LR[C06]=0.020
OMEGA[C07]=0.20; STAGE2_LR[C07]=0.005
OMEGA[C08]=0.20; STAGE2_LR[C08]=0.010
OMEGA[C09]=0.20; STAGE2_LR[C09]=0.020
OMEGA[C10]=0.30; STAGE2_LR[C10]=0.005
OMEGA[C11]=0.30; STAGE2_LR[C11]=0.010
OMEGA[C12]=0.30; STAGE2_LR[C12]=0.020

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_LABELS=()

cleanup() {
  local pid
  trap - INT TERM
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

terminal_summary_exists() {
  local summary="$1"
  [[ -f "$summary" ]] && grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "$summary"
}

output_dir_for() {
  local budget="$1" combo="$2"
  printf '%s/budget_%s/%s/omega_%s_lr_%s/train-validation' \
    "$RESULT_ROOT" "$budget" "$ANCHOR_TAG" "${OMEGA[$combo]}" "${STAGE2_LR[$combo]}"
}

launch_task() {
  local gpu="$1" phase="$2" budget="$3" combo="$4"
  local output_dir label log_path
  local -a command

  output_dir="$(output_dir_for "$budget" "$combo")"
  label="${phase}_${combo}_visda-c_b${budget}_train-validation"
  log_path="$LAUNCH_ROOT/logs/${label}_gpu${gpu}.log"

  if terminal_summary_exists "$output_dir/summary.json"; then
    echo "[GPU $gpu][$phase] skip terminal: $label"
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    echo "Refusing existing non-terminal output; auto-resume is disabled: $output_dir" >&2
    return 1
  fi

  command=(
    "$PYTHON" -m transformer.group_lbi transfer
    --config "$CONFIG" --device cuda
    --dataset visda-c --source train --target validation
    --budget "$budget" --output-dir "$output_dir"
    --lbi-alpha 0.200 --lbi-kappa 1.0 --lbi-nu 1.00
    --lbi-omega "${OMEGA[$combo]}"
    --lbi-prox-lambda 1.0 --lbi-tau-g 1e-4
    --lbi-stage1-max-steps 3000
    --lbi-stage2-lr "${STAGE2_LR[$combo]}"
    --no-progress
  )

  echo "[GPU $gpu][$phase] start: $label omega=${OMEGA[$combo]} LR2=${STAGE2_LR[$combo]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${command[@]}"
    printf '\n'
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" > "$log_path" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LABELS+=("$label (GPU $gpu; log $log_path)")
  printf '%s\t%s\t%s\t%s\n' "$!" "$gpu" "$label" "$output_dir" >> "$LAUNCH_ROOT/pids.tsv"
}

wait_for_wave() {
  local phase="$1" index status=0
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[$phase] dry-run wave validated"
    ACTIVE_PIDS=()
    ACTIVE_LABELS=()
    return 0
  fi
  for index in "${!ACTIVE_PIDS[@]}"; do
    if wait "${ACTIVE_PIDS[$index]}"; then
      echo "[$phase] done: ${ACTIVE_LABELS[$index]}"
    else
      echo "[$phase] failed: ${ACTIVE_LABELS[$index]}" >&2
      status=1
    fi
  done
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
  [[ "$status" -eq 0 ]] || {
    echo "[$phase] failure detected; later waves were not launched" >&2
    return "$status"
  }
  echo "[$phase] wave completed"
}

write_plan() {
  local gpu combo
  {
    echo -e 'phase\tgpu\tcombo\talpha\tkappa\tnu\tomega\tstage2_lr\tdataset\trho\tK\ttransfer'
    for gpu in {0..7}; do
      combo="${COMBO_IDS[$((gpu + 3))]}"
      echo -e "visda002_wave1\t$gpu\t$combo\t0.200\t1.0\t1.00\t${OMEGA[$combo]}\t${STAGE2_LR[$combo]}\tvisda-c\t0.002\t13\ttrain-validation"
    done
    combo=C12
    echo -e "visda_mixed_wave2\t0\t$combo\t0.200\t1.0\t1.00\t${OMEGA[$combo]}\t${STAGE2_LR[$combo]}\tvisda-c\t0.002\t13\ttrain-validation"
    for gpu in {1..7}; do
      combo="${COMBO_IDS[$((gpu - 1))]}"
      echo -e "visda_mixed_wave2\t$gpu\t$combo\t0.200\t1.0\t1.00\t${OMEGA[$combo]}\t${STAGE2_LR[$combo]}\tvisda-c\t0.001\t6\ttrain-validation"
    done
    for gpu in {1..5}; do
      combo="${COMBO_IDS[$((gpu + 6))]}"
      echo -e "visda001_wave3\t$gpu\t$combo\t0.200\t1.0\t1.00\t${OMEGA[$combo]}\t${STAGE2_LR[$combo]}\tvisda-c\t0.001\t6\ttrain-validation"
    done
  } > "$LAUNCH_ROOT/plan.tsv"
}

write_plan
printf 'pid\tgpu\tlabel\toutput_dir\n' > "$LAUNCH_ROOT/pids.tsv"
[[ "$(tail -n +2 "$LAUNCH_ROOT/plan.tsv" | wc -l)" -eq 21 ]] || { echo "Expected 21 tasks" >&2; exit 1; }
[[ "$(awk -F '\t' 'NR>1 && $10=="0.002" {n++} END {print n+0}' "$LAUNCH_ROOT/plan.tsv")" -eq 9 ]]
[[ "$(awk -F '\t' 'NR>1 && $10=="0.001" {n++} END {print n+0}' "$LAUNCH_ROOT/plan.tsv")" -eq 12 ]]

if [[ "$DRY_RUN" == "0" ]]; then
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 1; }
  for gpu in {0..7}; do
    pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
    [[ -z "${pids//[[:space:]]/}" ]] || {
      echo "GPU $gpu is no longer idle; refusing launch. PIDs: ${pids//$'\n'/,}" >&2
      exit 1
    }
  done
fi

echo "VisDA-C selected-anchor Stage-2 remaining-21 launcher"
echo "Fixed: alpha=.200 kappa=1.0 nu=1.00 Stage-2 steps=1 seed=2026"
echo "Tasks: rho=.002 9; rho=.001 12; Office 0"
echo "Auto-resume: disabled"
echo "Plan/logs: $LAUNCH_ROOT"

# Wave 1: C04-C11 at rho=.002 on GPU 0-7.
for gpu in {0..7}; do
  launch_task "$gpu" visda002_wave1 0.002 "${COMBO_IDS[$((gpu + 3))]}"
done
wait_for_wave visda002_wave1

# Wave 2: C12 rho=.002 on GPU0; C01-C07 rho=.001 on GPU1-7.
launch_task 0 visda_mixed_wave2 0.002 C12
for gpu in {1..7}; do
  launch_task "$gpu" visda_mixed_wave2 0.001 "${COMBO_IDS[$((gpu - 1))]}"
done
wait_for_wave visda_mixed_wave2

# Wave 3: C08-C12 rho=.001 on GPU1-5.
for gpu in {1..5}; do
  launch_task "$gpu" visda001_wave3 0.001 "${COMBO_IDS[$((gpu + 6))]}"
done
wait_for_wave visda001_wave3

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry-run complete: 21 commands validated; no training was started."
else
  echo "All 21 remaining VisDA-C Stage-2 conditions reached terminal state."
fi
