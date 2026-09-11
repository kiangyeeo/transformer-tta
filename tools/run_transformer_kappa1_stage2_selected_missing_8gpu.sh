#!/usr/bin/env bash
set -euo pipefail

# Complete the missing Transformer Group-LBI Stage-2 search for the selected
# kappa=1, alpha=0.200, nu=1.00 anchor.  The already-complete Office-31
# alpha=0.150, nu=1.00 grid is intentionally absent.
#
# Scheduling is frozen to the requested order:
#   1. VisDA-C rho=.002 combinations C01-C08 on GPU 0-7.
#   2. VisDA-C rho=.002 C09-C12 on GPU 0-3, and rho=.001 C01-C04 on GPU 4-7.
#   3. VisDA-C rho=.001 C05-C12 on GPU 0-7.
#   4. Office-31 rho=.002 (12 combinations x 6 transfers) on eight queues.

PROJECT_DIR="/home/nas3/biod/wangkangyi/transformer-tta"
PYTHON="/home/nas3/biod/wangkangyi/envs/lbi/bin/python"
CONFIG="$PROJECT_DIR/transformer/group_lbi/config.yaml"
STAGE2_ROOT="${STAGE2_ROOT:-/home/nas3/biod/wangkangyi/results/transformer_otta_group_lbi_tuning/stage2}"
ANCHOR_TAG="kappa1_alpha-0.200_nu-1.00"
RESULT_ROOT="$STAGE2_ROOT/runs"
STAMP="$(date -u +%Y%m%dT%H%M%S.%6NZ)"
LAUNCH_ROOT="${LAUNCH_ROOT:-$STAGE2_ROOT/launchers/${ANCHOR_TAG}_missing96/launcher_${STAMP}}"
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
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || {
  echo "DRY_RUN must be 0 or 1" >&2
  exit 1
}

# Refuse to launch if another experiment has changed a frozen value that is
# supplied by the shared config rather than a command-line override.
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

mkdir -p "$LAUNCH_ROOT/logs" "$LAUNCH_ROOT/queues" "$TMPDIR"

# Keep a stable pointer for later progress checks.  It points only to launcher
# metadata; scientific outputs remain in the canonical Stage-2 run tree.
mkdir -p "$STAGE2_ROOT/launchers/${ANCHOR_TAG}_missing96"
ln -sfn "$LAUNCH_ROOT" "$STAGE2_ROOT/launchers/${ANCHOR_TAG}_missing96/latest"
exec > >(tee -a "$LAUNCH_ROOT/launcher.log") 2>&1

ALPHA="0.200"
KAPPA="1.0"
NU="1.00"

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

OFFICE_TRANSFERS=(
  amazon:dslr amazon:webcam dslr:amazon
  dslr:webcam webcam:amazon webcam:dslr
)

COMMON_ARGS=(
  --config "$CONFIG"
  --device cuda
  --lbi-alpha "$ALPHA"
  --lbi-kappa "$KAPPA"
  --lbi-nu "$NU"
  --lbi-prox-lambda 1.0
  --lbi-tau-g 1e-4
  --lbi-stage1-max-steps 3000
  --no-progress
)

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_LABELS=()
declare -a OFFICE_QUEUE_PIDS=()

cleanup() {
  local pid
  trap - INT TERM
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${OFFICE_QUEUE_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

terminal_summary_exists() {
  local summary="$1"
  [[ -f "$summary" ]] &&
    grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "$summary"
}

combo_dir() {
  local combo="$1"
  printf 'omega_%s_lr_%s' "${OMEGA[$combo]}" "${STAGE2_LR[$combo]}"
}

launch_task() {
  local gpu="$1" phase="$2" combo="$3" dataset="$4"
  local source="$5" target="$6" budget="$7"
  local output_dir label log_path action
  local -a command

  output_dir="$RESULT_ROOT/$dataset/budget_${budget}/$ANCHOR_TAG/$(combo_dir "$combo")/${source}-${target}"
  label="${phase}_${combo}_${dataset}_b${budget}_${source}-${target}"
  log_path="$LAUNCH_ROOT/logs/${label}_gpu${gpu}.log"

  if terminal_summary_exists "$output_dir/summary.json"; then
    echo "[GPU $gpu][$phase] skip terminal: $label"
    return 0
  fi

  if [[ -f "$output_dir/.stream_checkpoint/state.pt" ]]; then
    action="resume"
    command=(
      "$PYTHON" -m transformer.group_lbi transfer
      --resume-run-dir "$output_dir" --no-progress
    )
  else
    if [[ -e "$output_dir" ]]; then
      echo "Refusing non-resumable existing output: $output_dir" >&2
      return 1
    fi
    action="start"
    command=(
      "$PYTHON" -m transformer.group_lbi transfer
      "${COMMON_ARGS[@]}"
      --dataset "$dataset" --source "$source" --target "$target"
      --budget "$budget" --output-dir "$output_dir"
      --lbi-omega "${OMEGA[$combo]}"
      --lbi-stage2-lr "${STAGE2_LR[$combo]}"
    )
  fi

  echo "[GPU $gpu][$phase] $action: $label"
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
  if [[ "$status" -ne 0 ]]; then
    echo "[$phase] at least one task failed; later waves were not launched" >&2
    return "$status"
  fi
  echo "[$phase] wave completed"
}

write_plan() {
  local gpu combo transfer source target task_index
  {
    echo -e 'phase\tgpu\tcombo\talpha\tkappa\tnu\tomega\tstage2_lr\tdataset\trho\tK\ttransfer'

    for gpu in {0..7}; do
      combo="${COMBO_IDS[$gpu]}"
      echo -e "visda002_wave1\t$gpu\t$combo\t$ALPHA\t$KAPPA\t$NU\t${OMEGA[$combo]}\t${STAGE2_LR[$combo]}\tvisda-c\t0.002\t13\ttrain-validation"
    done
    for gpu in {0..3}; do
      combo="${COMBO_IDS[$((gpu + 8))]}"
      echo -e "visda_mixed_wave2\t$gpu\t$combo\t$ALPHA\t$KAPPA\t$NU\t${OMEGA[$combo]}\t${STAGE2_LR[$combo]}\tvisda-c\t0.002\t13\ttrain-validation"
    done
    for gpu in {4..7}; do
      combo="${COMBO_IDS[$((gpu - 4))]}"
      echo -e "visda_mixed_wave2\t$gpu\t$combo\t$ALPHA\t$KAPPA\t$NU\t${OMEGA[$combo]}\t${STAGE2_LR[$combo]}\tvisda-c\t0.001\t6\ttrain-validation"
    done
    for gpu in {0..7}; do
      combo="${COMBO_IDS[$((gpu + 4))]}"
      echo -e "visda001_wave3\t$gpu\t$combo\t$ALPHA\t$KAPPA\t$NU\t${OMEGA[$combo]}\t${STAGE2_LR[$combo]}\tvisda-c\t0.001\t6\ttrain-validation"
    done

    task_index=0
    for combo in "${COMBO_IDS[@]}"; do
      for transfer in "${OFFICE_TRANSFERS[@]}"; do
        gpu=$((task_index % 8))
        source="${transfer%%:*}"
        target="${transfer##*:}"
        echo -e "office\t$gpu\t$combo\t$ALPHA\t$KAPPA\t$NU\t${OMEGA[$combo]}\t${STAGE2_LR[$combo]}\toffice31\t0.002\t13\t${source}-${target}"
        task_index=$((task_index + 1))
      done
    done
  } > "$LAUNCH_ROOT/plan.tsv"
}

write_office_queues() {
  local gpu combo transfer task_index=0
  for gpu in {0..7}; do
    : > "$LAUNCH_ROOT/queues/office_gpu${gpu}.tsv"
  done
  for combo in "${COMBO_IDS[@]}"; do
    for transfer in "${OFFICE_TRANSFERS[@]}"; do
      gpu=$((task_index % 8))
      printf '%s\t%s\t%s\n' "$combo" "${transfer%%:*}" "${transfer##*:}" \
        >> "$LAUNCH_ROOT/queues/office_gpu${gpu}.tsv"
      task_index=$((task_index + 1))
    done
  done
}

run_office_queue() {
  local gpu="$1" queue="$LAUNCH_ROOT/queues/office_gpu${gpu}.tsv"
  local combo source target
  while IFS=$'\t' read -r combo source target; do
    launch_task "$gpu" office "$combo" office31 "$source" "$target" 0.002
    wait_for_wave "office_gpu${gpu}"
  done < "$queue"
}

write_plan
write_office_queues
printf 'pid\tgpu\tlabel\toutput_dir\n' > "$LAUNCH_ROOT/pids.tsv"

# Invariants catch accidental omissions, duplicate combinations, or accidental
# inclusion of the already-complete Office alpha=.150 anchor.
[[ "${#COMBO_IDS[@]}" -eq 12 ]] || { echo "Expected exactly 12 omega/LR2 combinations" >&2; exit 1; }
[[ "$(tail -n +2 "$LAUNCH_ROOT/plan.tsv" | wc -l)" -eq 96 ]] || {
  echo "Expected exactly 96 task-level conditions" >&2
  exit 1
}
[[ "$(awk -F '\t' 'NR > 1 && $9 == "visda-c" && $10 == "0.002" {n++} END {print n+0}' "$LAUNCH_ROOT/plan.tsv")" -eq 12 ]]
[[ "$(awk -F '\t' 'NR > 1 && $9 == "visda-c" && $10 == "0.001" {n++} END {print n+0}' "$LAUNCH_ROOT/plan.tsv")" -eq 12 ]]
[[ "$(awk -F '\t' 'NR > 1 && $9 == "office31" && $10 == "0.002" {n++} END {print n+0}' "$LAUNCH_ROOT/plan.tsv")" -eq 72 ]]
for gpu in {0..7}; do
  [[ "$(wc -l < "$LAUNCH_ROOT/queues/office_gpu${gpu}.tsv")" -eq 9 ]] || {
    echo "Expected nine Office jobs on GPU $gpu" >&2
    exit 1
  }
done

echo "Transformer kappa=1 selected-anchor Stage-2 missing-grid launcher"
echo "Fixed anchor: alpha=$ALPHA kappa=$KAPPA nu=$NU; Stage-2 steps=1; seed=2026"
echo "Grid: omega={0.05,0.10,0.20,0.30} x LR2={0.005,0.010,0.020}"
echo "Conditions: 96 (VisDA .002: 12; VisDA .001: 12; Office .002: 72)"
echo "Excluded as already complete: Office .002 alpha=.150 kappa=1 nu=1.00 (12 x 6 transfers)"
echo "Scientific results: $RESULT_ROOT"
echo "Launch plan/logs: $LAUNCH_ROOT"

# Wave 1: first eight rho=.002 VisDA-C combinations, one per GPU.
for gpu in {0..7}; do
  launch_task "$gpu" visda002_wave1 "${COMBO_IDS[$gpu]}" visda-c train validation 0.002
done
wait_for_wave visda002_wave1

# Wave 2: final four rho=.002 combinations on GPU 0-3 and first four
# rho=.001 combinations on GPU 4-7.
for gpu in {0..3}; do
  launch_task "$gpu" visda_mixed_wave2 "${COMBO_IDS[$((gpu + 8))]}" visda-c train validation 0.002
done
for gpu in {4..7}; do
  launch_task "$gpu" visda_mixed_wave2 "${COMBO_IDS[$((gpu - 4))]}" visda-c train validation 0.001
done
wait_for_wave visda_mixed_wave2

# Wave 3: final eight rho=.001 combinations, one per GPU.
for gpu in {0..7}; do
  launch_task "$gpu" visda001_wave3 "${COMBO_IDS[$((gpu + 4))]}" visda-c train validation 0.001
done
wait_for_wave visda001_wave3

# Office starts only after all three VisDA waves have completed.  Each GPU gets
# nine sequential transfer-level jobs.
if [[ "$DRY_RUN" == "1" ]]; then
  for gpu in {0..7}; do
    run_office_queue "$gpu"
  done
else
  for gpu in {0..7}; do
    run_office_queue "$gpu" > "$LAUNCH_ROOT/logs/office_queue_gpu${gpu}.log" 2>&1 &
    OFFICE_QUEUE_PIDS+=("$!")
  done
  office_status=0
  for pid in "${OFFICE_QUEUE_PIDS[@]}"; do
    wait "$pid" || office_status=1
  done
  OFFICE_QUEUE_PIDS=()
  [[ "$office_status" -eq 0 ]] || {
    echo "At least one Office GPU queue failed" >&2
    exit "$office_status"
  }
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry-run complete: all 96 commands and queues validated; no training was started."
else
  echo "All requested missing Stage-2 conditions reached terminal state."
fi
