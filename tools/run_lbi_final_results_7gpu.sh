#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/nas3/biod/wangkangyi/transformer-tta"
PYTHON="/home/nas3/biod/wangkangyi/envs/lbi/bin/python"
CONFIG="$PROJECT_DIR/transformer/group_lbi/config.yaml"
FINAL_ROOT="${FINAL_ROOT:-/home/nas3/biod/wangkangyi/results/transformer_otta_final_result}"
RUN_TAG="${RUN_TAG:-final_seed2026_$(date -u +%Y%m%dT%H%M%S.%NZ)}"
RUN_ROOT="$FINAL_ROOT/$RUN_TAG"
DRY_RUN="${DRY_RUN:-0}"
GPUS=(0 1 2 3 4 5 7)

export HF_HOME="/home/nas3/biod/wangkangyi/hf-cache"
export TORCH_HOME="/home/nas3/biod/wangkangyi/hf-cache/torch"
export PIP_CACHE_DIR="/home/nas3/biod/wangkangyi/pip-cache"
export CONDA_PKGS_DIRS="/home/nas3/biod/wangkangyi/conda-pkgs"
export TMPDIR="/home/nas3/biod/wangkangyi/tmp"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 1; }

QUEUE_ROOT="$RUN_ROOT/queues"
LOG_ROOT="$RUN_ROOT/logs"
mkdir -p "$QUEUE_ROOT" "$LOG_ROOT" "$TMPDIR"
for gpu in "${GPUS[@]}"; do : > "$QUEUE_ROOT/gpu${gpu}.tsv"; done

# Columns: dataset, budget, anchor, alpha, kappa, nu, omega, stage2_lr,
#          source, target
enqueue() {
  local gpu="$1"
  shift
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" \
    >> "$QUEUE_ROOT/gpu${gpu}.tsv"
}

# Put the three long VisDA-C jobs on separate GPUs.  Budget 0.002 is kept by
# itself because its measured Stage-1 runtime dominates the complete matrix.
enqueue 0 visda-c 0.002  A4 0.10 1.5 0.50 0.20 0.005 train validation
enqueue 1 visda-c 0.001  A5 0.10 2.0 0.50 0.20 0.005 train validation
enqueue 2 visda-c 0.0005 A3 0.20 1.0 0.50 0.20 0.005 train validation

OFFICE_TRANSFERS=(
  amazon:dslr amazon:webcam dslr:amazon dslr:webcam webcam:amazon webcam:dslr
)
OFFICE_GPUS=(3 4 5 7)

enqueue_office_budget() {
  local budget="$1" anchor="$2" alpha="$3" kappa="$4" nu="$5"
  local omega="$6" lr="$7" i transfer source target gpu
  for i in "${!OFFICE_TRANSFERS[@]}"; do
    transfer="${OFFICE_TRANSFERS[$i]}"
    source="${transfer%%:*}"
    target="${transfer##*:}"
    # GPUs 0/1/2 are exclusively reserved for the three VisDA-C budgets.
    # Rotate Office jobs only across GPUs 3/4/5/7.
    gpu="${OFFICE_GPUS[$(((i + office_rotation) % 4))]}"
    enqueue "$gpu" office31 "$budget" "$anchor" "$alpha" "$kappa" "$nu" \
      "$omega" "$lr" "$source" "$target"
  done
  office_rotation=$(((office_rotation + 1) % 4))
}

office_rotation=0
enqueue_office_budget 0.0005 A7 0.10 1.0 1.00 0.10 0.005
enqueue_office_budget 0.001  A1 0.10 1.0 0.50 0.05 0.005
enqueue_office_budget 0.002  A4 0.10 1.5 0.50 0.05 0.010

job_count=0
for gpu in "${GPUS[@]}"; do
  count="$(wc -l < "$QUEUE_ROOT/gpu${gpu}.tsv")"
  job_count=$((job_count + count))
  printf 'GPU %s: %s conditions\n' "$gpu" "$count"
done
[[ "$job_count" -eq 21 ]] || { echo "Expected 21 jobs, got $job_count" >&2; exit 1; }

echo "Validated 21 final conditions."
echo "Run root: $RUN_ROOT"
echo "Progress logs: $LOG_ROOT"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: queues only; no experiment was started."
  exit 0
fi

run_gpu_queue() {
  local gpu="$1"
  local dataset budget anchor alpha kappa nu omega lr source target
  local output_dir log_path label
  while IFS=$'\t' read -r dataset budget anchor alpha kappa nu omega lr source target; do
    label="${dataset}_b${budget}_${anchor}_o${omega}_lr${lr}_${source}-${target}"
    output_dir="$RUN_ROOT/runs/$dataset/budget_${budget}/$anchor/omega_${omega}_lr_${lr}/${source}-${target}"
    log_path="$LOG_ROOT/${label}.log"

    if [[ -f "$output_dir/summary.json" ]] && \
       grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "$output_dir/summary.json"; then
      echo "[GPU $gpu] skip completed: $label"
      continue
    fi
    if [[ -f "$output_dir/.stream_checkpoint/state.pt" ]]; then
      echo "[GPU $gpu] resume: $label"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m transformer.group_lbi transfer \
        --resume-run-dir "$output_dir" > "$log_path" 2>&1
      echo "[GPU $gpu] completed: $label"
      continue
    fi
    if [[ -e "$output_dir" ]]; then
      echo "[GPU $gpu] refusing non-resumable existing output: $output_dir" >&2
      return 1
    fi

    echo "[GPU $gpu] start: $label"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m transformer.group_lbi transfer \
      --config "$CONFIG" --dataset "$dataset" --source "$source" --target "$target" \
      --budget "$budget" --device cuda --output-dir "$output_dir" \
      --lbi-alpha "$alpha" --lbi-kappa "$kappa" --lbi-nu "$nu" \
      --lbi-omega "$omega" --lbi-prox-lambda 1.0 --lbi-tau-g 1e-4 \
      --lbi-stage1-max-steps 3000 --lbi-stage2-lr "$lr" \
      > "$log_path" 2>&1
    echo "[GPU $gpu] completed: $label"
  done < "$QUEUE_ROOT/gpu${gpu}.tsv"
}

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup INT TERM

for gpu in "${GPUS[@]}"; do
  run_gpu_queue "$gpu" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -eq 0 ]]; then
  echo "All 21 final conditions completed: $RUN_ROOT"
else
  echo "At least one GPU queue failed. Inspect: $LOG_ROOT" >&2
fi
exit "$status"
