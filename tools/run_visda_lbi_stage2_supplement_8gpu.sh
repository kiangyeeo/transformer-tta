#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/nas3/biod/wangkangyi/transformer-tta"
PYTHON="/home/nas3/biod/wangkangyi/envs/lbi/bin/python"
CONFIG="$PROJECT_DIR/transformer/group_lbi/config.yaml"
STAGE2_ROOT="${STAGE2_ROOT:-/home/nas3/biod/wangkangyi/results/transformer_otta_group_lbi_tuning/stage2}"
DRY_RUN="${DRY_RUN:-0}"

export HF_HOME="/home/nas3/biod/wangkangyi/hf-cache"
export TORCH_HOME="/home/nas3/biod/wangkangyi/hf-cache/torch"
export PIP_CACHE_DIR="/home/nas3/biod/wangkangyi/pip-cache"
export CONDA_PKGS_DIRS="/home/nas3/biod/wangkangyi/conda-pkgs"
export TMPDIR="/home/nas3/biod/wangkangyi/tmp"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

cd "$PROJECT_DIR"
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 1; }

QUEUE_ROOT="$STAGE2_ROOT/queues_visda_supplement"
LOG_ROOT="$STAGE2_ROOT/logs_visda_supplement"
mkdir -p "$QUEUE_ROOT" "$LOG_ROOT" "$TMPDIR"
for gpu in {0..7}; do : > "$QUEUE_ROOT/gpu${gpu}.tsv"; done

declare -A ALPHA KAPPA NU
ALPHA[A3]=0.20; KAPPA[A3]=1.0; NU[A3]=0.50
ALPHA[A4]=0.10; KAPPA[A4]=1.5; NU[A4]=0.50
ALPHA[A5]=0.10; KAPPA[A5]=2.0; NU[A5]=0.50
ALPHA[A8]=0.15; KAPPA[A8]=1.0; NU[A8]=1.00

OMEGAS=(0.05 0.10 0.20 0.30)
STAGE2_LRS=(0.005 0.010 0.020)
job_count=0

enqueue_budget() {
  local budget="$1" anchors="$2" anchor omega lr gpu
  for anchor in $anchors; do
    for omega in "${OMEGAS[@]}"; do
      for lr in "${STAGE2_LRS[@]}"; do
        gpu=$((job_count % 8))
        printf '%s\t%s\t%s\t%s\n' "$budget" "$anchor" "$omega" "$lr" \
          >> "$QUEUE_ROOT/gpu${gpu}.tsv"
        job_count=$((job_count + 1))
      done
    done
  done
}

enqueue_budget 0.0005 "A3"
enqueue_budget 0.001  "A4"
enqueue_budget 0.002  "A5 A8 A4"

[[ "$job_count" -eq 60 ]] || { echo "Expected 60 jobs, got $job_count" >&2; exit 1; }
echo "Prepared $job_count VisDA-C Stage-2 supplement conditions under: $STAGE2_ROOT"
for gpu in {0..7}; do
  printf 'GPU %s: %s conditions\n' "$gpu" "$(wc -l < "$QUEUE_ROOT/gpu${gpu}.tsv")"
done
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: queue validated; no experiment was started."
  exit 0
fi

run_gpu_queue() {
  local gpu="$1" budget anchor omega lr output_dir log_path
  while IFS=$'\t' read -r budget anchor omega lr; do
    output_dir="$STAGE2_ROOT/runs/visda-c/budget_${budget}/$anchor/omega_${omega}_lr_${lr}/train-validation"
    log_path="$LOG_ROOT/visda-c_b${budget}_${anchor}_o${omega}_lr${lr}_train-validation.log"

    if [[ -f "$output_dir/summary.json" ]] && grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "$output_dir/summary.json"; then
      echo "[GPU $gpu] skip completed: $output_dir"
      continue
    fi
    if [[ -f "$output_dir/.stream_checkpoint/state.pt" ]]; then
      echo "[GPU $gpu] resume: $output_dir"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m transformer.group_lbi transfer \
        --resume-run-dir "$output_dir" > "$log_path" 2>&1
      continue
    fi
    if [[ -e "$output_dir" ]]; then
      echo "[GPU $gpu] refusing non-resumable existing output: $output_dir" >&2
      return 1
    fi

    echo "[GPU $gpu] start: VisDA-C b=$budget $anchor omega=$omega lr=$lr"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m transformer.group_lbi transfer \
      --config "$CONFIG" --dataset visda-c --source train --target validation \
      --budget "$budget" --device cuda --output-dir "$output_dir" \
      --lbi-alpha "${ALPHA[$anchor]}" --lbi-kappa "${KAPPA[$anchor]}" \
      --lbi-nu "${NU[$anchor]}" --lbi-omega "$omega" \
      --lbi-prox-lambda 1.0 --lbi-tau-g 1e-4 \
      --lbi-stage1-max-steps 3000 --lbi-stage2-lr "$lr" \
      > "$log_path" 2>&1
  done < "$QUEUE_ROOT/gpu${gpu}.tsv"
}

pids=()
cleanup() { local pid; for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM
for gpu in {0..7}; do run_gpu_queue "$gpu" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
