#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/nas3/biod/wangkangyi/transformer-tta"
PYTHON="/home/nas3/biod/wangkangyi/envs/lbi/bin/python"
CONFIG="$PROJECT_DIR/transformer/group_lbi/config.yaml"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SWEEP_ROOT="${SWEEP_ROOT:-/home/nas3/biod/wangkangyi/results/transformer_lbi_stage2_joint_${STAMP}}"
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

mkdir -p "$SWEEP_ROOT/queues" "$SWEEP_ROOT/logs" "$TMPDIR"
for gpu in {0..7}; do
  : > "$SWEEP_ROOT/queues/gpu${gpu}.tsv"
done

declare -A ALPHA KAPPA NU
ALPHA[A1]=0.10; KAPPA[A1]=1.0; NU[A1]=0.50
ALPHA[A2]=0.15; KAPPA[A2]=1.0; NU[A2]=0.50
ALPHA[A3]=0.20; KAPPA[A3]=1.0; NU[A3]=0.50
ALPHA[A4]=0.10; KAPPA[A4]=1.5; NU[A4]=0.50
ALPHA[A5]=0.10; KAPPA[A5]=2.0; NU[A5]=0.50
ALPHA[A6]=0.10; KAPPA[A6]=1.0; NU[A6]=0.25
ALPHA[A7]=0.10; KAPPA[A7]=1.0; NU[A7]=1.00
ALPHA[A8]=0.15; KAPPA[A8]=1.0; NU[A8]=1.00

OMEGAS=(0.05 0.10 0.20 0.30)
STAGE2_LRS=(0.005 0.010 0.020)
OFFICE_TRANSFERS=(amazon:dslr amazon:webcam dslr:amazon dslr:webcam webcam:amazon webcam:dslr)

job_count=0
enqueue() {
  local dataset="$1" budget="$2" anchor="$3" source="$4" target="$5" omega="$6" lr="$7"
  local gpu=$((job_count % 8))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$dataset" "$budget" "$anchor" "$source" "$target" "$omega" "$lr" \
    >> "$SWEEP_ROOT/queues/gpu${gpu}.tsv"
  job_count=$((job_count + 1))
}

add_office() {
  local budget="$1" anchors="$2" anchor transfer source target omega lr
  for anchor in $anchors; do
    for omega in "${OMEGAS[@]}"; do
      for lr in "${STAGE2_LRS[@]}"; do
        for transfer in "${OFFICE_TRANSFERS[@]}"; do
          source="${transfer%%:*}"; target="${transfer##*:}"
          enqueue office31 "$budget" "$anchor" "$source" "$target" "$omega" "$lr"
        done
      done
    done
  done
}

add_visda() {
  local budget="$1" anchors="$2" anchor omega lr
  for anchor in $anchors; do
    for omega in "${OMEGAS[@]}"; do
      for lr in "${STAGE2_LRS[@]}"; do
        enqueue visda-c "$budget" "$anchor" train validation "$omega" "$lr"
      done
    done
  done
}

add_office 0.0005 "A6 A1 A4"
add_office 0.001  "A6 A4 A8"
add_office 0.002  "A2 A3"
add_visda  0.0005 "A5 A6"
add_visda  0.001  "A5 A3"
add_visda  0.002  "A5"

if [[ "$job_count" -ne 636 ]]; then
  echo "Internal error: expected 636 jobs, generated $job_count" >&2
  exit 1
fi

echo "Prepared $job_count conditions under: $SWEEP_ROOT"
for gpu in {0..7}; do
  printf 'GPU %s: %s conditions\n' "$gpu" "$(wc -l < "$SWEEP_ROOT/queues/gpu${gpu}.tsv")"
done
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: queue validated; no tuning process was started."
  exit 0
fi

run_gpu_queue() {
  local gpu="$1" dataset budget anchor source target omega lr output_dir log_path
  while IFS=$'\t' read -r dataset budget anchor source target omega lr; do
    output_dir="$SWEEP_ROOT/runs/$dataset/budget_${budget}/$anchor/omega_${omega}_lr_${lr}/${source}-${target}"
    log_path="$SWEEP_ROOT/logs/${dataset}_b${budget}_${anchor}_o${omega}_lr${lr}_${source}-${target}.log"

    if [[ -f "$output_dir/summary.json" ]] && grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "$output_dir/summary.json"; then
      echo "[GPU $gpu] skip completed: $output_dir"
      continue
    fi

    if [[ -f "$output_dir/.stream_checkpoint/state.pt" ]]; then
      echo "[GPU $gpu] resume: $output_dir"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m transformer.group_lbi transfer \
        --resume-run-dir "$output_dir" --no-progress > "$log_path" 2>&1
      continue
    fi

    if [[ -e "$output_dir" ]]; then
      echo "[GPU $gpu] refusing non-resumable existing output: $output_dir" >&2
      return 1
    fi

    echo "[GPU $gpu] start: $dataset b=$budget $anchor omega=$omega lr=$lr $source->$target"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m transformer.group_lbi transfer \
      --config "$CONFIG" \
      --dataset "$dataset" --source "$source" --target "$target" \
      --budget "$budget" --device cuda --output-dir "$output_dir" \
      --lbi-alpha "${ALPHA[$anchor]}" \
      --lbi-kappa "${KAPPA[$anchor]}" \
      --lbi-nu "${NU[$anchor]}" \
      --lbi-omega "$omega" \
      --lbi-prox-lambda 1.0 \
      --lbi-tau-g 1e-4 \
      --lbi-stage1-max-steps 3000 \
      --lbi-stage2-lr "$lr" \
      --no-progress > "$log_path" 2>&1
  done < "$SWEEP_ROOT/queues/gpu${gpu}.tsv"
}

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for gpu in {0..7}; do
  run_gpu_queue "$gpu" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
