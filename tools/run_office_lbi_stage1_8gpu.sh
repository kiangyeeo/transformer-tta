#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/nas3/biod/wangkangyi/transformer-tta"
PYTHON="/home/nas3/biod/wangkangyi/envs/lbi/bin/python"
CONFIG="$PROJECT_DIR/transformer/group_lbi/config.yaml"
BASE_ROOT="/home/nas3/biod/wangkangyi/results/transformer_otta_group_lbi_tuning/stage1/office31"
STAMP="$(date -u +%Y%m%dT%H%M%S.%6NZ)"
LAUNCH_ROOT="$BASE_ROOT/launcher_${STAMP}"

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

mkdir -p "$LAUNCH_ROOT/queues" "$LAUNCH_ROOT/logs" "$TMPDIR"
for gpu in {0..7}; do
  : > "$LAUNCH_ROOT/queues/gpu${gpu}.tsv"
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

ANCHORS=(A1 A2 A3 A4 A5 A6 A7 A8)
BUDGETS=(0.0005 0.001 0.002)
TRANSFERS=(
  "amazon:dslr" "amazon:webcam"
  "dslr:amazon" "dslr:webcam"
  "webcam:amazon" "webcam:dslr"
)

job_count=0
for budget in "${BUDGETS[@]}"; do
  for anchor in "${ANCHORS[@]}"; do
    run_root="$BASE_ROOT/rho-${budget}/${anchor}/group_lbi_seed2026_${STAMP}"
    for transfer in "${TRANSFERS[@]}"; do
      source="${transfer%%:*}"
      target="${transfer##*:}"
      gpu=$((job_count % 8))
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "$budget" "$anchor" "$source" "$target" "$run_root" \
        >> "$LAUNCH_ROOT/queues/gpu${gpu}.tsv"
      job_count=$((job_count + 1))
    done
  done
done

[[ "$job_count" -eq 144 ]] || {
  echo "Expected 144 jobs, generated $job_count" >&2
  exit 1
}

echo "Office Group-LBI Stage-1: $job_count conditions"
echo "Results: $BASE_ROOT"
echo "Launcher logs: $LAUNCH_ROOT/logs"
for gpu in {0..7}; do
  echo "GPU $gpu: $(wc -l < "$LAUNCH_ROOT/queues/gpu${gpu}.tsv") conditions"
done

run_gpu_queue() {
  local gpu="$1" queue total local_index=0
  local budget anchor source target run_root output_dir job_log
  queue="$LAUNCH_ROOT/queues/gpu${gpu}.tsv"
  total="$(wc -l < "$queue")"

  while IFS=$'\t' read -r budget anchor source target run_root; do
    local_index=$((local_index + 1))
    output_dir="$run_root/results/rho-${budget}/office31/${source}-${target}"
    job_log="$run_root/logs/rho-${budget}_office31_${source}-${target}.log"
    mkdir -p "$(dirname "$job_log")"

    echo "[GPU $gpu][$local_index/$total] $anchor rho=$budget $source->$target"

    if [[ -f "$output_dir/summary.json" ]] &&
       grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "$output_dir/summary.json"; then
      echo "[GPU $gpu] skip completed: $output_dir"
      continue
    fi

    if [[ -f "$output_dir/.stream_checkpoint/state.pt" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m transformer.group_lbi transfer \
        --resume-run-dir "$output_dir" 2>&1 | tee "$job_log"
      continue
    fi

    if [[ -e "$output_dir" ]]; then
      echo "Refusing non-resumable output: $output_dir" >&2
      return 1
    fi

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m transformer.group_lbi transfer \
      --config "$CONFIG" \
      --dataset office31 --source "$source" --target "$target" \
      --budget "$budget" --device cuda --output-dir "$output_dir" \
      --lbi-alpha "${ALPHA[$anchor]}" \
      --lbi-kappa "${KAPPA[$anchor]}" \
      --lbi-nu "${NU[$anchor]}" \
      --lbi-omega 0.20 \
      --lbi-prox-lambda 1.0 \
      --lbi-tau-g 1e-4 \
      --lbi-stage1-max-steps 3000 \
      --lbi-stage2-lr 1e-5 \
      2>&1 | tee "$job_log"
  done < "$queue"

  echo "[GPU $gpu] queue completed"
}

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for gpu in {0..7}; do
  run_gpu_queue "$gpu" > "$LAUNCH_ROOT/logs/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

echo "All GPU queues finished with status=$status"
echo "Results: $BASE_ROOT"
echo "Logs: $LAUNCH_ROOT/logs"
exit "$status"
