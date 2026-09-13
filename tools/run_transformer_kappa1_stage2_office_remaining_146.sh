#!/usr/bin/env bash
set -euo pipefail

# Snapshot all currently missing Office-31 alpha=.200 Stage-2 transfer jobs,
# split them evenly across GPU 1/4/6, and run each per-GPU queue serially.

PROJECT_DIR="/home/nas3/biod/wangkangyi/transformer-tta"
PYTHON="/home/nas3/biod/wangkangyi/envs/lbi/bin/python"
CONFIG="$PROJECT_DIR/transformer/group_lbi/config.yaml"
STAGE2_ROOT="/home/nas3/biod/wangkangyi/results/transformer_otta_group_lbi_tuning/stage2"
RESULT_ROOT="$STAGE2_ROOT/runs/office31/budget_0.002/kappa1_alpha-0.200_nu-1.00"
LAUNCH_PARENT="$STAGE2_ROOT/launchers/kappa1_alpha-0.200_nu-1.00_office_dynamic"
STAMP="$(date -u +%Y%m%dT%H%M%S.%6NZ)"
LAUNCH_ROOT="$LAUNCH_PARENT/remaining146_${STAMP}"
GPUS=(1 4 6)

export HF_HOME="/home/nas3/biod/wangkangyi/hf-cache"
export TORCH_HOME="/home/nas3/biod/wangkangyi/hf-cache/torch"
export PIP_CACHE_DIR="/home/nas3/biod/wangkangyi/pip-cache"
export CONDA_PKGS_DIRS="/home/nas3/biod/wangkangyi/conda-pkgs"
export TMPDIR="/home/nas3/biod/wangkangyi/tmp/transformer_group_lbi_kappa1_stage2"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

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
TRANSFERS=(amazon:dslr amazon:webcam dslr:amazon dslr:webcam webcam:amazon webcam:dslr)

terminal_summary_exists() {
  local summary="$1"
  [[ -f "$summary" ]] &&
    grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "$summary"
}

cd "$PROJECT_DIR"
[[ -x "$PYTHON" ]]
[[ -f "$CONFIG" ]]
mkdir -p "$LAUNCH_ROOT/logs" "$LAUNCH_ROOT/queues" "$LAUNCH_PARENT" "$TMPDIR"
ln -sfn "$LAUNCH_ROOT" "$LAUNCH_PARENT/latest_remaining146"
exec > >(tee -a "$LAUNCH_ROOT/launcher.log") 2>&1

for gpu in "${GPUS[@]}"; do
  pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -n "${pids//[[:space:]]/}" ]]; then
    echo "GPU${gpu} is not idle; refusing launch. PIDs: ${pids//$'\n'/,}" >&2
    exit 1
  fi
  : > "$LAUNCH_ROOT/queues/gpu${gpu}.tsv"
done

missing=0
for combo in "${COMBO_IDS[@]}"; do
  for transfer in "${TRANSFERS[@]}"; do
    source="${transfer%%:*}"
    target="${transfer##*:}"
    output_dir="$RESULT_ROOT/omega_${OMEGA[$combo]}_lr_${STAGE2_LR[$combo]}/${source}-${target}"
    if terminal_summary_exists "$output_dir/summary.json"; then
      continue
    fi
    if [[ -e "$output_dir" ]]; then
      echo "Refusing non-terminal existing output: $output_dir" >&2
      exit 1
    fi
    gpu="${GPUS[$((missing % ${#GPUS[@]}))]}"
    printf '%s\t%s\t%s\n' "$combo" "$source" "$target" >> "$LAUNCH_ROOT/queues/gpu${gpu}.tsv"
    missing=$((missing + 1))
  done
done

echo "Office-31 remaining-job launcher on GPU 1/4/6"
echo "Fixed: rho=.002 alpha=.200 kappa=1.0 nu=1.00 stage2_steps=1 seed=2026"
echo "Missing jobs assigned: $missing"
for gpu in "${GPUS[@]}"; do
  echo "GPU${gpu} queue: $(wc -l < "$LAUNCH_ROOT/queues/gpu${gpu}.tsv") jobs"
done

run_queue() {
  local gpu="$1" queue="$LAUNCH_ROOT/queues/gpu${gpu}.tsv"
  local combo source target omega stage2_lr output_dir summary log
  while IFS=$'\t' read -r combo source target; do
    omega="${OMEGA[$combo]}"
    stage2_lr="${STAGE2_LR[$combo]}"
    output_dir="$RESULT_ROOT/omega_${omega}_lr_${stage2_lr}/${source}-${target}"
    summary="$output_dir/summary.json"
    log="$LAUNCH_ROOT/logs/GPU${gpu}_${combo}_${source}-${target}.log"

    if terminal_summary_exists "$summary"; then
      echo "[GPU${gpu}] skip completed: $combo $source->$target"
      continue
    fi
    if [[ -e "$output_dir" ]]; then
      echo "[GPU${gpu}] refusing newly existing non-terminal output: $output_dir" >&2
      return 1
    fi

    echo "[GPU${gpu}] start: $combo omega=$omega LR2=$stage2_lr $source->$target"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m transformer.group_lbi transfer \
      --config "$CONFIG" --device cuda \
      --dataset office31 --source "$source" --target "$target" \
      --budget 0.002 --output-dir "$output_dir" \
      --lbi-alpha 0.200 --lbi-kappa 1.0 --lbi-nu 1.00 \
      --lbi-omega "$omega" --lbi-prox-lambda 1.0 --lbi-tau-g 1e-4 \
      --lbi-stage1-max-steps 3000 --lbi-stage2-lr "$stage2_lr" \
      --no-progress > "$log" 2>&1

    if ! terminal_summary_exists "$summary"; then
      echo "[GPU${gpu}] task exited without completed summary: $combo $source->$target" >&2
      return 1
    fi
    echo "[GPU${gpu}] completed: $combo $source->$target"
  done < "$queue"
  echo "[GPU${gpu}] queue completed"
}

worker_pids=()
printf 'gpu\tworker_pid\tqueue\n' > "$LAUNCH_ROOT/workers.tsv"
for gpu in "${GPUS[@]}"; do
  run_queue "$gpu" > "$LAUNCH_ROOT/logs/GPU${gpu}_queue.log" 2>&1 &
  worker_pids+=("$!")
  printf '%s\t%s\t%s\n' "$gpu" "$!" "$LAUNCH_ROOT/queues/gpu${gpu}.tsv" >> "$LAUNCH_ROOT/workers.tsv"
done

status=0
for pid in "${worker_pids[@]}"; do
  wait "$pid" || status=1
done
[[ "$status" -eq 0 ]] || { echo "At least one Office queue failed" >&2; exit 1; }
echo "All $missing assigned Office jobs completed."
