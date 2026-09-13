#!/usr/bin/env bash
set -euo pipefail

# Run one GPU's original nine-job slice of the frozen Office-31 Stage-2 plan.

GPU="${1:?usage: $0 GPU_ID}"
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU_ID must be 0..7" >&2; exit 2; }

PROJECT_DIR="/home/nas3/biod/wangkangyi/transformer-tta"
PYTHON="/home/nas3/biod/wangkangyi/envs/lbi/bin/python"
CONFIG="$PROJECT_DIR/transformer/group_lbi/config.yaml"
STAGE2_ROOT="/home/nas3/biod/wangkangyi/results/transformer_otta_group_lbi_tuning/stage2"
RESULT_ROOT="$STAGE2_ROOT/runs/office31/budget_0.002/kappa1_alpha-0.200_nu-1.00"
LAUNCH_PARENT="$STAGE2_ROOT/launchers/kappa1_alpha-0.200_nu-1.00_office_dynamic"
STAMP="$(date -u +%Y%m%dT%H%M%S.%6NZ)"
LAUNCH_ROOT="$LAUNCH_PARENT/gpu${GPU}_${STAMP}"

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

QUEUE=()
task_index=0
for combo in "${COMBO_IDS[@]}"; do
  for transfer in "${TRANSFERS[@]}"; do
    if (( task_index % 8 == GPU )); then
      QUEUE+=("$combo|${transfer%%:*}|${transfer##*:}")
    fi
    task_index=$((task_index + 1))
  done
done
[[ "${#QUEUE[@]}" -eq 9 ]]

terminal_summary_exists() {
  local summary="$1"
  [[ -f "$summary" ]] &&
    grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "$summary"
}

cd "$PROJECT_DIR"
[[ -x "$PYTHON" ]]
[[ -f "$CONFIG" ]]
mkdir -p "$LAUNCH_ROOT/logs" "$LAUNCH_PARENT" "$TMPDIR"
ln -sfn "$LAUNCH_ROOT" "$LAUNCH_PARENT/latest"
exec > >(tee -a "$LAUNCH_ROOT/queue.log") 2>&1

gpu_pids="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${gpu_pids//[[:space:]]/}" ]]; then
  echo "GPU${GPU} is not idle; refusing to start. PIDs: ${gpu_pids//$'\n'/,}" >&2
  exit 1
fi

echo "Office-31 Stage-2 GPU${GPU} serial queue"
echo "Fixed: rho=.002 alpha=.200 kappa=1.0 nu=1.00 stage2_steps=1 seed=2026"
echo "Queued transfer jobs: ${#QUEUE[@]}"

for entry in "${QUEUE[@]}"; do
  IFS='|' read -r combo source target <<< "$entry"
  omega="${OMEGA[$combo]}"
  stage2_lr="${STAGE2_LR[$combo]}"
  output_dir="$RESULT_ROOT/omega_${omega}_lr_${stage2_lr}/${source}-${target}"
  summary="$output_dir/summary.json"
  log="$LAUNCH_ROOT/logs/${combo}_${source}-${target}.log"

  if terminal_summary_exists "$summary"; then
    echo "[GPU${GPU}] skip completed: $combo $source->$target"
    continue
  fi
  if [[ -e "$output_dir" ]]; then
    echo "[GPU${GPU}] refusing non-terminal existing output: $output_dir" >&2
    exit 1
  fi

  echo "[GPU${GPU}] start: $combo omega=$omega LR2=$stage2_lr $source->$target"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m transformer.group_lbi transfer \
    --config "$CONFIG" --device cuda \
    --dataset office31 --source "$source" --target "$target" \
    --budget 0.002 --output-dir "$output_dir" \
    --lbi-alpha 0.200 --lbi-kappa 1.0 --lbi-nu 1.00 \
    --lbi-omega "$omega" --lbi-prox-lambda 1.0 --lbi-tau-g 1e-4 \
    --lbi-stage1-max-steps 3000 --lbi-stage2-lr "$stage2_lr" \
    --no-progress > "$log" 2>&1

  if ! terminal_summary_exists "$summary"; then
    echo "[GPU${GPU}] task exited without completed summary: $combo $source->$target" >&2
    exit 1
  fi
  echo "[GPU${GPU}] completed: $combo $source->$target"
done

echo "GPU${GPU} Office queue completed all nine assigned jobs."
