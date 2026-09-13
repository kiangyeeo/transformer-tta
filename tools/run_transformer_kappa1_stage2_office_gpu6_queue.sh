#!/usr/bin/env bash
set -euo pipefail

# Original GPU6 slice of the frozen Office-31 rho=.002 Stage-2 plan.
# Nine transfer-level jobs run serially so other GPU slices can be launched
# later without overlapping this queue.

PROJECT_DIR="/home/nas3/biod/wangkangyi/transformer-tta"
PYTHON="/home/nas3/biod/wangkangyi/envs/lbi/bin/python"
CONFIG="$PROJECT_DIR/transformer/group_lbi/config.yaml"
STAGE2_ROOT="/home/nas3/biod/wangkangyi/results/transformer_otta_group_lbi_tuning/stage2"
RESULT_ROOT="$STAGE2_ROOT/runs/office31/budget_0.002/kappa1_alpha-0.200_nu-1.00"
LAUNCH_PARENT="$STAGE2_ROOT/launchers/kappa1_alpha-0.200_nu-1.00_office_dynamic"
STAMP="$(date -u +%Y%m%dT%H%M%S.%6NZ)"
LAUNCH_ROOT="$LAUNCH_PARENT/gpu6_${STAMP}"
GPU=6

export HF_HOME="/home/nas3/biod/wangkangyi/hf-cache"
export TORCH_HOME="/home/nas3/biod/wangkangyi/hf-cache/torch"
export PIP_CACHE_DIR="/home/nas3/biod/wangkangyi/pip-cache"
export CONDA_PKGS_DIRS="/home/nas3/biod/wangkangyi/conda-pkgs"
export TMPDIR="/home/nas3/biod/wangkangyi/tmp/transformer_group_lbi_kappa1_stage2"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# combo|omega|stage2_lr|source|target
QUEUE=(
  'C02|0.05|0.010|amazon|dslr'
  'C03|0.05|0.020|dslr|amazon'
  'C04|0.10|0.005|webcam|amazon'
  'C06|0.10|0.020|amazon|dslr'
  'C07|0.20|0.005|dslr|amazon'
  'C08|0.20|0.010|webcam|amazon'
  'C10|0.30|0.005|amazon|dslr'
  'C11|0.30|0.010|dslr|amazon'
  'C12|0.30|0.020|webcam|amazon'
)

terminal_summary_exists() {
  local summary="$1"
  [[ -f "$summary" ]] &&
    grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "$summary"
}

cd "$PROJECT_DIR"
[[ -x "$PYTHON" ]]
[[ -f "$CONFIG" ]]
mkdir -p "$LAUNCH_ROOT/logs" "$LAUNCH_PARENT" "$TMPDIR"
ln -sfn "$LAUNCH_ROOT" "$LAUNCH_PARENT/latest_gpu6"
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
  IFS='|' read -r combo omega stage2_lr source target <<< "$entry"
  combo_dir="omega_${omega}_lr_${stage2_lr}"
  output_dir="$RESULT_ROOT/$combo_dir/${source}-${target}"
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
