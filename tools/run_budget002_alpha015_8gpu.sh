#!/usr/bin/env bash
set -euo pipefail

# Run the seven scientifically distinct rho=0.002 Group-LBI conditions.
# VisDA-C is launched first on GPU 0; the six Office-31 transfers then occupy
# six of the remaining seven GPUs.  A single causal OTTA stream cannot be
# split across GPUs, so seven conditions imply at most seven useful workers.

PROJECT_DIR="/home/nas3/biod/wangkangyi/transformer-tta"
PYTHON="/home/nas3/biod/wangkangyi/envs/lbi/bin/python"
CONFIG="$PROJECT_DIR/transformer/group_lbi/config.yaml"
RESULT_ROOT="${RESULT_ROOT:-/home/nas3/biod/wangkangyi/results/transformer_otta_budget002_alpha015}"
RUN_TAG="${RUN_TAG:-seed2026_$(date -u +%Y%m%dT%H%M%S.%NZ)}"
RUN_ROOT="$RESULT_ROOT/$RUN_TAG"
DRY_RUN="${DRY_RUN:-0}"

# Override these arrays before editing the script if the visible card order is
# different. GPU 0 is deliberately reserved for the longer VisDA-C run.
VISDA_GPU="${VISDA_GPU:-0}"
OFFICE_GPUS=(1 2 3 4 5 6)
SPARE_GPU="${SPARE_GPU:-7}"

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
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results" "$TMPDIR"

COMMON_ARGS=(
  --config "$CONFIG"
  --budget 0.002
  --device cuda
  --lbi-alpha 0.15
  --lbi-kappa 1
  --lbi-nu 0.5
  --lbi-prox-lambda 1.0
  --lbi-tau-g 1e-4
  --lbi-stage1-max-steps 3000
)

declare -a PIDS=()
declare -a LABELS=()

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

launch_condition() {
  local gpu="$1" dataset="$2" source="$3" target="$4" omega="$5" stage2_lr="$6"
  local label="${dataset}_${source}-${target}"
  local output_dir="$RUN_ROOT/results/rho-0.002/$dataset/${source}-${target}"
  local log_path="$RUN_ROOT/logs/${label}_gpu${gpu}.log"
  local -a command

  if [[ -f "$output_dir/summary.json" ]] &&
     grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "$output_dir/summary.json"; then
    echo "[GPU $gpu] skip completed: $label"
    return 0
  fi

  if [[ -f "$output_dir/.stream_checkpoint/state.pt" ]]; then
    command=("$PYTHON" -m transformer.group_lbi transfer --resume-run-dir "$output_dir")
    echo "[GPU $gpu] resume: $label"
  else
    if [[ -e "$output_dir" ]]; then
      echo "Refusing non-resumable existing output: $output_dir" >&2
      return 1
    fi
    command=(
      "$PYTHON" -m transformer.group_lbi transfer
      "${COMMON_ARGS[@]}"
      --dataset "$dataset" --source "$source" --target "$target"
      --output-dir "$output_dir"
      --lbi-omega "$omega" --lbi-stage2-lr "$stage2_lr"
    )
    echo "[GPU $gpu] start: $label (omega=$omega, stage2_lr=$stage2_lr)"
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${command[@]}"
    printf '\n'
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" > "$log_path" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$label (GPU $gpu; log $log_path)")
}

echo "Run root: $RUN_ROOT"
echo "VisDA-C priority GPU: $VISDA_GPU"
echo "Office GPUs: ${OFFICE_GPUS[*]}"
echo "Unused failover/spare GPU: $SPARE_GPU"

# Priority launch: the longest condition receives its card before Office jobs.
launch_condition "$VISDA_GPU" visda-c train validation 0.20 0.005

OFFICE_TRANSFERS=(
  amazon:dslr amazon:webcam dslr:amazon
  dslr:webcam webcam:amazon webcam:dslr
)
for index in "${!OFFICE_TRANSFERS[@]}"; do
  transfer="${OFFICE_TRANSFERS[$index]}"
  launch_condition "${OFFICE_GPUS[$index]}" office31 \
    "${transfer%%:*}" "${transfer##*:}" 0.05 0.010
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete: 7 commands validated; no training started."
  exit 0
fi

status=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "[done] ${LABELS[$index]}"
  else
    echo "[failed] ${LABELS[$index]}" >&2
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "At least one condition failed. Re-run with the same RUN_TAG to resume." >&2
  exit "$status"
fi

"$PYTHON" -m transformer.group_lbi summarize \
  --run-root "$RUN_ROOT" --datasets all --budgets 0.002
echo "All seven budget=0.002 conditions completed: $RUN_ROOT"
