#!/usr/bin/env bash
set -euo pipefail

# Complete only the 12 previously untested (alpha, nu) points from the
# constrained kappa=1 Transformer Group-LBI Stage-1 screen.  VisDA-C follows
# the explicitly frozen three-wave GPU order; Office-31 starts only after both
# VisDA-C settings have reached a terminal state.

PROJECT_DIR="/home/nas3/biod/wangkangyi/transformer-tta"
PYTHON="/home/nas3/biod/wangkangyi/envs/lbi/bin/python"
CONFIG="$PROJECT_DIR/transformer/group_lbi/config.yaml"
BASE_ROOT="/home/nas3/biod/wangkangyi/results/transformer_otta_group_lbi_tuning/stage1/kappa1_constrained_missing12"
STAMP="$(date -u +%Y%m%dT%H%M%S.%6NZ)"
LAUNCH_ROOT="$BASE_ROOT/launchers/launcher_${STAMP}"
DRY_RUN="${DRY_RUN:-0}"

export HF_HOME="/home/nas3/biod/wangkangyi/hf-cache"
export TORCH_HOME="/home/nas3/biod/wangkangyi/hf-cache/torch"
export PIP_CACHE_DIR="/home/nas3/biod/wangkangyi/pip-cache"
export CONDA_PKGS_DIRS="/home/nas3/biod/wangkangyi/conda-pkgs"
export TMPDIR="/home/nas3/biod/wangkangyi/tmp/transformer_group_lbi_kappa1_screen"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd "$PROJECT_DIR"
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 1; }
mkdir -p "$LAUNCH_ROOT/logs" "$LAUNCH_ROOT/queues" "$TMPDIR"

# Point order is the order requested by the user.  The six prior points
# (0.10,0.50), (0.15,0.50), (0.20,0.50), (0.10,0.25), (0.10,1.00), and
# (0.15,1.00) are deliberately absent.
POINT_IDS=(P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11 P12)
declare -A ALPHA NU
ALPHA[P01]=0.025; NU[P01]=0.25
ALPHA[P02]=0.025; NU[P02]=0.50
ALPHA[P03]=0.025; NU[P03]=1.00
ALPHA[P04]=0.050; NU[P04]=0.25
ALPHA[P05]=0.050; NU[P05]=0.50
ALPHA[P06]=0.050; NU[P06]=1.00
ALPHA[P07]=0.125; NU[P07]=0.25
ALPHA[P08]=0.125; NU[P08]=0.50
ALPHA[P09]=0.125; NU[P09]=1.00
ALPHA[P10]=0.150; NU[P10]=0.25
ALPHA[P11]=0.200; NU[P11]=0.25
ALPHA[P12]=0.200; NU[P12]=1.00

COMMON_ARGS=(
  --config "$CONFIG"
  --device cuda
  --lbi-kappa 1.0
  --lbi-omega 0.20
  --lbi-prox-lambda 1.0
  --lbi-tau-g 1e-4
  --lbi-stage1-max-steps 3000
  --lbi-stage2-lr 1e-5
  --no-progress
)

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_LABELS=()
declare -a OFFICE_QUEUE_PIDS=()

cleanup() {
  local pid
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${OFFICE_QUEUE_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

point_tag() {
  local point="$1"
  printf '%s_alpha-%s_nu-%s' "$point" "${ALPHA[$point]}" "${NU[$point]}"
}

terminal_summary_exists() {
  local summary="$1"
  [[ -f "$summary" ]] &&
    grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "$summary"
}

launch_task() {
  local gpu="$1" phase="$2" point="$3" dataset="$4"
  local source="$5" target="$6" rho="$7"
  local point_name output_dir log_path label
  local -a command

  point_name="$(point_tag "$point")"
  output_dir="$BASE_ROOT/$point_name/rho-$rho/$dataset/${source}-${target}"
  label="${phase}_${point}_rho-${rho}_${dataset}_${source}-${target}"
  log_path="$LAUNCH_ROOT/logs/${label}_gpu${gpu}.log"

  if terminal_summary_exists "$output_dir/summary.json"; then
    echo "[GPU $gpu][$phase] skip terminal: $label"
    return 0
  fi

  if [[ -f "$output_dir/.stream_checkpoint/state.pt" ]]; then
    command=(
      "$PYTHON" -m transformer.group_lbi transfer
      --resume-run-dir "$output_dir" --no-progress
    )
    echo "[GPU $gpu][$phase] resume: $label"
  else
    if [[ -e "$output_dir" ]]; then
      echo "Refusing non-resumable existing output: $output_dir" >&2
      return 1
    fi
    command=(
      "$PYTHON" -m transformer.group_lbi transfer
      "${COMMON_ARGS[@]}"
      --dataset "$dataset" --source "$source" --target "$target"
      --budget "$rho" --output-dir "$output_dir"
      --lbi-alpha "${ALPHA[$point]}" --lbi-nu "${NU[$point]}"
    )
    echo "[GPU $gpu][$phase] start: $label"
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${command[@]}"
    printf '\n'
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" > "$log_path" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LABELS+=("$label (GPU $gpu; log $log_path)")
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
  local point transfer
  {
    echo -e 'phase\tgpu\tpoint\talpha\tnu\tdataset\trho\tK\ttransfer'
    for gpu in {0..7}; do
      point="${POINT_IDS[$gpu]}"
      echo -e "visda002_wave1\t$gpu\t$point\t${ALPHA[$point]}\t${NU[$point]}\tvisda-c\t0.002\t13\ttrain-validation"
    done
    for gpu in {0..3}; do
      point="${POINT_IDS[$((gpu + 8))]}"
      echo -e "visda_mixed_wave2\t$gpu\t$point\t${ALPHA[$point]}\t${NU[$point]}\tvisda-c\t0.002\t13\ttrain-validation"
    done
    for gpu in {4..7}; do
      point="${POINT_IDS[$((gpu - 4))]}"
      echo -e "visda_mixed_wave2\t$gpu\t$point\t${ALPHA[$point]}\t${NU[$point]}\tvisda-c\t0.001\t6\ttrain-validation"
    done
    for gpu in {0..7}; do
      point="${POINT_IDS[$((gpu + 4))]}"
      echo -e "visda001_wave3\t$gpu\t$point\t${ALPHA[$point]}\t${NU[$point]}\tvisda-c\t0.001\t6\ttrain-validation"
    done
    local task_index=0 gpu source target
    for point in "${POINT_IDS[@]}"; do
      for transfer in amazon:dslr amazon:webcam dslr:amazon dslr:webcam webcam:amazon webcam:dslr; do
        gpu=$((task_index % 8))
        source="${transfer%%:*}"
        target="${transfer##*:}"
        echo -e "office\t$gpu\t$point\t${ALPHA[$point]}\t${NU[$point]}\toffice31\t0.002\t13\t${source}-${target}"
        task_index=$((task_index + 1))
      done
    done
  } > "$LAUNCH_ROOT/plan.tsv"
}

run_office_queue() {
  local gpu="$1" queue="$LAUNCH_ROOT/queues/office_gpu${gpu}.tsv"
  local point source target
  while IFS=$'\t' read -r point source target; do
    launch_task "$gpu" office "$point" office31 "$source" "$target" 0.002
    wait_for_wave "office_gpu${gpu}"
  done < "$queue"
}

write_plan

# Invariants catch accidental additions, omissions, or old-point duplication.
[[ "${#POINT_IDS[@]}" -eq 12 ]] || { echo "Expected exactly 12 points" >&2; exit 1; }
[[ "$(tail -n +2 "$LAUNCH_ROOT/plan.tsv" | wc -l)" -eq 96 ]] || {
  echo "Expected exactly 96 conditions" >&2
  exit 1
}
[[ "$(awk -F '\t' 'NR > 1 && $6 == "visda-c" && $7 == "0.002" {n++} END {print n+0}' "$LAUNCH_ROOT/plan.tsv")" -eq 12 ]]
[[ "$(awk -F '\t' 'NR > 1 && $6 == "visda-c" && $7 == "0.001" {n++} END {print n+0}' "$LAUNCH_ROOT/plan.tsv")" -eq 12 ]]
[[ "$(awk -F '\t' 'NR > 1 && $6 == "office31" && $7 == "0.002" {n++} END {print n+0}' "$LAUNCH_ROOT/plan.tsv")" -eq 72 ]]

for gpu in {0..7}; do
  : > "$LAUNCH_ROOT/queues/office_gpu${gpu}.tsv"
done
task_index=0
for point in "${POINT_IDS[@]}"; do
  for transfer in amazon:dslr amazon:webcam dslr:amazon dslr:webcam webcam:amazon webcam:dslr; do
    gpu=$((task_index % 8))
    printf '%s\t%s\t%s\n' "$point" "${transfer%%:*}" "${transfer##*:}" \
      >> "$LAUNCH_ROOT/queues/office_gpu${gpu}.tsv"
    task_index=$((task_index + 1))
  done
done

echo "Transformer kappa=1 constrained Stage-1 missing-point screen"
echo "Conditions: 96 (VisDA rho=.002: 12; VisDA rho=.001: 12; Office rho=.002: 72)"
echo "Results: $BASE_ROOT"
echo "Launch metadata/logs: $LAUNCH_ROOT"

# Wave 1: first eight rho=.002 VisDA-C points, one per GPU.
for gpu in {0..7}; do
  launch_task "$gpu" visda002_wave1 "${POINT_IDS[$gpu]}" visda-c train validation 0.002
done
wait_for_wave visda002_wave1

# Wave 2: remaining four rho=.002 points on GPU 0-3 and first four rho=.001
# points on GPU 4-7.
for gpu in {0..3}; do
  launch_task "$gpu" visda_mixed_wave2 "${POINT_IDS[$((gpu + 8))]}" visda-c train validation 0.002
done
for gpu in {4..7}; do
  launch_task "$gpu" visda_mixed_wave2 "${POINT_IDS[$((gpu - 4))]}" visda-c train validation 0.001
done
wait_for_wave visda_mixed_wave2

# Wave 3: remaining eight rho=.001 VisDA-C points, one per GPU.
for gpu in {0..7}; do
  launch_task "$gpu" visda001_wave3 "${POINT_IDS[$((gpu + 4))]}" visda-c train validation 0.001
done
wait_for_wave visda001_wave3

# Office starts only after every VisDA wave completes. Each GPU receives nine
# sequential jobs, maintaining one experiment process per card.
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

echo "All requested constrained Stage-1 conditions reached terminal state."
