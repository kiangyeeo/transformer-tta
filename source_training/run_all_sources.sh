#!/usr/bin/env bash
set -euo pipefail

# Train all four source domains concurrently. Each job sees two GPUs and the
# trainer uses torch.nn.DataParallel without changing the configured global
# batch size or learning rates.

PROJECT_ROOT=/home/nas3/biod/wangkangyi/transformer-tta
PYTHON_BIN=/home/nas3/biod/wangkangyi/envs/lbi/bin/python
LOG_ROOT=/home/nas3/biod/wangkangyi/results/transformer_source_training_logs
TMP_ROOT=/home/nas3/biod/wangkangyi/tmp/transformer_source_training
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
RUN_LOG_DIR="$LOG_ROOT/$RUN_ID"
RUN_TMP_DIR="$TMP_ROOT/$RUN_ID"

export HF_HOME=/home/nas3/biod/wangkangyi/hf-cache
export TORCH_HOME=/home/nas3/biod/wangkangyi/hf-cache/torch
export PIP_CACHE_DIR=/home/nas3/biod/wangkangyi/pip-cache
export CONDA_PKGS_DIRS=/home/nas3/biod/wangkangyi/conda-pkgs
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4

cd "$PROJECT_ROOT"
mkdir -p "$RUN_LOG_DIR" "$RUN_TMP_DIR"

GPU_COUNT=$(
  "$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())'
)
if (( GPU_COUNT < 8 )); then
  echo "Expected at least 8 visible GPUs, found $GPU_COUNT" >&2
  exit 1
fi

declare -a JOB_NAMES=()
declare -a JOB_PIDS=()

launch_job() {
  local job_name=$1
  local gpu_pair=$2
  shift 2
  local job_tmp="$RUN_TMP_DIR/$job_name"
  local job_log="$RUN_LOG_DIR/$job_name.log"
  mkdir -p "$job_tmp"
  (
    export CUDA_VISIBLE_DEVICES="$gpu_pair"
    export TMPDIR="$job_tmp"
    "$PYTHON_BIN" train_source_deit.py "$@"
  ) >"$job_log" 2>&1 &
  JOB_NAMES+=("$job_name")
  JOB_PIDS+=("$!")
  echo "started $job_name on GPUs $gpu_pair: pid=$!, log=$job_log"
}

launch_job amazon 0,1 \
  --config configs/source_deit_office31.yaml --source-domain amazon
launch_job dslr 2,3 \
  --config configs/source_deit_office31.yaml --source-domain dslr
launch_job webcam 4,5 \
  --config configs/source_deit_office31.yaml --source-domain webcam
launch_job visda_train 6,7 \
  --config configs/source_deit_visda.yaml

echo "All jobs launched. Follow logs with: tail -F $RUN_LOG_DIR/*.log"

exit_status=0
for index in "${!JOB_PIDS[@]}"; do
  if wait "${JOB_PIDS[$index]}"; then
    echo "completed ${JOB_NAMES[$index]}"
  else
    child_status=$?
    echo "FAILED ${JOB_NAMES[$index]} (exit=$child_status)" >&2
    exit_status=1
  fi
done

echo "logs: $RUN_LOG_DIR"
exit "$exit_status"
