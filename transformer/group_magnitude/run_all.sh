#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/nas3/biod/wangkangyi/transformer-tta
PYTHON_BIN=/home/nas3/biod/wangkangyi/envs/lbi/bin/python
CONFIG_PATH="$PROJECT_ROOT/transformer/group_magnitude/config.yaml"
OUTPUT_ROOT=/home/nas3/biod/wangkangyi/results/transformer_otta_group_magnitude
GROUP_MAGNITUDE_GPUS=${GROUP_MAGNITUDE_GPUS:-0,1,2,3,4,5,6,7}
GROUP_MAGNITUDE_RHOS=${GROUP_MAGNITUDE_RHOS:-all}

export HF_HOME=/home/nas3/biod/wangkangyi/hf-cache
export TORCH_HOME=/home/nas3/biod/wangkangyi/hf-cache/torch
export PIP_CACHE_DIR=/home/nas3/biod/wangkangyi/pip-cache
export CONDA_PKGS_DIRS=/home/nas3/biod/wangkangyi/conda-pkgs
export TMPDIR=/home/nas3/biod/wangkangyi/tmp/transformer_group_magnitude
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4

mkdir -p "$OUTPUT_ROOT" "$TMPDIR"
cd "$PROJECT_ROOT"

exec "$PYTHON_BIN" -m transformer.group_magnitude matrix \
  --config "$CONFIG_PATH" \
  --datasets all \
  --rhos "$GROUP_MAGNITUDE_RHOS" \
  --devices "$GROUP_MAGNITUDE_GPUS" \
  --output-root "$OUTPUT_ROOT"
