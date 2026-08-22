#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/nas3/biod/wangkangyi/transformer-tta
PYTHON_BIN=/home/nas3/biod/wangkangyi/envs/lbi/bin/python
CONFIG_PATH="$PROJECT_ROOT/transformer/candidate_dense/config.yaml"
OUTPUT_ROOT=/home/nas3/biod/wangkangyi/results/transformer_otta_candidate_dense
CANDIDATE_DENSE_GPUS=${CANDIDATE_DENSE_GPUS:-0,1,2,3,4,5,6}

export HF_HOME=/home/nas3/biod/wangkangyi/hf-cache
export TORCH_HOME=/home/nas3/biod/wangkangyi/hf-cache/torch
export PIP_CACHE_DIR=/home/nas3/biod/wangkangyi/pip-cache
export CONDA_PKGS_DIRS=/home/nas3/biod/wangkangyi/conda-pkgs
export TMPDIR=/home/nas3/biod/wangkangyi/tmp/transformer_candidate_dense
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4

mkdir -p "$OUTPUT_ROOT" "$TMPDIR"
cd "$PROJECT_ROOT"

exec "$PYTHON_BIN" -m transformer.candidate_dense matrix \
  --config "$CONFIG_PATH" \
  --datasets all \
  --devices "$CANDIDATE_DENSE_GPUS" \
  --output-root "$OUTPUT_ROOT"

