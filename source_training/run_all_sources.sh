#!/usr/bin/env bash
set -euo pipefail

# Sequentially train the four source checkpoints on one GPU.
# Usage: bash source_training/run_all_sources.sh [GPU_ID]

PROJECT_ROOT=/home/nas3/biod/wangkangyi/transformer-tta
PYTHON_BIN=/home/nas3/biod/wangkangyi/envs/lbi/bin/python
GPU_ID=${1:-0}

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME=/home/nas3/biod/wangkangyi/hf-cache
export TORCH_HOME=/home/nas3/biod/wangkangyi/hf-cache/torch
export PIP_CACHE_DIR=/home/nas3/biod/wangkangyi/pip-cache
export CONDA_PKGS_DIRS=/home/nas3/biod/wangkangyi/conda-pkgs
export TMPDIR=/home/nas3/biod/wangkangyi/tmp

cd "$PROJECT_ROOT"

for SOURCE_DOMAIN in amazon dslr webcam; do
  "$PYTHON_BIN" train_source_deit.py \
    --config configs/source_deit_office31.yaml \
    --source-domain "$SOURCE_DOMAIN"
done

"$PYTHON_BIN" train_source_deit.py \
  --config configs/source_deit_visda.yaml

