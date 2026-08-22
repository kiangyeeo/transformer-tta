#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/home/nas3/biod/wangkangyi/hf-cache
export TORCH_HOME=/home/nas3/biod/wangkangyi/hf-cache/torch
export PIP_CACHE_DIR=/home/nas3/biod/wangkangyi/pip-cache
export CONDA_PKGS_DIRS=/home/nas3/biod/wangkangyi/conda-pkgs
export TMPDIR=/home/nas3/biod/wangkangyi/tmp

FULL_DENSE_GPUS=${FULL_DENSE_GPUS:-0,1,2,3,4,5,6}

exec /home/nas3/biod/wangkangyi/envs/lbi/bin/python \
  -m transformer.full_dense matrix \
  --datasets all \
  --devices "${FULL_DENSE_GPUS}"
