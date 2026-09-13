#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/nas3/biod/wangkangyi/transformer-tta
PYTHON=/home/nas3/biod/wangkangyi/envs/lbi/bin/python
PILOT_TMP=/home/nas3/biod/wangkangyi/tmp/transformer_come_mitigation

cd "$PROJECT_ROOT"
mkdir -p "$PILOT_TMP"
export TMPDIR="$PILOT_TMP"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID

exec "$PYTHON" -m transformer_come_mitigation matrix \
  --devices 0,1,2,3,4,5,6,7 \
  --output-root /home/nas3/biod/wangkangyi/results/transformer_come_mitigation
