#!/usr/bin/env bash
# Run all five formal COME-Transformer protocol-v2 baseline matrices.
# Group-LBI is separate because its six profiles are still provisional.
# Each child script defaults to GPUs 0,1,2 and writes a new timestamped root.
set -euo pipefail

SCRIPT_ROOT=/home/nas3/biod/wangkangyi/transformer-tta/transformer_come/scripts

for script in \
  run_full_dense.sh \
  run_candidate_dense.sh \
  run_group_random.sh \
  run_group_magnitude.sh \
  run_group_saliency.sh; do
  "$SCRIPT_ROOT/$script"
done
