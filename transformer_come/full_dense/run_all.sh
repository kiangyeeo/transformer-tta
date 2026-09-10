#!/usr/bin/env bash
# COME-OTTA full-dense (non-head unrestricted reference): all 7 formal transfers (6 Office-31 + 1 VisDA-C).
# One process per GPU; each condition is a complete one-pass target stream
# plus an independent full-target FO evaluation.
set -euo pipefail

source /home/nas3/biod/wangkangyi/transformer-tta/transformer_come/scripts/come_env.sh

VARIANT=full_dense
CONFIG_PATH="$PROJECT_ROOT/transformer_come/full_dense/config.yaml"
OUTPUT_ROOT=${COME_FULL_DENSE_OUTPUT_ROOT:-$RESULTS_ROOT/transformer_come_otta_full_dense}
GPUS=${COME_FULL_DENSE_GPUS:-0,1,2,3,4,5,6,7}
DATASETS=${COME_FULL_DENSE_DATASETS:-all}   # all | office31 | visda-c
DRY_RUN=${COME_DRY_RUN:-0}

export TMPDIR=/home/nas3/biod/wangkangyi/tmp/transformer_come_full_dense

come_preflight "$VARIANT" "$GPUS" "$CONFIG_PATH" "$OUTPUT_ROOT" "$DRY_RUN"
cd "$PROJECT_ROOT"

ARGS=(-m transformer_come.full_dense matrix
      --config "$CONFIG_PATH"
      --datasets "$DATASETS"
      --devices "$GPUS"
      --output-root "$OUTPUT_ROOT")
[[ "$DRY_RUN" == "1" ]] && ARGS+=(--dry-run)

exec "$PYTHON_BIN" "${ARGS[@]}"
