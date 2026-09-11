#!/usr/bin/env bash
# COME-OTTA structural-group Random: 3 budgets x 7 formal transfers = 21 conditions, each running 3 real child
# trajectories (mask seeds 202600/202601/202602) = 63 child runs.
# rho in {0.0005, 0.001, 0.002} -> exact K_G in {3, 6, 13} of 6912 groups.
# One process per GPU; each condition is a complete one-pass target stream
# plus an independent full-target FO evaluation.
set -euo pipefail

source /home/nas3/biod/wangkangyi/transformer-tta/transformer_come/scripts/come_env.sh

VARIANT=group_random
CONFIG_PATH="$PROJECT_ROOT/transformer_come/config.yaml"
OUTPUT_ROOT=${COME_GROUP_RANDOM_OUTPUT_ROOT:-$RESULTS_ROOT/transformer_come_otta_group_random}
GPUS=${COME_GROUP_RANDOM_GPUS:-0,1,2,3,4,5,6,7}
DATASETS=${COME_GROUP_RANDOM_DATASETS:-all}   # all | office31 | visda-c
RHOS=${COME_GROUP_RANDOM_RHOS:-all}
DRY_RUN=${COME_DRY_RUN:-0}

export TMPDIR=/home/nas3/biod/wangkangyi/tmp/transformer_come_group_random

come_preflight "$VARIANT" "$GPUS" "$CONFIG_PATH" "$OUTPUT_ROOT" "$DRY_RUN" "$RHOS"
cd "$PROJECT_ROOT"

ARGS=(-m transformer_come matrix
      --config "$CONFIG_PATH"
      --variants "$VARIANT"
      --datasets "$DATASETS"
      --rhos "$RHOS"
      --devices "$GPUS"
      --output-root "$OUTPUT_ROOT")
[[ "$DRY_RUN" == "1" ]] && ARGS+=(--dry-run)

exec "$PYTHON_BIN" "${ARGS[@]}"
