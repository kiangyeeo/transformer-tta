#!/usr/bin/env bash
# COME Group-LBI implementation/search launcher. Profiles are provisional;
# set COME_ALLOW_PROVISIONAL_LBI=1 explicitly. Do not use these artifacts as
# formal results until the six dataset-by-budget profiles are frozen.
set -euo pipefail

source /home/nas3/biod/wangkangyi/transformer-tta/transformer_come/scripts/come_env.sh

VARIANT=group_lbi
CONFIG_PATH="$PROJECT_ROOT/transformer_come/config.yaml"
OUTPUT_ROOT=${COME_GROUP_LBI_OUTPUT_ROOT:-$RESULTS_ROOT/transformer_come_otta_group_lbi}
GPUS=${COME_GROUP_LBI_GPUS:-0,1,2}
DATASETS=${COME_GROUP_LBI_DATASETS:-all}
RHOS=${COME_GROUP_LBI_RHOS:-all}
DRY_RUN=${COME_DRY_RUN:-0}

if [[ "${COME_ALLOW_PROVISIONAL_LBI:-0}" != "1" ]]; then
  echo "COME Group-LBI profiles are provisional." >&2
  echo "Set COME_ALLOW_PROVISIONAL_LBI=1 only for search/smoke execution." >&2
  exit 1
fi

export TMPDIR=/home/nas3/biod/wangkangyi/tmp/transformer_come_group_lbi

come_preflight "$VARIANT" "$GPUS" "$CONFIG_PATH" "$OUTPUT_ROOT" "$DRY_RUN" "$RHOS"
cd "$PROJECT_ROOT"

ARGS=(-m transformer_come matrix
      --config "$CONFIG_PATH"
      --variants "$VARIANT"
      --datasets "$DATASETS"
      --rhos "$RHOS"
      --devices "$GPUS"
      --output-root "$OUTPUT_ROOT"
      --allow-provisional-lbi)
[[ "$DRY_RUN" == "1" ]] && ARGS+=(--dry-run)

exec "$PYTHON_BIN" "${ARGS[@]}"
