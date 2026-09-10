"""Frozen configuration, budgets, and identity for COME Group-Magnitude."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from transformer.source_only.config import FORMAL_SEED, TRANSFERS

from transformer_come.budget import (
    BUDGET_TO_K,
    FORMAL_BUDGETS,
    GROUP_SIZE,
    TOTAL_GROUPS,
    budget_group_count,
    budget_key,
    budget_tag,
    normalize_budget,
    parse_budgets,
)
from transformer_come.common_config import (
    CANDIDATE_ADAPTATION,
    finalize_identity,
    load_config,
    resolve_sparse_transfer_config,
    select_transfers,
    validate_common_fields,
    validate_selection_block,
)
from transformer_come.identity import IMPLEMENTATION_REVISIONS, PROTOCOL_REVISIONS


VARIANT = "group_magnitude"
PROTOCOL_REVISION = PROTOCOL_REVISIONS[VARIANT]
IMPLEMENTATION_REVISION = IMPLEMENTATION_REVISIONS[VARIANT]

EXPECTED_SELECTION = {
    "type": "source_w0_structural_group_l2",
    "score": "paired_group_l2_norm",
    "score_device": "cpu",
    "score_dtype": "float64",
    "tie_break": "ascending_canonical_group_id",
    "group_order": "block_then_qk_vo_ffn_then_coordinate",
    "total_groups": TOTAL_GROUPS,
    "group_size": GROUP_SIZE,
    "budgets": list(FORMAL_BUDGETS),
    "integer_rule": "floor",
    "integer_budgets": [BUDGET_TO_K[item] for item in FORMAL_BUDGETS],
    "ranking_source": "source_checkpoint_pre_adaptation",
    "mask_refresh_policy": "once_before_adaptation",
}


def _validate_frozen_fields(config: dict[str, Any]) -> None:
    validate_common_fields(config, variant=VARIANT, adaptation=CANDIDATE_ADAPTATION)
    validate_selection_block(config, EXPECTED_SELECTION)


def resolve_transfer_config(
    raw_config: dict[str, Any],
    *,
    project_root: Path,
    dataset: str,
    source: str,
    target: str,
    budget: float | str,
    device: str,
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Resolve one formal Magnitude condition and verify all local assets."""

    _validate_frozen_fields(raw_config)
    resolved = resolve_sparse_transfer_config(
        raw_config,
        variant=VARIANT,
        project_root=project_root,
        dataset=dataset,
        source=source,
        target=target,
        budget=budget,
        device=device,
        output_dir=output_dir,
    )
    return finalize_identity(
        resolved, variant=VARIANT, tag=budget_tag(resolved["selection"]["requested_budget"])
    )


__all__ = [
    "BUDGET_TO_K",
    "EXPECTED_SELECTION",
    "FORMAL_BUDGETS",
    "FORMAL_SEED",
    "GROUP_SIZE",
    "IMPLEMENTATION_REVISION",
    "PROTOCOL_REVISION",
    "TOTAL_GROUPS",
    "TRANSFERS",
    "VARIANT",
    "_validate_frozen_fields",
    "budget_group_count",
    "budget_key",
    "budget_tag",
    "load_config",
    "normalize_budget",
    "parse_budgets",
    "resolve_transfer_config",
    "select_transfers",
]
