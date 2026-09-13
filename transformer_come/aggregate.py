"""Aggregate completed COME conditions with the SHOT aggregation semantics.

Each variant reuses its matched SHOT aggregator - same run-root layout, same
equal-weight Office mean, same Random mean/population-std over the three mask
children - and this module only refuses to mix in non-COME summaries and
stamps the COME revisions onto the result.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from .config import (
    FORMAL_BUDGETS,
    IMPLEMENTATION_REVISIONS,
    PROTOCOL_REVISIONS,
    SPARSE_VARIANTS,
    GROUP_LBI,
    TRANSFERS,
    require_supported_variant,
)


def _shot_aggregate_module(variant: str):
    return importlib.import_module(f"transformer.{variant}.aggregate")


def aggregate_matrix(
    run_root: Path,
    *,
    variant: str,
    transfers=TRANSFERS,
    budgets=None,
) -> dict:
    """Aggregate one variant's run root and refuse non-COME summaries."""

    require_supported_variant(variant)
    module = _shot_aggregate_module(variant)
    if variant in SPARSE_VARIANTS:
        result = module.aggregate_matrix(
            run_root,
            transfers=transfers,
            budgets=FORMAL_BUDGETS if budgets is None else budgets,
        )
    else:
        if budgets is not None:
            raise ValueError("dense COME variants have no structural-group budgets")
        result = module.aggregate_matrix(run_root, transfers=transfers)
    for row in result["transfers"]:
        if row.get("method") != "come":
            raise ValueError(f"Summary is not a COME run: {row.get('output_dir')}")
    result["method"] = "come"
    result["protocol_revision"] = PROTOCOL_REVISIONS[variant]
    result["implementation_revision"] = IMPLEMENTATION_REVISIONS[variant]
    if variant == GROUP_LBI:
        result["formal_eligible"] = all(
            bool(row.get("lbi", {}).get("formal_eligible"))
            for row in result["transfers"]
        )
        result["result_scope"] = (
            "formal" if result["formal_eligible"] else "search_or_smoke_nonformal"
        )
    return result


def write_aggregate(run_root: Path, aggregate: dict, *, variant: str) -> None:
    require_supported_variant(variant)
    _shot_aggregate_module(variant).write_aggregate(run_root, aggregate)


def print_aggregate(aggregate: dict, *, variant: str) -> None:
    require_supported_variant(variant)
    _shot_aggregate_module(variant).print_aggregate(aggregate)


__all__ = ["aggregate_matrix", "print_aggregate", "write_aggregate"]
