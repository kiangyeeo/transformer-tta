"""Aggregate COME Group-Random budgets using SHOT aggregation semantics."""

from __future__ import annotations

from pathlib import Path

from transformer.group_random.aggregate import (
    aggregate_matrix as _aggregate_matrix,
    print_aggregate,
    write_aggregate,
)

from transformer_come.identity import IMPLEMENTATION_REVISIONS, PROTOCOL_REVISIONS

from .config import FORMAL_BUDGETS, TRANSFERS


def aggregate_matrix(run_root: Path, *, transfers=TRANSFERS, budgets=FORMAL_BUDGETS) -> dict:
    result = _aggregate_matrix(run_root, transfers=transfers, budgets=budgets)
    for row in result["transfers"]:
        if row.get("method") != "come":
            raise ValueError(f"Summary is not a COME run: {row.get('output_dir')}")
    result["method"] = "come"
    result["protocol_revision"] = PROTOCOL_REVISIONS["group_random"]
    result["implementation_revision"] = IMPLEMENTATION_REVISIONS["group_random"]
    return result


__all__ = ["aggregate_matrix", "print_aggregate", "write_aggregate"]
