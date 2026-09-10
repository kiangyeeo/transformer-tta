"""Aggregate COME candidate-dense transfers using SHOT aggregation semantics."""

from __future__ import annotations

from pathlib import Path

from transformer.candidate_dense.aggregate import (
    aggregate_matrix as _aggregate_matrix,
    print_aggregate,
    write_aggregate,
)

from transformer_come.identity import IMPLEMENTATION_REVISIONS, PROTOCOL_REVISIONS

from .config import TRANSFERS


def aggregate_matrix(run_root: Path, transfers=TRANSFERS) -> dict:
    result = _aggregate_matrix(run_root, transfers=transfers)
    for row in result["transfers"]:
        if row.get("method") != "come":
            raise ValueError(f"Summary is not a COME run: {row.get('output_dir')}")
    result["method"] = "come"
    result["protocol_revision"] = PROTOCOL_REVISIONS["candidate_dense"]
    result["implementation_revision"] = IMPLEMENTATION_REVISIONS["candidate_dense"]
    return result


__all__ = ["aggregate_matrix", "print_aggregate", "write_aggregate"]
