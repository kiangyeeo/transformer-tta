"""Aggregate COME full-dense transfers using the SHOT aggregation semantics."""

from __future__ import annotations

from pathlib import Path

from transformer.full_dense.aggregate import (
    aggregate_matrix as _aggregate_matrix,
    print_aggregate,
    write_aggregate,
)

from transformer_come.identity import IMPLEMENTATION_REVISIONS, PROTOCOL_REVISIONS

from .config import TRANSFERS


def aggregate_matrix(run_root: Path, transfers=TRANSFERS) -> dict:
    """Reuse the SHOT aggregation and refuse to mix in non-COME summaries."""

    result = _aggregate_matrix(run_root, transfers=transfers)
    for row in result["transfers"]:
        if row.get("method") != "come":
            raise ValueError(
                f"Summary is not a COME run: {row.get('output_dir')}"
            )
    result["method"] = "come"
    result["protocol_revision"] = PROTOCOL_REVISIONS["full_dense"]
    result["implementation_revision"] = IMPLEMENTATION_REVISIONS["full_dense"]
    return result


__all__ = ["aggregate_matrix", "print_aggregate", "write_aggregate"]
