#!/usr/bin/env python3
"""Regression coverage for IST sparse-budget matrix aggregation."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer_ist.aggregate import aggregate_matrix  # noqa: E402


TRANSFERS = (
    "amazon->dslr",
    "amazon->webcam",
    "dslr->amazon",
    "dslr->webcam",
    "webcam->amazon",
    "webcam->dslr",
)


def _write_summary(
    root: Path,
    *,
    variant: str,
    transfer: str,
    budget=None,
    top_level_budget=False,
) -> None:
    path = (
        root
        / "results"
        / variant
        / ("dense" if budget is None else f"rho-{budget}")
        / transfer.replace("->", "-")
        / "summary.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "completed",
        "variant": variant,
        "dataset": "office31",
        "transfer": transfer,
        "selection": (
            None if budget is None else {"requested_budget": budget}
        ),
        "PU-Acc": 50.0,
        "FO-Acc": 60.0,
        "output_dir": str(path.parent),
    }
    if top_level_budget:
        summary["requested_budget"] = budget
    path.write_text(json.dumps(summary), encoding="utf-8")


def check_nested_sparse_budgets_remain_distinct() -> None:
    with tempfile.TemporaryDirectory(prefix="transformer_ist_aggregate_") as value:
        root = Path(value)
        for budget in (0.0005, 0.001, 0.002):
            for transfer in TRANSFERS:
                _write_summary(
                    root,
                    variant="group_magnitude",
                    transfer=transfer,
                    budget=budget,
                )
        for transfer in TRANSFERS:
            _write_summary(
                root,
                variant="full_dense",
                transfer=transfer,
            )
        aggregate = aggregate_matrix(root)

    office = aggregate["office31_equal_transfer_mean"]
    assert set(office) == {
        "full_dense",
        "group_magnitude@0.0005",
        "group_magnitude@0.001",
        "group_magnitude@0.002",
    }
    assert all(record["transfer_count"] == 6 for record in office.values())
    assert {row["budget"] for row in aggregate["rows"]} == {
        None,
        0.0005,
        0.001,
        0.002,
    }


def check_top_level_budget_schema_still_works() -> None:
    with tempfile.TemporaryDirectory(prefix="transformer_ist_aggregate_") as value:
        root = Path(value)
        for transfer in TRANSFERS:
            _write_summary(
                root,
                variant="group_random",
                transfer=transfer,
                budget=0.001,
                top_level_budget=True,
            )
        aggregate = aggregate_matrix(root)
    assert set(aggregate["office31_equal_transfer_mean"]) == {
        "group_random@0.001"
    }


def main() -> int:
    check_nested_sparse_budgets_remain_distinct()
    check_top_level_budget_schema_still_works()
    print("Transformer IST aggregate tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
