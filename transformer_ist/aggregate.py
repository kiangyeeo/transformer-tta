"""Aggregate completed IST-Transformer conditions without changing science."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


def _read(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if value.get("status") != "completed":
        raise RuntimeError(f"incomplete IST result: {path}")
    return value


def _requested_budget(summary: dict):
    """Read both current and already-emitted sparse-summary schemas."""
    top_level = summary.get("requested_budget")
    selection = summary.get("selection")
    nested = (
        selection.get("requested_budget")
        if isinstance(selection, dict)
        else None
    )
    if top_level is not None and nested is not None and top_level != nested:
        raise RuntimeError(
            "IST summary has conflicting requested budgets: "
            f"top-level={top_level}, selection={nested}"
        )
    return top_level if top_level is not None else nested


def aggregate_matrix(run_root: Path) -> dict:
    summaries = [
        _read(path)
        for path in sorted((run_root / "results").glob("**/summary.json"))
        if "/mask_" not in str(path)
    ]
    rows = []
    for summary in summaries:
        rows.append(
            {
                "variant": summary["variant"],
                "dataset": summary["dataset"],
                "transfer": summary["transfer"],
                "budget": _requested_budget(summary),
                "PU-Acc": summary["PU-Acc"],
                "FO-Acc": summary["FO-Acc"],
                "summary_path": str(
                    Path(summary["output_dir"]) / "summary.json"
                ),
            }
        )
    office = {}
    for variant in sorted({row["variant"] for row in rows}):
        budgets = sorted(
            {
                row["budget"]
                for row in rows
                if row["variant"] == variant
            },
            key=lambda value: -1.0 if value is None else value,
        )
        for budget in budgets:
            selected = [
                row
                for row in rows
                if row["dataset"] == "office31"
                and row["variant"] == variant
                and row["budget"] == budget
            ]
            if selected:
                if len(selected) != 6:
                    raise RuntimeError(
                        "Office aggregate requires exactly six equal-weight transfers "
                        f"for {variant}@{budget}, got {len(selected)}"
                    )
                key = variant if budget is None else f"{variant}@{budget}"
                office[key] = {
                    "transfer_count": len(selected),
                    "PU-Acc": statistics.fmean(row["PU-Acc"] for row in selected),
                    "FO-Acc": statistics.fmean(row["FO-Acc"] for row in selected),
                }
    return {"rows": rows, "office31_equal_transfer_mean": office}


def write_aggregate(run_root: Path, aggregate: dict) -> None:
    with open(run_root / "aggregate.json", "w", encoding="utf-8") as file_obj:
        json.dump(aggregate, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")
    with open(
        run_root / "results.csv", "w", encoding="utf-8", newline=""
    ) as file_obj:
        fields = [
            "variant",
            "dataset",
            "transfer",
            "budget",
            "PU-Acc",
            "FO-Acc",
            "summary_path",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate["rows"])
