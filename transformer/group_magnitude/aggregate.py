"""Aggregate Group-Magnitude budgets, transfers, and VisDA classes."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from .config import FORMAL_BUDGETS, TRANSFERS, budget_tag


def _load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        summary = json.load(file_obj)
    if summary.get("status") != "completed":
        raise ValueError(f"Magnitude condition is not completed: {path}")
    if summary.get("variant") != "group_magnitude":
        raise ValueError(f"Condition is not group-magnitude: {path}")
    return summary


def aggregate_matrix(run_root: Path, *, transfers=TRANSFERS, budgets=FORMAL_BUDGETS) -> dict:
    summaries = []
    for budget in budgets:
        for dataset, source, target in transfers:
            summaries.append(
                _load_summary(
                    run_root
                    / "results"
                    / budget_tag(budget)
                    / dataset
                    / f"{source}-{target}"
                    / "summary.json"
                )
            )
    result = {
        "status": "completed",
        "variant": "group_magnitude",
        "formal_seed": 2026,
        "condition_count": len(summaries),
        "transfers": summaries,
        "budgets": {},
    }
    for budget in budgets:
        rows = [row for row in summaries if row["requested_budget"] == budget]
        office = [row for row in rows if row["dataset"] == "office31"]
        visda = [row for row in rows if row["dataset"] == "visda-c"]
        budget_result = {
            "requested_budget": budget,
            "requested_group_count": rows[0]["requested_group_count"],
            "active_candidate_scalars": rows[0]["active_candidate_scalars"],
        }
        if office:
            budget_result["office31"] = {
                "transfer_count": len(office),
                "aggregation": "equal-weight mean of six transfer sample accuracies",
                "PU-Acc": statistics.fmean(row["PU-Acc"] for row in office),
                "FO-Acc": statistics.fmean(row["FO-Acc"] for row in office),
                "per-transfer": {
                    row["transfer"]: {
                        "PU-Acc": row["PU-Acc"],
                        "FO-Acc": row["FO-Acc"],
                    }
                    for row in office
                },
            }
        if visda:
            if len(visda) != 1:
                raise ValueError("Each budget requires exactly one VisDA transfer")
            row = visda[0]
            budget_result["visda-c"] = {
                "transfer": row["transfer"],
                "primary_metric": "fixed_12_class_macro_accuracy",
                "PU-Acc": row["PU-Acc"],
                "FO-Acc": row["FO-Acc"],
                "PU-overall-Acc": row["PU-overall-Acc"],
                "FO-overall-Acc": row["FO-overall-Acc"],
                "PU-worst-class-Acc": row["PU-worst-class-Acc"],
                "FO-worst-class-Acc": row["FO-worst-class-Acc"],
                "PU-class-std": row["PU-class-std"],
                "FO-class-std": row["FO-class-std"],
                "PU-Acc-per-class": row["PU-Acc-per-class-by-name"],
                "FO-Acc-per-class": row["FO-Acc-per-class-by-name"],
            }
        result["budgets"][f"{budget:.3f}"] = budget_result
    return result


def write_aggregate(run_root: Path, aggregate: dict) -> None:
    with open(run_root / "aggregate.json", "w", encoding="utf-8") as file_obj:
        json.dump(aggregate, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")

    rows = []
    for summary in aggregate["transfers"]:
        rows.append(
            {
                "requested_budget": summary["requested_budget"],
                "requested_group_count": summary["requested_group_count"],
                "dataset": summary["dataset"],
                "transfer": summary["transfer"],
                "formal_seed": summary["formal_seed"],
                "primary_metric": summary["primary_metric"],
                "PU-Acc": summary["PU-Acc"],
                "FO-Acc": summary["FO-Acc"],
                "PU-overall-Acc": summary["PU-overall-Acc"],
                "FO-overall-Acc": summary["FO-overall-Acc"],
                "mask_sha256": summary["selection"]["mask_sha256"],
                "summary_path": str(Path(summary["output_dir"]) / "summary.json"),
            }
        )
    if not rows:
        raise ValueError("Cannot write an empty Group-Magnitude aggregate")
    with open(run_root / "results.csv", "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    class_rows = []
    for budget, record in aggregate["budgets"].items():
        visda = record.get("visda-c")
        if not visda:
            continue
        for name, pu_value in visda["PU-Acc-per-class"].items():
            class_rows.append(
                {
                    "requested_budget": budget,
                    "class": name,
                    "PU-Acc": pu_value,
                    "FO-Acc": visda["FO-Acc-per-class"][name],
                }
            )
    if class_rows:
        with open(
            run_root / "visda_per_class.csv", "w", encoding="utf-8", newline=""
        ) as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(class_rows[0]))
            writer.writeheader()
            writer.writerows(class_rows)


def print_aggregate(aggregate: dict) -> None:
    print("rho     dataset   transfer           PU-Acc    FO-Acc")
    print("------  --------  -----------------  --------  --------")
    for row in aggregate["transfers"]:
        print(
            f"{row['requested_budget']:<6.3f}  {row['dataset']:<8}  "
            f"{row['transfer']:<17}  {row['PU-Acc']:8.3f}  {row['FO-Acc']:8.3f}"
        )
    for budget, record in aggregate["budgets"].items():
        office = record.get("office31")
        if office:
            print(
                f"rho={budget} Office equal-transfer mean: "
                f"PU={office['PU-Acc']:.3f}, FO={office['FO-Acc']:.3f}"
            )
        visda = record.get("visda-c")
        if visda:
            print(
                f"rho={budget} VisDA fixed-12 macro: "
                f"PU={visda['PU-Acc']:.3f}, FO={visda['FO-Acc']:.3f}"
            )

