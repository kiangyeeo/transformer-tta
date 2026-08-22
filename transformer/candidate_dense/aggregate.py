"""Aggregate candidate-dense transfers with protocol-correct metrics."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from .config import TRANSFERS


def _load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        summary = json.load(file_obj)
    if summary.get("status") != "completed":
        raise ValueError(f"Transfer is not completed: {path}")
    if summary.get("variant") != "candidate_dense":
        raise ValueError(f"Transfer is not candidate-dense: {path}")
    return summary


def aggregate_matrix(run_root: Path, transfers=TRANSFERS) -> dict:
    summaries = []
    for dataset, source, target in transfers:
        summaries.append(
            _load_summary(
                run_root / "results" / dataset / f"{source}-{target}" / "summary.json"
            )
        )
    office = [row for row in summaries if row["dataset"] == "office31"]
    visda = [row for row in summaries if row["dataset"] == "visda-c"]
    result = {
        "status": "completed",
        "variant": "candidate_dense",
        "formal_seed": 2026,
        "transfer_count": len(summaries),
        "transfers": summaries,
    }
    if office:
        result["office31"] = {
            "transfer_count": len(office),
            "aggregation": "equal-weight mean of transfer sample accuracies",
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
            raise ValueError("Formal VisDA-C aggregation requires exactly one transfer")
        row = visda[0]
        result["visda-c"] = {
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
    return result


def write_aggregate(run_root: Path, aggregate: dict) -> None:
    with open(run_root / "aggregate.json", "w", encoding="utf-8") as file_obj:
        json.dump(aggregate, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")

    rows = [
        {
            "dataset": summary["dataset"],
            "transfer": summary["transfer"],
            "formal_seed": summary["formal_seed"],
            "primary_metric": summary["primary_metric"],
            "PU-Acc": summary["PU-Acc"],
            "FO-Acc": summary["FO-Acc"],
            "PU-overall-Acc": summary["PU-overall-Acc"],
            "FO-overall-Acc": summary["FO-overall-Acc"],
            "summary_path": summary["output_dir"] + "/summary.json",
        }
        for summary in aggregate["transfers"]
    ]
    if not rows:
        raise ValueError("Cannot write an empty candidate-dense aggregate")
    with open(run_root / "results.csv", "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    visda = aggregate.get("visda-c")
    if visda:
        class_rows = [
            {
                "class": name,
                "PU-Acc": visda["PU-Acc-per-class"][name],
                "FO-Acc": visda["FO-Acc-per-class"][name],
            }
            for name in visda["PU-Acc-per-class"]
        ]
        with open(
            run_root / "visda_per_class.csv", "w", encoding="utf-8", newline=""
        ) as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=["class", "PU-Acc", "FO-Acc"])
            writer.writeheader()
            writer.writerows(class_rows)


def print_aggregate(aggregate: dict) -> None:
    print("dataset   transfer           PU-Acc   FO-Acc")
    print("--------  -----------------  -------  -------")
    for row in aggregate["transfers"]:
        print(
            f"{row['dataset']:<8}  {row['transfer']:<17}  "
            f"{row['PU-Acc']:7.3f}  {row['FO-Acc']:7.3f}"
        )
    if "office31" in aggregate:
        office = aggregate["office31"]
        print(
            f"Office equal-transfer mean: PU={office['PU-Acc']:.3f}, "
            f"FO={office['FO-Acc']:.3f}"
        )
    if "visda-c" in aggregate:
        visda = aggregate["visda-c"]
        print(
            f"VisDA fixed-12 macro: PU={visda['PU-Acc']:.3f}, "
            f"FO={visda['FO-Acc']:.3f}"
        )
        print("VisDA per class:")
        for name, pu_value in visda["PU-Acc-per-class"].items():
            print(
                f"  {name:<12} PU={pu_value:7.3f} "
                f"FO={visda['FO-Acc-per-class'][name]:7.3f}"
            )

