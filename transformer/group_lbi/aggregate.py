"""Aggregate Group-LBI budgets, transfers, diagnostics, and VisDA classes."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from .config import FORMAL_BUDGETS, TRANSFERS, budget_key, budget_tag


def _tuning_diagnostics(selection: dict) -> dict:
    return {
        "batch_count": int(selection["online_batch_count"]),
        "average_utilization": float(selection["utilization_mean"]),
        "minimum_utilization": float(selection["utilization_min"]),
        "utilization_ge_90_rate": float(selection["utilization_ge_90_rate"]),
        "utilization_ge_95_rate": float(selection["utilization_ge_95_rate"]),
        "stage1_3000_step_hit_rate": float(selection["stage1_3000_step_hit_rate"]),
        "stage1_step_cap": int(selection["stage1_step_cap"]),
        "stage1_step_cap_hit_rate": float(selection["stage1_step_cap_hit_rate"]),
        "stage1_mean_steps": float(selection["stage1_steps_mean"]),
        "stage1_max_steps": int(selection["stage1_steps_max"]),
        "rollback_rate": float(selection["rollback_rate"]),
        "average_selected_groups": float(selection["average_selected_groups"]),
        "budget_violation_rate": float(selection["budget_violation_rate"]),
        "failure_rate": float(selection["failure_rate"]),
    }


def _pooled_tuning_diagnostics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Pooled tuning diagnostics require at least one condition")
    selections = [row["selection"] for row in rows]
    total_batches = sum(int(item["online_batch_count"]) for item in selections)
    if total_batches <= 0:
        raise ValueError("Pooled tuning diagnostics require completed online batches")
    caps = {int(item["stage1_step_cap"]) for item in selections}
    if len(caps) != 1:
        raise ValueError("Pooled conditions use different Stage-1 step caps")
    total = lambda field: sum(float(item[field]) for item in selections)
    return {
        "aggregation": "pooled over all online batches",
        "condition_count": len(rows),
        "batch_count": total_batches,
        "average_utilization": total("utilization_sum") / total_batches,
        "minimum_utilization": min(float(item["utilization_min"]) for item in selections),
        "utilization_ge_90_rate": total("utilization_ge_90_count") / total_batches,
        "utilization_ge_95_rate": total("utilization_ge_95_count") / total_batches,
        "stage1_3000_step_hit_rate": total("stage1_3000_step_hit_count") / total_batches,
        "stage1_step_cap": caps.pop(),
        "stage1_step_cap_hit_rate": total("stage1_step_cap_hit_count") / total_batches,
        "stage1_mean_steps": total("stage1_steps_sum") / total_batches,
        "stage1_max_steps": max(int(item["stage1_steps_max"]) for item in selections),
        "rollback_rate": total("rollback_count") / total_batches,
        "average_selected_groups": total("selected_groups_sum") / total_batches,
        "budget_violation_rate": total("budget_violation_count") / total_batches,
        "failure_rate": total("failure_batch_count") / total_batches,
    }


def _load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        summary = json.load(file_obj)
    if summary.get("status") != "completed":
        raise ValueError(f"Group-LBI condition is not completed: {path}")
    if summary.get("valid_lbi_run") is False:
        raise ValueError(f"Group-LBI condition is scientifically invalid: {path}")
    if summary.get("variant") != "group_lbi":
        raise ValueError(f"Condition is not Group-LBI: {path}")
    selection = summary.get("selection", {})
    if "utilization_ge_90_rate" not in selection:
        metrics_path = path.with_name("metrics.jsonl")
        if not metrics_path.is_file():
            raise ValueError(
                f"Legacy summary needs metrics.jsonl for tuning diagnostics: {path}"
            )
        records = []
        with open(metrics_path, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("event") == "online_batch":
                    records.append(record)
        from .runner import _selection_summary

        derived = _selection_summary(
            records,
            stage1_step_cap=int(summary["lbi"]["stage1_max_steps"]),
        )
        summary["selection"] = {**selection, **derived}
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
        "variant": "group_lbi",
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
            "realized_group_count_mean_across_conditions": statistics.fmean(
                row["selection"]["realized_group_count_mean"] for row in rows
            ),
            "utilization_mean_across_conditions": statistics.fmean(
                row["selection"]["utilization_mean"] for row in rows
            ),
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
                        "realized-K-mean": row["selection"]["realized_group_count_mean"],
                        "utilization-mean": row["selection"]["utilization_mean"],
                    }
                    for row in office
                },
                "tuning-diagnostics-pooled": _pooled_tuning_diagnostics(office),
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
                "realized-K-mean": row["selection"]["realized_group_count_mean"],
                "utilization-mean": row["selection"]["utilization_mean"],
                "tuning-diagnostics": _tuning_diagnostics(row["selection"]),
            }
        result["budgets"][budget_key(budget)] = budget_result
    return result


def write_aggregate(run_root: Path, aggregate: dict) -> None:
    with open(run_root / "aggregate.json", "w", encoding="utf-8") as file_obj:
        json.dump(aggregate, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")

    rows = []
    for summary in aggregate["transfers"]:
        selection = summary["selection"]
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
                "realized-K-mean": selection["realized_group_count_mean"],
                "realized-K-min": selection["realized_group_count_min"],
                "realized-K-max": selection["realized_group_count_max"],
                "utilization-mean": selection["utilization_mean"],
                "rollback-count": selection["rollback_count"],
                "rollback-rate": selection["rollback_rate"],
                "max-steps-count": selection["max_steps_reached_count"],
                "utilization-ge-90-rate": selection["utilization_ge_90_rate"],
                "utilization-ge-95-rate": selection["utilization_ge_95_rate"],
                "stage1-3000-step-hit-rate": selection["stage1_3000_step_hit_rate"],
                "stage1-step-cap": selection["stage1_step_cap"],
                "stage1-step-cap-hit-rate": selection["stage1_step_cap_hit_rate"],
                "stage1-mean-steps": selection["stage1_steps_mean"],
                "stage1-max-steps": selection["stage1_steps_max"],
                "budget-violation-rate": selection["budget_violation_rate"],
                "failure-rate": selection["failure_rate"],
                "summary_path": str(Path(summary["output_dir"]) / "summary.json"),
            }
        )
    if not rows:
        raise ValueError("Cannot write an empty Group-LBI aggregate")
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
        with open(run_root / "visda_per_class.csv", "w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(class_rows[0]))
            writer.writeheader()
            writer.writerows(class_rows)

    tuning_rows = []
    for summary in aggregate["transfers"]:
        tuning_rows.append(
            {
                "requested_budget": summary["requested_budget"],
                "requested_group_count": summary["requested_group_count"],
                "dataset": summary["dataset"],
                "transfer": summary["transfer"],
                "aggregation": "single transfer over all online batches",
                "condition_count": 1,
                **_tuning_diagnostics(summary["selection"]),
            }
        )
    for budget, record in aggregate["budgets"].items():
        office = record.get("office31")
        if office:
            tuning_rows.append(
                {
                    "requested_budget": budget,
                    "requested_group_count": record["requested_group_count"],
                    "dataset": "office31",
                    "transfer": "OFFICE_POOLED",
                    **office["tuning-diagnostics-pooled"],
                }
            )
    with open(
        run_root / "tuning_diagnostics.csv", "w", encoding="utf-8", newline=""
    ) as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(tuning_rows[0]))
        writer.writeheader()
        writer.writerows(tuning_rows)


def print_aggregate(aggregate: dict) -> None:
    def print_tuning(diagnostics: dict) -> None:
        print(
            "  tuning: "
            f"batches={diagnostics['batch_count']}, "
            f"util(mean/min)={diagnostics['average_utilization']:.4f}/"
            f"{diagnostics['minimum_utilization']:.4f}, "
            f"u90/u95={diagnostics['utilization_ge_90_rate']:.4f}/"
            f"{diagnostics['utilization_ge_95_rate']:.4f}, "
            f"selected-mean={diagnostics['average_selected_groups']:.2f}, "
            f"steps(mean/max)={diagnostics['stage1_mean_steps']:.1f}/"
            f"{diagnostics['stage1_max_steps']}, "
            f"hit3000={diagnostics['stage1_3000_step_hit_rate']:.4f}, "
            f"cap={diagnostics['stage1_step_cap']}, "
            f"cap-hit={diagnostics['stage1_step_cap_hit_rate']:.4f}, "
            f"rollback={diagnostics['rollback_rate']:.4f}, "
            f"violation={diagnostics['budget_violation_rate']:.4f}, "
            f"failure={diagnostics['failure_rate']:.4f}"
        )

    print("rho     dataset   transfer           PU-Acc    FO-Acc   realized-K")
    print("------  --------  -----------------  --------  --------  ----------")
    for row in aggregate["transfers"]:
        print(
            f"{budget_key(row['requested_budget']):<6}  {row['dataset']:<8}  "
            f"{row['transfer']:<17}  {row['PU-Acc']:8.3f}  {row['FO-Acc']:8.3f}  "
            f"{row['selection']['realized_group_count_mean']:10.2f}"
        )
    for budget, record in aggregate["budgets"].items():
        office = record.get("office31")
        if office:
            print(
                f"rho={budget} Office equal-transfer mean: "
                f"PU={office['PU-Acc']:.3f}, FO={office['FO-Acc']:.3f}"
            )
            print_tuning(office["tuning-diagnostics-pooled"])
        visda = record.get("visda-c")
        if visda:
            print(
                f"rho={budget} VisDA fixed-12 macro: "
                f"PU={visda['PU-Acc']:.3f}, FO={visda['FO-Acc']:.3f}"
            )
            print_tuning(visda["tuning-diagnostics"])
