#!/usr/bin/env python3
"""Finalize 04E: VISDA-C budget-0.001 small-omega Stage-2 LR search."""

import argparse
import csv
import json
import math
from pathlib import Path


SEGMENTS = (
    ("1-50", 1, 50),
    ("51-100", 51, 100),
    ("101-150", 101, 150),
    ("151-end", 151, None),
)
NEW_BASE_CANDIDATES = ("D4", "D4", "D5", "D5")


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def average(values):
    values = [value for value in values if finite(value)]
    return sum(values) / len(values) if values else None


def has_nonfinite(value):
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(has_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(has_nonfinite(item) for item in value)
    return False


def online_events(summary_path):
    metrics_path = Path(summary_path).with_name("metrics.jsonl")
    events = []
    with metrics_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "online_step":
                events.append(record)
    return metrics_path, events


def is_fully_valid(summary, events):
    required_metrics = (
        "FO-overall-Acc", "FO-mean-class-Acc", "FO-worst-class-Acc",
        "FO-class-std", "PU-overall-Acc", "PU-mean-class-Acc",
    )
    return all((
        summary.get("status") == "completed",
        summary.get("budget_reached_all_steps") is True,
        summary.get("max_steps_hit_count") == 0,
        summary.get("valid_lbi_run") is True,
        len(events) == summary.get("online_steps"),
        all(finite(summary.get(key)) for key in required_metrics),
        not has_nonfinite(summary),
        not any(has_nonfinite(event) for event in events),
    ))


def make_row(base_candidate, role, summary_path, summary, events, plan_status):
    return {
        "base_candidate": base_candidate,
        "role": role,
        "plan_status": plan_status,
        "alpha": summary["lbi_alpha"],
        "kappa": summary["lbi_kappa"],
        "nu": summary["lbi_nu"],
        "omega": summary["lbi_omega"],
        "stage2_lr": summary["stage2_lr"],
        "stage2_steps": summary["stage2_steps_requested"],
        "implementation_revision": summary.get("implementation_revision"),
        "experiment_key": summary["experiment_key"],
        "experiment_config_sha256": summary["experiment_config_sha256"],
        "status": summary["status"],
        "fully_valid": is_fully_valid(summary, events),
        "budget_hit_rate": summary.get("budget_hit_rate"),
        "budget_reached_all_steps": summary.get("budget_reached_all_steps"),
        "max_steps_hit_count": summary.get("max_steps_hit_count"),
        "stage1_steps_completed_mean": summary.get("stage1_steps_completed_mean"),
        "stage1_steps_completed_max": summary.get("stage1_steps_completed_max"),
        "FO_overall": summary.get("FO-overall-Acc"),
        "FO_mean_per_class": summary.get("FO-mean-class-Acc"),
        "FO_worst_class": summary.get("FO-worst-class-Acc"),
        "FO_classwise_std": summary.get("FO-class-std"),
        "PU_overall": summary.get("PU-overall-Acc"),
        "PU_mean_per_class": summary.get("PU-mean-class-Acc"),
        "runtime_seconds": summary.get("runtime"),
        "online_steps": summary.get("online_steps"),
        "summary_path": str(Path(summary_path).resolve()),
        "metrics_path": str(Path(summary_path).with_name("metrics.jsonl").resolve()),
    }


def segment_rows(row, events):
    rows = []
    for name, first, last in SEGMENTS:
        subset = [
            event for event in events
            if event["iteration"] >= first
            and (last is None or event["iteration"] <= last)
        ]
        support = [event.get("stage1_support_count") for event in subset]
        rows.append({
            "base_candidate": row["base_candidate"],
            "role": row["role"],
            "stage2_lr": row["stage2_lr"],
            "experiment_key": row["experiment_key"],
            "segment": name,
            "first_batch": first,
            "last_batch": last or len(events),
            "online_steps": len(subset),
            "budget_hit_rate": average(
                [1.0 if event.get("budget_reached") else 0.0 for event in subset]
            ),
            "max_steps_hit_count": sum(
                event.get("max_steps_hit") is True for event in subset
            ),
            "stage1_steps_completed_mean": average(
                [event.get("stage1_steps_completed") for event in subset]
            ),
            "stage1_support_count_min": min(
                (value for value in support if finite(value)), default=None
            ),
            "stage1_support_count_max": max(
                (value for value in support if finite(value)), default=None
            ),
        })
    return rows


def rank_key(row):
    return (
        -row["FO_mean_per_class"], -row["FO_worst_class"],
        row["FO_classwise_std"], -row["FO_overall"],
        row["stage1_steps_completed_max"],
        row["stage1_steps_completed_mean"], row["runtime_seconds"],
    )


def markdown_table(headers, data):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in data)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--selected-stage1-config", required=True)
    parser.add_argument("--source-baseline-summary", required=True)
    parser.add_argument("--module-dense-summary", required=True)
    parser.add_argument("--sparse-reference-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-config", required=True)
    args = parser.parse_args()

    plan = read_json(args.plan)
    if plan.get("experiment_count") != 4:
        raise ValueError("04E must contain exactly four planned experiments")
    status = read_json(args.status_json)
    status_by_key = {row["experiment_key"]: row for row in status["experiments"]}
    rows, segments = [], []
    for index, experiment in enumerate(plan["experiments"]):
        status_row = status_by_key[experiment["experiment_key"]]
        if not (
            status_row["plan_status"] == "completed"
            and status_row["matching_completed_count"] == 1
            and not status_row["hash_mismatch_paths"]
            and not status_row["invalid_summary_paths"]
        ):
            raise ValueError("planned run is not uniquely valid: " + experiment["experiment_key"])
        summary_path = Path(status_row["matching_summary_paths"][0])
        summary = read_json(summary_path)
        if summary.get("experiment_config_sha256") != experiment["experiment_config_sha256"]:
            raise ValueError("planned run hash mismatch: " + experiment["experiment_key"])
        metrics_path, events = online_events(summary_path)
        if not metrics_path.is_file():
            raise ValueError("metrics missing: " + str(metrics_path))
        row = make_row(NEW_BASE_CANDIDATES[index], "04E_new", summary_path, summary, events, "completed")
        rows.append(row)
        segments.extend(segment_rows(row, events))

    stage1_selected = read_json(args.selected_stage1_config)
    references = stage1_selected.get("selected")
    if not isinstance(references, list) or len(references) != 2:
        raise ValueError("04D selected configuration must contain D4 and D5")
    for reference in references:
        base = reference.get("candidate")
        if base not in ("D4", "D5"):
            raise ValueError("invalid 04D lr=0.02 reference")
        summary_path = Path(reference["summary_path"])
        summary = read_json(summary_path)
        if summary.get("experiment_config_sha256") != reference["experiment_config_sha256"]:
            raise ValueError("04D reference hash mismatch: " + base)
        if summary.get("stage2_lr") != 0.02:
            raise ValueError("04D reference is not lr=0.02: " + base)
        _, events = online_events(summary_path)
        row = make_row(base, "04D_reference_lr_0.020", summary_path, summary, events, "completed")
        rows.append(row)
        segments.extend(segment_rows(row, events))

    if len(rows) != 6:
        raise AssertionError("expected exactly six aggregate rows")
    valid_rows = sorted([row for row in rows if row["fully_valid"]], key=rank_key)
    if len(valid_rows) != 6:
        raise ValueError("all six 04D/04E configurations must be fully valid")
    for rank, row in enumerate(valid_rows, 1):
        row["rank"] = rank
    best_new = min(
        [row for row in valid_rows if row["role"] == "04E_new"], key=rank_key
    )
    best_reference = min(
        [row for row in valid_rows if row["role"] == "04D_reference_lr_0.020"], key=rank_key
    )
    winner = valid_rows[0]

    source = read_json(args.source_baseline_summary)
    dense = read_json(args.module_dense_summary)
    with Path(args.sparse_reference_csv).open(newline="", encoding="utf-8") as handle:
        sparse = next(
            row for row in csv.DictReader(handle)
            if float(row["requested_budget"]) == 0.001
        )
    baseline_comparison = {
        "winner": winner["experiment_key"],
        "winner_base_candidate": winner["base_candidate"],
        "winner_stage2_lr": winner["stage2_lr"],
        "source_only_FO_mean_per_class": source["FO-mean-class-Acc"],
        "best_sparse_method": sparse["best_sparse_method"],
        "best_sparse_FO_mean_per_class": float(sparse["FO-mean-class-Acc"]),
        "module_dense_FO_mean_per_class": dense["FO-mean-class-Acc"],
        "winner_FO_mean_per_class": winner["FO_mean_per_class"],
        "delta_vs_source_only": winner["FO_mean_per_class"] - source["FO-mean-class-Acc"],
        "delta_vs_best_sparse": winner["FO_mean_per_class"] - float(sparse["FO-mean-class-Acc"]),
        "delta_vs_module_dense": winner["FO_mean_per_class"] - dense["FO-mean-class-Acc"],
        "source_only_summary_path": str(Path(args.source_baseline_summary).resolve()),
        "best_sparse_summary_path": sparse["source_summary_path"],
        "module_dense_summary_path": str(Path(args.module_dense_summary).resolve()),
    }
    selection = {
        "winner": winner,
        "best_new_lr": best_new,
        "best_lr_0.020_reference": best_reference,
        "new_lr_beats_best_lr_0.020_reference": rank_key(best_new) < rank_key(best_reference),
        "rationale": (
            "A new Stage-2 LR ranks above both lr=0.020 references under the required validity-first ordering."
            if winner["role"] == "04E_new" else
            "No new Stage-2 LR beats the best lr=0.020 reference; retain the lr=0.020 configuration."
        ),
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "all_six_configurations.csv", rows)
    write_json(output / "all_six_configurations.json", rows)
    write_csv(output / "segmented_stage1_reachability.csv", segments)
    write_json(output / "segmented_stage1_reachability.json", segments)
    write_csv(output / "final_ranking.csv", valid_rows)
    write_json(output / "final_ranking.json", valid_rows)
    write_json(output / "baseline_comparison.json", baseline_comparison)
    write_csv(output / "baseline_comparison.csv", [baseline_comparison])
    write_json(output / "selection.json", selection)

    selected = {
        "schema_version": 1,
        "phase": "04E_VISDA_STAGE2_LR_SMALL_OMEGA_BUDGET_001",
        "budget": 0.001,
        "omega": 0.0125,
        "selection_rule": "validity first; FO mean per-class desc; FO worst-class desc; FO class-wise std asc; FO overall desc; Stage-1 max steps asc; Stage-1 mean steps asc; runtime asc",
        "winner": winner,
        "selection": selection,
        "all_six_configurations": rows,
        "baseline_comparison": baseline_comparison,
        "04D_provenance": {
            "selected_stage1_config_path": str(Path(args.selected_stage1_config).resolve()),
            "04d_plan": stage1_selected.get("04d_plan"),
            "04d_status": stage1_selected.get("04d_status"),
            "references": references,
        },
        "04E_provenance": {
            "plan_path": str(Path(args.plan).resolve()),
            "status_path": str(Path(args.status_json).resolve()),
            "result_dir": str(output.resolve()),
        },
    }
    write_json(args.selected_config, selected)

    completion = status["status_counts"]
    table = [[
        row["rank"], row["base_candidate"], row["role"], f"{row['stage2_lr']:.3f}",
        row["fully_valid"], f"{row['FO_mean_per_class']:.4f}",
        f"{row['FO_worst_class']:.4f}", f"{row['FO_classwise_std']:.4f}",
        f"{row['FO_overall']:.4f}", f"{row['runtime_seconds'] / 3600:.2f}",
    ] for row in valid_rows]
    report = [
        "# 04E Stage-2 LR small-omega — FINALIZE", "",
        "## Completion", "",
        f"Plan status: {completion['completed']} completed, {completion['missing']} missing, 0 failed, {completion['duplicate_completed']} duplicate, {completion['hash_mismatch']} hash mismatch, {completion['invalid_summary']} invalid summary.", "",
        "## Six-Configuration Ranking", "",
        markdown_table(["Rank", "Base", "Role", "Stage-2 LR", "Valid", "FO mean", "FO worst", "FO std", "FO overall", "Runtime h"], table), "",
        "## Selection", "",
        selection["rationale"], "",
        f"Winner: {winner['base_candidate']} with stage2_lr={winner['stage2_lr']:.3f}; FO mean per-class={winner['FO_mean_per_class']:.6f}.", "",
        "## Baselines", "",
        f"FO mean-class deltas: source-only {baseline_comparison['delta_vs_source_only']:+.6f}; best sparse ({baseline_comparison['best_sparse_method']}) {baseline_comparison['delta_vs_best_sparse']:+.6f}; module dense {baseline_comparison['delta_vs_module_dense']:+.6f}.", "",
        "All six configurations satisfy the every-step budget, no-max-step-failure, and finite-result validity gate. The segmented reachability data is retained in segmented_stage1_reachability.{csv,json}. No training was launched, retried, or rerun during FINALIZE.",
    ]
    (output / "FINALIZE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
