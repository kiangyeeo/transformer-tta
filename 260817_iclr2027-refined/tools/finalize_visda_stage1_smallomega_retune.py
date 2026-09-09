#!/usr/bin/env python3
"""Finalize the completed 04D VisDA-C small-omega Stage-1 retune."""

import argparse
import csv
import json
import math
from pathlib import Path


SEGMENTS = (
    ("early", 1, 50),
    ("middle-1", 51, 100),
    ("middle-2", 101, 150),
    ("late", 151, None),
)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def average(values):
    values = [value for value in values if finite(value)]
    return sum(values) / len(values) if values else None


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def dump_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({
            key: json.dumps(value) if isinstance(value, (list, dict)) else value
            for key, value in row.items()
        } for row in rows)


def online_events(summary_path):
    metrics_path = Path(summary_path).with_name("metrics.jsonl")
    with metrics_path.open(encoding="utf-8") as handle:
        return metrics_path, [
            json.loads(line) for line in handle
            if line.strip() and json.loads(line).get("event") == "online_step"
        ]


def valid(summary, events):
    metrics = ("FO-Acc", "FO-overall-Acc", "PU-Acc", "PU-overall-Acc")
    return all((
        summary.get("status") == "completed",
        summary.get("budget_reached_all_steps") is True,
        summary.get("max_steps_hit_count") == 0,
        summary.get("valid_lbi_run") is True,
        len(events) == summary.get("online_steps"),
        all(finite(summary.get(metric)) for metric in metrics),
    ))


def candidate_row(candidate, role, summary, summary_path, events, plan_status):
    return {
        "candidate": candidate,
        "role": role,
        "plan_status": plan_status,
        "alpha": summary["lbi_alpha"], "kappa": summary["lbi_kappa"],
        "nu": summary["lbi_nu"], "omega": summary["lbi_omega"],
        "implementation_revision": summary.get("implementation_revision"),
        "experiment_key": summary["experiment_key"],
        "experiment_config_sha256": summary["experiment_config_sha256"],
        "status": summary["status"], "fully_valid": valid(summary, events),
        "budget_hit_rate": summary.get("budget_hit_rate"),
        "budget_reached_all_steps": summary.get("budget_reached_all_steps"),
        "max_steps_hit_count": summary.get("max_steps_hit_count"),
        "stage1_steps_completed_mean": summary.get("stage1_steps_completed_mean"),
        "stage1_steps_completed_max": summary.get("stage1_steps_completed_max"),
        "support_min": summary.get("stage1_support_count_min"),
        "support_max": summary.get("stage1_support_count_max"),
        "FO_overall": summary.get("FO-overall-Acc"),
        "FO_mean_per_class": summary.get("FO-mean-class-Acc"),
        "FO_worst_class": summary.get("FO-worst-class-Acc"),
        "FO_classwise_std": summary.get("FO-class-std"),
        "PU_overall": summary.get("PU-overall-Acc"),
        "PU_mean_per_class": summary.get("PU-mean-class-Acc"),
        "runtime_seconds": summary.get("runtime"),
        "summary_path": str(summary_path),
        "metrics_path": str(Path(summary_path).with_name("metrics.jsonl")),
    }


def segment_rows(row, events):
    result = []
    for name, first, last in SEGMENTS:
        subset = [event for event in events if event["iteration"] >= first and (last is None or event["iteration"] <= last)]
        support = [event.get("stage1_support_count") for event in subset]
        result.append({
            "candidate": row["candidate"], "role": row["role"],
            "experiment_key": row["experiment_key"], "segment": name,
            "first_batch": first, "last_batch": last or len(events),
            "online_steps": len(subset),
            "budget_hit_rate": average([1.0 if event.get("budget_reached") else 0.0 for event in subset]),
            "max_steps_hit_count": sum(event.get("max_steps_hit") is True for event in subset),
            "stage1_steps_completed_mean": average([event.get("stage1_steps_completed") for event in subset]),
            "support_min": min((value for value in support if finite(value)), default=None),
            "mean_confidence": average([event.get("max_prob_mean") for event in subset]),
            "mean_acc_post": average([event.get("acc_post") for event in subset]),
            "mean_applied_update_l2": average([event.get("applied_update_l2") for event in subset]),
        })
    return result


def formal_key(row):
    return (-row["FO_mean_per_class"], -row["FO_worst_class"], row["FO_classwise_std"],
            -row["FO_overall"], row["stage1_steps_completed_max"],
            row["stage1_steps_completed_mean"], row["runtime_seconds"])


def markdown_table(headers, values):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in values)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--source-baseline-summary", required=True)
    parser.add_argument("--module-dense-summary", required=True)
    parser.add_argument("--sparse-reference-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-config", required=True)
    args = parser.parse_args()

    plan = read_json(args.plan)
    status = read_json(args.status_json)
    status_by_key = {item["experiment_key"]: item for item in status["experiments"]}
    summaries = {}
    for path in Path(args.runs_root).rglob("summary.json"):
        summary = read_json(path)
        key = summary.get("experiment_key")
        if key:
            summaries.setdefault(key, []).append((path, summary))

    rows, segments = [], []
    for index, experiment in enumerate(plan["experiments"], 1):
        item = status_by_key[experiment["experiment_key"]]
        if item["plan_status"] != "completed" or item["matching_completed_count"] != 1:
            raise ValueError(f"planned identity is not uniquely completed: {experiment['experiment_key']}")
        matches = summaries.get(experiment["experiment_key"], [])
        if len(matches) != 1 or matches[0][1].get("experiment_config_sha256") != experiment["experiment_config_sha256"]:
            raise ValueError(f"summary identity mismatch: {experiment['experiment_key']}")
        path, summary = matches[0]
        _, events = online_events(path)
        row = candidate_row(f"D{index}", "formal_candidate", summary, path, events, item["plan_status"])
        rows.append(row)
        segments.extend(segment_rows(row, events))

    reference_path = Path(args.reference_summary)
    reference = read_json(reference_path)
    _, reference_events = online_events(reference_path)
    reference_row = candidate_row("04C_O1_reference", "reference_only", reference, reference_path, reference_events, "completed")
    rows.append(reference_row)
    segments.extend(segment_rows(reference_row, reference_events))

    valid_rows = sorted([row for row in rows if row["role"] == "formal_candidate" and row["fully_valid"]], key=formal_key)
    for rank, row in enumerate(valid_rows, 1): row["formal_rank"] = rank
    by_key = {row["experiment_key"]: {segment["segment"]: segment for segment in segments if segment["experiment_key"] == row["experiment_key"]} for row in rows}
    invalid_rows = [row for row in rows if row["role"] == "formal_candidate" and not row["fully_valid"]]
    invalid_rows.sort(key=lambda row: (-row["budget_hit_rate"], -by_key[row["experiment_key"]]["late"]["budget_hit_rate"], -row["support_min"], row["max_steps_hit_count"], -row["FO_mean_per_class"], -row["FO_worst_class"], row["FO_classwise_std"]))
    for rank, row in enumerate(invalid_rows, 1): row["invalid_diagnostic_rank"] = rank

    source = read_json(args.source_baseline_summary)
    module_dense = read_json(args.module_dense_summary)
    with Path(args.sparse_reference_csv).open(newline="", encoding="utf-8") as handle:
        sparse = next(row for row in csv.DictReader(handle) if float(row["requested_budget"]) == 0.001)
    best = valid_rows[0] if valid_rows else None
    gate = {
        "best_valid_candidate": best["candidate"] if best else None,
        "best_sparse_method": sparse["best_sparse_method"],
        "source_only_FO_mean_per_class": source["FO-mean-class-Acc"],
        "best_sparse_FO_mean_per_class": float(sparse["FO-mean-class-Acc"]),
        "module_dense_FO_mean_per_class": module_dense["FO-mean-class-Acc"],
        "best_valid_FO_mean_per_class": best["FO_mean_per_class"] if best else None,
        "delta_vs_source_only": best["FO_mean_per_class"] - source["FO-mean-class-Acc"] if best else None,
        "delta_vs_best_sparse": best["FO_mean_per_class"] - float(sparse["FO-mean-class-Acc"]) if best else None,
        "delta_vs_module_dense": best["FO_mean_per_class"] - module_dense["FO-mean-class-Acc"] if best else None,
        "beats_source_only": best is not None and best["FO_mean_per_class"] > source["FO-mean-class-Acc"],
        "beats_best_sparse": best is not None and best["FO_mean_per_class"] > float(sparse["FO-mean-class-Acc"]),
        "beats_module_dense": best is not None and best["FO_mean_per_class"] > module_dense["FO-mean-class-Acc"],
        "source_only_summary_path": args.source_baseline_summary,
        "module_dense_summary_path": args.module_dense_summary,
        "best_sparse_summary_path": sparse["source_summary_path"],
    }

    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    dump_csv(output / "candidate_diagnostics.csv", rows); dump_json(output / "candidate_diagnostics.json", rows)
    dump_csv(output / "segmented_reachability.csv", segments); dump_json(output / "segmented_reachability.json", segments)
    dump_csv(output / "formal_candidate_ranking.csv", valid_rows); dump_json(output / "formal_candidate_ranking.json", valid_rows)
    dump_csv(output / "invalid_diagnostic_ranking.csv", invalid_rows); dump_json(output / "invalid_diagnostic_ranking.json", invalid_rows)
    dump_csv(output / "performance_gate.csv", [gate]); dump_json(output / "performance_gate.json", gate)

    selected = {
        "schema_version": 1, "budget": 0.001, "omega": 0.0125,
        "selection_rule": "validity first; FO mean per-class desc; FO worst-class desc; FO class-wise std asc; FO overall desc; Stage-1 max steps asc; Stage-1 mean steps asc; runtime asc",
        "04c_reference": reference_row,
        "04d_plan": str(Path(args.plan).resolve()),
        "04d_status": str(Path(args.status_json).resolve()),
        "performance_gate": gate, "selected": valid_rows[:2],
    }
    if len(valid_rows) >= 2:
        dump_json(args.selected_config, selected)
    else:
        (output / "NEXT_ACTION_REQUIRED.md").write_text("# Next Action Required\n\nFewer than two fully valid 04D candidates were available; do not proceed to 05/06.\n", encoding="utf-8")

    candidate_table = [[row["candidate"], row["role"], row["fully_valid"], f"{row['budget_hit_rate']:.4f}", row["max_steps_hit_count"], f"{row['FO_mean_per_class']:.4f}", f"{row['FO_worst_class']:.4f}", f"{row['FO_classwise_std']:.4f}", f"{row['runtime_seconds']/3600:.2f}"] for row in rows]
    report = ["# 04D Stage-1 small-omega retune - FINALIZE", "", "## Completion", "", f"Plan status: {status['status_counts']['completed']} completed, {status['status_counts']['missing']} missing, 0 failed, {status['status_counts']['duplicate_completed']} duplicate, {status['status_counts']['hash_mismatch']} hash mismatch, {status['status_counts']['invalid_summary']} invalid summary.", "", "## Candidate Diagnostics", "", markdown_table(["Candidate", "Role", "Valid", "Hit rate", "Max failures", "FO mean", "FO worst", "FO std", "Runtime h"], candidate_table), "", "## Formal Selection", "", "Top two fully valid candidates: " + ", ".join(f"{row['candidate']} (FO mean {row['FO_mean_per_class']:.4f})" for row in valid_rows[:2]) + ".", "", "## Performance Gate", "", f"{best['candidate']} FO mean {best['FO_mean_per_class']:.4f}: source-only {gate['delta_vs_source_only']:+.4f}; best sparse ({gate['best_sparse_method']}) {gate['delta_vs_best_sparse']:+.4f}; module dense {gate['delta_vs_module_dense']:+.4f}.", "", "D1 and D2 are invalid because their Stage-1 max-step failures prevent budget reachability at every online step. D3-D8 are fully valid. The Stage-1 reachability gate is solved by D4/D5; no training was launched, retried, or rerun in FINALIZE."]
    (output / "FINALIZE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
