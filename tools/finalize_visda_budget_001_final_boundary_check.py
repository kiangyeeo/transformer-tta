#!/usr/bin/env python3
"""Finalize the 04F VisDA-C budget-0.001 boundary check without training."""

import argparse
import csv
import json
import math
from pathlib import Path


SEGMENTS = (("1-50", 1, 50), ("51-100", 51, 100),
            ("101-150", 101, 150), ("151-end", 151, None))


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
                if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def has_nonfinite(value):
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(has_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(has_nonfinite(item) for item in value)
    return False


def average(values):
    values = [value for value in values if finite(value)]
    return sum(values) / len(values) if values else None


def read_metrics(summary_path):
    metrics_path = Path(summary_path).with_name("metrics.jsonl")
    all_records, online = [], []
    with metrics_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            all_records.append(record)
            if record.get("event") == "online_step":
                online.append(record)
    return metrics_path, all_records, online


def valid(summary, all_records, online):
    required = ("FO-overall-Acc", "FO-mean-class-Acc", "FO-worst-class-Acc",
                "FO-class-std", "PU-overall-Acc", "PU-mean-class-Acc")
    return all((
        summary.get("status") == "completed",
        summary.get("valid_lbi_run") is True,
        summary.get("budget_reached_all_steps") is True,
        summary.get("max_steps_hit_count") == 0,
        len(online) == summary.get("online_steps"),
        bool(online),
        all(event.get("valid_lbi_step") is True for event in online),
        not any(event.get("max_steps_hit") is True for event in online),
        not any(record.get("event") == "error" for record in all_records),
        all(finite(summary.get(key)) for key in required),
        not has_nonfinite(summary),
        not any(has_nonfinite(record) for record in all_records),
    ))


def base_candidate(summary):
    if (summary.get("lbi_alpha"), summary.get("lbi_kappa"), summary.get("lbi_nu")) == (0.15, 1.0, 1.0):
        return "D4"
    if (summary.get("lbi_alpha"), summary.get("lbi_kappa"), summary.get("lbi_nu")) == (0.15, 1.0, 0.5):
        return "D5"
    raise ValueError("summary is outside the D4/D5 candidate scope: " + summary["experiment_key"])


def resume_metadata(experiment_key, launcher_logs):
    events = []
    for path in Path(launcher_logs).glob("**/launcher_summary.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("experiment_key") != experiment_key:
                    continue
                resume_dir = row.get("resume_run_dir") or None
                if resume_dir:
                    events.append({
                        "launcher_summary_path": str(path.resolve()),
                        "resume_run_dir": resume_dir,
                        "attempt": int(row["attempt"]) if row.get("attempt") else None,
                        "started_at_utc": row.get("started_at_utc") or None,
                        "completed_at_utc": row.get("completed_at_utc") or None,
                        "return_code": int(row["return_code"]) if row.get("return_code") else None,
                    })
    return {
        "resumed": bool(events),
        "resume_count": len(events),
        "resume_run_dirs": sorted({event["resume_run_dir"] for event in events}),
        "resume_events": events,
    }


def make_row(summary_path, summary, all_records, online, role, plan_status, launcher_logs):
    support = [event.get("stage1_support_count") for event in online]
    row = {
        "base_candidate": base_candidate(summary),
        "role": role,
        "plan_status": plan_status,
        "alpha": summary["lbi_alpha"], "kappa": summary["lbi_kappa"],
        "nu": summary["lbi_nu"], "omega": summary["lbi_omega"],
        "stage2_lr": summary["stage2_lr"],
        "stage2_steps": summary["stage2_steps_requested"],
        "implementation_revision": summary.get("implementation_revision"),
        "experiment_key": summary["experiment_key"],
        "experiment_config_sha256": summary["experiment_config_sha256"],
        "status": summary["status"],
        "fully_valid": valid(summary, all_records, online),
        "budget_hit_rate": summary.get("budget_hit_rate"),
        "budget_reached_all_steps": summary.get("budget_reached_all_steps"),
        "max_steps_hit_count": summary.get("max_steps_hit_count"),
        "stage1_steps_completed_mean": summary.get("stage1_steps_completed_mean"),
        "stage1_steps_completed_max": summary.get("stage1_steps_completed_max"),
        "support_min": min((value for value in support if finite(value)), default=None),
        "support_max": max((value for value in support if finite(value)), default=None),
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
    row.update(resume_metadata(row["experiment_key"], launcher_logs))
    return row


def segment_rows(row, online):
    result = []
    for name, first, last in SEGMENTS:
        subset = [event for event in online if event["iteration"] >= first and (last is None or event["iteration"] <= last)]
        support = [event.get("stage1_support_count") for event in subset]
        result.append({
            "base_candidate": row["base_candidate"], "role": row["role"],
            "omega": row["omega"], "stage2_lr": row["stage2_lr"],
            "experiment_key": row["experiment_key"], "segment": name,
            "first_batch": first, "last_batch": last or row["online_steps"],
            "online_steps": len(subset),
            "budget_hit_rate": average([1.0 if event.get("budget_reached") else 0.0 for event in subset]),
            "max_steps_hit_count": sum(event.get("max_steps_hit") is True for event in subset),
            "stage1_steps_completed_mean": average([event.get("stage1_steps_completed") for event in subset]),
            "support_min": min((value for value in support if finite(value)), default=None),
            "support_max": max((value for value in support if finite(value)), default=None),
        })
    return result


def rank_key(row):
    return (-row["FO_mean_per_class"], -row["FO_worst_class"],
            row["FO_classwise_std"], -row["FO_overall"],
            row["stage1_steps_completed_max"],
            row["stage1_steps_completed_mean"], row["runtime_seconds"])


def markdown_table(headers, values):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in values)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--selected-04e-config", required=True)
    parser.add_argument("--source-baseline-summary", required=True)
    parser.add_argument("--module-dense-summary", required=True)
    parser.add_argument("--sparse-reference-csv", required=True)
    parser.add_argument("--launcher-logs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-config", required=True)
    args = parser.parse_args()

    plan, status = read_json(args.plan), read_json(args.status_json)
    if plan.get("experiment_count") != 6 or len(plan.get("experiments", [])) != 6:
        raise ValueError("04F must contain exactly six planned experiments")
    status_by_key = {row["experiment_key"]: row for row in status["experiments"]}
    rows, segments = [], []
    for experiment in plan["experiments"]:
        status_row = status_by_key.get(experiment["experiment_key"])
        if not status_row or not (status_row["plan_status"] == "completed" and status_row["matching_completed_count"] == 1 and not status_row["hash_mismatch_paths"] and not status_row["invalid_summary_paths"]):
            raise ValueError("planned run is not uniquely complete and valid: " + experiment["experiment_key"])
        summary_path = Path(status_row["matching_summary_paths"][0])
        summary = read_json(summary_path)
        if summary.get("experiment_config_sha256") != experiment["experiment_config_sha256"]:
            raise ValueError("planned run hash mismatch: " + experiment["experiment_key"])
        _, all_records, online = read_metrics(summary_path)
        row = make_row(summary_path, summary, all_records, online, "04F_new", "completed", args.launcher_logs)
        rows.append(row)
        segments.extend(segment_rows(row, online))

    prior = read_json(args.selected_04e_config)
    references = [row for row in prior.get("all_six_configurations", [])
                  if row.get("base_candidate") in ("D4", "D5")
                  and row.get("omega") == 0.0125 and row.get("stage2_lr") == 0.005]
    if len(references) != 2 or {row["base_candidate"] for row in references} != {"D4", "D5"}:
        raise ValueError("04E must provide exactly the D4/D5 omega=0.0125, lr=0.005 references")
    for reference in references:
        summary_path = Path(reference["summary_path"])
        summary = read_json(summary_path)
        if summary.get("experiment_config_sha256") != reference["experiment_config_sha256"]:
            raise ValueError("04E reference hash mismatch: " + reference["base_candidate"])
        _, all_records, online = read_metrics(summary_path)
        row = make_row(summary_path, summary, all_records, online, "04E_reused_reference", "completed", args.launcher_logs)
        rows.append(row)
        segments.extend(segment_rows(row, online))

    if len(rows) != 8:
        raise AssertionError("expected exactly eight aggregate configurations")
    valid_rows = sorted([row for row in rows if row["fully_valid"]], key=rank_key)
    if len(valid_rows) != 8:
        raise ValueError("all eight configurations must pass the required validity gate")
    for rank, row in enumerate(valid_rows, 1):
        row["rank"] = rank
    if any(segment["budget_hit_rate"] != 1.0 or segment["max_steps_hit_count"] != 0 for segment in segments):
        raise ValueError("segmented reachability failed")

    winner = valid_rows[0]
    source, dense = read_json(args.source_baseline_summary), read_json(args.module_dense_summary)
    with Path(args.sparse_reference_csv).open(newline="", encoding="utf-8") as handle:
        sparse = next(row for row in csv.DictReader(handle) if float(row["requested_budget"]) == 0.001)
    baseline_comparison = {
        "winner": winner["experiment_key"], "winner_base_candidate": winner["base_candidate"],
        "winner_omega": winner["omega"], "winner_stage2_lr": winner["stage2_lr"],
        "winner_FO_mean_per_class": winner["FO_mean_per_class"],
        "source_only_FO_mean_per_class": source["FO-mean-class-Acc"],
        "best_sparse_method": sparse["best_sparse_method"],
        "best_sparse_FO_mean_per_class": float(sparse["FO-mean-class-Acc"]),
        "module_dense_FO_mean_per_class": dense["FO-mean-class-Acc"],
        "previous_04E_winner_FO_mean_per_class": prior["winner"]["FO_mean_per_class"],
        "delta_vs_source_only": winner["FO_mean_per_class"] - source["FO-mean-class-Acc"],
        "delta_vs_best_sparse": winner["FO_mean_per_class"] - float(sparse["FO-mean-class-Acc"]),
        "delta_vs_module_dense": winner["FO_mean_per_class"] - dense["FO-mean-class-Acc"],
        "delta_vs_previous_04E_winner": winner["FO_mean_per_class"] - prior["winner"]["FO_mean_per_class"],
        "source_only_summary_path": str(Path(args.source_baseline_summary).resolve()),
        "best_sparse_summary_path": sparse["source_summary_path"],
        "module_dense_summary_path": str(Path(args.module_dense_summary).resolve()),
        "previous_04E_winner_summary_path": prior["winner"]["summary_path"],
    }

    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "all_eight_configurations.csv", rows)
    write_json(output / "all_eight_configurations.json", rows)
    write_csv(output / "segmented_reachability.csv", segments)
    write_json(output / "segmented_reachability.json", segments)
    write_csv(output / "final_ranking.csv", valid_rows)
    write_json(output / "final_ranking.json", valid_rows)
    write_csv(output / "baseline_comparison.csv", [baseline_comparison])
    write_json(output / "baseline_comparison.json", baseline_comparison)

    selected = {
        "schema_version": 1,
        "phase": "04F_VISDA_BUDGET_001_FINAL_BOUNDARY_CHECK",
        "budget": 0.001,
        "tuning_frozen_after_04f": True,
        "freeze_statement": "Budget 0.001 tuning is frozen after 04F; do not perform another alpha/kappa/nu/omega/stage2_lr search, expand the grid downward, or tune stage1_max_steps.",
        "selection_rule": "validity first; FO mean per-class desc; FO worst-class desc; FO class-wise std asc; FO overall desc; Stage-1 max steps asc; Stage-1 mean steps asc; runtime asc",
        "winner": winner,
        "ranking": valid_rows,
        "all_eight_configurations": rows,
        "baseline_comparison": baseline_comparison,
        "04D_provenance": prior.get("04D_provenance"),
        "04E_provenance": {"selected_config_path": str(Path(args.selected_04e_config).resolve()), "winner": prior.get("winner"), "provenance": prior.get("04E_provenance")},
        "04F_provenance": {"plan_path": str(Path(args.plan).resolve()), "status_path": str(Path(args.status_json).resolve()), "launcher_logs": str(Path(args.launcher_logs).resolve()), "result_dir": str(output.resolve())},
        "next_recommended_phase": {"budgets": [0.0005, 0.002], "warm_start_center": {key: winner[key] for key in ("base_candidate", "alpha", "kappa", "nu", "omega", "stage2_lr", "stage2_steps")}},
    }
    write_json(args.selected_config, selected)

    completion = status["status_counts"]
    table = [[row["rank"], row["base_candidate"], row["role"], f"{row['omega']:.5f}",
              f"{row['stage2_lr']:.4f}", f"{row['FO_mean_per_class']:.4f}",
              f"{row['FO_worst_class']:.4f}", f"{row['FO_classwise_std']:.4f}",
              f"{row['FO_overall']:.4f}", row["resumed"]] for row in valid_rows]
    report = [
        "# 04F budget 0.001 final boundary check — FINALIZE", "", "## Completion", "",
        f"Plan status: {completion['completed']} completed, {completion['missing']} missing, 0 failed, {completion['duplicate_completed']} duplicate, {completion['hash_mismatch']} hash mismatch, {completion['invalid_summary']} invalid summary.", "",
        "## Eight-Configuration Ranking", "",
        markdown_table(["Rank", "Base", "Role", "Omega", "Stage-2 LR", "FO mean", "FO worst", "FO std", "FO overall", "Resumed"], table), "",
        "## Final Selection", "",
        f"Winner: {winner['base_candidate']}, omega={winner['omega']:.5f}, stage2_lr={winner['stage2_lr']:.4f}; FO mean per-class={winner['FO_mean_per_class']:.6f}.", "",
        f"FO mean-class deltas: source-only {baseline_comparison['delta_vs_source_only']:+.6f}; best sparse ({baseline_comparison['best_sparse_method']}) {baseline_comparison['delta_vs_best_sparse']:+.6f}; module dense {baseline_comparison['delta_vs_module_dense']:+.6f}; previous 04E winner {baseline_comparison['delta_vs_previous_04E_winner']:+.6f}.", "",
        "All eight configurations satisfy every-step budget reachability, zero Stage-1 max-step failures, and finite/no-error validity. Segmented reachability for 1-50, 51-100, 101-150, and 151-end is recorded in segmented_reachability.{csv,json}. Budget 0.001 tuning is frozen after 04F; the next work moves to budgets 0.0005 and 0.002 using this winner as the warm-start center. No training was launched, retried, or rerun during FINALIZE.",
    ]
    (output / "FINALIZE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
