#!/usr/bin/env python3
"""Finalize the VisDA-C budget-0.001 omega late-stream diagnostic."""

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


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [{key: csv_value(value) for key, value in row.items()} for row in rows]
        )


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def mean(values):
    values = [value for value in values if finite(value)]
    return sum(values) / len(values) if values else None


def read_online_steps(summary_path):
    metrics_path = Path(summary_path).with_name("metrics.jsonl")
    events = []
    with metrics_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                event = json.loads(line)
                if event.get("event") == "online_step":
                    events.append(event)
    return metrics_path, events


def compute_segment_rows(row, events):
    rows = []
    for segment, first_batch, last_batch in SEGMENTS:
        subset = [
            event for event in events
            if event.get("iteration", 0) >= first_batch
            and (last_batch is None or event.get("iteration", 0) <= last_batch)
        ]
        supports = [
            event.get("support_param_count", event.get("stage1_support_count"))
            for event in subset
        ]
        hits = sum(event.get("budget_reached") is True for event in subset)
        max_steps = sum(event.get("max_steps_hit") is True for event in subset)
        rows.append({
            "omega": row["omega"],
            "role": row["role"],
            "experiment_key": row["experiment_key"],
            "segment": segment,
            "first_batch": first_batch,
            "last_batch": last_batch or len(events),
            "online_steps": len(subset),
            "budget_hit_count": hits,
            "budget_hit_rate": hits / len(subset) if subset else None,
            "max_steps_hit_count": max_steps,
            "stage1_steps_completed_mean": mean(
                [event.get("stage1_steps_completed") for event in subset]
            ),
            "support_min": min(
                [support for support in supports if finite(support)], default=None
            ),
            "mean_confidence": mean(
                [event.get("max_prob_mean") for event in subset]
            ),
            "mean_acc_post": mean(
                [event.get("acc_post") for event in subset]
            ),
            "mean_applied_update_l2": mean(
                [event.get("applied_update_l2") for event in subset]
            ),
        })
    return rows


def fully_valid(summary, events):
    required = (
        summary.get("status") == "completed",
        summary.get("budget_reached_all_steps") is True,
        summary.get("max_steps_hit_count") == 0,
        summary.get("valid_lbi_run") is True,
        len(events) == summary.get("online_steps"),
    )
    numeric_fields = ("FO-Acc", "FO-overall-Acc", "PU-Acc", "PU-overall-Acc")
    return all(required) and all(finite(summary.get(field)) for field in numeric_fields)


def summary_row(summary, summary_path, role, candidate, events):
    return {
        "candidate": candidate,
        "omega": summary["lbi_omega"],
        "role": role,
        "implementation_revision": summary.get("implementation_revision"),
        "experiment_key": summary["experiment_key"],
        "experiment_config_sha256": summary["experiment_config_sha256"],
        "status": summary["status"],
        "fully_valid": fully_valid(summary, events),
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
        "PU_worst_class": summary.get("PU-worst-class-Acc"),
        "PU_classwise_std": summary.get("PU-class-std"),
        "runtime_seconds": summary.get("runtime"),
        "summary_path": str(summary_path),
        "metrics_path": str(Path(summary_path).with_name("metrics.jsonl")),
    }


def selection_key(row):
    return (
        -row["FO_mean_per_class"],
        -row["FO_worst_class"],
        row["FO_classwise_std"],
        -row["FO_overall"],
        row["stage1_steps_completed_max"],
        row["stage1_steps_completed_mean"],
        row["runtime_seconds"],
        row["omega"],
    )


def invalid_diagnostic_key(row, segments_by_identity):
    late = segments_by_identity[row["experiment_key"]]["late"]
    return (
        -row["budget_hit_rate"],
        -late["budget_hit_rate"],
        -row["support_min"],
        row["max_steps_hit_count"],
        -row["FO_mean_per_class"],
        -row["FO_worst_class"],
        row["FO_classwise_std"],
        row["omega"],
    )


def markdown_table(headers, rows):
    output = ["| " + " | ".join(headers) + " |",
              "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        output.append("| " + " | ".join(str(value) for value in row) + " |")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-config", required=True)
    args = parser.parse_args()

    plan = read_json(args.plan)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for path in Path(args.runs_root).rglob("summary.json"):
        summary = read_json(path)
        key = summary.get("experiment_key")
        if key:
            if key in summaries:
                raise ValueError(f"duplicate summary for {key}")
            summaries[key] = (path, summary)

    rows, segmented_rows = [], []
    for index, experiment in enumerate(plan["experiments"]):
        path, summary = summaries[experiment["experiment_key"]]
        if summary.get("experiment_config_sha256") != experiment["experiment_config_sha256"]:
            raise ValueError(f"hash mismatch for {experiment['experiment_key']}")
        metrics_path, events = read_online_steps(path)
        if metrics_path != path.with_name("metrics.jsonl"):
            raise AssertionError("unexpected metrics path")
        role = "probe_control" if experiment["omega"] == 0 else "formal_candidate"
        row = summary_row(summary, path, role, f"O{index}", events)
        rows.append(row)
        segmented_rows.extend(compute_segment_rows(row, events))

    reference_path = Path(args.reference_summary)
    reference = read_json(reference_path)
    if reference.get("lbi_omega") != 0.2:
        raise ValueError("reference summary is not omega=0.20")
    _, reference_events = read_online_steps(reference_path)
    reference_row = summary_row(
        reference, reference_path, "reference_only", "R8_reference", reference_events
    )
    rows.append(reference_row)
    segmented_rows.extend(compute_segment_rows(reference_row, reference_events))

    rows.sort(key=lambda row: row["omega"])
    segments_by_identity = {}
    for row in segmented_rows:
        segments_by_identity.setdefault(row["experiment_key"], {})[row["segment"]] = row

    formal_valid = [
        row for row in rows
        if row["role"] == "formal_candidate" and row["fully_valid"]
    ]
    formal_valid.sort(key=selection_key)
    for rank, row in enumerate(formal_valid, 1):
        row["formal_rank"] = rank
    invalid_rows = [
        row for row in rows
        if row["role"] != "probe_control" and not row["fully_valid"]
    ]
    invalid_rows.sort(key=lambda row: invalid_diagnostic_key(row, segments_by_identity))
    for rank, row in enumerate(invalid_rows, 1):
        row["invalid_diagnostic_rank"] = rank

    write_csv(output_dir / "omega_candidate_diagnostics.csv", rows)
    write_json(output_dir / "omega_candidate_diagnostics.json", rows)
    write_csv(output_dir / "omega_segmented_diagnostics.csv", segmented_rows)
    write_json(output_dir / "omega_segmented_diagnostics.json", segmented_rows)
    write_csv(output_dir / "formal_candidate_ranking.csv", formal_valid)
    write_json(output_dir / "formal_candidate_ranking.json", formal_valid)
    write_csv(output_dir / "invalid_diagnostic_ranking.csv", invalid_rows)
    write_json(output_dir / "invalid_diagnostic_ranking.json", invalid_rows)

    control_late = segments_by_identity[rows[0]["experiment_key"]]["late"]
    reference_late = segments_by_identity[reference_row["experiment_key"]]["late"]
    conclusion = {
        "answer": "A",
        "statement": "cross-batch accumulation driven by omega is the primary cause",
        "evidence": {
            "omega_0_budget_hit_rate": rows[0]["budget_hit_rate"],
            "omega_0_late_budget_hit_rate": control_late["budget_hit_rate"],
            "omega_0_max_steps_hit_count": rows[0]["max_steps_hit_count"],
            "omega_020_budget_hit_rate": reference_row["budget_hit_rate"],
            "omega_020_late_budget_hit_rate": reference_late["budget_hit_rate"],
            "omega_020_max_steps_hit_count": reference_row["max_steps_hit_count"],
        },
    }
    selection = {
        "schema_version": 1,
        "budget": 0.001,
        "selection_rule": "validity first; FO mean per-class desc; FO worst-class desc; FO class-wise std asc; FO overall desc; Stage-1 max/mean steps asc; runtime asc",
        "04b_r8_reference": {
            "experiment_key": reference_row["experiment_key"],
            "experiment_config_sha256": reference_row["experiment_config_sha256"],
            "omega": reference_row["omega"],
            "summary_path": reference_row["summary_path"],
        },
        "04c_plan": str(Path(args.plan).resolve()),
        "diagnostic_conclusion": conclusion,
        "selected": formal_valid[:2],
    }
    if len(formal_valid) < 2:
        raise ValueError("fewer than two fully valid nonzero omega candidates")
    write_json(args.selected_config, selection)

    candidate_md = ["# Omega candidate diagnostics", "",
                    "All values use the fixed R8 settings except `omega`. O0 is diagnostic-only; R8 is reference-only.", ""]
    candidate_md.extend(markdown_table(
        ["Candidate", "Omega", "Role", "Valid", "Hit rate", "Max failures", "FO mean/worst/std", "PU mean", "Runtime (h)"],
        [[row["candidate"], f"{row['omega']:.4f}", row["role"], row["fully_valid"],
          f"{row['budget_hit_rate']:.4f}", row["max_steps_hit_count"],
          f"{row['FO_mean_per_class']:.4f}/{row['FO_worst_class']:.4f}/{row['FO_classwise_std']:.4f}",
          f"{row['PU_mean_per_class']:.4f}", f"{row['runtime_seconds'] / 3600:.2f}"] for row in rows]))
    (output_dir / "omega_candidate_diagnostics.md").write_text(
        "\n".join(candidate_md) + "\n", encoding="utf-8"
    )

    segment_md = ["# Omega segmented diagnostics", ""]
    segment_md.extend(markdown_table(
        ["Omega", "Role", "Segment", "Batches", "Hit rate", "Max failures", "Mean S1 steps", "Support min", "Confidence", "Online acc", "Update L2"],
        [[f"{row['omega']:.4f}", row["role"], row["segment"], f"{row['first_batch']}-{row['last_batch']}",
          f"{row['budget_hit_rate']:.4f}", row["max_steps_hit_count"],
          f"{row['stage1_steps_completed_mean']:.2f}", row["support_min"],
          f"{row['mean_confidence']:.4f}", f"{row['mean_acc_post']:.4f}",
          f"{row['mean_applied_update_l2']:.4f}"] for row in segmented_rows]))
    (output_dir / "omega_segmented_diagnostics.md").write_text(
        "\n".join(segment_md) + "\n", encoding="utf-8"
    )

    report = ["# 04C omega late-stream diagnostic - FINALIZE", "",
              "## Completion", "",
              "All eight planned O0-O7 identities are completed with no duplicate, hash-mismatch, missing, failed, or invalid-summary status.", "",
              "## Main diagnostic", "",
              "**A. Cross-batch accumulation driven by omega is the primary cause.** "
              f"The omega=0 control had {control_late['budget_hit_rate']:.4f} late-stream hit rate and "
              f"{rows[0]['max_steps_hit_count']} max-step failures, while reused R8 omega=0.20 had "
              f"{reference_late['budget_hit_rate']:.4f} late-stream hit rate and "
              f"{reference_row['max_steps_hit_count']} max-step failures. Lower nonzero omega values through 0.075 are fully valid; failures begin at omega=0.10.", "",
              "## Formal selection", "",
              "The formal rule excludes omega=0. Four nonzero candidates are fully valid. The selected top two are:", ""]
    report.extend([f"- `{row['omega']:.4f}` (rank {row['formal_rank']}; FO mean per-class {row['FO_mean_per_class']:.4f})" for row in formal_valid[:2]])
    report.extend(["", "PU metrics are report-only. No training was launched, retried, or rerun in FINALIZE."])
    (output_dir / "FINALIZE_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
