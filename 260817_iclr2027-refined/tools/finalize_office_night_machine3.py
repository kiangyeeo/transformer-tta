#!/usr/bin/env python3
"""Finalize Office Night Machine 3 from existing summaries and batch logs."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.check_experiment_status import check_status, scan_run_summaries


K = 1049
TRANSFERS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
ANCHORS = (
    ("A1", 0.10, 1.0, 0.50),
    ("A2", 0.15, 1.0, 0.50),
    ("A3", 0.20, 1.0, 0.50),
    ("A4", 0.10, 1.5, 0.50),
    ("A5", 0.10, 2.0, 0.50),
    ("A6", 0.10, 1.0, 0.25),
    ("A7", 0.10, 1.0, 1.00),
    ("A8", 0.15, 1.0, 1.00),
)
ANCHOR_BY_TUPLE = {
    (alpha, kappa, nu): candidate_id
    for candidate_id, alpha, kappa, nu in ANCHORS
}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _number(value):
    if value is None or value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value: {value}")
    return number


def _percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _plans(root):
    return {
        "baseline_machine1": root / "plans/baseline_machine1/plan.json",
        "baseline_machine2": root / "plans/baseline_machine2/plan.json",
        "baseline_machine3": root / "plans/machine3/baseline/plan.json",
        "stage1_machine3": root / "plans/machine3/stage1/plan.json",
    }


def _summary_index(runs_root):
    records, invalid = scan_run_summaries(str(runs_root))
    index = {}
    for record in records:
        identity = (
            record["experiment_key"],
            record["experiment_config_sha256"],
        )
        index.setdefault(identity, []).append(record)
    return index, invalid


def _status_snapshot(plan_path, runs_root):
    result = check_status(_read_json(plan_path), str(runs_root))
    return result["status_counts"], result


def _launcher_failed_count(root, phase):
    manifest = _read_json(
        root / f"launcher_logs/machine3/{phase}/launcher_manifest.json"
    )
    return sum(
        record.get("status") != "completed"
        for record in manifest.get("records", [])
    )


def _planned_summary(plan_entry, summary_index):
    identity = (
        plan_entry["experiment_key"],
        plan_entry["experiment_config_sha256"],
    )
    records = summary_index.get(identity, [])
    completed = [record for record in records if record["status"] == "completed"]
    if len(completed) != 1:
        raise ValueError(
            f"expected one completed summary for {identity}, found {len(completed)}"
        )
    summary_path = Path(completed[0]["summary_path"])
    return summary_path, _read_json(summary_path)


def _metrics(summary_path):
    metrics_path = summary_path.with_name("metrics.jsonl")
    online = []
    errors = []
    finite = True
    with metrics_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            finite = finite and _finite(record)
            if record.get("event") == "online_step":
                online.append(record)
            if record.get("event") == "error":
                errors.append({"line": line_number, "record": record})
    if not online:
        raise ValueError(f"no online_step records in {metrics_path}")
    return metrics_path, online, errors, finite


def _run_diagnostics(plan_entry, summary_path, summary):
    metrics_path, online, errors, finite = _metrics(summary_path)
    selected = [_number(step.get("selected_param_count")) for step in online]
    utilizations = [value / K for value in selected]
    gaps = [K - value for value in selected]
    rollback_count = sum(bool(step.get("stage1_rollback_used")) for step in online)
    max_step_failures = sum(
        bool(step.get("max_steps_hit"))
        or step.get("stage1_stop_reason") in {"max_steps", "max_steps_reached"}
        for step in online
    )
    budget_violations = sum(value > K for value in selected)
    no_nan_inf_or_error = finite and not errors
    scientific_valid = bool(
        summary.get("status") == "completed"
        and no_nan_inf_or_error
        and budget_violations == 0
        and max_step_failures == 0
        and summary.get("valid_lbi_run") is True
        and len(online) == int(summary.get("online_steps", len(online)))
    )
    hard_gate = scientific_valid and all(value >= 0.90 for value in utilizations)
    stage1_steps = [_number(step.get("stage1_steps_completed")) for step in online]
    return {
        "candidate_id": plan_entry["candidate_id"],
        "experiment_key": plan_entry["experiment_key"],
        "experiment_config_sha256": plan_entry["experiment_config_sha256"],
        "summary_path": str(summary_path),
        "metrics_path": str(metrics_path),
        "source": plan_entry["source"],
        "target": plan_entry["target"],
        "alpha": plan_entry["alpha"],
        "kappa": plan_entry["kappa"],
        "nu": plan_entry["nu"],
        "omega": plan_entry["omega"],
        "stage2_lr": plan_entry["stage2_lr"],
        "online_steps": len(online),
        "no_nan_inf_or_error": no_nan_inf_or_error,
        "budget_violations": budget_violations,
        "max_steps_failure_count": max_step_failures,
        "scientific_valid": scientific_valid,
        "hard_90pct_valid": hard_gate,
        "selected_count_min": min(selected),
        "selected_count_mean": sum(selected) / len(selected),
        "selected_count_max": max(selected),
        "support_utilization_mean": sum(utilizations) / len(utilizations),
        "support_utilization_min": min(utilizations),
        "support_utilization_p05": _percentile(utilizations, 0.05),
        "under_95pct_batch_count": sum(value < 0.95 for value in utilizations),
        "under_90pct_batch_count": sum(value < 0.90 for value in utilizations),
        "mean_budget_gap": sum(gaps) / len(gaps),
        "max_budget_gap": max(gaps),
        "rollback_count": rollback_count,
        "rollback_rate": rollback_count / len(online),
        "stage1_steps_mean": sum(stage1_steps) / len(stage1_steps),
        "stage1_steps_max": max(stage1_steps),
        "max_steps_hit_count": max_step_failures,
        "PU": summary.get("PU-Acc"),
        "FO": summary.get("FO-Acc"),
    }


def _baseline_reference(plan_paths, runs_root, summary_index):
    rows = []
    for machine, plan_path in plan_paths.items():
        plan = _read_json(plan_path)
        for entry in plan["experiments"]:
            summary_path, summary = _planned_summary(entry, summary_index)
            rows.append(
                {
                    "machine": machine,
                    "experiment_key": entry["experiment_key"],
                    "experiment_config_sha256": entry["experiment_config_sha256"],
                    "source": entry["source"],
                    "target": entry["target"],
                    "transfer": f"{entry['source']}->{entry['target']}",
                    "variant": entry["variant"],
                    "requested_budget": entry["requested_budget"],
                    "PU": summary.get("PU-Acc"),
                    "FO": summary.get("FO-Acc"),
                    "summary_path": str(summary_path),
                }
            )
    rows.sort(key=lambda row: (row["source"], row["target"], row["variant"], str(row["requested_budget"])))
    return rows


def _baseline_sparse_reference(rows):
    grouped = {}
    for row in rows:
        key = (row["source"], row["target"], float(row["requested_budget"] or 0.0))
        grouped.setdefault(key, {})[row["variant"]] = row
    output = {}
    for (source, target, budget), variants in grouped.items():
        if {"module_random", "module_magnitude", "module_saliency"} <= set(variants):
            candidates = {
                "module_random": variants["module_random"]["FO"],
                "module_magnitude": variants["module_magnitude"]["FO"],
                "module_saliency": variants["module_saliency"]["FO"],
            }
            best_variant = max(candidates, key=lambda variant: candidates[variant])
            output[(source, target, budget)] = {
                "best_sparse_variant": best_variant,
                "best_sparse_FO": candidates[best_variant],
                "random_3mask_mean_FO": candidates["module_random"],
                "magnitude_FO": candidates["module_magnitude"],
                "saliency_FO": candidates["module_saliency"],
            }
    return output


def _candidate_aggregate(rows, baseline_reference_complete, sparse_reference):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["candidate_id"], []).append(row)
    aggregates = []
    for candidate_id, candidate_rows in grouped.items():
        candidate_rows.sort(key=lambda row: (row["source"], row["target"]))
        utilizations = []
        gaps = []
        steps = []
        for row in candidate_rows:
            utilizations.extend(row["_utilizations"])
            gaps.extend(row["_gaps"])
            steps.extend(row["_stage1_steps"])
        margins = []
        for row in candidate_rows:
            reference = sparse_reference.get((row["source"], row["target"], 0.002))
            if reference is not None and row["FO"] is not None:
                margins.append(row["FO"] - reference["best_sparse_FO"])
        aggregate = {
            "candidate_id": candidate_id,
            "alpha": candidate_rows[0]["alpha"],
            "kappa": candidate_rows[0]["kappa"],
            "nu": candidate_rows[0]["nu"],
            "omega": candidate_rows[0]["omega"],
            "stage2_lr": candidate_rows[0]["stage2_lr"],
            "completed_transfer_count": len(candidate_rows),
            "scientific_valid_transfer_count": sum(row["scientific_valid"] for row in candidate_rows),
            "hard_90pct_valid_transfer_count": sum(row["hard_90pct_valid"] for row in candidate_rows),
            "global_min_utilization": min(utilizations),
            "aggregate_p05_utilization": _percentile(utilizations, 0.05),
            "aggregate_mean_utilization": sum(utilizations) / len(utilizations),
            "under_95pct_batch_count": sum(row["under_95pct_batch_count"] for row in candidate_rows),
            "under_90pct_batch_count": sum(row["under_90pct_batch_count"] for row in candidate_rows),
            "mean_budget_gap": sum(gaps) / len(gaps),
            "max_budget_gap": max(gaps),
            "rollback_rate": sum(row["rollback_count"] for row in candidate_rows) / len(utilizations),
            "stage1_steps_mean": sum(steps) / len(steps),
            "stage1_steps_max": max(steps),
            "max_steps_hit_count": sum(row["max_steps_hit_count"] for row in candidate_rows),
            "mean_PU": sum(row["PU"] for row in candidate_rows) / len(candidate_rows),
            "mean_FO": sum(row["FO"] for row in candidate_rows) / len(candidate_rows),
            "baseline_reference_complete": baseline_reference_complete,
            "mean_FO_margin": sum(margins) / len(margins) if margins else None,
            "worst_transfer_FO_margin": min(margins) if margins else None,
        }
        aggregates.append(aggregate)
    return aggregates


def _rank(aggregates):
    baseline_complete = bool(aggregates and aggregates[0]["baseline_reference_complete"])

    def descending(value):
        return -(float(value) if value is not None else float("-inf"))

    def ascending(value):
        return float(value) if value is not None else float("inf")

    def key(row):
        fields = [
            -row["scientific_valid_transfer_count"],
            -row["hard_90pct_valid_transfer_count"],
            descending(row["global_min_utilization"]),
            descending(row["aggregate_p05_utilization"]),
            descending(row["aggregate_mean_utilization"]),
            row["under_95pct_batch_count"],
            row["max_budget_gap"],
            row["mean_budget_gap"],
        ]
        if baseline_complete:
            fields.extend(
                [
                    descending(row["mean_FO_margin"]),
                    descending(row["worst_transfer_FO_margin"]),
                ]
            )
        fields.extend(
            [
                descending(row["mean_FO"]),
                row["stage1_steps_max"],
                row["stage1_steps_mean"],
                row["candidate_id"],
            ]
        )
        return tuple(fields)

    ranked = sorted(aggregates, key=key)
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    runs_root = root / "runs"
    paths = _plans(root)
    summary_index, invalid_records = _summary_index(runs_root)

    status = {}
    for name, plan_path in paths.items():
        status[name] = _status_snapshot(plan_path, runs_root)[0]
    status["baseline_machine3"]["failed"] = _launcher_failed_count(
        root, "baseline"
    )
    status["stage1_machine3"]["failed"] = _launcher_failed_count(
        root, "stage1"
    )

    baseline_rows = _baseline_reference(
        {name: paths[name] for name in ("baseline_machine1", "baseline_machine2", "baseline_machine3")},
        runs_root,
        summary_index,
    )
    baseline_reference_complete = len(baseline_rows) == 72 and not invalid_records
    sparse_reference = _baseline_sparse_reference(baseline_rows)

    stage1_plan = _read_json(paths["stage1_machine3"])
    stage1_rows = []
    for entry in stage1_plan["experiments"]:
        candidate_id = ANCHOR_BY_TUPLE[
            (float(entry["alpha"]), float(entry["kappa"]), float(entry["nu"]))
        ]
        entry = dict(entry)
        entry["candidate_id"] = candidate_id
        summary_path, summary = _planned_summary(entry, summary_index)
        row = _run_diagnostics(entry, summary_path, summary)
        _, online, _, _ = _metrics(summary_path)
        row["_utilizations"] = [float(step["selected_param_count"]) / K for step in online]
        row["_gaps"] = [K - float(step["selected_param_count"]) for step in online]
        row["_stage1_steps"] = [float(step["stage1_steps_completed"]) for step in online]
        stage1_rows.append(row)

    candidate_rows = _candidate_aggregate(
        stage1_rows,
        baseline_reference_complete,
        sparse_reference,
    )
    ranked = _rank(candidate_rows)

    results_root = root / "results/lbi_stage1/budget_002/machine3"
    public_run_fields = [
        key for key in stage1_rows[0]
        if not key.startswith("_")
    ]
    _write_csv(results_root / "per_run_diagnostics.csv", stage1_rows, public_run_fields)
    _write_json(results_root / "per_run_diagnostics.json", stage1_rows)
    _write_csv(results_root / "candidate_ranking.csv", ranked, list(ranked[0]))
    _write_json(results_root / "candidate_ranking.json", ranked)
    _write_csv(
        results_root / "baseline_reference_seed2026.csv",
        baseline_rows,
        list(baseline_rows[0]),
    )
    _write_json(results_root / "baseline_reference_seed2026.json", baseline_rows)
    _write_json(
        results_root / "finalize_status.json",
        {
            "status": status,
            "baseline_reference_complete": baseline_reference_complete,
            "baseline_reference_count": len(baseline_rows),
            "stage1_count": len(stage1_rows),
            "invalid_scanned_records": len(invalid_records),
            "K": K,
            "ranking_rule": [
                "scientific_valid_transfer_count desc",
                "hard_90pct_valid_transfer_count desc",
                "global_min_utilization desc",
                "aggregate_p05_utilization desc",
                "aggregate_mean_utilization desc",
                "under_95pct_batch_count asc",
                "max_budget_gap asc",
                "mean_budget_gap asc",
                "mean_FO_margin desc if baseline reference complete",
                "worst_transfer_FO_margin desc if baseline reference complete",
                "mean_FO desc",
                "stage1_steps_max asc",
                "stage1_steps_mean asc",
            ],
        },
    )

    report = root / "phase_records/machine3/FINALIZE.md"
    lines = [
        "# FINALIZE — Office Night Machine 3",
        "",
        "Date: 2026-08-18",
        "Mode: `FINALIZE`",
        "Machine: `machine3`",
        "",
        "## No rerun policy",
        "",
        "No launch, retry, resume, or training rerun was performed in FINALIZE.",
        "",
        "## Completion status",
        "",
        "| plan | completed | missing | failed | duplicate | hash mismatch | invalid summary |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline_machine3", "stage1_machine3"):
        counts = status[name]
        lines.append(
            f"| {name} | {counts.get('completed', 0)} | {counts.get('missing', 0)} | "
            f"{counts.get('failed', 0)} | "
            f"{counts.get('duplicate_completed', 0)} | {counts.get('hash_mismatch', 0)} | "
            f"{counts.get('invalid_summary', 0)} |"
        )
    lines.extend(
        [
            "",
            "Baseline machine3 is complete: 24/24. Stage-1 machine3 is complete: 48/48.",
            "",
            "## Baseline reference",
            "",
            f"The new seed-2026 baseline reference contains {len(baseline_rows)}/72 planned identities across machines 1–3; `baseline_reference_complete={str(baseline_reference_complete).lower()}`.",
            "Only seed-2026 summaries were used. For each transfer and budget 0.002, sparse reference is the maximum of Random 3-mask mean, Magnitude, and Saliency.",
            "",
            "## Stage-1 candidate ranking",
            "",
            "| rank | candidate | alpha | kappa | nu | valid transfers | hard 90% transfers | global min u | aggregate p05 u | mean u | under 95% | under 90% | mean gap | max gap | rollback rate | mean FO | mean FO margin | worst FO margin | stage1 max | stage1 mean |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ranked:
        values = [
            row["rank"], row["candidate_id"], f"{row['alpha']:.2f}",
            f"{row['kappa']:.2f}", f"{row['nu']:.2f}",
            f"{row['scientific_valid_transfer_count']}/6",
            f"{row['hard_90pct_valid_transfer_count']}/6",
            f"{row['global_min_utilization']:.6f}",
            f"{row['aggregate_p05_utilization']:.6f}",
            f"{row['aggregate_mean_utilization']:.6f}",
            row["under_95pct_batch_count"], row["under_90pct_batch_count"],
            f"{row['mean_budget_gap']:.3f}", row["max_budget_gap"],
            f"{row['rollback_rate']:.6f}", f"{row['mean_FO']:.6f}",
            "" if row["mean_FO_margin"] is None else f"{row['mean_FO_margin']:.6f}",
            "" if row["worst_transfer_FO_margin"] is None else f"{row['worst_transfer_FO_margin']:.6f}",
            f"{row['stage1_steps_max']:.3f}", f"{row['stage1_steps_mean']:.3f}",
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    hard_winners = [row for row in ranked if row["hard_90pct_valid_transfer_count"] == 6]
    lines.extend(
        [
            "",
            f"Hard utilization gate candidates: {', '.join(row['candidate_id'] for row in hard_winners) or 'none'}.",
            "",
            "## Required conclusion",
            "",
            "No Stage-1 configuration is ready for omega tuning." if not hard_winners else f"Stage-1 configurations passing the hard gate: {', '.join(row['candidate_id'] for row in hard_winners)}.",
            "No candidate was automatically promoted and no omega search was launched.",
            "",
            "## Scientifically invalid completed runs",
            "",
            "Completed runs below the scientific validity gate were retained as evidence and were not rerun:",
            "",
            "| candidate | transfer | experiment_key | reason | summary |",
            "|---|---|---|---|---|",
        ]
    )
    invalid_rows = [row for row in stage1_rows if not row["scientific_valid"]]
    for row in invalid_rows:
        reasons = []
        if row["max_steps_failure_count"]:
            reasons.append(
                f"max_steps_failure_count={row['max_steps_failure_count']}"
            )
        if row["budget_violations"]:
            reasons.append(f"budget_violations={row['budget_violations']}")
        if not row["no_nan_inf_or_error"]:
            reasons.append("NaN/Inf/error event")
        if not reasons:
            reasons.append("valid_lbi_run=false")
        lines.append(
            f"| {row['candidate_id']} | {row['source']}->{row['target']} | "
            f"`{row['experiment_key']}` | {'; '.join(reasons)} | "
            f"`{row['summary_path']}` |"
        )
    if not invalid_rows:
        lines.append("| none | — | — | — | — |")
    lines.extend(
        [
            "",
            "## Output files",
            "",
            f"- Status: `{results_root / 'finalize_status.json'}`",
            f"- Machine3 baseline summary: `{root / 'results/baseline/machine3'}`",
            f"- Per-run diagnostics: `{results_root / 'per_run_diagnostics.csv'}`",
            f"- Candidate ranking: `{results_root / 'candidate_ranking.csv'}`",
            f"- Baseline reference: `{results_root / 'baseline_reference_seed2026.csv'}`",
            "- Metrics source: each completed Stage-1 run's `metrics.jsonl` online-step records.",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"baseline_reference_complete": baseline_reference_complete, "stage1_runs": len(stage1_rows), "hard_gate_candidates": [row["candidate_id"] for row in hard_winners], "ranking": [{"rank": row["rank"], "candidate_id": row["candidate_id"], "scientific_valid_transfer_count": row["scientific_valid_transfer_count"], "hard_90pct_valid_transfer_count": row["hard_90pct_valid_transfer_count"]} for row in ranked]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
