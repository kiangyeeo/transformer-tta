#!/usr/bin/env python3
"""Finalize the Office machine2 / rho=0.001 Stage-1 shard.

This utility is read-only with respect to training artifacts.  It consumes the
plans, status outputs, summaries, and per-batch metrics produced by the
non-interactive launcher and writes only FINALIZE reports.
"""

import csv
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiment_logs" / "shot_otta_office_seed2026_stage1_20260818"
RUNS = EXP / "runs"
K = 524
TRANSFERS = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
ANCHORS = {
    (0.10, 1.0, 0.50): "A1",
    (0.15, 1.0, 0.50): "A2",
    (0.20, 1.0, 0.50): "A3",
    (0.10, 1.5, 0.50): "A4",
    (0.10, 2.0, 0.50): "A5",
    (0.10, 1.0, 0.25): "A6",
    (0.10, 1.0, 1.00): "A7",
    (0.15, 1.0, 1.00): "A8",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def pctl(values, q):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    return float(values[lo] + (values[hi] - values[lo]) * (pos - lo))


def status(path):
    data = load(path)
    return data, {row["experiment_key"]: row for row in data["experiments"]}


def add_failed_count(status_data, plan):
    planned = {row["experiment_key"] for row in plan["experiments"]}
    failed = 0
    for path in RUNS.rglob("summary.json"):
        try:
            summary = load(path)
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("experiment_key") in planned and summary.get("status") in {"failed", "incomplete"}:
            failed += 1
    result = dict(status_data)
    result["status_counts"] = dict(status_data["status_counts"])
    result["status_counts"]["failed"] = failed
    return result


def plan_rows(plan_path, status_path):
    plan = load(plan_path)
    status_data, by_key = status(status_path)
    rows = []
    for exp in plan["experiments"]:
        row = by_key[exp["experiment_key"]]
        rows.append((exp, row))
    return plan, status_data, rows


def read_summary(path):
    return load(path)


def read_online_metrics(summary_path):
    metric_path = Path(summary_path).parent / "metrics.jsonl"
    records = []
    parse_errors = []
    if not metric_path.exists():
        return records, [f"missing metrics.jsonl: {metric_path}"]
    for line_no, line in enumerate(metric_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            parse_errors.append(f"{metric_path}:{line_no}: {exc}")
            continue
        if record.get("event") == "online_step":
            records.append(record)
        if not finite(record):
            parse_errors.append(f"non-finite value: {metric_path}:{line_no}")
    return records, parse_errors


def baseline_rows():
    plans = [
        (EXP / "plans" / "baseline_machine1" / "plan.json", EXP / "status" / "all_machines" / "baseline_machine1" / "experiment_status.json"),
        (EXP / "plans" / "baseline_machine2" / "plan.json", EXP / "status" / "machine2" / "baseline" / "experiment_status.json"),
        (EXP / "plans" / "machine3" / "baseline" / "plan.json", EXP / "status" / "all_machines" / "baseline_machine3" / "experiment_status.json"),
    ]
    all_rows = []
    for plan_path, status_path in plans:
        plan, status_data, rows = plan_rows(plan_path, status_path)
        all_rows.extend((plan_path, exp, st, status_data) for exp, st in rows)
    return all_rows


def baseline_reference(all_baselines):
    ref = {}
    for plan_path, exp, st, _ in all_baselines:
        source, target = int(exp["source"]), int(exp["target"])
        summary_paths = st.get("matching_summary_paths", [])
        if st["plan_status"] != "completed" or len(summary_paths) != 1:
            continue
        summary = read_summary(summary_paths[0])
        key = (source, target, int(exp["seed"]), float(exp["requested_budget"] or 0.0))
        bucket = ref.setdefault(key, [])
        if exp["variant"] == "module_random":
            fo = summary.get("mean_FO-Acc")
        elif exp["variant"] in {"module_magnitude", "module_saliency"}:
            fo = summary.get("FO-Acc")
        else:
            fo = None
        if fo is not None:
            bucket.append({"variant": exp["variant"], "FO-Acc": float(fo), "summary_path": summary_paths[0], "experiment_key": exp["experiment_key"]})
    result = {}
    for key, entries in ref.items():
        sparse = [x for x in entries if x["variant"] in {"module_random", "module_magnitude", "module_saliency"}]
        if len(sparse) < 3:
            continue
        best = max(sparse, key=lambda x: x["FO-Acc"])
        result[key] = {"best_sparse_FO": best["FO-Acc"], "best_sparse_variant": best["variant"], "entries": sparse}
    return result


def make_baseline_report(machine_rows, all_baselines, reference):
    report_rows = []
    for exp, st in machine_rows:
        path = st.get("matching_summary_paths", [None])[0]
        summary = read_summary(path) if path else {}
        report_rows.append({
            "experiment_key": exp["experiment_key"],
            "experiment_config_sha256": exp["experiment_config_sha256"],
            "source": exp["source"], "target": exp["target"],
            "transfer": f"{exp['source']}->{exp['target']}",
            "variant": exp["variant"],
            "requested_budget": exp.get("requested_budget"),
            "selection_seed": exp.get("selection_seed"),
            "plan_status": st["plan_status"],
            "summary_path": path,
            "PU": summary.get("PU-Acc"),
            "FO": summary.get("mean_FO-Acc", summary.get("FO-Acc")),
            "online_batch_runtime_mean_sec": summary.get("online_batch_runtime_mean_sec"),
            "online_batch_runtime_p95_sec": summary.get("online_batch_runtime_p95_sec"),
            "online_compute_runtime_sec": summary.get("online_compute_runtime_sec"),
            "adapt_runtime_total_sec": summary.get("adapt_runtime_total_sec"),
            "pu_runtime_total_sec": summary.get("pu_runtime_total_sec"),
            "gpu_peak_allocated_mean_mb": summary.get("gpu_peak_allocated_mean_mb"),
            "gpu_peak_allocated_max_mb": summary.get("gpu_peak_allocated_max_mb"),
            "gpu_peak_reserved_mean_mb": summary.get("gpu_peak_reserved_mean_mb"),
            "gpu_peak_reserved_max_mb": summary.get("gpu_peak_reserved_max_mb"),
            "runtime_comparable": summary.get("runtime_comparable"),
            "efficiency_protocol_revision": summary.get("efficiency_protocol_revision"),
        })
    fields = list(report_rows[0]) if report_rows else []
    return report_rows, fields


def stage1_run(exp, st):
    path = st.get("matching_summary_paths", [None])[0]
    summary = read_summary(path) if path else {}
    metrics, errors = read_online_metrics(path) if path else ([], ["missing summary"])
    selected = []
    stage_steps = []
    rollback = []
    max_hits = []
    valid_steps = []
    violation = []
    for record in metrics:
        count = record.get("support_param_count", record.get("stage1_support_count", record.get("selected_param_count")))
        if count is None:
            errors.append(f"missing selected/support count at batch {record.get('batch_index')}")
            continue
        count = float(count)
        selected.append(count)
        stage_steps.append(float(record.get("stage1_steps_completed")) if record.get("stage1_steps_completed") is not None else float("nan"))
        rollback.append(bool(record.get("stage1_rollback_used", False)))
        max_hits.append(bool(record.get("max_steps_hit", False)))
        valid_steps.append(bool(record.get("valid_lbi_step", False)))
        if count > K:
            violation.append(count)
    util = [x / K for x in selected]
    gaps = [K - x for x in selected]
    scientific_valid = bool(
        st["plan_status"] == "completed" and summary.get("status") == "completed" and metrics
        and not errors and not violation and not any(max_hits)
        and all(valid_steps) and all(math.isfinite(x) for x in stage_steps)
    )
    hard_valid = bool(scientific_valid and all(x >= 0.90 for x in util))
    anchor = ANCHORS[(round(float(exp["alpha"]), 2), round(float(exp["kappa"]), 2), round(float(exp["nu"]), 2))]
    return {
        "candidate_id": anchor,
        "alpha": exp["alpha"], "kappa": exp["kappa"], "nu": exp["nu"],
        "omega": exp["omega"], "stage2_lr": exp["stage2_lr"],
        "source": exp["source"], "target": exp["target"],
        "transfer": f"{exp['source']}->{exp['target']}",
        "experiment_key": exp["experiment_key"],
        "experiment_config_sha256": exp["experiment_config_sha256"],
        "summary_path": path,
        "metrics_path": str(Path(path).parent / "metrics.jsonl") if path else None,
        "plan_status": st["plan_status"],
        "completed": st["plan_status"] == "completed",
        "scientific_valid": scientific_valid,
        "hard_90pct_valid": hard_valid,
        "error_count": len(errors),
        "errors": errors,
        "batch_count": len(selected),
        "support_utilization_mean": sum(util) / len(util) if util else None,
        "support_utilization_min": min(util) if util else None,
        "support_utilization_p05": pctl(util, 0.05),
        "under_95pct_batch_count": sum(x < 0.95 for x in util),
        "under_90pct_batch_count": sum(x < 0.90 for x in util),
        "mean_budget_gap": sum(gaps) / len(gaps) if gaps else None,
        "max_budget_gap": max(gaps) if gaps else None,
        "rollback_count": sum(rollback),
        "rollback_rate": sum(rollback) / len(rollback) if rollback else None,
        "stage1_steps_mean": sum(stage_steps) / len(stage_steps) if stage_steps else None,
        "stage1_steps_max": max(stage_steps) if stage_steps else None,
        "stage1_max_steps_hit_count": sum(max_hits),
        "PU": summary.get("PU-Acc"), "FO": summary.get("FO-Acc"),
        "requested_budget": exp["requested_budget"], "K": K,
        "stage1_max_steps": exp["stage1_max_steps"],
        "batch_size": exp.get("effective_overrides", {}).get("data", {}).get("batch_size", 64),
        "seed": exp["seed"], "variant": exp["variant"],
    }


def candidate_ranking(per_run, reference_complete, reference):
    groups = {}
    for row in per_run:
        groups.setdefault(row["candidate_id"], []).append(row)
    ranked = []
    for candidate_id, rows in groups.items():
        utils = []
        gaps = []
        for row in rows:
            summary = read_summary(row["summary_path"])
            metrics, _ = read_online_metrics(row["summary_path"])
            for rec in metrics:
                count = rec.get("support_param_count", rec.get("stage1_support_count", rec.get("selected_param_count")))
                if count is not None:
                    utils.append(float(count) / K)
                    gaps.append(K - float(count))
        vals = {k: rows[0][k] for k in ("candidate_id", "alpha", "kappa", "nu", "omega", "stage2_lr")}
        vals.update({
            "completed_transfer_count": sum(r["completed"] for r in rows),
            "scientific_valid_transfer_count": sum(r["scientific_valid"] for r in rows),
            "hard_90pct_valid_transfer_count": sum(r["hard_90pct_valid"] for r in rows),
            "global_min_utilization": min(utils) if utils else None,
            "aggregate_p05_utilization": pctl(utils, 0.05),
            "aggregate_mean_utilization": sum(utils) / len(utils) if utils else None,
            "under_95pct_batch_count": sum(r["under_95pct_batch_count"] for r in rows),
            "under_90pct_batch_count": sum(r["under_90pct_batch_count"] for r in rows),
            "mean_budget_gap": sum(gaps) / len(gaps) if gaps else None,
            "max_budget_gap": max(gaps) if gaps else None,
            "rollback_rate": sum(r["rollback_count"] for r in rows) / sum(r["batch_count"] for r in rows) if sum(r["batch_count"] for r in rows) else None,
            "stage1_steps_mean": sum(r["stage1_steps_mean"] * r["batch_count"] for r in rows if r["stage1_steps_mean"] is not None) / sum(r["batch_count"] for r in rows if r["stage1_steps_mean"] is not None) if any(r["stage1_steps_mean"] is not None for r in rows) else None,
            "stage1_steps_max": max((r["stage1_steps_max"] for r in rows if r["stage1_steps_max"] is not None), default=None),
            "max_steps_hit_count": sum(r["stage1_max_steps_hit_count"] for r in rows),
            "mean_PU": sum(r["PU"] for r in rows if r["PU"] is not None) / sum(r["PU"] is not None for r in rows) if any(r["PU"] is not None for r in rows) else None,
            "mean_FO": sum(r["FO"] for r in rows if r["FO"] is not None) / sum(r["FO"] is not None for r in rows) if any(r["FO"] is not None for r in rows) else None,
            "transfer_rows": [r["experiment_key"] for r in rows],
            "hard_gate_failures": [r["transfer"] for r in rows if not r["hard_90pct_valid"]],
        })
        margins = []
        for r in rows:
            key = (int(r["source"]), int(r["target"]), 2026, 0.001)
            if reference_complete and key in reference and r["FO"] is not None:
                margins.append((r["FO"] - reference[key]["best_sparse_FO"], r["transfer"]))
        vals["baseline_reference_complete"] = reference_complete
        vals["mean_FO_margin"] = sum(x[0] for x in margins) / len(margins) if margins else None
        vals["worst_transfer_FO_margin"] = min((x[0] for x in margins), default=None)
        ranked.append(vals)

    def low(value): return float("-inf") if value is None else value
    ranked.sort(key=lambda r: (
        -r["scientific_valid_transfer_count"],
        -r["hard_90pct_valid_transfer_count"],
        -low(r["global_min_utilization"]),
        -low(r["aggregate_p05_utilization"]),
        -low(r["aggregate_mean_utilization"]),
        r["under_95pct_batch_count"], r["max_budget_gap"], r["mean_budget_gap"],
        -low(r["mean_FO_margin"] if reference_complete else None),
        -low(r["worst_transfer_FO_margin"] if reference_complete else None),
        -low(r["mean_FO"]),
        r["stage1_steps_max"] if r["stage1_steps_max"] is not None else float("inf"),
        r["stage1_steps_mean"] if r["stage1_steps_mean"] is not None else float("inf"),
    ))
    for i, row in enumerate(ranked, 1):
        row["rank"] = i
    return ranked


def main():
    base_plan = EXP / "plans" / "baseline_machine2" / "plan.json"
    base_status = EXP / "status" / "machine2" / "baseline" / "experiment_status.json"
    lbi_plan = EXP / "plans" / "lbi_stage1_machine2" / "plan.json"
    lbi_status = EXP / "status" / "machine2" / "stage1" / "experiment_status.json"
    base_plan_data, base_status_data, base_rows_machine2 = plan_rows(base_plan, base_status)
    lbi_plan_data, lbi_status_data, lbi_rows = plan_rows(lbi_plan, lbi_status)
    base_status_data = add_failed_count(base_status_data, base_plan_data)
    lbi_status_data = add_failed_count(lbi_status_data, lbi_plan_data)
    all_baselines = baseline_rows()
    reference = baseline_reference(all_baselines)
    reference_complete = all(
        (s["plan_status"] == "completed" and len(s.get("matching_summary_paths", [])) == 1)
        for _, _, s, _ in all_baselines
    ) and len(all_baselines) == 72 and all((source, target, 2026, 0.001) in reference for source, target in TRANSFERS)

    reports = EXP / "reports" / "machine2"
    results_base = EXP / "results" / "baseline" / "machine2"
    results_lbi = EXP / "results" / "lbi_stage1" / "budget_001"
    phase = EXP / "phase_records" / "machine2"
    reports.mkdir(parents=True, exist_ok=True)
    results_base.mkdir(parents=True, exist_ok=True)
    results_lbi.mkdir(parents=True, exist_ok=True)

    base_report_rows, base_fields = make_baseline_report(base_rows_machine2, all_baselines, reference)
    write_csv(reports / "baseline_summary.csv", base_report_rows, base_fields)
    transfer_aggregates = {}
    for transfer in ((1, 0), (1, 2)):
        rows = [r for r in base_report_rows if (int(r["source"]), int(r["target"])) == transfer]
        transfer_aggregates[f"{transfer[0]}->{transfer[1]}"] = {
            "count": len(rows),
            "mean_PU": sum(r["PU"] for r in rows if r["PU"] is not None) / sum(r["PU"] is not None for r in rows),
            "mean_FO": sum(r["FO"] for r in rows if r["FO"] is not None) / sum(r["FO"] is not None for r in rows),
            "runtime_comparable_count": sum(r["runtime_comparable"] is True for r in rows),
            "online_batch_runtime_mean_sec_mean": sum(r["online_batch_runtime_mean_sec"] for r in rows if r["online_batch_runtime_mean_sec"] is not None) / sum(r["online_batch_runtime_mean_sec"] is not None for r in rows),
            "online_compute_runtime_sec_sum": sum(r["online_compute_runtime_sec"] or 0 for r in rows),
        }
    baseline_json = {
        "machine": "machine2", "seed": 2026, "transfers": ["D->A", "D->W"],
        "status": base_status_data["status_counts"], "planned_count": 24,
        "reference_all_three_machines": {"complete": reference_complete, "planned_count": len(all_baselines), "completed_count": sum(s["plan_status"] == "completed" for _, _, s, _ in all_baselines), "same_budget": 0.001, "per_transfer": {f"{s}->{t}": reference.get((s, t, 2026, 0.001)) for s, t in sorted(TRANSFERS)}},
        "transfer_aggregates": transfer_aggregates, "rows": base_report_rows,
    }
    dump(reports / "baseline_summary.json", baseline_json)
    lines = ["# Machine2 baseline summary", "", "Baseline shard: 24 planned identities; D->A and D->W; seed=2026.", "", f"Status: `{base_status_data['status_counts']}`.", "", "| transfer | identities | mean PU | mean FO | runtime-comparable |", "|---|---:|---:|---:|---:|"]
    for transfer, agg in transfer_aggregates.items():
        lines.append(f"| {transfer} | {agg['count']} | {agg['mean_PU']:.6f} | {agg['mean_FO']:.6f} | {agg['runtime_comparable_count']}/{agg['count']} |")
    lines += ["", "Formal efficiency fields are retained per identity in `baseline_summary.csv`; no historical seed-2020 result is used.", f"", f"Full 72-baseline reference complete: `{str(reference_complete).lower()}`."]
    (reports / "baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump(results_base / "baseline_reference.json", {"complete": reference_complete, "same_budget": 0.001, "per_transfer": {f"{s}->{t}": reference.get((s, t, 2026, 0.001)) for s, t in sorted(TRANSFERS)}})

    per_run = [stage1_run(exp, st) for exp, st in lbi_rows]
    per_fields = ["candidate_id", "alpha", "kappa", "nu", "omega", "stage2_lr", "source", "target", "transfer", "experiment_key", "experiment_config_sha256", "summary_path", "metrics_path", "plan_status", "completed", "scientific_valid", "hard_90pct_valid", "error_count", "batch_count", "support_utilization_mean", "support_utilization_min", "support_utilization_p05", "under_95pct_batch_count", "under_90pct_batch_count", "mean_budget_gap", "max_budget_gap", "rollback_count", "rollback_rate", "stage1_steps_mean", "stage1_steps_max", "stage1_max_steps_hit_count", "PU", "FO", "requested_budget", "K", "stage1_max_steps", "batch_size", "seed", "variant", "errors"]
    write_csv(reports / "stage1_per_run.csv", per_run, per_fields)
    dump(results_lbi / "stage1_per_run.json", {"K": K, "requested_budget": 0.001, "runs": per_run})
    ranked = candidate_ranking(per_run, reference_complete, reference)
    rank_fields = ["rank", "candidate_id", "alpha", "kappa", "nu", "omega", "stage2_lr", "completed_transfer_count", "scientific_valid_transfer_count", "hard_90pct_valid_transfer_count", "global_min_utilization", "aggregate_p05_utilization", "aggregate_mean_utilization", "under_95pct_batch_count", "under_90pct_batch_count", "mean_budget_gap", "max_budget_gap", "rollback_rate", "stage1_steps_mean", "stage1_steps_max", "max_steps_hit_count", "mean_PU", "mean_FO", "baseline_reference_complete", "mean_FO_margin", "worst_transfer_FO_margin", "hard_gate_failures", "transfer_rows"]
    write_csv(reports / "stage1_candidate_ranking.csv", ranked, rank_fields)
    dump(reports / "stage1_candidate_ranking.json", {"ranking_rule": ["scientific_valid_transfer_count desc", "hard_90pct_valid_transfer_count desc", "global_min_utilization desc", "aggregate_p05_utilization desc", "aggregate_mean_utilization desc", "under_95pct_batch_count asc", "max_budget_gap asc", "mean_budget_gap asc", "mean_FO_margin desc if baseline complete", "worst_transfer_FO_margin desc if baseline complete", "mean_FO desc", "stage1_steps_max asc", "stage1_steps_mean asc"], "baseline_reference_complete": reference_complete, "candidates": ranked})
    md = ["# Stage-1 candidate ranking", "", "Ranking uses PU only as report-only; it is not a ranking field.", "", "| rank | candidate | valid transfers | hard >=90% | global min u | p05 u | mean u | under 90% | mean FO |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in ranked:
        md.append(f"| {row['rank']} | {row['candidate_id']} (a={row['alpha']}, k={row['kappa']}, n={row['nu']}) | {row['scientific_valid_transfer_count']}/{row['completed_transfer_count']} | {row['hard_90pct_valid_transfer_count']}/6 | {row['global_min_utilization']:.6f} | {row['aggregate_p05_utilization']:.6f} | {row['aggregate_mean_utilization']:.6f} | {row['under_90pct_batch_count']} | {row['mean_FO']:.6f} |")
    md += ["", "Hard gate requires every batch on every transfer to have u >= 0.90; 0.95 is preference only.", f"Full baseline reference complete: `{str(reference_complete).lower()}`."]
    (reports / "stage1_candidate_ranking.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    ready = bool(ranked and ranked[0]["scientific_valid_transfer_count"] == 6 and ranked[0]["hard_90pct_valid_transfer_count"] == 6)
    selection = {
        "machine": "machine2", "budget": 0.001, "K": K, "ready_for_omega_tuning": ready,
        "selected_candidate": ranked[0] if ready else None,
        "preferred_candidate": ranked[0] if ranked else None,
        "reason": "Top candidate passes scientific validity and hard 90% utilization gate across all six transfers." if ready else "No Stage-1 configuration is ready for omega tuning.",
        "training_launched_by_finalize": False,
    }
    dump(reports / "stage1_selection.json", selection)

    status_commands = [
        "python tools/check_experiment_status.py experiment_logs/shot_otta_office_seed2026_stage1_20260818/plans/baseline_machine2/plan.json experiment_logs/shot_otta_office_seed2026_stage1_20260818/runs --output-dir experiment_logs/shot_otta_office_seed2026_stage1_20260818/status/machine2/baseline",
        "python tools/check_experiment_status.py experiment_logs/shot_otta_office_seed2026_stage1_20260818/plans/lbi_stage1_machine2/plan.json experiment_logs/shot_otta_office_seed2026_stage1_20260818/runs --output-dir experiment_logs/shot_otta_office_seed2026_stage1_20260818/status/machine2/stage1",
        "python tools/check_experiment_status.py experiment_logs/shot_otta_office_seed2026_stage1_20260818/plans/baseline_machine1/plan.json experiment_logs/shot_otta_office_seed2026_stage1_20260818/runs --output-dir experiment_logs/shot_otta_office_seed2026_stage1_20260818/status/all_machines/baseline_machine1",
        "python tools/check_experiment_status.py experiment_logs/shot_otta_office_seed2026_stage1_20260818/plans/machine3/baseline/plan.json experiment_logs/shot_otta_office_seed2026_stage1_20260818/runs --output-dir experiment_logs/shot_otta_office_seed2026_stage1_20260818/status/all_machines/baseline_machine3",
    ]
    invalid_runs = [r for r in per_run if not r["scientific_valid"]]
    invalid_lines = []
    for row in invalid_runs:
        reasons = []
        if row["stage1_max_steps_hit_count"]:
            reasons.append(f"stage1_max_steps_hit_count={row['stage1_max_steps_hit_count']}")
        if row["under_90pct_batch_count"]:
            reasons.append(f"under_90pct_batch_count={row['under_90pct_batch_count']}")
        if row["error_count"]:
            reasons.append(f"record_errors={row['error_count']}")
        invalid_lines.append(f"- `{row['candidate_id']}` {row['transfer']}: {', '.join(reasons) or 'scientific validity predicate failed'}; identity `{row['experiment_key']}`; summary `{row['summary_path']}`; metrics `{row['metrics_path']}`.")
    final_lines = [
        "# FINALIZE — Office Night Machine2 / budget 0.001", "", "No training, retry, resume, or tuning was launched in FINALIZE.", "", "## Status commands", "", *[f"- `{cmd}`" for cmd in status_commands], "", "## Completion", "", f"- Baseline plan: {base_status_data['status_counts']} (24 planned; D->A and D->W).", f"- Stage-1 plan: {lbi_status_data['status_counts']} (48 planned; 8 anchors x 6 transfers).", f"- All three baseline shards: {len(all_baselines)} planned, {sum(s['plan_status'] == 'completed' for _, _, s, _ in all_baselines)} completed; reference complete={reference_complete}.", "", "## Diagnostics", "", "- For each completed Stage-1 run, u_t = selected_count_t / 524, with selected_count_t taken from the real per-batch `support_param_count` (fallback: `stage1_support_count`, then `selected_param_count`).", "- Scientific validity requires finite batch records, completed status, selected_count <= 524, no max-step hit, and valid LBI steps.", "- Hard utilization validity requires u_t >= 0.90 for every batch; u_t >= 0.95 is preference only.", "- Full per-run diagnostics: `reports/machine2/stage1_per_run.csv`; candidate aggregation and ranking: `reports/machine2/stage1_candidate_ranking.*`.", "", "## Ranking and decision", "", *[f"{r['rank']}. {r['candidate_id']}: scientific_valid={r['scientific_valid_transfer_count']}/6, hard_90pct={r['hard_90pct_valid_transfer_count']}/6, global_min_u={r['global_min_utilization']}, p05_u={r['aggregate_p05_utilization']}, mean_FO={r['mean_FO']}." for r in ranked], "", "### Eliminated transfer runs", "", *(invalid_lines or ["- None; all 48 completed transfer runs passed scientific validity."]), "", f"- Preferred candidate: `{ranked[0]['candidate_id'] if ranked else None}`.", f"- Ready for omega tuning: `{ready}`.", "- No omega tuning, Stage-2 LR tuning, final LBI rerun, baseline retry, or LBI retry was started.", "", "## Paths", "", f"- Baseline plan: `{base_plan}`", f"- Stage-1 plan: `{lbi_plan}`", f"- Runs: `{RUNS}`", f"- Launcher logs: `{EXP / 'launcher_logs'}`", f"- Machine command history: `{phase / 'COMMAND_HISTORY.sh'}`", f"- Real launcher shell: `{ROOT / 'tools' / 'run_office_night_machine2_noninteractive.sh'}`", "", "FINALIZE complete for machine2 / budget 0.001.", "No additional training was launched.",
    ]
    (phase / "FINALIZE.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": base_status_data["status_counts"], "stage1": lbi_status_data["status_counts"], "baseline_reference_complete": reference_complete, "ready_for_omega_tuning": ready, "preferred_candidate": ranked[0]["candidate_id"] if ranked else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
