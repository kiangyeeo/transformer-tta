#!/usr/bin/env python3
"""Pure artifact-only FINALIZE for the Office LBI joint sweep, machine3."""

import argparse
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools.check_experiment_status import scan_run_summaries


K = 1049
TRANSFERS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
TRANSFER_NAMES = {(0, 1): "A->D", (0, 2): "A->W", (1, 0): "D->A", (1, 2): "D->W", (2, 0): "W->A", (2, 1): "W->D"}
ANCHORS = {
    (0.15, 1.0, 0.50): "A2",
    (0.20, 1.0, 0.50): "A3",
}
ANCHOR_PAIRS = {
    "A2": (0.15, 1.0, 0.50),
    "A3": (0.20, 1.0, 0.50),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row.get(field), ensure_ascii=False) if isinstance(row.get(field), (dict, list)) else row.get(field) for field in fields})


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def number(value):
    if value is None or value == "":
        raise ValueError("missing numeric value")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite numeric value")
    return value


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[low]
    weight = position - low
    return values[low] * (1.0 - weight) + values[high] * weight


def index_summaries(runs_root):
    records, invalid = scan_run_summaries(str(runs_root))
    by_key = {}
    for record in records:
        by_key.setdefault(record["experiment_key"], []).append(record)
    return by_key, invalid


def status_for_plan(plan, by_key, invalid):
    invalid_by_key = {}
    for item in invalid:
        if item.get("experiment_key") is not None:
            invalid_by_key.setdefault(item["experiment_key"], []).append(item)
    rows = []
    for entry in plan["experiments"]:
        key = entry["experiment_key"]
        expected = entry["experiment_config_sha256"]
        records = by_key.get(key, [])
        exact = [r for r in records if r["experiment_config_sha256"] == expected]
        mismatch = [r for r in records if r["experiment_config_sha256"] != expected]
        completed = [r for r in exact if r["status"] == "completed"]
        failed = [r for r in exact if r["status"] in {"failed", "incomplete"}]
        if invalid_by_key.get(key):
            state = "invalid_summary"
        elif mismatch:
            state = "hash_mismatch"
        elif len(exact) > 1:
            state = "duplicate"
        elif len(completed) == 1:
            state = "completed"
        elif failed:
            state = "failed"
        else:
            state = "missing"
        rows.append({"entry": entry, "state": state, "records": exact, "summary_record": completed[0] if len(completed) == 1 else None})
    return rows


def counts(status_rows, invalid_count=0):
    result = {"completed": 0, "missing": 0, "failed": 0, "duplicate": 0, "hash_mismatch": 0, "invalid_summary": 0}
    for row in status_rows:
        result[row["state"]] += 1
    result["invalid_summary_artifacts_scanned"] = invalid_count
    return result


def load_metrics(summary_path):
    metrics_path = Path(summary_path).with_name("metrics.jsonl")
    online = []
    errors = []
    finite_records = True
    parse_error = None
    try:
        with metrics_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                finite_records = finite_records and finite(record)
                if record.get("event") == "online_step":
                    online.append(record)
                if record.get("event") == "error":
                    errors.append({"line": line_number, "record": record})
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parse_error = str(exc)
    return metrics_path, online, errors, finite_records, parse_error


def anchor_for(entry):
    key = (round(float(entry["alpha"]), 2), round(float(entry["kappa"]), 2), round(float(entry["nu"]), 2))
    return ANCHORS[key]


def run_diagnostics(entry, summary_path, summary, reference=False):
    metrics_path, online, errors, finite_records, parse_error = load_metrics(summary_path)
    reasons = []
    selected = []
    support = []
    gaps = []
    stage_steps = []
    rollbacks = []
    max_hits = []
    valid_steps = []
    for record in online:
        try:
            selected_value = number(record.get("selected_param_count"))
            support_value = number(record.get("support_param_count", record.get("stage1_support_count", selected_value)))
            selected.append(selected_value)
            support.append(support_value / K)
            gaps.append(K - selected_value)
            stage_steps.append(number(record.get("stage1_steps_completed")))
            rollbacks.append(bool(record.get("stage1_rollback_used", False)))
            max_hits.append(bool(record.get("max_steps_hit", False)) or record.get("stage1_stop_reason") in {"max_steps", "max_steps_reached"})
            valid_steps.append(record.get("valid_lbi_step") is True)
        except (TypeError, ValueError) as exc:
            reasons.append(f"invalid online_step: {exc}")
    if parse_error:
        reasons.append(f"metrics parse error: {parse_error}")
    if not online:
        reasons.append("no online_step records")
    if not finite_records:
        reasons.append("NaN/Inf in metrics")
    if errors:
        reasons.append(f"error events={len(errors)}")
    if summary.get("status") != "completed":
        reasons.append(f"summary status={summary.get('status')}")
    if summary.get("valid_lbi_run") is not True:
        reasons.append("valid_lbi_run is not true")
    if selected and any(value > K for value in selected):
        reasons.append("selected_count exceeds K")
    if max_hits and any(max_hits):
        reasons.append("Stage-1 max_steps failure")
    if valid_steps and not all(valid_steps):
        reasons.append("invalid LBI step")
    expected_steps = summary.get("online_steps")
    if expected_steps is not None and len(online) != int(expected_steps):
        reasons.append("online step count mismatch")
    if summary.get("FO-Acc") is None or not finite(summary.get("FO-Acc")):
        reasons.append("invalid FO")
    if summary.get("PU-Acc") is None or not finite(summary.get("PU-Acc")):
        reasons.append("invalid PU")
    scientific_valid = bool(online and not reasons)
    hard_valid = bool(scientific_valid and support and all(value >= 0.90 for value in support))
    if scientific_valid and any(value < 0.90 for value in support):
        reasons.append("support utilization below 90%")
    source, target = int(entry["source"]), int(entry["target"])
    alpha, kappa, nu = float(entry["alpha"]), float(entry["kappa"]), float(entry["nu"])
    omega, stage2_lr = float(entry["omega"]), float(entry["stage2_lr"])
    return {
        "result_kind": "reference" if reference else "new",
        "budget": float(entry.get("requested_budget", summary.get("requested_budget", 0.002))),
        "anchor_id": anchor_for(entry),
        "alpha": alpha, "kappa": kappa, "nu": nu, "omega": omega, "stage2_lr": stage2_lr,
        "source": source, "target": target, "transfer": TRANSFER_NAMES[(source, target)],
        "experiment_key": entry["experiment_key"],
        "experiment_config_sha256": entry["experiment_config_sha256"],
        "summary_path": str(summary_path), "metrics_path": str(metrics_path),
        "plan_status": "reused_reference" if reference else "completed",
        "completed": True, "scientific_valid": scientific_valid, "hard_90pct_valid": hard_valid,
        "support_utilization_min": min(support) if support else None,
        "support_utilization_p05": percentile(support, 0.05),
        "support_utilization_mean": sum(support) / len(support) if support else None,
        "under_95pct_batch_count": sum(value < 0.95 for value in support),
        "under_90pct_batch_count": sum(value < 0.90 for value in support),
        "mean_budget_gap": sum(gaps) / len(gaps) if gaps else None,
        "max_budget_gap": max(gaps) if gaps else None,
        "rollback_count": sum(rollbacks), "rollback_rate": sum(rollbacks) / len(rollbacks) if rollbacks else None,
        "stage1_steps_mean": sum(stage_steps) / len(stage_steps) if stage_steps else None,
        "stage1_steps_max": max(stage_steps) if stage_steps else None,
        "stage1_max_steps_hit_count": sum(max_hits),
        "batch_count": len(online), "selected_count_min": min(selected) if selected else None,
        "selected_count_max": max(selected) if selected else None,
        "PU": summary.get("PU-Acc"), "FO": summary.get("FO-Acc"),
        "eligibility_reasons": reasons,
    }


def reference_entries(ref_spec):
    entries = []
    for ref in ref_spec["references"]:
        source, target = {"A": 0, "D": 1, "W": 2}[ref["transfer"][0]], {"A": 0, "D": 1, "W": 2}[ref["transfer"][3]]
        entries.append({
            "experiment_key": ref["experiment_key"], "experiment_config_sha256": ref["experiment_config_sha256"],
            "summary_path": ref["summary_path"], "source": source, "target": target,
            "requested_budget": 0.002, "alpha": ref["alpha"], "kappa": 1.0, "nu": 0.50,
            "omega": 0.20, "stage2_lr": 0.020, "anchor": ref["anchor"],
        })
    return entries


def baseline_reference(base_root, base_plans):
    by_key, invalid = index_summaries(base_root / "runs")
    rows = []
    for plan_path in base_plans:
        plan = read_json(plan_path)
        for entry in plan["experiments"]:
            if entry.get("seed") != 2026:
                continue
            matches = [r for r in by_key.get(entry["experiment_key"], []) if r["experiment_config_sha256"] == entry["experiment_config_sha256"] and r["status"] == "completed"]
            if len(matches) != 1:
                rows.append({"source": entry["source"], "target": entry["target"], "variant": entry["variant"], "budget": entry.get("requested_budget"), "FO": None, "summary_path": None, "valid": False, "reason": f"completed matches={len(matches)}"})
                continue
            summary = matches[0]["summary"]
            rows.append({"source": entry["source"], "target": entry["target"], "variant": entry["variant"], "budget": entry.get("requested_budget"), "FO": summary.get("FO-Acc"), "summary_path": matches[0]["summary_path"], "valid": summary.get("status") == "completed" and finite(summary.get("FO-Acc")), "reason": None})
    sparse = {}
    for source, target in TRANSFERS:
        candidates = {}
        for variant in ("module_random", "module_magnitude", "module_saliency"):
            match = [r for r in rows if r["source"] == source and r["target"] == target and r["variant"] == variant and float(r["budget"] or -1) == 0.002]
            if len(match) == 1 and match[0]["valid"]:
                candidates[variant] = match[0]["FO"]
        if len(candidates) == 3:
            sparse[(source, target)] = {"best_sparse_FO": max(candidates.values()), **{f"{key}_FO": value for key, value in candidates.items()}}
    complete = len(rows) == 72 and len(sparse) == 6 and not invalid and all(row["valid"] for row in rows)
    return rows, sparse, complete, invalid


def aggregate(config_rows, sparse):
    groups = {}
    for row in config_rows:
        key = (row["anchor_id"], row["alpha"], row["kappa"], row["nu"], row["omega"], row["stage2_lr"])
        groups.setdefault(key, []).append(row)
    result = []
    for key, rows in sorted(groups.items()):
        utils = [value for row in rows for value in ([row["support_utilization_min"]] if row["support_utilization_min"] is not None else [])]
        # Aggregate diagnostics use every real batch; recompute from saved metrics.
        all_utils, all_gaps, all_steps = [], [], []
        for row in rows:
            _, online, _, _, _ = load_metrics(row["summary_path"])
            for record in online:
                selected = number(record.get("selected_param_count"))
                support = number(record.get("support_param_count", record.get("stage1_support_count", selected)))
                all_utils.append(support / K); all_gaps.append(K - selected); all_steps.append(number(record.get("stage1_steps_completed")))
        margins = []
        for row in rows:
            reference = sparse.get((row["source"], row["target"]))
            if reference is not None and row["FO"] is not None:
                margins.append(row["FO"] - reference["best_sparse_FO"])
        result.append({
            "anchor_id": key[0], "budget": rows[0]["budget"], "alpha": key[1], "kappa": key[2], "nu": key[3], "omega": key[4], "stage2_lr": key[5],
            "valid_transfer_count": sum(bool(row["scientific_valid"]) for row in rows),
            "hard_90pct_valid_transfer_count": sum(bool(row["hard_90pct_valid"]) for row in rows),
            "global_min_utilization": min(all_utils) if all_utils else None, "aggregate_p05_utilization": percentile(all_utils, 0.05), "aggregate_mean_utilization": sum(all_utils) / len(all_utils) if all_utils else None,
            "under_95pct_batch_count": sum(row["under_95pct_batch_count"] for row in rows), "under_90pct_batch_count": sum(row["under_90pct_batch_count"] for row in rows),
            "mean_budget_gap": sum(all_gaps) / len(all_gaps) if all_gaps else None, "max_budget_gap": max(all_gaps) if all_gaps else None,
            "rollback_rate": sum(row["rollback_count"] for row in rows) / len(all_utils) if all_utils else None,
            "stage1_steps_mean": sum(all_steps) / len(all_steps) if all_steps else None, "stage1_steps_max": max(all_steps) if all_steps else None,
            "mean_FO": sum(row["FO"] for row in rows) / len(rows), "mean_PU": sum(row["PU"] for row in rows) / len(rows),
            "mean_FO_margin": sum(margins) / len(margins) if margins else None, "worst_transfer_FO_margin": min(margins) if margins else None,
            "transfer_rows": [row["experiment_key"] for row in sorted(rows, key=lambda item: (item["source"], item["target"]))],
            "hard_gate_failures": [row["transfer"] for row in rows if not row["hard_90pct_valid"]],
        })
    return result


def rank(configs):
    def desc(value):
        return -(float(value) if value is not None else float("inf"))
    ranked = sorted(configs, key=lambda row: (
        -row["valid_transfer_count"], -row["hard_90pct_valid_transfer_count"], desc(row["mean_FO_margin"]), desc(row["worst_transfer_FO_margin"]), desc(row["mean_FO"]),
        row["under_95pct_batch_count"], desc(row["aggregate_p05_utilization"]), desc(row["global_min_utilization"]), desc(row["aggregate_mean_utilization"]),
        row["stage1_steps_max"], row["stage1_steps_mean"], row["anchor_id"], row["omega"], row["stage2_lr"],
    ))
    for index, row in enumerate(ranked, 1):
        row["rank"] = index
    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    plan_path = root / "plans/machine3/plan.json"
    plan = read_json(plan_path)
    by_key, invalid = index_summaries(root / "runs")
    status_rows = status_for_plan(plan, by_key, invalid)
    new_counts = counts(status_rows, len(invalid))

    new_rows = []
    for item in status_rows:
        if item["state"] != "completed":
            continue
        entry = item["entry"]
        record = item["summary_record"]
        new_rows.append(run_diagnostics(entry, record["summary_path"], record["summary"], reference=False))

    ref_spec_path = root / "matrices/machine3/reference_runs.json"
    ref_spec = read_json(ref_spec_path)
    references = reference_entries(ref_spec)
    ref_rows, ref_invalid = [], []
    for entry in references:
        path = Path(entry["summary_path"])
        try:
            summary = read_json(path)
            if summary.get("experiment_key") != entry["experiment_key"] or summary.get("experiment_config_sha256") != entry["experiment_config_sha256"] or summary.get("status") != "completed":
                raise ValueError("reference identity/status mismatch")
            ref_rows.append(run_diagnostics(entry, path, summary, reference=True))
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
            ref_invalid.append({"experiment_key": entry["experiment_key"], "summary_path": str(path), "reason": str(exc)})

    base_root = root.parent / "shot_otta_office_seed2026_stage1_20260818"
    base_plans = sorted((base_root / "plans").glob("baseline_*/plan.json"))
    base_plans.append(base_root / "plans/machine3/baseline/plan.json")
    baseline_rows, sparse, baseline_complete, baseline_invalid = baseline_reference(base_root, base_plans)

    all_rows = new_rows + ref_rows
    configs = aggregate(all_rows, sparse) if all_rows else []
    ranked = rank(configs) if configs else []
    reports = root / "reports/machine3"
    run_fields = ["result_kind", "budget", "anchor_id", "alpha", "kappa", "nu", "omega", "stage2_lr", "source", "target", "transfer", "experiment_key", "experiment_config_sha256", "summary_path", "metrics_path", "plan_status", "completed", "scientific_valid", "hard_90pct_valid", "batch_count", "support_utilization_min", "support_utilization_p05", "support_utilization_mean", "under_95pct_batch_count", "under_90pct_batch_count", "mean_budget_gap", "max_budget_gap", "rollback_count", "rollback_rate", "stage1_steps_mean", "stage1_steps_max", "stage1_max_steps_hit_count", "PU", "FO", "eligibility_reasons"]
    config_fields = ["anchor_id", "budget", "alpha", "kappa", "nu", "omega", "stage2_lr", "valid_transfer_count", "hard_90pct_valid_transfer_count", "global_min_utilization", "aggregate_p05_utilization", "aggregate_mean_utilization", "under_95pct_batch_count", "under_90pct_batch_count", "mean_budget_gap", "max_budget_gap", "rollback_rate", "stage1_steps_mean", "stage1_steps_max", "mean_FO", "mean_PU", "mean_FO_margin", "worst_transfer_FO_margin", "hard_gate_failures", "transfer_rows"]
    write_csv(reports / "per_run_144.csv", all_rows, run_fields)
    write_csv(reports / "per_config_24.csv", configs, config_fields)
    write_json(reports / "per_config_24.json", {"configuration_count": len(configs), "configurations": configs})
    write_csv(reports / "ranking.csv", ranked, ["rank"] + config_fields)
    md = ["# Office LBI joint sweep — machine3 ranking", "", "No training, retry, resume, or rerun was performed by FINALIZE.", "", "Ranking: all-six scientific validity, all-six hard 90% validity, mean FO margin, worst-transfer FO margin, mean FO, 95%-utilization diagnostics, Stage-1 cost.", "", "| rank | anchor | omega | stage2_lr | valid | hard90 | mean FO margin | worst FO margin | mean FO | under95 batches | stage1 max | stage1 mean |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in ranked:
        fmt = lambda value: "" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
        md.append(f"| {row['rank']} | {row['anchor_id']} | {row['omega']:.3f} | {row['stage2_lr']:.3f} | {row['valid_transfer_count']}/6 | {row['hard_90pct_valid_transfer_count']}/6 | {fmt(row['mean_FO_margin'])} | {fmt(row['worst_transfer_FO_margin'])} | {fmt(row['mean_FO'])} | {row['under_95pct_batch_count']} | {fmt(row['stage1_steps_max'])} | {fmt(row['stage1_steps_mean'])} |")
    (reports / "ranking.md").parent.mkdir(parents=True, exist_ok=True)
    (reports / "ranking.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    local_complete = new_counts["completed"] == 132 and all(new_counts[key] == 0 for key in ("missing", "failed", "duplicate", "hash_mismatch", "invalid_summary")) and len(ref_rows) == 12 and not ref_invalid and len(all_rows) == 144 and len(configs) == 24
    validly_summarized = local_complete and baseline_complete and all(row["batch_count"] > 0 for row in all_rows) and all(row["FO"] is not None and row["PU"] is not None for row in all_rows) and not baseline_invalid
    ready = local_complete and validly_summarized
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    summary = {
        "machine_tag": "machine3", "mode": "FINALIZE", "timestamp": timestamp,
        "plan_path": str(plan_path), "new_run_count": len(new_rows), "reference_run_count": len(ref_rows), "local_run_count": len(all_rows), "configuration_count": len(configs),
        "new_status": new_counts, "reference_invalid_count": len(ref_invalid), "invalid_reference_artifacts": ref_invalid,
        "baseline_reference": {"complete": baseline_complete, "planned_rows": len(baseline_rows), "invalid_artifacts": len(baseline_invalid), "source_root": str(base_root), "best_sparse_definition": "max(module_random, module_magnitude, module_saliency) at budget=0.002 per transfer"},
        "local_result_set_complete": local_complete, "validly_summarized": validly_summarized, "finalize_ready": ready,
        "eligibility": "scientific_valid requires completion, finite/no-error metrics, no Stage-1 max_steps failure, selected_count<=K, valid LBI steps; hard_90pct additionally requires support utilization>=0.90 on every batch.",
        "ranking_rule": ["all 6 transfers scientific-valid", "all 6 transfers pass 90% utilization hard gate", "mean_FO_margin descending", "worst_transfer_FO_margin descending", "mean_FO descending", "95%-utilization diagnostics", "stage1_steps_max ascending", "stage1_steps_mean ascending"],
        "training_launched_by_finalize": False,
        "reports": {"per_run": str(reports / "per_run_144.csv"), "per_config_csv": str(reports / "per_config_24.csv"), "per_config_json": str(reports / "per_config_24.json"), "ranking_csv": str(reports / "ranking.csv"), "ranking_md": str(reports / "ranking.md")},
    }
    write_json(reports / "FINALIZE_SUMMARY.json", summary)

    phase = root / "phase_records/machine3"
    lines = ["# FINALIZE — Office LBI joint sweep machine3", "", "Mode: `FINALIZE`", "Machine: `machine3`", "", "No training, launcher, retry, resume, or rerun was performed by FINALIZE.", "", "## Completion", "", "| set | completed | missing | failed | duplicate | hash mismatch | invalid summary |", "|---|---:|---:|---:|---:|---:|---:|"]
    lines.append(f"| new plan (132) | {new_counts['completed']} | {new_counts['missing']} | {new_counts['failed']} | {new_counts['duplicate']} | {new_counts['hash_mismatch']} | {new_counts['invalid_summary']} |")
    lines.append(f"| reused references (12) | {len(ref_rows)} | {12-len(ref_rows)} | 0 | 0 | 0 | {len(ref_invalid)} |")
    lines += ["", f"Local result set: {len(all_rows)}/144 rows; configurations: {len(configs)}/24; `local_result_set_complete={str(local_complete).lower()}`.", f"Baseline reference: {len(baseline_rows)}/72 rows; `baseline_reference_complete={str(baseline_complete).lower()}`; only seed-2026 artifacts were used.", "", "## Reused Stage-1 references", "", "| anchor | transfer | experiment_key | summary_path |", "|---|---|---|---|"]
    for entry in references:
        lines.append(f"| {entry['anchor']} | {TRANSFER_NAMES[(entry['source'], entry['target'])]} | `{entry['experiment_key']}` | `{entry['summary_path']}` |")
    lines += ["", "## Eligibility and ranking", "", "Per-batch selected/support records were read from saved `metrics.jsonl`; PU is report-only and was not used for ranking.", "", *[f"{row['rank']}. `{row['anchor_id']}` omega={row['omega']}, stage2_lr={row['stage2_lr']}: valid={row['valid_transfer_count']}/6, hard90={row['hard_90pct_valid_transfer_count']}/6, mean_FO_margin={row['mean_FO_margin']}, worst_transfer_FO_margin={row['worst_transfer_FO_margin']}." for row in ranked], "", "## Outputs", "", f"- `{reports / 'per_run_144.csv'}`", f"- `{reports / 'per_config_24.csv'}`", f"- `{reports / 'per_config_24.json'}`", f"- `{reports / 'ranking.csv'}`", f"- `{reports / 'ranking.md'}`", f"- `{reports / 'FINALIZE_SUMMARY.json'}`", "", "Global final Office tuple selection is not owned by machine3.", "", "FINALIZE complete for machine3.", "No training was launched by FINALIZE."]
    (phase / "FINALIZE.md").parent.mkdir(parents=True, exist_ok=True)
    (phase / "FINALIZE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if ready:
        write_json(phase / "FINALIZE_READY.json", {"machine_tag": "machine3", "mode": "FINALIZE", "timestamp": timestamp, "plan_path": str(plan_path), "new_run_count": 132, "reference_run_count": 12, "local_run_count": 144, "configuration_count": 24, "new_status": new_counts, "reference_invalid_count": 0, "baseline_reference_complete": baseline_complete, "local_result_set_complete": True, "validly_summarized": True, "training_launched_by_finalize": False, "reports": summary["reports"]})
    print(json.dumps({"new_status": new_counts, "references": len(ref_rows), "rows": len(all_rows), "configs": len(configs), "baseline_reference_complete": baseline_complete, "finalize_ready": ready}, sort_keys=True))


if __name__ == "__main__":
    main()
