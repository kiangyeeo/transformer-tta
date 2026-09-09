#!/usr/bin/env python3
"""Artifact-only FINALIZE for the machine1 Office LBI joint sweep.

This module uses only JSON, JSONL, CSV, and path artifacts.  It never imports
training, model, CUDA, or launcher code and never starts an experiment.
"""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOINT = ROOT / "experiment_logs/shot_otta_office_lbi_joint_sweep_20260818"
PLAN_PATH = JOINT / "plans/machine1/plan.json"
LAUNCHER_SUMMARY = JOINT / "launcher_logs/machine1/launcher_summary.csv"
REFERENCE_MATRIX = JOINT / "matrices/machine1/reference_runs.json"
STAGE1_ROOT = ROOT / "experiment_logs/shot_otta_office_seed2026_stage1_20260818"
RUNS_ROOT = JOINT / "runs"
REPORT_ROOT = JOINT / "reports/machine1"
PHASE_ROOT = JOINT / "phase_records/machine1"
K = 262
REFERENCE_POINT = (0.20, 0.020)
TRANSFERS = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
PAIR_TO_ANCHOR = {
    (0.0005, 0.10, 1.0, 0.25): "A6",
    (0.0005, 0.10, 1.0, 0.50): "A1",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


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
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def pctl(values, quantile):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * quantile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(values[low])
    return float(values[low] + (values[high] - values[low]) * (position - low))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value):
    return float(value) if value is not None else None


def transfer_name(source, target):
    labels = {0: "A", 1: "D", 2: "W"}
    return f"{labels[int(source)]}->{labels[int(target)]}"


def read_metrics(summary_path):
    metrics_path = Path(summary_path).parent / "metrics.jsonl"
    records = []
    errors = []
    if not metrics_path.exists():
        return records, [f"missing metrics.jsonl: {metrics_path}"]
    for line_number, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{metrics_path}:{line_number}: {exc}")
            continue
        if not finite(record):
            errors.append(f"non-finite value: {metrics_path}:{line_number}")
        if record.get("error") or record.get("exception"):
            errors.append(f"error field: {metrics_path}:{line_number}")
        if record.get("event") == "online_step":
            records.append(record)
    if not records:
        errors.append(f"no online_step records: {metrics_path}")
    return records, errors


def load_launcher_rows():
    with LAUNCHER_SUMMARY.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_new_matches(plan):
    planned = {entry["experiment_key"]: entry for entry in plan["experiments"]}
    launcher = defaultdict(list)
    for row in load_launcher_rows():
        launcher[row.get("experiment_key", "")].append(row)
    summaries = {}
    for path in JOINT.glob("runs/**/summary.json"):
        try:
            summary = load(path)
        except (OSError, json.JSONDecodeError):
            continue
        key = summary.get("experiment_key")
        if key in planned:
            summaries.setdefault(key, []).append((path, summary))
    matches = {}
    status_counts = Counter()
    for key, entry in planned.items():
        launch_rows = launcher.get(key, [])
        summary_rows = summaries.get(key, [])
        issues = []
        if not launch_rows or not summary_rows:
            issues.append("missing")
        if len(launch_rows) > 1 or len(summary_rows) > 1:
            issues.append("duplicate")
        if launch_rows and any(row.get("experiment_config_sha256") != entry["experiment_config_sha256"] for row in launch_rows):
            issues.append("hash mismatch")
        if summary_rows and any(summary.get("experiment_config_sha256") != entry["experiment_config_sha256"] for _, summary in summary_rows):
            issues.append("hash mismatch")
        if launch_rows and any(row.get("status") != "completed" or row.get("return_code") != "0" for row in launch_rows):
            issues.append("failed")
        if summary_rows:
            path, summary = summary_rows[0]
            if summary.get("experiment_key") != key or summary.get("status") != "completed" or not finite(summary):
                issues.append("invalid summary")
        if issues:
            for issue in set(issues):
                status_counts[issue] += 1
            continue
        path, summary = summary_rows[0]
        matches[key] = (entry, path, summary)
    status_counts["completed"] = len(matches)
    for name in ("missing", "failed", "duplicate", "hash mismatch", "invalid summary"):
        status_counts.setdefault(name, 0)
    return matches, dict(status_counts)


def load_reference_rows():
    matrix = load(REFERENCE_MATRIX)
    references = matrix.get("references", [])
    if len(references) != 12:
        raise RuntimeError(f"expected 12 reference rows, found {len(references)}")
    status_path = ROOT / matrix["source_status"]
    status = {row["experiment_key"]: row for row in load(status_path)["experiments"]}
    rows = []
    seen = set()
    for reference in references:
        key = reference["experiment_key"]
        if key in seen:
            raise RuntimeError(f"duplicate reference identity: {key}")
        seen.add(key)
        state = status.get(key)
        summary_path = ROOT / reference["summary_path"]
        issues = []
        if state is None or state.get("plan_status") != "completed":
            issues.append("missing")
        if not summary_path.exists():
            issues.append("missing")
        if state and state.get("matching_summary_paths") != [str(summary_path)]:
            issues.append("duplicate")
        if summary_path.exists():
            summary = load(summary_path)
            if summary.get("experiment_key") != key or summary.get("experiment_config_sha256") != reference["scientific_hash"]:
                issues.append("hash mismatch")
            if summary.get("status") != "completed" or not finite(summary):
                issues.append("invalid summary")
        if issues:
            raise RuntimeError(f"reference {key} invalid: {sorted(set(issues))}")
        summary = load(summary_path)
        pair = (float(reference["budget"]), float(reference["alpha"]), float(reference["kappa"]), float(reference["nu"]))
        if pair not in PAIR_TO_ANCHOR or (float(summary.get("omega")), float(summary.get("stage2_lr"))) != REFERENCE_POINT:
            raise RuntimeError(f"reference tuple mismatch: {key}")
        rows.append((reference, summary_path, summary))
    if len({row[0]["experiment_key"] for row in rows}) != 12:
        raise RuntimeError("reference identities are not unique")
    return rows


def baseline_reference():
    shards = [
        ("plans/baseline_machine1/plan.json", "status/baseline_machine1/experiment_status.json"),
        ("plans/baseline_machine2/plan.json", "status/baseline_machine2/experiment_status.json"),
        ("plans/machine3/baseline/plan.json", "status/all_machines/baseline_machine3/experiment_status.json"),
    ]
    entries = []
    for plan_rel, status_rel in shards:
        plan = load(STAGE1_ROOT / plan_rel)
        status = {row["experiment_key"]: row for row in load(STAGE1_ROOT / status_rel)["experiments"]}
        for entry in plan["experiments"]:
            if float(entry.get("requested_budget") or 0.0) != 0.0005:
                continue
            if entry.get("variant") not in {"module_random", "module_magnitude", "module_saliency"}:
                continue
            state = status[entry["experiment_key"]]
            paths = state.get("matching_summary_paths", [])
            if state.get("plan_status") != "completed" or len(paths) != 1:
                raise RuntimeError(f"incomplete baseline reference: {entry['experiment_key']}")
            summary_path = Path(paths[0])
            summary = load(summary_path)
            if summary.get("status") != "completed" or not finite(summary):
                raise RuntimeError(f"invalid baseline summary: {summary_path}")
            if entry.get("variant") == "module_random":
                fo = summary.get("mean_FO-Acc")
            else:
                fo = summary.get("FO-Acc")
            if fo is None or not math.isfinite(float(fo)):
                raise RuntimeError(f"missing baseline FO: {summary_path}")
            entries.append({
                "source": int(entry["source"]),
                "target": int(entry["target"]),
                "variant": entry["variant"],
                "FO": float(fo),
                "experiment_key": entry["experiment_key"],
                "summary_path": str(summary_path),
            })
    grouped = defaultdict(list)
    for row in entries:
        grouped[(row["source"], row["target"])].append(row)
    if len(entries) != 18 or set(grouped) != TRANSFERS or any(len(rows) != 3 for rows in grouped.values()):
        raise RuntimeError("seed-2026 Office baseline reference is incomplete")
    result = {}
    for transfer, rows in grouped.items():
        best = max(rows, key=lambda row: row["FO"])
        result[transfer] = {
            "best_sparse_FO": best["FO"],
            "best_sparse_variant": best["variant"],
            "entries": rows,
        }
    return result


def metric_row(entry, summary_path, summary, reused_reference):
    records, errors = read_metrics(summary_path)
    counts = []
    steps = []
    rollback = []
    max_hits = []
    for record in records:
        count = record.get("support_param_count", record.get("stage1_support_count", record.get("selected_param_count")))
        if count is None:
            errors.append(f"missing selected/support count at batch {record.get('batch_index')}")
            continue
        counts.append(float(count))
        if record.get("stage1_steps_completed") is None:
            errors.append(f"missing stage1 steps at batch {record.get('batch_index')}")
        else:
            steps.append(float(record["stage1_steps_completed"]))
        rollback.append(bool(record.get("stage1_rollback_used", False)))
        max_hits.append(bool(record.get("max_steps_hit", False)))
        if record.get("valid_lbi_step") is not True:
            errors.append(f"invalid LBI step at batch {record.get('batch_index')}")
    if len(steps) != len(counts):
        errors.append("stage1 step count does not match batch count")
    util = [count / K for count in counts]
    gaps = [K - count for count in counts]
    over_budget = sum(count > K for count in counts)
    if over_budget:
        errors.append(f"budget violations: {over_budget}")
    if any(hit for hit in max_hits):
        errors.append(f"stage1 max_steps hits: {sum(max_hits)}")
    if any(value < 0.90 for value in util):
        errors.append(f"under 90 percent batches: {sum(value < 0.90 for value in util)}")
    scientific_valid = bool(
        summary.get("status") == "completed"
        and records
        and not errors
        and all(math.isfinite(value) for value in counts + steps)
    )
    pair = (float(entry["budget"] if reused_reference else entry["requested_budget"]), float(entry["alpha"]), float(entry["kappa"]), float(entry["nu"]))
    anchor = entry["anchor_id"] if reused_reference else PAIR_TO_ANCHOR[pair]
    row = {
        "reused_reference": reused_reference,
        "anchor_id": anchor,
        "budget": pair[0],
        "alpha": pair[1],
        "kappa": pair[2],
        "nu": pair[3],
        "omega": float(summary.get("omega", entry.get("omega"))),
        "stage2_lr": float(summary.get("stage2_lr", entry.get("stage2_lr"))),
        "source": int(entry["source"] if reused_reference else entry["source"]),
        "target": int(entry["target"] if reused_reference else entry["target"]),
        "transfer": entry["transfer"] if reused_reference else transfer_name(entry["source"], entry["target"]),
        "experiment_key": entry["experiment_key"],
        "experiment_config_sha256": entry.get("scientific_hash", entry.get("experiment_config_sha256")),
        "summary_path": str(summary_path),
        "metrics_path": str(Path(summary_path).parent / "metrics.jsonl"),
        "batch_count": len(counts),
        "scientific_valid": scientific_valid,
        "support_utilization_min": min(util) if util else None,
        "support_utilization_p05": pctl(util, 0.05),
        "support_utilization_mean": sum(util) / len(util) if util else None,
        "under_95pct_batch_count": sum(value < 0.95 for value in util),
        "under_90pct_batch_count": sum(value < 0.90 for value in util),
        "mean_budget_gap": sum(gaps) / len(gaps) if gaps else None,
        "max_budget_gap": max(gaps) if gaps else None,
        "rollback_rate": sum(rollback) / len(rollback) if rollback else None,
        "stage1_steps_mean": sum(steps) / len(steps) if steps else None,
        "stage1_steps_max": max(steps) if steps else None,
        "stage1_max_steps_hit_count": sum(max_hits),
        "PU": summary.get("PU-Acc"),
        "FO": summary.get("FO-Acc"),
        "errors": errors,
    }
    return row


def aggregate_config(rows, baseline):
    first = rows[0]
    utilities = [value for row in rows for value in [row["support_utilization_min"]] if value is not None]
    # Aggregate p05/mean and batch diagnostics from the actual per-run values.
    all_utils = []
    total_batches = sum(row["batch_count"] for row in rows)
    for row in rows:
        metrics, _ = read_metrics(row["summary_path"])
        for record in metrics:
            count = record.get("support_param_count", record.get("stage1_support_count", record.get("selected_param_count")))
            if count is not None:
                all_utils.append(float(count) / K)
    margins = []
    for row in rows:
        ref = baseline.get((row["source"], row["target"]))
        if ref is not None and row["FO"] is not None:
            margins.append(float(row["FO"]) - ref["best_sparse_FO"])
    valid_count = sum(bool(row["scientific_valid"]) for row in rows)
    hard_count = sum(bool(row["scientific_valid"] and row["under_90pct_batch_count"] == 0) for row in rows)
    return {
        "anchor_id": first["anchor_id"],
        "budget": first["budget"],
        "alpha": first["alpha"],
        "kappa": first["kappa"],
        "nu": first["nu"],
        "omega": first["omega"],
        "stage2_lr": first["stage2_lr"],
        "transfer_count": len(rows),
        "valid_transfer_count": valid_count,
        "hard_90pct_valid_transfer_count": hard_count,
        "eligible": valid_count == 6 and hard_count == 6,
        "global_min_utilization": min(all_utils) if all_utils else None,
        "aggregate_p05_utilization": pctl(all_utils, 0.05),
        "aggregate_mean_utilization": sum(all_utils) / len(all_utils) if all_utils else None,
        "under_95pct_batch_count": sum(row["under_95pct_batch_count"] for row in rows),
        "under_90pct_batch_count": sum(row["under_90pct_batch_count"] for row in rows),
        "mean_budget_gap": sum(row["mean_budget_gap"] * row["batch_count"] for row in rows if row["mean_budget_gap"] is not None) / total_batches if total_batches else None,
        "max_budget_gap": max((row["max_budget_gap"] for row in rows if row["max_budget_gap"] is not None), default=None),
        "rollback_rate": sum((row["rollback_rate"] or 0.0) * row["batch_count"] for row in rows) / total_batches if total_batches else None,
        "stage1_steps_mean": sum((row["stage1_steps_mean"] or 0.0) * row["batch_count"] for row in rows) / total_batches if total_batches else None,
        "stage1_steps_max": max((row["stage1_steps_max"] for row in rows if row["stage1_steps_max"] is not None), default=None),
        "stage1_max_steps_hit_count": sum(row["stage1_max_steps_hit_count"] for row in rows),
        "mean_FO": sum(row["FO"] for row in rows if row["FO"] is not None) / sum(row["FO"] is not None for row in rows),
        "mean_PU": sum(row["PU"] for row in rows if row["PU"] is not None) / sum(row["PU"] is not None for row in rows),
        "mean_FO_margin": sum(margins) / len(margins) if margins else None,
        "worst_transfer_FO_margin": min(margins) if margins else None,
        "transfer_rows": [row["experiment_key"] for row in rows],
        "hard_gate_failures": [row["transfer"] for row in rows if not (row["scientific_valid"] and row["under_90pct_batch_count"] == 0)],
    }


def rank_configs(configs):
    def low(value):
        return float("-inf") if value is None else float(value)

    def ranking_key(row):
        return (
            -int(row["valid_transfer_count"] == 6),
            -int(row["hard_90pct_valid_transfer_count"] == 6),
            -low(row["mean_FO_margin"]),
            -low(row["worst_transfer_FO_margin"]),
            -low(row["mean_FO"]),
            row["under_95pct_batch_count"],
            -low(row["aggregate_p05_utilization"]),
            -low(row["global_min_utilization"]),
            row["stage1_steps_max"] if row["stage1_steps_max"] is not None else float("inf"),
            row["stage1_steps_mean"] if row["stage1_steps_mean"] is not None else float("inf"),
        )

    grouped = defaultdict(list)
    for row in configs:
        grouped[row["anchor_id"]].append(row)
    ranked = []
    for anchor_id in ("A6", "A1"):
        rows = sorted(grouped.get(anchor_id, []), key=ranking_key)
        for index, row in enumerate(rows, 1):
            row["rank"] = index
        ranked.extend(rows)
    return ranked


def main():
    plan = load(PLAN_PATH)
    if plan.get("experiment_count") != 132 or len(plan.get("experiments", [])) != 132:
        raise RuntimeError("machine1 plan is not exactly 132 entries")
    matches, new_status = load_new_matches(plan)
    reference_rows = load_reference_rows()
    baseline = baseline_reference()
    new_rows = [metric_row(entry, path, summary, False) for entry, path, summary in matches.values()]
    reused_rows = []
    for reference, path, summary in reference_rows:
        entry = dict(reference)
        entry["source"], entry["target"] = next((source, target) for source, target in TRANSFERS if transfer_name(source, target) == reference["transfer"])
        reused_rows.append(metric_row(entry, path, summary, True))
    per_run = new_rows + reused_rows
    pair_counts = Counter(row["anchor_id"] for row in per_run)
    config_groups = defaultdict(list)
    for row in per_run:
        config_groups[(row["anchor_id"], row["omega"], row["stage2_lr"])].append(row)
    if len(per_run) != 144 or len(config_groups) != 24 or any(len(rows) != 6 for rows in config_groups.values()):
        raise RuntimeError("full 144-row result set is incomplete")
    if pair_counts != Counter({"A6": 72, "A1": 72}):
        raise RuntimeError(f"anchor row counts are wrong: {pair_counts}")
    configs = [aggregate_config(rows, baseline) for rows in config_groups.values()]
    ranking = rank_configs(configs)

    per_run_fields = [
        "reused_reference", "anchor_id", "budget", "alpha", "kappa", "nu", "omega", "stage2_lr",
        "source", "target", "transfer", "experiment_key", "experiment_config_sha256", "summary_path",
        "metrics_path", "batch_count", "scientific_valid", "support_utilization_min", "support_utilization_p05",
        "support_utilization_mean", "under_95pct_batch_count", "under_90pct_batch_count", "mean_budget_gap",
        "max_budget_gap", "rollback_rate", "stage1_steps_mean", "stage1_steps_max", "stage1_max_steps_hit_count",
        "PU", "FO", "errors",
    ]
    config_fields = [
        "rank", "anchor_id", "budget", "alpha", "kappa", "nu", "omega", "stage2_lr", "transfer_count",
        "valid_transfer_count", "hard_90pct_valid_transfer_count", "eligible", "global_min_utilization",
        "aggregate_p05_utilization", "aggregate_mean_utilization", "under_95pct_batch_count", "under_90pct_batch_count",
        "mean_budget_gap", "max_budget_gap", "rollback_rate", "stage1_steps_mean", "stage1_steps_max",
        "stage1_max_steps_hit_count", "mean_FO", "mean_PU", "mean_FO_margin", "worst_transfer_FO_margin",
        "hard_gate_failures", "transfer_rows",
    ]
    write_csv(REPORT_ROOT / "per_run_144.csv", per_run, per_run_fields)
    write_csv(REPORT_ROOT / "per_config_24.csv", ranking, config_fields)
    dump(REPORT_ROOT / "per_config_24.json", {"configuration_count": 24, "configurations": ranking})
    write_csv(REPORT_ROOT / "ranking.csv", ranking, config_fields)
    ranking_lines = [
        "# Machine1 Office LBI joint-sweep ranking", "",
        "Ranking is local to A1 and A6. PU is report-only and is not used for ranking.", "",
        "| rank | anchor | omega | stage2_lr | valid/6 | hard >=90%/6 | mean FO margin | worst FO margin | mean FO | under 95% | stage1 max |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranking:
        fmt = lambda value: "" if value is None else f"{float(value):.6f}"
        ranking_lines.append(
            f"| {row['rank']} | {row['anchor_id']} | {row['omega']:.3f} | {row['stage2_lr']:.3f} | "
            f"{row['valid_transfer_count']}/6 | {row['hard_90pct_valid_transfer_count']}/6 | "
            f"{fmt(row['mean_FO_margin'])} | {fmt(row['worst_transfer_FO_margin'])} | {fmt(row['mean_FO'])} | "
            f"{row['under_95pct_batch_count']} | {fmt(row['stage1_steps_max'])} |"
        )
    (REPORT_ROOT / "ranking.md").parent.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "ranking.md").write_text("\n".join(ranking_lines) + "\n", encoding="utf-8")

    ready = bool(
        len(matches) == 132
        and new_status["missing"] == 0
        and new_status["failed"] == 0
        and new_status["duplicate"] == 0
        and new_status["hash mismatch"] == 0
        and new_status["invalid summary"] == 0
        and len(reference_rows) == 12
        and len(per_run) == 144
        and all(len(rows) == 6 for rows in config_groups.values())
        and all(not row["errors"] for row in per_run)
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    finalize_summary = {
        "schema_version": 1,
        "machine_tag": "machine1",
        "plan_path": str(PLAN_PATH),
        "plan_sha256": sha256(PLAN_PATH),
        "new_run_status": new_status,
        "reference_run_count": len(reference_rows),
        "full_result_row_count": len(per_run),
        "configuration_count": len(config_groups),
        "anchor_row_counts": dict(pair_counts),
        "scientific_valid_run_count": sum(row["scientific_valid"] for row in per_run),
        "eligible_configuration_count": sum(row["eligible"] for row in configs),
        "baseline_reference": {
            transfer_name(source, target): value for (source, target), value in sorted(baseline.items())
        },
        "finalize_ready": ready,
        "training_launched_by_finalize": False,
        "timestamp": timestamp,
    }
    dump(REPORT_ROOT / "FINALIZE_SUMMARY.json", finalize_summary)

    phase_lines = [
        "# FINALIZE - Office LBI joint sweep machine1", "",
        "FINALIZE used saved plans, launcher status, summaries, and real per-batch metrics only.",
        "No training, retry, resume, or rerun was launched by FINALIZE.", "",
        "## Completion", "",
        f"- New plan entries: 132; completed={new_status['completed']}; missing={new_status['missing']}; failed={new_status['failed']}; duplicate={new_status['duplicate']}; hash mismatch={new_status['hash mismatch']}; invalid summary={new_status['invalid summary']}.",
        f"- Reused Stage-1 reference rows: {len(reference_rows)}.",
        f"- Full local result set: {len(per_run)} rows across {len(config_groups)} configurations; A1={pair_counts['A1']}, A6={pair_counts['A6']}.",
        f"- Scientific-valid runs: {sum(row['scientific_valid'] for row in per_run)}/{len(per_run)}; eligible configurations: {sum(row['eligible'] for row in configs)}/{len(configs)}.",
        "",
        "## Rules", "",
        "- Eligibility requires completed status, finite artifacts, valid LBI steps, no Stage-1 max-step hit, no support-count budget violation, and support utilization >= 0.90 on every real batch.",
        "- 95% utilization is a diagnostic preference only.",
        "- PU is report-only and was not used for ranking.",
        "- Baseline margins use complete seed-2026 Office budget=0.0005 Random 3-mask mean, Magnitude, and Saliency references; no seed-2020 results are used.",
        "",
        "## Outputs", "",
        f"- Plan: `{PLAN_PATH}`",
        f"- Reports: `{REPORT_ROOT}`",
        f"- Plan SHA256: `{finalize_summary['plan_sha256']}`",
        "",
        "FINALIZE complete for machine1.",
        "No training was launched by FINALIZE.",
    ]
    PHASE_ROOT.mkdir(parents=True, exist_ok=True)
    (PHASE_ROOT / "FINALIZE.md").write_text("\n".join(phase_lines) + "\n", encoding="utf-8")
    if ready:
        dump(PHASE_ROOT / "FINALIZE_READY.json", {
            "machine_tag": "machine1",
            "plan_path": str(PLAN_PATH),
            "plan_sha256": finalize_summary["plan_sha256"],
            "new_run_count": 132,
            "reference_run_count": 12,
            "full_result_row_count": 144,
            "configuration_count": 24,
            "anchor_row_counts": dict(pair_counts),
            "new_run_status": new_status,
            "scientific_valid_run_count": finalize_summary["scientific_valid_run_count"],
            "eligible_configuration_count": finalize_summary["eligible_configuration_count"],
            "finalize_ready": True,
            "training_launched_by_finalize": False,
            "timestamp": timestamp,
        })
    print(json.dumps({"finalize_ready": ready, "new_run_status": new_status, "reference_run_count": len(reference_rows), "result_rows": len(per_run), "configurations": len(config_groups)}, sort_keys=True))


if __name__ == "__main__":
    main()
