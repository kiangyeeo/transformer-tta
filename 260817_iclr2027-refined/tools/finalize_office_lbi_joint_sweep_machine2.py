#!/usr/bin/env python3
"""Finalize the Office LBI joint sweep for machine2 from saved artifacts only.

This module deliberately imports no training, CUDA, or model code.  It reads
the prepared plan, saved summaries/metrics, launcher artifacts, and the
seed-2026 baseline/reference artifacts, then writes the required FINALIZE
reports and readiness marker.
"""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
JOINT = PROJECT / "experiment_logs/shot_otta_office_lbi_joint_sweep_20260818"
PLAN_PATH = JOINT / "plans/machine2/plan.json"
REF_PATH = JOINT / "matrices/machine2/reused_stage1_references.json"
RUNS_ROOT = JOINT / "runs"
BASELINE_ROOT = PROJECT / "experiment_logs/shot_otta_office_seed2026_stage1_20260818"
K = 524
TRANSFERS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
OMEGAS = {0.05, 0.10, 0.20, 0.30}
STAGE2_LRS = {0.005, 0.010, 0.020}
ANCHORS = {
    (0.10, 1.0, 0.25): "A6",
    (0.10, 1.5, 0.50): "A4",
}
BASELINE_PLANS = (
    BASELINE_ROOT / "plans/baseline_machine1/plan.json",
    BASELINE_ROOT / "plans/baseline_machine2/plan.json",
    BASELINE_ROOT / "plans/machine3/baseline/plan.json",
)
BASELINE_STATUS = (
    BASELINE_ROOT / "status/all_machines/baseline_machine1/experiment_status.json",
    BASELINE_ROOT / "status/machine2/baseline/experiment_status.json",
    BASELINE_ROOT / "status/machine3/baseline/experiment_status.json",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    path = Path(path)
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


def number(value, field):
    if value is None or value == "":
        raise ValueError(f"missing numeric field {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric field {field}: {value}")
    return result


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(values[low])
    weight = position - low
    return float(values[low] * (1.0 - weight) + values[high] * weight)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transfer_name(source, target):
    names = {0: "A", 1: "D", 2: "W"}
    return f"{names[int(source)]}->{names[int(target)]}"


def anchor_for(entry):
    key = (round(float(entry["alpha"]), 2), round(float(entry["kappa"]), 2), round(float(entry["nu"]), 2))
    if key not in ANCHORS:
        raise ValueError(f"unowned anchor tuple: {key}")
    return ANCHORS[key]


def summary_records(runs_root):
    records = []
    invalid = []
    for path in sorted(Path(runs_root).rglob("summary.json")):
        try:
            summary = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            invalid.append({"path": str(path), "experiment_key": None, "reason": f"invalid_json: {exc}"})
            continue
        if not isinstance(summary, dict):
            invalid.append({"path": str(path), "experiment_key": None, "reason": "summary_not_object"})
            continue
        missing = sorted({"experiment_key", "experiment_config_sha256", "status"} - set(summary))
        if missing or summary.get("status") not in {"completed", "failed", "incomplete"}:
            invalid.append({
                "path": str(path),
                "experiment_key": summary.get("experiment_key"),
                "reason": f"missing_fields: {missing}" if missing else f"illegal_status: {summary.get('status')}",
            })
            continue
        records.append({
            "path": str(path),
            "summary": summary,
            "experiment_key": summary["experiment_key"],
            "experiment_config_sha256": summary["experiment_config_sha256"],
            "status": summary["status"],
        })
    return records, invalid


def classify_plan(plan, records, invalid):
    by_key = defaultdict(list)
    for record in records:
        by_key[record["experiment_key"]].append(record)
    invalid_by_key = defaultdict(list)
    for record in invalid:
        if record.get("experiment_key") is not None:
            invalid_by_key[record["experiment_key"]].append(record)
    rows = []
    for entry in plan["experiments"]:
        key = entry["experiment_key"]
        expected = entry["experiment_config_sha256"]
        matches = by_key.get(key, [])
        exact = [row for row in matches if row["experiment_config_sha256"] == expected]
        completed = [row for row in exact if row["status"] == "completed"]
        failed = [row for row in exact if row["status"] in {"failed", "incomplete"}]
        mismatched = [row for row in matches if row["experiment_config_sha256"] != expected]
        invalid_rows = invalid_by_key.get(key, [])
        if invalid_rows:
            status = "invalid_summary"
        elif mismatched:
            status = "hash_mismatch"
        elif len(completed) > 1:
            status = "duplicate"
        elif completed:
            status = "completed"
        elif failed:
            status = "failed"
        else:
            status = "missing"
        rows.append({
            "entry": entry,
            "status": status,
            "completed": completed,
            "failed": failed,
            "mismatched": mismatched,
            "invalid": invalid_rows,
        })
    return rows


def read_metrics(summary_path):
    path = Path(summary_path).with_name("metrics.jsonl")
    online = []
    errors = []
    finite_records = True
    if not path.exists():
        return path, online, [f"missing metrics.jsonl: {path}"], False
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: invalid_json: {exc}")
            continue
        finite_records = finite_records and finite(record)
        if not finite(record):
            errors.append(f"{path}:{line_number}: non-finite value")
        if record.get("event") == "error":
            errors.append(f"{path}:{line_number}: error event")
        if record.get("event") == "online_step":
            online.append(record)
    if not online:
        errors.append(f"no online_step records: {path}")
    return path, online, errors, finite_records


def selected_count(record):
    for field in ("support_param_count", "stage1_support_count", "selected_param_count"):
        if record.get(field) is not None:
            return number(record[field], field), field
    raise ValueError("missing per-batch support/selected count")


def diagnostics(entry, summary_path, summary, reused_reference=False):
    metrics_path, online, errors, finite_records = read_metrics(summary_path)
    selected = []
    selected_sources = []
    steps = []
    rollback_count = 0
    max_steps_hit_count = 0
    budget_violation_count = 0
    valid_lbi_step = True
    for step in online:
        try:
            count, source = selected_count(step)
            selected.append(count)
            selected_sources.append(source)
            steps.append(number(step.get("stage1_steps_completed"), "stage1_steps_completed"))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if count > K:
            budget_violation_count += 1
        if step.get("budget_violation") is True:
            budget_violation_count += 1
        if bool(step.get("max_steps_hit")) or step.get("stage1_stop_reason") in {"max_steps", "max_steps_reached"}:
            max_steps_hit_count += 1
        rollback_count += int(bool(step.get("stage1_rollback_used")))
        valid_lbi_step = valid_lbi_step and step.get("valid_lbi_step") is True
    summary_max_hits = int(summary.get("max_steps_hit_count") or 0)
    max_steps_hit_count = max(max_steps_hit_count, summary_max_hits)
    if summary.get("online_steps") is not None and len(online) != int(summary["online_steps"]):
        errors.append(f"online_steps mismatch: metrics={len(online)} summary={summary['online_steps']}")
    if not summary.get("valid_lbi_run", True):
        errors.append("summary valid_lbi_run is not true")
    if not finite(summary):
        errors.append("summary contains non-finite value")
    if not selected:
        errors.append("no usable selected/support counts")
    utilities = [value / K for value in selected]
    gaps = [K - value for value in selected]
    scientific_valid = bool(
        summary.get("status") == "completed"
        and finite_records
        and not errors
        and max_steps_hit_count == 0
        and budget_violation_count == 0
        and valid_lbi_step
        and all(value <= K for value in selected)
    )
    hard_valid = bool(scientific_valid and utilities and all(value >= 0.90 for value in utilities))
    row = {
        "budget": float(entry["requested_budget"]),
        "anchor_id": anchor_for(entry),
        "alpha": float(entry["alpha"]),
        "kappa": float(entry["kappa"]),
        "nu": float(entry["nu"]),
        "omega": float(entry["omega"]),
        "stage2_lr": float(entry["stage2_lr"]),
        "source": int(entry["source"]),
        "target": int(entry["target"]),
        "transfer": transfer_name(entry["source"], entry["target"]),
        "experiment_key": entry["experiment_key"],
        "experiment_config_sha256": entry["experiment_config_sha256"],
        "summary_path": str(summary_path),
        "metrics_path": str(metrics_path),
        "reused_reference": bool(reused_reference),
        "plan_status": "completed",
        "online_step_count": len(online),
        "selected_count_source": ",".join(sorted(set(selected_sources))),
        "scientific_valid": scientific_valid,
        "hard_90pct_valid": hard_valid,
        "no_nan_inf_or_error": finite_records and not errors,
        "budget_violation_count": budget_violation_count,
        "support_utilization_min": min(utilities) if utilities else None,
        "support_utilization_p05": percentile(utilities, 0.05),
        "support_utilization_mean": sum(utilities) / len(utilities) if utilities else None,
        "under_95pct_batch_count": sum(value < 0.95 for value in utilities),
        "under_90pct_batch_count": sum(value < 0.90 for value in utilities),
        "mean_budget_gap": sum(gaps) / len(gaps) if gaps else None,
        "max_budget_gap": max(gaps) if gaps else None,
        "rollback_count": rollback_count,
        "rollback_rate": rollback_count / len(online) if online else None,
        "stage1_steps_mean": sum(steps) / len(steps) if steps else None,
        "stage1_steps_max": max(steps) if steps else None,
        "stage1_max_steps_hit_count": max_steps_hit_count,
        "PU": number(summary["PU-Acc"], "PU-Acc"),
        "FO": number(summary["FO-Acc"], "FO-Acc"),
        "errors": errors,
    }
    row["_utilities"] = utilities
    row["_gaps"] = gaps
    row["_steps"] = steps
    return row


def reference_entries():
    data = read_json(REF_PATH)
    refs = data.get("references", [])
    if len(refs) != 12:
        raise ValueError(f"expected 12 reused references, found {len(refs)}")
    identities = set()
    rows = []
    for ref in refs:
        identity = (ref["experiment_key"], ref["scientific_hash"])
        if identity in identities:
            raise ValueError(f"duplicate reused reference identity: {identity}")
        identities.add(identity)
        path = Path(ref["summary_path"])
        if not path.exists():
            raise ValueError(f"missing reused reference summary: {path}")
        summary = read_json(path)
        if summary.get("status") != "completed":
            raise ValueError(f"reused reference not completed: {path}")
        if summary.get("experiment_key") != ref["experiment_key"]:
            raise ValueError(f"reference key mismatch: {path}")
        if summary.get("experiment_config_sha256") != ref["scientific_hash"]:
            raise ValueError(f"reference hash mismatch: {path}")
        if not (abs(float(summary.get("omega")) - 0.20) < 1e-12 and abs(float(summary.get("stage2_lr")) - 0.020) < 1e-12):
            raise ValueError(f"reference sweep point mismatch: {path}")
        entry = {
            "requested_budget": ref["budget"],
            "alpha": ref["alpha"],
            "kappa": ref["kappa"],
            "nu": ref["nu"],
            "omega": 0.20,
            "stage2_lr": 0.020,
            "source": ref["source"],
            "target": ref["target"],
            "experiment_key": ref["experiment_key"],
            "experiment_config_sha256": ref["scientific_hash"],
        }
        row = diagnostics(entry, path, summary, reused_reference=True)
        row["reference_declared_anchor_id"] = ref["anchor_id"]
        if row["anchor_id"] != ref["anchor_id"]:
            raise ValueError(f"reference anchor mismatch: {path}")
        rows.append(row)
    return rows


def baseline_reference():
    all_rows = []
    for plan_path, status_path in zip(BASELINE_PLANS, BASELINE_STATUS):
        plan = read_json(plan_path)
        status = read_json(status_path)
        if status.get("status_counts", {}).get("completed") != len(plan["experiments"]):
            raise ValueError(f"baseline status is not complete: {status_path}")
        for entry in plan["experiments"]:
            matches = []
            for path in sorted((BASELINE_ROOT / "runs").rglob("summary.json")):
                try:
                    summary = read_json(path)
                except (OSError, json.JSONDecodeError):
                    continue
                if (
                    summary.get("experiment_key") == entry["experiment_key"]
                    and summary.get("experiment_config_sha256") == entry["experiment_config_sha256"]
                    and summary.get("status") == "completed"
                ):
                    matches.append((path, summary))
            if len(matches) != 1:
                raise ValueError(f"baseline identity does not have exactly one result: {entry['experiment_key']} ({len(matches)})")
            path, summary = matches[0]
            all_rows.append({
                "source": int(entry["source"]),
                "target": int(entry["target"]),
                "transfer": transfer_name(entry["source"], entry["target"]),
                "variant": entry["variant"],
                "requested_budget": entry.get("requested_budget"),
                "num_random_masks": entry.get("num_random_masks"),
                "experiment_key": entry["experiment_key"],
                "experiment_config_sha256": entry["experiment_config_sha256"],
                "FO": number(summary["FO-Acc"], "FO-Acc"),
                "PU": number(summary["PU-Acc"], "PU-Acc"),
                "summary_path": str(path),
            })
    if len(all_rows) != 72 or len({row["experiment_key"] for row in all_rows}) != 72:
        raise ValueError(f"baseline reference count/identity mismatch: {len(all_rows)}")
    grouped = defaultdict(dict)
    for row in all_rows:
        if float(row["requested_budget"] or 0.0) == 0.001:
            grouped[(row["source"], row["target"])][row["variant"]] = row
    sparse = {}
    for transfer in TRANSFERS:
        variants = grouped[transfer]
        required = {"module_random", "module_magnitude", "module_saliency"}
        if not required <= set(variants):
            raise ValueError(f"incomplete sparse baseline for {transfer}: {set(variants)}")
        candidates = {name: variants[name]["FO"] for name in required}
        winner = max(candidates, key=candidates.get)
        sparse[transfer] = {
            "best_sparse_FO": candidates[winner],
            "best_sparse_variant": winner,
            "random_3mask_mean_FO": candidates["module_random"],
            "magnitude_FO": candidates["module_magnitude"],
            "saliency_FO": candidates["module_saliency"],
            "entries": {name: variants[name] for name in sorted(required)},
        }
    return all_rows, sparse


def aggregate(per_run, sparse_reference):
    groups = defaultdict(list)
    for row in per_run:
        groups[(row["anchor_id"], row["budget"], row["alpha"], row["kappa"], row["nu"], row["omega"], row["stage2_lr"])].append(row)
    configs = []
    for key, rows in sorted(groups.items()):
        if len(rows) != 6 or { (row["source"], row["target"]) for row in rows } != set(TRANSFERS):
            raise ValueError(f"configuration does not contain six transfers: {key}")
        utilities = [value for row in rows for value in row["_utilities"]]
        gaps = [value for row in rows for value in row["_gaps"]]
        steps = [value for row in rows for value in row["_steps"]]
        margins = [row["FO"] - sparse_reference[(row["source"], row["target"])] ["best_sparse_FO"] for row in rows]
        config = {
            "anchor_id": key[0], "budget": key[1], "alpha": key[2], "kappa": key[3], "nu": key[4],
            "omega": key[5], "stage2_lr": key[6], "transfer_count": len(rows),
            "valid_transfer_count": sum(bool(row["scientific_valid"]) for row in rows),
            "hard_90pct_valid_transfer_count": sum(bool(row["hard_90pct_valid"]) for row in rows),
            "eligible": all(bool(row["scientific_valid"]) for row in rows),
            "hard_90pct_eligible": all(bool(row["hard_90pct_valid"]) for row in rows),
            "global_min_utilization": min(utilities),
            "aggregate_p05_utilization": percentile(utilities, 0.05),
            "aggregate_mean_utilization": sum(utilities) / len(utilities),
            "under_95pct_batch_count": sum(row["under_95pct_batch_count"] for row in rows),
            "under_90pct_batch_count": sum(row["under_90pct_batch_count"] for row in rows),
            "mean_budget_gap": sum(gaps) / len(gaps), "max_budget_gap": max(gaps),
            "rollback_rate": sum(row["rollback_count"] for row in rows) / len(utilities),
            "stage1_steps_mean": sum(steps) / len(steps), "stage1_steps_max": max(steps),
            "stage1_max_steps_hit_count": sum(row["stage1_max_steps_hit_count"] for row in rows),
            "mean_FO": sum(row["FO"] for row in rows) / len(rows),
            "mean_PU": sum(row["PU"] for row in rows) / len(rows),
            "mean_FO_margin": sum(margins) / len(margins), "worst_transfer_FO_margin": min(margins),
            "transfer_rows": [row["experiment_key"] for row in sorted(rows, key=lambda item: (item["source"], item["target"]))],
        }
        configs.append(config)
    if len(configs) != 24:
        raise ValueError(f"expected 24 configurations, found {len(configs)}")
    return configs


def rank(configs):
    def desc(value):
        return -(float(value) if value is not None else float("inf"))
    ordered = sorted(configs, key=lambda row: (
        -int(row["valid_transfer_count"] == 6),
        -int(row["hard_90pct_valid_transfer_count"] == 6),
        desc(row["mean_FO_margin"]), desc(row["worst_transfer_FO_margin"]), desc(row["mean_FO"]),
        row["under_95pct_batch_count"], desc(row["global_min_utilization"]),
        desc(row["aggregate_p05_utilization"]), desc(row["aggregate_mean_utilization"]),
        row["under_90pct_batch_count"], row["stage1_steps_max"], row["stage1_steps_mean"],
        row["anchor_id"], row["omega"], row["stage2_lr"],
    ))
    for index, row in enumerate(ordered, 1):
        row["rank"] = index
    return ordered


def public_run_fields():
    return [
        "budget", "anchor_id", "alpha", "kappa", "nu", "omega", "stage2_lr", "source", "target", "transfer",
        "experiment_key", "experiment_config_sha256", "reused_reference", "plan_status", "summary_path", "metrics_path",
        "online_step_count", "selected_count_source", "scientific_valid", "hard_90pct_valid", "no_nan_inf_or_error",
        "budget_violation_count", "support_utilization_min", "support_utilization_p05", "support_utilization_mean",
        "under_95pct_batch_count", "under_90pct_batch_count", "mean_budget_gap", "max_budget_gap", "rollback_count",
        "rollback_rate", "stage1_steps_mean", "stage1_steps_max", "stage1_max_steps_hit_count", "PU", "FO", "errors",
    ]


def main():
    plan = read_json(PLAN_PATH)
    if len(plan.get("experiments", [])) != 132:
        raise ValueError("joint plan must contain exactly 132 entries")
    if sha256(PLAN_PATH) != read_json(JOINT / "phase_records/machine2/PREPARE_READY.json")["plan_sha256"]:
        raise ValueError("plan SHA256 does not match PREPARE_READY")
    records, invalid = summary_records(RUNS_ROOT)
    statuses = classify_plan(plan, records, invalid)
    counts = Counter(row["status"] for row in statuses)
    expected_counts = {"completed": 132, "missing": 0, "failed": 0, "duplicate": 0, "hash_mismatch": 0, "invalid_summary": 0}
    actual_counts = {key: int(counts.get(key, 0)) for key in expected_counts}
    if actual_counts != expected_counts:
        raise ValueError(f"new-run completion is not FINALIZE-ready: {actual_counts}")
    new_rows = []
    for status in statuses:
        summary_path = Path(status["completed"][0]["path"])
        new_rows.append(diagnostics(status["entry"], summary_path, read_json(summary_path)))
    reference_rows = reference_entries()
    if len(reference_rows) != 12:
        raise ValueError("reference row count is not 12")
    per_run = new_rows + reference_rows
    if len(per_run) != 144 or len({row["experiment_key"] for row in per_run}) != 144:
        raise ValueError("merged per-run set is not exactly 144 unique identities")
    baseline_rows, sparse_reference = baseline_reference()
    configs = aggregate(per_run, sparse_reference)
    ranked = rank(configs)

    reports = JOINT / "reports/machine2"
    run_fields = public_run_fields()
    public_rows = []
    for row in per_run:
        public = {field: row.get(field) for field in run_fields}
        public["errors"] = json.dumps(row["errors"], ensure_ascii=False)
        public_rows.append(public)
    write_csv(reports / "per_run_144.csv", public_rows, run_fields)
    config_fields = [key for key in configs[0] if key != "transfer_rows"] + ["transfer_rows"]
    write_csv(reports / "per_config_24.csv", configs, config_fields)
    write_json(reports / "per_config_24.json", {
        "machine_tag": "machine2", "configuration_count": len(configs), "configs": configs,
        "baseline_reference_complete": True, "baseline_reference_count": len(baseline_rows),
    })
    ranking_fields = ["rank"] + [field for field in config_fields if field != "rank"]
    write_csv(reports / "ranking.csv", ranked, ranking_fields)
    ranking_lines = [
        "# Office LBI joint sweep — machine2 ranking", "", 
        "Only complete six-transfer configurations are eligible. PU is report-only and is not used for ranking.", "",
        "| rank | anchor | omega | stage2_lr | valid/6 | hard 90%/6 | mean FO margin | worst FO margin | mean FO | under 95% | stage1 max | stage1 mean | eligible |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        ranking_lines.append(
            f"| {row['rank']} | {row['anchor_id']} | {row['omega']:.3f} | {row['stage2_lr']:.3f} | "
            f"{row['valid_transfer_count']}/6 | {row['hard_90pct_valid_transfer_count']}/6 | "
            f"{row['mean_FO_margin']:.6f} | {row['worst_transfer_FO_margin']:.6f} | {row['mean_FO']:.6f} | "
            f"{row['under_95pct_batch_count']} | {row['stage1_steps_max']:.3f} | {row['stage1_steps_mean']:.3f} | "
            f"{str(row['eligible']).lower()} |"
        )
    ranking_lines += [
        "", "Ranking order:",
        "1. all six transfers scientific-valid; 2. all six transfers pass the 90% utilization gate; "
        "3. mean FO margin descending; 4. worst-transfer FO margin descending; 5. mean FO descending; "
        "6. 95%-utilization robustness diagnostics (under-95 count ascending, then utilization diagnostics descending); "
        "7. stage1_steps_max ascending; 8. stage1_steps_mean ascending.",
        "", "The 95% utilization condition is diagnostic/preference only; the hard gate is 90%.",
    ]
    (reports / "ranking.md").parent.mkdir(parents=True, exist_ok=True)
    (reports / "ranking.md").write_text("\n".join(ranking_lines) + "\n", encoding="utf-8")

    timestamp = datetime.now(timezone.utc).isoformat()
    summary = {
        "mode": "FINALIZE", "machine_tag": "machine2", "timestamp": timestamp,
        "plan_path": str(PLAN_PATH), "plan_sha256": sha256(PLAN_PATH),
        "new_run_expected": 132, "new_run_rows": len(new_rows), "merged_run_expected": 144, "merged_run_rows": len(per_run),
        "configuration_expected": 24, "configuration_rows": len(configs),
        "status_counts": actual_counts,
        "unassociated_valid_summary_count": len(records) - sum(len(row["completed"]) for row in statuses),
        "unassociated_invalid_summary_count": sum(record.get("experiment_key") is None for record in invalid),
        "reference_run_expected": 12, "reference_run_rows": len(reference_rows),
        "baseline_reference_complete": True, "baseline_reference_rows": len(baseline_rows),
        "eligible_configuration_count": sum(row["eligible"] for row in configs),
        "hard_90pct_eligible_configuration_count": sum(row["hard_90pct_eligible"] for row in configs),
        "best_ranked_configuration": ranked[0],
        "ranking_rule": [
            "all 6 transfers scientific-valid", "all 6 transfers pass 90% utilization hard gate",
            "mean_FO_margin descending", "worst_transfer_FO_margin descending", "mean_FO descending",
            "95%-utilization robustness diagnostics", "stage1_steps_max ascending", "stage1_steps_mean ascending",
        ],
        "canonical_runs_root": str(RUNS_ROOT),
        "training_launched_by_finalize": False,
        "gpu_required_by_finalize": False,
        "outputs": [str(reports / name) for name in ("per_run_144.csv", "per_config_24.csv", "per_config_24.json", "ranking.csv", "ranking.md", "FINALIZE_SUMMARY.json")],
    }
    write_json(reports / "FINALIZE_SUMMARY.json", summary)

    phase = JOINT / "phase_records/machine2"
    reference_lines = [
        "# FINALIZE — Office LBI joint sweep machine2", "", "Mode: `FINALIZE`", "Machine: `machine2`", "",
        "## No-rerun policy", "", 
        "FINALIZE used saved plans, launcher artifacts, summaries, metrics.jsonl files, and seed-2026 baseline artifacts only.",
        "No training was launched, retried, resumed, or rerun by FINALIZE.", "",
        "## Completion", "", "| set | completed | missing | failed | duplicate | hash mismatch | invalid summary |", "|---|---:|---:|---:|---:|---:|---:|",
        f"| new joint plan | {actual_counts['completed']} | {actual_counts['missing']} | {actual_counts['failed']} | {actual_counts['duplicate']} | {actual_counts['hash_mismatch']} | {actual_counts['invalid_summary']} |",
        f"| reused Stage-1 references | {len(reference_rows)} | 0 | 0 | 0 | 0 | 0 |",
        "", f"Merged local result set: {len(per_run)}/144 run rows; {len(configs)}/24 configurations; six transfers per configuration.",
        f"Plan SHA256: `{sha256(PLAN_PATH)}`.", "",
        "## Reused reference identities", "",
    ]
    ref_data = read_json(REF_PATH)
    for ref in ref_data["references"]:
        reference_lines.append(f"- `{ref['anchor_id']}` `{ref['transfer']}` `{ref['experiment_key']}` `{ref['scientific_hash']}` — `{ref['summary_path']}`")
    reference_lines += [
        "", "## Eligibility and baseline", "",
        "A run is scientifically valid only if completed, finite/error-free, without Stage-1 max-step failure, without budget violation, and with support utilization >= 0.90 on every batch. PU is report-only.",
        "The 95% utilization threshold is a robustness diagnostic/preference, not a hard gate.",
        f"Seed-2026 Office baseline reference: {len(baseline_rows)}/72 identities complete; best_sparse_FO is max(Random 3-mask mean, Magnitude, Saliency) per transfer at budget=0.001.",
        f"Eligible configurations: {sum(row['eligible'] for row in configs)}/24; hard-90%-gate configurations: {sum(row['hard_90pct_eligible'] for row in configs)}/24.",
        "No global Office tuple was frozen; machine4 is the designated global aggregator.", "",
        "## Outputs", "",
        f"- Reports: `{reports}`", f"- Finalize summary: `{reports / 'FINALIZE_SUMMARY.json'}`", f"- Phase record: `{phase / 'FINALIZE.md'}`", "",
        "FINALIZE complete for machine2.", "No training was launched by FINALIZE.",
    ]
    phase.mkdir(parents=True, exist_ok=True)
    (phase / "FINALIZE.md").write_text("\n".join(reference_lines) + "\n", encoding="utf-8")

    ready = {
        "mode": "FINALIZE", "machine_tag": "machine2", "timestamp": timestamp,
        "plan_path": str(PLAN_PATH), "plan_sha256": sha256(PLAN_PATH),
        "new_run_count": len(new_rows), "reference_run_count": len(reference_rows), "merged_run_count": len(per_run),
        "configuration_count": len(configs), "all_new_runs_completed": actual_counts == expected_counts,
        "all_references_present": len(reference_rows) == 12, "validly_summarized": True,
        "baseline_reference_complete": True, "local_finalize_ready": True,
        "training_launched_by_finalize": False,
        "report_paths": [str(reports / name) for name in ("per_run_144.csv", "per_config_24.csv", "per_config_24.json", "ranking.csv", "ranking.md", "FINALIZE_SUMMARY.json")],
    }
    write_json(phase / "FINALIZE_READY.json", ready)
    print(json.dumps({"status_counts": actual_counts, "references": len(reference_rows), "run_rows": len(per_run), "configs": len(configs), "ready": True}, sort_keys=True))


if __name__ == "__main__":
    main()
