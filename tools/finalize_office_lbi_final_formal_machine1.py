#!/usr/bin/env python3
"""Artifact-only FINALIZE for formal Office LBI budget 0.0005 (machine1).

The program deliberately consumes only saved JSON/JSONL/CSV artifacts.  It
does not import training code, inspect CUDA, or start/resume/retry a run.
"""

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiment_logs/shot_otta_office_lbi_final_formal_20260818"
PLAN_PATH = EXP / "plans/machine1/plan.json"
STATUS_PATH = EXP / "status/machine1/experiment_status.json"
RUNS_ROOT = EXP / "runs"
REPORT_ROOT = EXP / "reports/machine1"
PHASE_ROOT = EXP / "phase_records/machine1"
RUNTIME_PROVENANCE = PHASE_ROOT / "runtime_provenance.json"
BASE = ROOT / "experiment_logs/shot_otta_office_seed2026_stage1_20260818"
K = 262
TRANSFERS = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
SPARSE_VARIANTS = {"module_random", "module_magnitude", "module_saliency"}
BASELINE_SHARDS = (
    ("plans/baseline_machine1/plan.json", "status/baseline_machine1/experiment_status.json"),
    ("plans/baseline_machine2/plan.json", "status/machine2/baseline/experiment_status.json"),
    ("plans/machine3/baseline/plan.json", "status/all_machines/baseline_machine3/experiment_status.json"),
)
FROZEN = {
    "requested_budget": 0.0005, "alpha": 0.10, "kappa": 1.0, "nu": 0.25,
    "omega": 0.30, "stage2_lr": 0.020, "stage1_max_steps": 3000,
    "stage2_steps_requested": 1, "lbi_support_threshold": 1e-4,
    "seed": 2026,
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def pctl(values, quantile):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * quantile
    low, high = math.floor(index), math.ceil(index)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (index - low)


def avg(values):
    values = [float(value) for value in values if value is not None]
    return statistics.fmean(values) if values else None


def std(values):
    values = [float(value) for value in values if value is not None]
    return statistics.pstdev(values) if values else None


def transfer(source, target):
    return f"{('A', 'D', 'W')[int(source)]}->{('A', 'D', 'W')[int(target)]}"


def score(summary, field, variant):
    # Random is a 3-mask aggregate in its own completed summary.
    key = f"mean_{field}" if variant == "module_random" else field
    return float(summary[key])


def read_metrics(summary_path):
    path = Path(summary_path).parent / "metrics.jsonl"
    records, errors = [], []
    if not path.exists():
        return records, [f"missing metrics.jsonl: {path}"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: {exc}")
            continue
        if not finite(record):
            errors.append(f"non-finite record: {path}:{line_number}")
        if record.get("event") == "online_step":
            records.append(record)
    if not records:
        errors.append(f"no online_step records: {path}")
    return records, errors


def validate_tuple(entry, summary):
    errors = []
    for name, expected in FROZEN.items():
        actual = summary.get(name, entry.get(name))
        if actual is None or not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
            errors.append(f"{name}={actual!r}, expected {expected!r}")
    for name, expected in (("batch_size", 64), ("workers", 4)):
        actual = summary.get(name)
        if actual is None:
            actual = entry.get("effective_overrides", {}).get(name)
        if actual != expected:
            errors.append(f"{name}={actual!r}, expected {expected!r}")
    return errors


def completed_matches(plan):
    status = {row["experiment_key"]: row for row in load(STATUS_PATH)["experiments"]}
    planned = {entry["experiment_key"]: entry for entry in plan["experiments"]}
    all_summaries = defaultdict(list)
    for path in RUNS_ROOT.rglob("summary.json"):
        try:
            summary = load(path)
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("experiment_key") in planned:
            all_summaries[summary["experiment_key"]].append((path, summary))
    results, details, counts = {}, [], Counter()
    for key, entry in planned.items():
        state = status.get(key)
        summaries = all_summaries.get(key, [])
        issues = []
        if state is None or state.get("plan_status") != "completed" or not summaries:
            issues.append("missing")
        if len(summaries) != 1 or (state and len(state.get("matching_summary_paths", [])) != 1):
            issues.append("duplicate")
        if summaries and any(summary.get("experiment_config_sha256") != entry["experiment_config_sha256"] for _, summary in summaries):
            issues.append("hash mismatch")
        if summaries:
            path, summary = summaries[0]
            if summary.get("status") in {"failed", "incomplete"}:
                issues.append("failed")
            if summary.get("status") != "completed" or not finite(summary):
                issues.append("invalid summary")
            tuple_errors = validate_tuple(entry, summary)
            if tuple_errors:
                issues.append("invalid summary")
        else:
            path, summary, tuple_errors = None, None, []
        for issue in set(issues):
            counts[issue] += 1
        details.append({"experiment_key": key, "expected_output_root": entry["expected_output_root"], "summary_path": str(path) if path else None, "issues": sorted(set(issues)), "tuple_errors": tuple_errors})
        if not issues:
            results[key] = (entry, path, summary)
    counts["completed"] = len(results)
    for name in ("missing", "failed", "duplicate", "hash mismatch", "invalid summary"):
        counts.setdefault(name, 0)
    return results, dict(counts), details


def lbi_row(entry, summary_path, summary):
    records, errors = read_metrics(summary_path)
    values = defaultdict(list)
    counts = []
    for record in records:
        count = record.get("support_param_count")
        if count is None:
            errors.append(f"missing support_param_count at batch {record.get('batch_index')}")
            continue
        counts.append(float(count))
        for name in ("online_runtime_sec", "adapt_runtime_sec", "pu_runtime_sec", "lbi_stage1_runtime_sec", "lbi_stage2_runtime_sec", "peak_gpu_memory_allocated_mb", "peak_gpu_memory_reserved_mb"):
            if record.get(name) is None:
                errors.append(f"missing {name} at batch {record.get('batch_index')}")
            else:
                values[name].append(float(record[name]))
        if record.get("valid_lbi_step") is not True:
            errors.append(f"invalid LBI step at batch {record.get('batch_index')}")
    utilization = [count / K for count in counts]
    gaps = [K - count for count in counts]
    row = {
        "transfer": transfer(entry["source"], entry["target"]), "source": entry["source"], "target": entry["target"],
        "experiment_key": entry["experiment_key"], "experiment_config_sha256": entry["experiment_config_sha256"],
        "summary_path": str(summary_path), "metrics_path": str(Path(summary_path).parent / "metrics.jsonl"),
        "PU": float(summary["PU-Acc"]), "FO": float(summary["FO-Acc"]), "batch_count": len(records),
        "support_utilization_min": min(utilization) if utilization else None,
        "support_utilization_p05": pctl(utilization, .05), "support_utilization_mean": avg(utilization),
        "under_95pct_batch_count": sum(value < .95 for value in utilization),
        "under_90pct_batch_count": sum(value < .90 for value in utilization),
        "mean_budget_gap": avg(gaps), "max_budget_gap": max(gaps) if gaps else None,
        "rollback_rate": avg([bool(record.get("stage1_rollback_used", False)) for record in records]),
        "stage1_steps_mean": avg([record.get("stage1_steps_completed") for record in records]),
        "stage1_steps_max": max((record.get("stage1_steps_completed") for record in records), default=None),
        "stage1_max_steps_hit_count": sum(bool(record.get("max_steps_hit", False)) for record in records),
        "scientific_valid": not errors and all(count <= K for count in counts) and not any(record.get("max_steps_hit", False) for record in records),
        "errors": errors,
    }
    for name, output in (("online_runtime_sec", "online_batch_runtime"), ("adapt_runtime_sec", "adapt_runtime"), ("pu_runtime_sec", "pu_runtime"), ("lbi_stage1_runtime_sec", "lbi_stage1_runtime"), ("lbi_stage2_runtime_sec", "lbi_stage2_runtime")):
        row[f"{output}_mean_sec"] = avg(values[name])
        row[f"{output}_std_sec"] = std(values[name])
        row[f"{output}_total_sec"] = sum(values[name])
    row.update({
        "online_batch_runtime_median_sec": pctl(values["online_runtime_sec"], .5),
        "online_batch_runtime_p95_sec": pctl(values["online_runtime_sec"], .95),
        "online_compute_runtime_sec": sum(values["online_runtime_sec"]),
        "gpu_peak_allocated_mean_mb": avg(values["peak_gpu_memory_allocated_mb"]),
        "gpu_peak_allocated_max_mb": max(values["peak_gpu_memory_allocated_mb"]) if values["peak_gpu_memory_allocated_mb"] else None,
        "gpu_peak_reserved_mean_mb": avg(values["peak_gpu_memory_reserved_mb"]),
        "gpu_peak_reserved_max_mb": max(values["peak_gpu_memory_reserved_mb"]) if values["peak_gpu_memory_reserved_mb"] else None,
        "fo_eval_runtime_sec": float(summary["fo_eval_runtime_sec"]), "wall_runtime_sec": float(summary["wall_runtime_sec"]),
        "runtime_resume_used": bool(summary.get("runtime_resume_used")), "runtime_segment_count": int(summary.get("runtime_segment_count", 1)),
        "gpu_name": summary.get("gpu_name"), "torch_version": summary.get("torch_version"), "cuda_version": summary.get("cuda_version"),
        "runtime_comparable": summary.get("runtime_comparable"),
    })
    return row


def baseline_rows():
    rows = []
    for plan_rel, status_rel in BASELINE_SHARDS:
        plan = load(BASE / plan_rel)
        status = {row["experiment_key"]: row for row in load(BASE / status_rel)["experiments"]}
        for entry in plan["experiments"]:
            if entry["variant"] not in SPARSE_VARIANTS | {"module_dense", "full_dense", "source_only"}:
                continue
            if entry["variant"] in SPARSE_VARIANTS and float(entry["requested_budget"]) != 0.0005:
                continue
            state = status.get(entry["experiment_key"])
            paths = state.get("matching_summary_paths", []) if state else []
            if state is None or state.get("plan_status") != "completed" or len(paths) != 1:
                raise RuntimeError(f"incomplete formal baseline: {entry['experiment_key']}")
            summary = load(paths[0])
            if summary.get("status") != "completed" or not finite(summary):
                raise RuntimeError(f"invalid formal baseline summary: {paths[0]}")
            rows.append({
                "source": int(entry["source"]), "target": int(entry["target"]), "transfer": transfer(entry["source"], entry["target"]),
                "variant": entry["variant"], "PU": score(summary, "PU-Acc", entry["variant"]), "FO": score(summary, "FO-Acc", entry["variant"]),
                "online_batch_runtime_mean_sec": float(summary["online_batch_runtime_mean_sec"]),
                "online_compute_runtime_sec": float(summary["online_compute_runtime_sec"]),
                "gpu_peak_allocated_max_mb": float(summary["gpu_peak_allocated_max_mb"]),
                "runtime_comparable": summary.get("runtime_comparable"), "gpu_name": summary.get("gpu_name"),
                "summary_path": paths[0], "experiment_key": entry["experiment_key"],
            })
    if len(rows) != 36:
        raise RuntimeError(f"expected 36 requested seed-2026 formal baseline identities, found {len(rows)}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["target"], row["variant"])].append(row)
    if set((source, target) for source, target, _ in grouped) != TRANSFERS or any(len(items) != 1 for items in grouped.values()):
        raise RuntimeError("baseline identities are incomplete or duplicated")
    return rows, grouped


def baseline_comparison(lbi_rows, groups):
    results = []
    for row in lbi_rows:
        key = (row["source"], row["target"])
        candidates = {variant: groups[key + (variant,)][0] for variant in SPARSE_VARIANTS}
        best_sparse = max(candidates.values(), key=lambda item: item["FO"])
        module_dense = groups[key + ("module_dense",)][0]
        full_dense = groups[key + ("full_dense",)][0]
        source_only = groups[key + ("source_only",)][0]
        results.append({
            "transfer": row["transfer"], "LBI_PU": row["PU"], "LBI_FO": row["FO"],
            "best_sparse_variant": best_sparse["variant"], "best_sparse_PU": best_sparse["PU"], "best_sparse_FO": best_sparse["FO"],
            "module_dense_PU": module_dense["PU"], "module_dense_FO": module_dense["FO"],
            "full_dense_PU": full_dense["PU"], "full_dense_FO": full_dense["FO"],
            "source_only_PU": source_only["PU"], "source_only_FO": source_only["FO"],
            "LBI_minus_best_sparse_PU": row["PU"] - best_sparse["PU"], "LBI_minus_best_sparse_FO": row["FO"] - best_sparse["FO"],
            "LBI_minus_module_dense_PU": row["PU"] - module_dense["PU"], "LBI_minus_module_dense_FO": row["FO"] - module_dense["FO"],
            "LBI_minus_full_dense_PU": row["PU"] - full_dense["PU"], "LBI_minus_full_dense_FO": row["FO"] - full_dense["FO"],
            "LBI_minus_source_only_PU": row["PU"] - source_only["PU"], "LBI_minus_source_only_FO": row["FO"] - source_only["FO"],
        })
    overall = {"transfer_count": len(results)}
    for name in results[0]:
        if name != "transfer" and all(isinstance(row[name], (int, float)) for row in results):
            overall[f"mean_{name}"] = avg([row[name] for row in results])
    by_variant = {}
    for variant in ("module_random", "module_magnitude", "module_saliency", "module_dense", "full_dense", "source_only"):
        subset = [item for item in groups.values() if item[0]["variant"] == variant]
        by_variant[variant] = {
            "mean_PU": avg([item[0]["PU"] for item in subset]), "mean_FO": avg([item[0]["FO"] for item in subset]),
            "online_batch_runtime_mean_sec": avg([item[0]["online_batch_runtime_mean_sec"] for item in subset]),
            "online_compute_runtime_sec": sum(item[0]["online_compute_runtime_sec"] for item in subset),
            "gpu_peak_allocated_max_mb": max(item[0]["gpu_peak_allocated_max_mb"] for item in subset),
        }
    return results, overall, by_variant


def main():
    plan = load(PLAN_PATH)
    if plan.get("experiment_count") != 6 or len(plan.get("experiments", [])) != 6:
        raise RuntimeError("formal plan is not exactly six entries")
    if len({entry["experiment_key"] for entry in plan["experiments"]}) != 6:
        raise RuntimeError("formal plan has duplicate experiment keys")
    matches, status, details = completed_matches(plan)
    if any(status[name] for name in ("missing", "failed", "duplicate", "hash mismatch", "invalid summary")):
        PHASE_ROOT.mkdir(parents=True, exist_ok=True)
        (PHASE_ROOT / "FINALIZE.md").write_text(
            "# FINALIZE — Office LBI final formal machine1\n\n"
            f"Completion is incomplete: `{status}`. Exact identities/paths are recorded in `status/machine1/experiment_status.json`; FINALIZE did not launch training.\n",
            encoding="utf-8",
        )
        print(json.dumps({"finalize_ready": False, "status": status, "details": details}, ensure_ascii=True))
        return
    rows = [lbi_row(entry, path, summary) for entry, path, summary in matches.values()]
    if len(rows) != 6 or {(row["source"], row["target"]) for row in rows} != TRANSFERS:
        raise RuntimeError("completed LBI set does not cover exactly six Office transfers")
    if any(not row["scientific_valid"] for row in rows):
        raise RuntimeError("completed formal LBI run has invalid batch records")
    baseline, baseline_groups = baseline_rows()
    comparison, comparison_overall, baseline_efficiency = baseline_comparison(rows, baseline_groups)
    all_batches = [record for row in rows for record in read_metrics(row["summary_path"])[0]]
    support = [float(record["support_param_count"]) / K for record in all_batches]
    gaps = [K - float(record["support_param_count"]) for record in all_batches]
    stage1 = [float(record["stage1_steps_completed"]) for record in all_batches]
    efficiency = {
        "online_batch_runtime_mean_sec": avg([record["online_runtime_sec"] for record in all_batches]),
        "online_batch_runtime_std_sec": std([record["online_runtime_sec"] for record in all_batches]),
        "online_batch_runtime_median_sec": pctl([record["online_runtime_sec"] for record in all_batches], .5),
        "online_batch_runtime_p95_sec": pctl([record["online_runtime_sec"] for record in all_batches], .95),
        "online_compute_runtime_sec": sum(record["online_runtime_sec"] for record in all_batches),
        "adapt_runtime_mean_sec": avg([record["adapt_runtime_sec"] for record in all_batches]), "adapt_runtime_std_sec": std([record["adapt_runtime_sec"] for record in all_batches]), "adapt_runtime_total_sec": sum(record["adapt_runtime_sec"] for record in all_batches),
        "pu_runtime_mean_sec": avg([record["pu_runtime_sec"] for record in all_batches]), "pu_runtime_std_sec": std([record["pu_runtime_sec"] for record in all_batches]), "pu_runtime_total_sec": sum(record["pu_runtime_sec"] for record in all_batches),
        "lbi_stage1_runtime_mean_sec": avg([record["lbi_stage1_runtime_sec"] for record in all_batches]), "lbi_stage1_runtime_std_sec": std([record["lbi_stage1_runtime_sec"] for record in all_batches]), "lbi_stage1_runtime_total_sec": sum(record["lbi_stage1_runtime_sec"] for record in all_batches),
        "lbi_stage2_runtime_mean_sec": avg([record["lbi_stage2_runtime_sec"] for record in all_batches]), "lbi_stage2_runtime_std_sec": std([record["lbi_stage2_runtime_sec"] for record in all_batches]), "lbi_stage2_runtime_total_sec": sum(record["lbi_stage2_runtime_sec"] for record in all_batches),
        "gpu_peak_allocated_mean_mb": avg([record["peak_gpu_memory_allocated_mb"] for record in all_batches]), "gpu_peak_allocated_max_mb": max(record["peak_gpu_memory_allocated_mb"] for record in all_batches),
        "gpu_peak_reserved_mean_mb": avg([record["peak_gpu_memory_reserved_mb"] for record in all_batches]), "gpu_peak_reserved_max_mb": max(record["peak_gpu_memory_reserved_mb"] for record in all_batches),
        "fo_eval_runtime_sec": sum(row["fo_eval_runtime_sec"] for row in rows), "wall_runtime_sec": sum(row["wall_runtime_sec"] for row in rows),
        "runtime_resume_used": any(row["runtime_resume_used"] for row in rows), "runtime_segment_count": sum(row["runtime_segment_count"] for row in rows),
        "batch_count": len(all_batches),
    }
    provenance = load(RUNTIME_PROVENANCE)
    baseline_models = sorted({row["gpu_name"] for row in baseline if row["gpu_name"]})
    lbi_models = sorted({row["gpu_name"] for row in rows if row["gpu_name"]})
    comparable = bool(all(row["runtime_comparable"] is True for row in rows) and provenance.get("gpu_model_uniform") is True and lbi_models == baseline_models)
    hardware = {"runtime_comparable": comparable, "reason": "all six LBI runs used the uniform baseline-matched GPU model with runtime_comparable=true" if comparable else "runtime flags or GPU model provenance do not match the formal baseline", "lbi_gpu_models": lbi_models, "baseline_gpu_models": baseline_models, "lbi_torch_versions": sorted({row["torch_version"] for row in rows}), "lbi_cuda_versions": sorted({row["cuda_version"] for row in rows}), "execution_provenance_path": str(RUNTIME_PROVENANCE)}
    aggregate = {
        "schema_version": 1, "machine_tag": "machine1", "budget": .0005, "K": K, "frozen_tuple": FROZEN,
        "completion": status, "PU": avg([row["PU"] for row in rows]), "FO": avg([row["FO"] for row in rows]),
        "support_utilization_min": min(support), "support_utilization_p05": pctl(support, .05), "support_utilization_mean": avg(support),
        "under_95pct_batch_count": sum(value < .95 for value in support), "under_90pct_batch_count": sum(value < .90 for value in support),
        "mean_budget_gap": avg(gaps), "max_budget_gap": max(gaps), "rollback_rate": avg([bool(record.get("stage1_rollback_used")) for record in all_batches]),
        "stage1_steps_mean": avg(stage1), "stage1_steps_max": max(stage1), "stage1_max_steps_hit_count": sum(bool(record.get("max_steps_hit")) for record in all_batches),
        "efficiency": efficiency, "hardware_comparability": hardware, "baseline_comparison": comparison_overall,
        "training_launched_by_finalize": False, "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    per_fields = list(rows[0])
    comp_fields = list(comparison[0])
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT_ROOT / "per_transfer.csv", rows, per_fields)
    dump(REPORT_ROOT / "per_transfer.json", {"transfers": rows, "baseline_comparison": comparison})
    write_csv(REPORT_ROOT / "final_budget_summary.csv", [aggregate], [key for key in aggregate if key not in {"frozen_tuple", "completion", "efficiency", "hardware_comparability", "baseline_comparison"}])
    dump(REPORT_ROOT / "final_budget_summary.json", aggregate)
    dump(REPORT_ROOT / "efficiency_summary.json", {"efficiency": efficiency, "baseline_efficiency": baseline_efficiency, "hardware_comparability": hardware})
    lines = ["# Office formal LBI final result — machine1 / budget 0.0005", "", "No training, retry, resume, or rerun was launched by FINALIZE.", "", "## Completion", "", f"completed={status['completed']}; missing={status['missing']}; failed={status['failed']}; duplicate={status['duplicate']}; hash mismatch={status['hash mismatch']}; invalid summary={status['invalid summary']}.", "", "## Results", "", "| transfer | PU | FO | best sparse | LBI - best sparse FO | LBI - module dense FO | LBI - full dense FO |", "|---|---:|---:|---|---:|---:|---:|"]
    for item in comparison:
        lines.append(f"| {item['transfer']} | {item['LBI_PU']:.6f} | {item['LBI_FO']:.6f} | {item['best_sparse_variant']} | {item['LBI_minus_best_sparse_FO']:+.6f} | {item['LBI_minus_module_dense_FO']:+.6f} | {item['LBI_minus_full_dense_FO']:+.6f} |")
    lines += ["", f"Office mean PU: `{aggregate['PU']:.6f}`. Office mean FO: `{aggregate['FO']:.6f}`.", f"Best-sparse mean FO: `{comparison_overall['mean_best_sparse_FO']:.6f}`; LBI margin: `{comparison_overall['mean_LBI_minus_best_sparse_FO']:+.6f}`.", f"Module-dense mean FO: `{comparison_overall['mean_module_dense_FO']:.6f}`; LBI margin: `{comparison_overall['mean_LBI_minus_module_dense_FO']:+.6f}`.", f"Full-dense mean FO: `{comparison_overall['mean_full_dense_FO']:.6f}`; LBI margin: `{comparison_overall['mean_LBI_minus_full_dense_FO']:+.6f}`.", "", "## Efficiency and hardware", "", f"Primary GPU-memory number (max allocated): `{efficiency['gpu_peak_allocated_max_mb']:.6f} MB`.", f"Online batch runtime mean/std/median/p95: `{efficiency['online_batch_runtime_mean_sec']:.6f}` / `{efficiency['online_batch_runtime_std_sec']:.6f}` / `{efficiency['online_batch_runtime_median_sec']:.6f}` / `{efficiency['online_batch_runtime_p95_sec']:.6f}` sec.", f"runtime_comparable = `{str(comparable).lower()}`: {hardware['reason']}.", "", "## Artifacts", "", "- `reports/machine1/per_transfer.csv` and `.json`: all six formal LBI transfers and margins.", "- `reports/machine1/final_budget_summary.*`: aggregate PU/FO, support, budget, and comparison results.", "- `reports/machine1/efficiency_summary.json`: batch-record efficiency and baseline comparison.", "", "FINALIZE complete for machine1 / budget 0.0005.", "No training was launched by FINALIZE."]
    efficiency_index = lines.index("## Efficiency and hardware")
    lines[efficiency_index:efficiency_index] = [
        f"Source-only mean FO: `{comparison_overall['mean_source_only_FO']:.6f}`; LBI margin: `{comparison_overall['mean_LBI_minus_source_only_FO']:+.6f}`.",
        f"Mean PU margins — best sparse: `{comparison_overall['mean_LBI_minus_best_sparse_PU']:+.6f}`; module dense: `{comparison_overall['mean_LBI_minus_module_dense_PU']:+.6f}`; full dense: `{comparison_overall['mean_LBI_minus_full_dense_PU']:+.6f}`; source only: `{comparison_overall['mean_LBI_minus_source_only_PU']:+.6f}`.",
        "",
    ]
    runtime_line = f"runtime_comparable = `{str(comparable).lower()}`: {hardware['reason']}."
    runtime_index = lines.index(runtime_line)
    lines[runtime_index:runtime_index] = [
        "", "| method | mean online-batch sec | total online-compute sec | primary GPU peak MB |",
        "|---|---:|---:|---:|",
        f"| LBI | {efficiency['online_batch_runtime_mean_sec']:.6f} | {efficiency['online_compute_runtime_sec']:.6f} | {efficiency['gpu_peak_allocated_max_mb']:.6f} |",
    ]
    for variant, values in baseline_efficiency.items():
        lines.insert(runtime_index + 4, f"| {variant} | {values['online_batch_runtime_mean_sec']:.6f} | {values['online_compute_runtime_sec']:.6f} | {values['gpu_peak_allocated_max_mb']:.6f} |")
        runtime_index += 1
    (REPORT_ROOT / "final_budget_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    PHASE_ROOT.mkdir(parents=True, exist_ok=True)
    (PHASE_ROOT / "FINALIZE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ready = {"machine_tag": "machine1", "budget": .0005, "plan_path": str(PLAN_PATH), "plan_sha256": sha256(PLAN_PATH), "completion": status, "summary_paths": [row["summary_path"] for row in rows], "reports": [str(REPORT_ROOT / name) for name in ("per_transfer.csv", "per_transfer.json", "final_budget_summary.csv", "final_budget_summary.json", "final_budget_summary.md", "efficiency_summary.json")], "runtime_comparable": comparable, "training_launched_by_finalize": False, "timestamp": aggregate["timestamp"]}
    dump(PHASE_ROOT / "FINALIZE_READY.json", ready)
    print(json.dumps({"finalize_ready": True, "completion": status, "mean_PU": aggregate["PU"], "mean_FO": aggregate["FO"], "runtime_comparable": comparable}, sort_keys=True))


if __name__ == "__main__":
    main()
