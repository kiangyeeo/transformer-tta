#!/usr/bin/env python3
"""Artifact-only FINALIZE for formal Office LBI budget 0.001 (machine2)."""

import csv
import datetime as dt
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools.check_experiment_status import scan_run_summaries


ROOT = PROJECT / "experiment_logs/shot_otta_office_lbi_final_formal_20260818"
BASELINE_ROOT = PROJECT / "experiment_logs/shot_otta_office_seed2026_stage1_20260818"
PLAN_PATH = ROOT / "plans/machine2/plan.json"
RUNS_ROOT = ROOT / "runs"
REPORTS = ROOT / "reports/machine2"
PHASE = ROOT / "phase_records/machine2"
HARDWARE_PATH = PHASE / "RUNTIME_HARDWARE.json"
PREPARE_READY_PATH = PHASE / "PREPARE_READY.json"
K = 524
TRANSFERS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
NAMES = {(0, 1): "A->D", (0, 2): "A->W", (1, 0): "D->A", (1, 2): "D->W", (2, 0): "W->A", (2, 1): "W->D"}
FROZEN = {
    "requested_budget": 0.001, "alpha": 0.10, "kappa": 1.0, "nu": 0.25,
    "omega": 0.30, "stage2_lr": 0.020, "stage1_max_steps": 3000,
    "stage2_steps_requested": 1, "lbi_support_threshold": 1e-4,
    "budget_tolerance": 1e-4, "delta_nonzero_tolerance": 1e-12, "seed": 2026,
}
BASELINE_PLANS = (
    BASELINE_ROOT / "plans/baseline_machine1/plan.json",
    BASELINE_ROOT / "plans/baseline_machine2/plan.json",
    BASELINE_ROOT / "plans/machine3/baseline/plan.json",
)
REQUIRED_VARIANTS = {"module_random", "module_magnitude", "module_saliency", "module_dense", "full_dense", "source_only"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def number(value, name):
    if isinstance(value, bool) or value is None:
        raise ValueError(f"missing/non-numeric {name}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}")
    return value


def mean(values):
    return statistics.fmean(values) if values else None


def std(values):
    return statistics.pstdev(values) if values else None


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (position - lo)


def close(actual, expected):
    return math.isclose(number(actual, "frozen value"), expected, rel_tol=0.0, abs_tol=1e-12)


def scan_index(root):
    records, invalid = scan_run_summaries(str(root))
    index = defaultdict(list)
    for record in records:
        index[record["experiment_key"]].append(record)
    return index, invalid


def completion(plan):
    index, invalid = scan_index(RUNS_ROOT)
    invalid_keys = {item.get("experiment_key") for item in invalid if item.get("experiment_key")}
    statuses = []
    for entry in plan["experiments"]:
        candidates = index.get(entry["experiment_key"], [])
        exact = [item for item in candidates if item.get("experiment_config_sha256") == entry["experiment_config_sha256"]]
        mismatch = [item for item in candidates if item.get("experiment_config_sha256") != entry["experiment_config_sha256"]]
        completed = [item for item in exact if item.get("status") == "completed"]
        failed = [item for item in exact if item.get("status") in {"failed", "incomplete"}]
        if entry["experiment_key"] in invalid_keys:
            state = "invalid summary"
        elif mismatch:
            state = "hash mismatch"
        elif len(exact) > 1:
            state = "duplicate"
        elif len(completed) == 1:
            state = "completed"
        elif failed:
            state = "failed"
        else:
            state = "missing"
        statuses.append({"entry": entry, "state": state, "record": completed[0] if state == "completed" else None,
                         "candidate_paths": [item.get("summary_path") for item in candidates]})
    counts = Counter(item["state"] for item in statuses)
    return statuses, {key: counts[key] for key in ("completed", "missing", "failed", "duplicate", "hash mismatch", "invalid summary")}


def frozen_errors(entry, summary, manifest, online):
    errors = []
    for key, expected in FROZEN.items():
        try:
            if not close(summary.get(key), expected):
                errors.append(f"summary {key}={summary.get(key)!r}")
        except ValueError:
            errors.append(f"summary {key} missing")
    if summary.get("variant") != "module_lbi":
        errors.append("summary variant")
    if summary.get("runtime_comparable") is not True or manifest.get("runtime_comparable") is not True:
        errors.append("runtime_comparable")
    data = manifest.get("scientific_config", {}).get("data", {})
    if data.get("batch_size") != 64 or data.get("workers") != 4:
        errors.append("manifest batch_size/workers")
    command = manifest.get("command", [])
    if "--no-save-model" not in (" ".join(command) if isinstance(command, list) else str(command)):
        errors.append("manifest save_model")
    if entry.get("effective_overrides", {}).get("save_model") is not False:
        errors.append("plan save_model")
    for record in online:
        try:
            for key, expected in FROZEN.items():
                if key != "seed" and not close(record.get(key), expected):
                    errors.append(f"batch {record.get('batch_index')} {key}")
                    break
            batch_size = record.get("batch_size")
            if not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
                errors.append(f"batch {record.get('batch_index')} size")
        except ValueError:
            errors.append(f"batch {record.get('batch_index')} frozen tuple missing")
    return sorted(set(errors))


def diagnose(entry, record):
    summary_path = Path(record["summary_path"])
    summary = record["summary"]
    manifest = read(summary_path.with_name("manifest.json"))
    online, errors = [], []
    try:
        for line in summary_path.with_name("metrics.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("event") == "online_step":
                online.append(item)
            elif item.get("event") == "error":
                errors.append("error event")
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"metrics parse: {exc}")
    if not online:
        errors.append("no online steps")
    if not finite(summary) or not all(finite(item) for item in online):
        errors.append("NaN/Inf")
    if summary.get("status") != "completed" or summary.get("valid_lbi_run") is not True:
        errors.append("summary completion/validity")
    if len(online) != summary.get("online_steps"):
        errors.append("online step count")
    for key in ("experiment_key", "experiment_config_sha256"):
        if summary.get(key) != entry.get(key) or manifest.get(key) != entry.get(key):
            errors.append(f"identity {key}")
    errors.extend(frozen_errors(entry, summary, manifest, online))
    selected, support, gaps, stages, rollbacks, maxhits = [], [], [], [], [], []
    batch = defaultdict(list)
    fields = ("online_runtime_sec", "adapt_runtime_sec", "pu_runtime_sec", "lbi_stage1_runtime_sec", "lbi_stage2_runtime_sec", "peak_gpu_memory_allocated_mb", "peak_gpu_memory_reserved_mb")
    for item in online:
        try:
            selected_count = number(item.get("selected_param_count"), "selected_param_count")
            support_count = number(item.get("support_param_count", item.get("stage1_support_count")), "support_param_count")
            selected.append(selected_count)
            support.append(support_count / K)
            gaps.append(K - selected_count)
            stages.append(number(item.get("stage1_steps_completed"), "stage1_steps_completed"))
            rollbacks.append(bool(item.get("stage1_rollback_used", False)))
            maxhits.append(bool(item.get("max_steps_hit", False)) or item.get("stage1_stop_reason") in {"max_steps", "max_steps_reached"})
            if selected_count > K or support_count > K:
                errors.append("support exceeds K")
            if item.get("valid_lbi_step") is not True:
                errors.append("invalid LBI batch")
            for field in fields:
                batch[field].append(number(item.get(field), field))
        except ValueError as exc:
            errors.append(str(exc))
    if any(maxhits):
        errors.append("stage1 max_steps hit")
    source, target = int(entry["source"]), int(entry["target"])
    return {
        "source": source, "target": target, "transfer": NAMES[(source, target)],
        "experiment_key": entry["experiment_key"], "experiment_config_sha256": entry["experiment_config_sha256"],
        "summary_path": str(summary_path), "metrics_path": str(summary_path.with_name("metrics.jsonl")),
        "PU": number(summary.get("PU-Acc"), "PU-Acc"), "FO": number(summary.get("FO-Acc"), "FO-Acc"),
        "batch_count": len(online), "support_utilization_min": min(support), "support_utilization_p05": percentile(support, .05),
        "support_utilization_mean": mean(support), "under_95pct_batch_count": sum(value < .95 for value in support),
        "under_90pct_batch_count": sum(value < .90 for value in support), "mean_budget_gap": mean(gaps), "max_budget_gap": max(gaps),
        "rollback_count": sum(rollbacks), "rollback_rate": mean(rollbacks), "stage1_steps_mean": mean(stages),
        "stage1_steps_max": max(stages), "stage1_max_steps_hit_count": sum(maxhits),
        "scientific_valid": not errors, "validation_errors": sorted(set(errors)), "runtime_comparable": summary.get("runtime_comparable"),
        "gpu_name": summary.get("gpu_name"), "torch_version": summary.get("torch_version"), "cuda_version": summary.get("cuda_version"),
        "runtime_resume_used": bool(summary.get("runtime_resume_used")), "runtime_segment_count": int(summary.get("runtime_segment_count", 1)),
        "fo_eval_runtime_sec": number(summary.get("fo_eval_runtime_sec"), "fo_eval_runtime_sec"), "wall_runtime_sec": number(summary.get("wall_runtime_sec"), "wall_runtime_sec"),
        "batch_metrics": batch, "support_utilizations": support, "budget_gaps": gaps, "stage1_steps": stages,
    }


def baseline_rows():
    index, invalid = scan_index(BASELINE_ROOT / "runs")
    if invalid:
        raise RuntimeError(f"invalid seed-2026 formal baseline summaries: {len(invalid)}")
    rows = []
    for plan_path in BASELINE_PLANS:
        for entry in read(plan_path)["experiments"]:
            exact = [item for item in index.get(entry["experiment_key"], []) if item.get("experiment_config_sha256") == entry["experiment_config_sha256"] and item.get("status") == "completed"]
            if len(exact) != 1:
                raise RuntimeError(f"baseline identity incomplete/duplicate: {entry['experiment_key']}")
            summary = exact[0]["summary"]
            if not finite(summary):
                raise RuntimeError(f"invalid baseline summary: {exact[0]['summary_path']}")
            accuracy = lambda metric: number(summary.get(f"mean_{metric}") if entry["variant"] == "module_random" else summary.get(metric), f"baseline {metric}")
            rows.append({"source": int(entry["source"]), "target": int(entry["target"]), "variant": entry["variant"], "budget": entry.get("requested_budget"),
                         "PU": accuracy("PU-Acc"), "FO": accuracy("FO-Acc"), "online_batch_runtime_mean_sec": number(summary.get("online_batch_runtime_mean_sec"), "baseline runtime"),
                         "online_compute_runtime_sec": number(summary.get("online_compute_runtime_sec"), "baseline compute"), "gpu_peak_allocated_max_mb": number(summary.get("gpu_peak_allocated_max_mb"), "baseline memory"),
                         "gpu_name": summary.get("gpu_name"), "torch_version": summary.get("torch_version"), "cuda_version": summary.get("cuda_version"), "runtime_comparable": summary.get("runtime_comparable")})
    if len(rows) != 72:
        raise RuntimeError(f"expected 72 seed-2026 baseline identities, found {len(rows)}")
    return rows


def add_baseline_comparisons(rows, baselines):
    grouped = defaultdict(dict)
    for item in baselines:
        key = (item["source"], item["target"])
        if item["variant"] in {"module_random", "module_magnitude", "module_saliency"} and math.isclose(float(item["budget"]), .001):
            grouped[key][item["variant"]] = item
        elif item["variant"] in {"module_dense", "full_dense", "source_only"}:
            grouped[key][item["variant"]] = item
    for row in rows:
        values = grouped[(row["source"], row["target"])]
        if set(values) != REQUIRED_VARIANTS:
            raise RuntimeError(f"baseline comparison incomplete: {row['transfer']}")
        best = max((values[name] for name in ("module_random", "module_magnitude", "module_saliency")), key=lambda item: item["FO"])
        labels = {"random_3mask_mean": values["module_random"], "magnitude": values["module_magnitude"], "saliency": values["module_saliency"], "best_sparse": best,
                  "module_dense": values["module_dense"], "full_dense": values["full_dense"], "source_only": values["source_only"]}
        row["best_sparse_variant"] = best["variant"]
        for label, item in labels.items():
            row[f"{label}_PU"], row[f"{label}_FO"] = item["PU"], item["FO"]
        for label in ("best_sparse", "module_dense", "full_dense", "source_only"):
            row[f"LBI_minus_{label}_PU"] = row["PU"] - row[f"{label}_PU"]
            row[f"LBI_minus_{label}_FO"] = row["FO"] - row[f"{label}_FO"]


def aggregate(rows):
    batch = defaultdict(list)
    for row in rows:
        for key, values in row["batch_metrics"].items():
            batch[key].extend(values)
    support = [value for row in rows for value in row["support_utilizations"]]
    gaps = [value for row in rows for value in row["budget_gaps"]]
    stages = [value for row in rows for value in row["stage1_steps"]]
    output = {"budget": .001, "K": K, "alpha": .10, "kappa": 1.0, "nu": .25, "omega": .30, "stage2_lr": .020,
              "completed_transfer_count": len(rows), "Office_mean_PU": mean([row["PU"] for row in rows]), "Office_mean_FO": mean([row["FO"] for row in rows]),
              "support_utilization_min": min(support), "support_utilization_p05": percentile(support, .05), "support_utilization_mean": mean(support),
              "under_95pct_batch_count": sum(row["under_95pct_batch_count"] for row in rows), "under_90pct_batch_count": sum(row["under_90pct_batch_count"] for row in rows),
              "mean_budget_gap": mean(gaps), "max_budget_gap": max(gaps), "rollback_rate": sum(row["rollback_count"] for row in rows) / sum(row["batch_count"] for row in rows),
              "stage1_steps_mean": mean(stages), "stage1_steps_max": max(stages), "stage1_max_steps_hit_count": sum(row["stage1_max_steps_hit_count"] for row in rows),
              "runtime_resume_used": any(row["runtime_resume_used"] for row in rows), "runtime_segment_count": sum(row["runtime_segment_count"] for row in rows),
              "fo_eval_runtime_sec": sum(row["fo_eval_runtime_sec"] for row in rows), "wall_runtime_sec": sum(row["wall_runtime_sec"] for row in rows)}
    for source, prefix in (("online_runtime_sec", "online_batch_runtime"), ("adapt_runtime_sec", "adapt_runtime"), ("pu_runtime_sec", "pu_runtime"), ("lbi_stage1_runtime_sec", "lbi_stage1_runtime"), ("lbi_stage2_runtime_sec", "lbi_stage2_runtime")):
        output[f"{prefix}_mean_sec"], output[f"{prefix}_std_sec"], output[f"{prefix}_total_sec"] = mean(batch[source]), std(batch[source]), sum(batch[source])
    output["online_batch_runtime_median_sec"] = percentile(batch["online_runtime_sec"], .5)
    output["online_batch_runtime_p95_sec"] = percentile(batch["online_runtime_sec"], .95)
    output["online_compute_runtime_sec"] = sum(batch["online_runtime_sec"])
    for source, prefix in (("peak_gpu_memory_allocated_mb", "gpu_peak_allocated"), ("peak_gpu_memory_reserved_mb", "gpu_peak_reserved")):
        output[f"{prefix}_mean_mb"], output[f"{prefix}_max_mb"] = mean(batch[source]), max(batch[source])
    for label in ("random_3mask_mean", "magnitude", "saliency", "best_sparse", "module_dense", "full_dense", "source_only"):
        output[f"{label}_mean_FO"] = mean([row[f"{label}_FO"] for row in rows])
        output[f"LBI_minus_{label}_mean_FO"] = output["Office_mean_FO"] - output[f"{label}_mean_FO"]
        output[f"{label}_mean_PU"] = mean([row[f"{label}_PU"] for row in rows])
        output[f"LBI_minus_{label}_mean_PU"] = output["Office_mean_PU"] - output[f"{label}_mean_PU"]
    return output


def baseline_efficiency(baselines):
    output = {}
    for label, variant in (("random_3mask_mean", "module_random"), ("magnitude", "module_magnitude"), ("saliency", "module_saliency"),
                           ("module_dense", "module_dense"), ("full_dense", "full_dense"), ("source_only", "source_only")):
        rows = [row for row in baselines if row["variant"] == variant and (variant not in {"module_random", "module_magnitude", "module_saliency"} or math.isclose(float(row["budget"]), .001))]
        if len(rows) != 6:
            raise RuntimeError(f"expected six baseline efficiency rows for {label}, found {len(rows)}")
        output[label] = {"online_batch_runtime_mean_sec": mean([row["online_batch_runtime_mean_sec"] for row in rows]),
                         "online_compute_runtime_sec": sum(row["online_compute_runtime_sec"] for row in rows),
                         "gpu_peak_allocated_max_mb": max(row["gpu_peak_allocated_max_mb"] for row in rows)}
    return output


def hardware_audit(rows, baselines):
    provenance = read(HARDWARE_PATH)
    new = {"gpu_models": sorted({row["gpu_name"] for row in rows}), "torch_versions": sorted({row["torch_version"] for row in rows}), "cuda_versions": sorted({row["cuda_version"] for row in rows})}
    baseline = {"gpu_models": sorted({row["gpu_name"] for row in baselines}), "torch_versions": sorted({row["torch_version"] for row in baselines}), "cuda_versions": sorted({row["cuda_version"] for row in baselines})}
    comparable = all(row["runtime_comparable"] is True for row in rows) and provenance.get("gpu_model_uniform") is not False and new == baseline
    reason = "same uniform GPU model and PyTorch/CUDA environment as seed-2026 formal baselines" if comparable else "GPU model and/or PyTorch/CUDA provenance differs from or is incomplete relative to seed-2026 formal baselines"
    return comparable, reason, {"formal_lbi": new, "baselines": baseline, "execution_provenance_path": str(HARDWARE_PATH)}


def main():
    plan = read(PLAN_PATH)
    if plan.get("experiment_count") != 6 or len(plan.get("experiments", [])) != 6 or len({entry["experiment_key"] for entry in plan["experiments"]}) != 6:
        raise RuntimeError("machine2 plan is not the exact unique six-entry formal plan")
    prepare = read(PREPARE_READY_PATH)
    plan_sha256 = sha256(PLAN_PATH)
    if prepare.get("plan_sha256") != plan_sha256:
        raise RuntimeError("formal plan SHA256 does not match machine2 PREPARE_READY record")
    statuses, counts = completion(plan)
    expected_counts = {"completed": 6, "missing": 0, "failed": 0, "duplicate": 0, "hash mismatch": 0, "invalid summary": 0}
    if counts != expected_counts:
        PHASE.mkdir(parents=True, exist_ok=True)
        details = [{"experiment_key": item["entry"]["experiment_key"], "expected_output_root": item["entry"]["expected_output_root"], "state": item["state"], "candidate_paths": item["candidate_paths"]} for item in statuses]
        (PHASE / "FINALIZE.md").write_text("# FINALIZE — Office formal LBI machine2 / budget 0.001\n\nCompletion is incomplete. No training was launched by FINALIZE.\n\n```json\n" + json.dumps({"completion": counts, "entries": details}, indent=2) + "\n```\n", encoding="utf-8")
        print(json.dumps({"finalize_ready": False, "completion": counts, "entries": details}, indent=2))
        return
    rows = [diagnose(item["entry"], item["record"]) for item in statuses]
    if {(row["source"], row["target"]) for row in rows} != set(TRANSFERS) or not all(row["scientific_valid"] for row in rows):
        raise RuntimeError("completed formal LBI runs failed frozen/scientific validation: " + json.dumps([{row["transfer"]: row["validation_errors"]} for row in rows if not row["scientific_valid"]]))
    baselines = baseline_rows()
    add_baseline_comparisons(rows, baselines)
    summary = aggregate(rows)
    baseline_runtime = baseline_efficiency(baselines)
    comparable, reason, hardware = hardware_audit(rows, baselines)
    public = [{key: value for key, value in row.items() if key not in {"batch_metrics", "support_utilizations", "budget_gaps", "stage1_steps"}} for row in rows]
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_csv(REPORTS / "per_transfer.csv", public)
    write_json(REPORTS / "per_transfer.json", {"completion": counts, "transfers": public})
    write_csv(REPORTS / "final_budget_summary.csv", [summary])
    write_json(REPORTS / "final_budget_summary.json", {"completion": counts, "frozen_tuple": FROZEN, "summary": summary, "hardware": {"runtime_comparable": comparable, "reason": reason}})
    efficiency = {key: value for key, value in summary.items() if "runtime" in key or "gpu_peak" in key or key in {"fo_eval_runtime_sec", "wall_runtime_sec"}}
    write_json(REPORTS / "efficiency_summary.json", {"runtime_comparable": comparable, "reason": reason, "hardware": hardware, "efficiency": efficiency, "baseline_efficiency": baseline_runtime})
    lines = ["# Office final formal LBI — machine2 / budget 0.001", "", "No training, retry, resume, or rerun was launched by FINALIZE.", "", "## Completion", "", "| completed | missing | failed | duplicate | hash mismatch | invalid summary |", "|---:|---:|---:|---:|---:|---:|", f"| {counts['completed']} | {counts['missing']} | {counts['failed']} | {counts['duplicate']} | {counts['hash mismatch']} | {counts['invalid summary']} |", "", "## Accuracy", "", "| transfer | PU | FO | best sparse FO | LBI − best sparse | LBI − module dense | LBI − full dense |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in public:
        lines.append(f"| {row['transfer']} | {row['PU']:.6f} | {row['FO']:.6f} | {row['best_sparse_FO']:.6f} ({row['best_sparse_variant']}) | {row['LBI_minus_best_sparse_FO']:+.6f} | {row['LBI_minus_module_dense_FO']:+.6f} | {row['LBI_minus_full_dense_FO']:+.6f} |")
    lines += ["", f"Office mean PU={summary['Office_mean_PU']:.6f}; mean FO={summary['Office_mean_FO']:.6f}.", "", "| baseline | mean PU | LBI − PU | mean FO | LBI − FO |", "|---|---:|---:|---:|---:|"]
    for label in ("random_3mask_mean", "magnitude", "saliency", "best_sparse", "module_dense", "full_dense", "source_only"):
        lines.append(f"| {label} | {summary[f'{label}_mean_PU']:.6f} | {summary[f'LBI_minus_{label}_mean_PU']:+.6f} | {summary[f'{label}_mean_FO']:.6f} | {summary[f'LBI_minus_{label}_mean_FO']:+.6f} |")
    lines += ["", "## Budget diagnostics and efficiency", "", f"Support utilization min/p05/mean={summary['support_utilization_min']:.6f}/{summary['support_utilization_p05']:.6f}/{summary['support_utilization_mean']:.6f}; under-95%={summary['under_95pct_batch_count']}; under-90%={summary['under_90pct_batch_count']}.", f"Budget gap mean/max={summary['mean_budget_gap']:.6f}/{summary['max_budget_gap']:.6f}; rollback rate={summary['rollback_rate']:.6f}; Stage-1 steps mean/max={summary['stage1_steps_mean']:.6f}/{summary['stage1_steps_max']:.0f}; max-step hits={summary['stage1_max_steps_hit_count']}.", f"Primary GPU-memory number (peak allocated max)={summary['gpu_peak_allocated_max_mb']:.6f} MB.", f"Online batch runtime mean/std/median/p95={summary['online_batch_runtime_mean_sec']:.6f}/{summary['online_batch_runtime_std_sec']:.6f}/{summary['online_batch_runtime_median_sec']:.6f}/{summary['online_batch_runtime_p95_sec']:.6f} sec.", "", "| method | mean online-batch sec | total online-compute sec | primary GPU peak MB |", "|---|---:|---:|---:|", f"| LBI | {summary['online_batch_runtime_mean_sec']:.6f} | {summary['online_compute_runtime_sec']:.6f} | {summary['gpu_peak_allocated_max_mb']:.6f} |"]
    for label, values in baseline_runtime.items():
        lines.append(f"| {label} | {values['online_batch_runtime_mean_sec']:.6f} | {values['online_compute_runtime_sec']:.6f} | {values['gpu_peak_allocated_max_mb']:.6f} |")
    lines += ["", "## Hardware comparability", "", f"runtime_comparable = {str(comparable).lower()}: {reason}.", "", "Only new seed-2026 formal Office baseline results were used. The Office LBI hyperparameter search remains closed.", "", "FINALIZE complete for machine2 / budget 0.001.", "No training was launched by FINALIZE."]
    (REPORTS / "final_budget_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    PHASE.mkdir(parents=True, exist_ok=True)
    (PHASE / "FINALIZE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(PHASE / "FINALIZE_READY.json", {"machine_tag": "machine2", "mode": "FINALIZE", "budget": .001, "timestamp": timestamp, "plan_path": str(PLAN_PATH), "plan_sha256": plan_sha256, "completion": counts, "frozen_tuple": FROZEN, "runtime_comparable": comparable, "runtime_comparability_reason": reason, "training_launched_by_finalize": False, "reports": {name: str(REPORTS / name) for name in ("per_transfer.csv", "per_transfer.json", "final_budget_summary.csv", "final_budget_summary.json", "final_budget_summary.md", "efficiency_summary.json")}})
    print(json.dumps({"finalize_ready": True, "completion": counts, "mean_PU": summary["Office_mean_PU"], "mean_FO": summary["Office_mean_FO"], "runtime_comparable": comparable}, sort_keys=True))


if __name__ == "__main__":
    main()
