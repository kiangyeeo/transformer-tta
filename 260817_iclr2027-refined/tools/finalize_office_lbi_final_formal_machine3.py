#!/usr/bin/env python3
"""Artifact-only FINALIZE for Office formal LBI, machine3 / budget 0.002."""

import csv
import datetime as dt
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
K = 1049
TRANSFERS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
NAMES = {(0, 1): "A->D", (0, 2): "A->W", (1, 0): "D->A", (1, 2): "D->W", (2, 0): "W->A", (2, 1): "W->D"}
FROZEN = {"requested_budget": 0.002, "alpha": 0.15, "kappa": 1.0, "nu": 0.50, "omega": 0.30, "stage2_lr": 0.005, "stage1_max_steps": 3000, "stage2_steps_requested": 1, "lbi_support_threshold": 1e-4, "budget_tolerance": 1e-4, "delta_nonzero_tolerance": 1e-12, "seed": 2026}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def num(value, name):
    if isinstance(value, bool) or value is None:
        raise ValueError(f"missing/non-numeric {name}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}")
    return value


def finite(value):
    if isinstance(value, float): return math.isfinite(value)
    if isinstance(value, dict): return all(finite(x) for x in value.values())
    if isinstance(value, list): return all(finite(x) for x in value)
    return True


def mean(values): return sum(values) / len(values) if values else None
def std(values): return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
def percentile(values, q):
    values = sorted(values)
    if not values: return None
    pos = (len(values) - 1) * q; lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (pos - lo)


def exact_close(actual, expected):
    return math.isclose(num(actual, "value"), expected, rel_tol=0.0, abs_tol=1e-15)


def scan_index(root):
    records, invalid = scan_run_summaries(str(root))
    index = defaultdict(list)
    for record in records: index[record["experiment_key"]].append(record)
    return index, invalid


def plan_status(plan, index, invalid):
    invalid_keys = {x.get("experiment_key") for x in invalid if x.get("experiment_key")}
    rows = []
    for entry in plan["experiments"]:
        candidates = index.get(entry["experiment_key"], [])
        exact = [x for x in candidates if x["experiment_config_sha256"] == entry["experiment_config_sha256"]]
        mismatched = [x for x in candidates if x["experiment_config_sha256"] != entry["experiment_config_sha256"]]
        completed = [x for x in exact if x["status"] == "completed"]
        failed = [x for x in exact if x["status"] in {"failed", "incomplete"}]
        if entry["experiment_key"] in invalid_keys: state = "invalid_summary"
        elif mismatched: state = "hash_mismatch"
        elif len(exact) > 1: state = "duplicate"
        elif len(completed) == 1: state = "completed"
        elif failed: state = "failed"
        else: state = "missing"
        rows.append({"entry": entry, "state": state, "record": completed[0] if state == "completed" else None,
                     "mismatch_paths": [x["summary_path"] for x in mismatched]})
    return rows


def frozen_errors(entry, summary, manifest, online):
    errors = []
    for key, expected in FROZEN.items():
        try:
            if not exact_close(summary.get(key), expected): errors.append(f"summary {key}={summary.get(key)!r}")
        except ValueError: errors.append(f"summary {key} missing")
    if summary.get("variant") != "module_lbi": errors.append("summary variant")
    if summary.get("runtime_comparable") is not True: errors.append("runtime_comparable")
    if manifest.get("runtime_comparable") is not True: errors.append("manifest runtime_comparable")
    data = manifest.get("scientific_config", {}).get("data", {})
    if data.get("batch_size") != 64 or data.get("workers") != 4: errors.append("manifest batch_size/workers")
    command = manifest.get("command", [])
    command_text = " ".join(command) if isinstance(command, list) else str(command)
    if "--no-save-model" not in command_text: errors.append("manifest save_model")
    if entry.get("effective_overrides", {}).get("save_model") is not False: errors.append("plan save_model")
    for record in online:
        try:
            for key, expected in FROZEN.items():
                if key == "seed": continue
                field = key
                if not exact_close(record.get(field), expected): errors.append(f"batch {record.get('batch_index')} {field}"); break
            # The formal loader is configured with batch_size=64; its final
            # non-empty batch may legitimately be smaller.
            batch_size = record.get("batch_size")
            if not isinstance(batch_size, int) or not 1 <= batch_size <= 64: errors.append(f"batch {record.get('batch_index')} size")
        except ValueError: errors.append(f"batch {record.get('batch_index')} tuple missing")
    return sorted(set(errors))


def diagnose(entry, record):
    summary_path = Path(record["summary_path"]); summary = record["summary"]
    manifest = read(summary_path.with_name("manifest.json"))
    online, errors, parse_error = [], [], None
    try:
        with summary_path.with_name("metrics.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                item = json.loads(line)
                if item.get("event") == "online_step": online.append(item)
                elif item.get("event") == "error": errors.append(item)
    except (OSError, json.JSONDecodeError) as exc: parse_error = str(exc)
    problems = []
    if parse_error: problems.append(f"metrics parse: {parse_error}")
    if not online: problems.append("no online steps")
    if errors: problems.append(f"error events={len(errors)}")
    if not finite(summary) or not all(finite(x) for x in online): problems.append("NaN/Inf")
    if summary.get("status") != "completed": problems.append("summary status")
    if summary.get("valid_lbi_run") is not True: problems.append("valid_lbi_run")
    if len(online) != summary.get("online_steps"): problems.append("online step count")
    for key in ("experiment_key", "experiment_config_sha256"):
        if summary.get(key) != entry.get(key) or manifest.get(key) != entry.get(key): problems.append(f"identity {key}")
    problems.extend(frozen_errors(entry, summary, manifest, online))
    selected=[]; support=[]; gaps=[]; stages=[]; rollbacks=[]; maxhits=[]
    for item in online:
        try:
            s = num(item.get("selected_param_count"), "selected_param_count")
            u = num(item.get("support_param_count", item.get("stage1_support_count")), "support_param_count")
            selected.append(s); support.append(u / K); gaps.append(K - s); stages.append(num(item.get("stage1_steps_completed"), "stage1_steps_completed"))
            rollbacks.append(bool(item.get("stage1_rollback_used", False)))
            maxhits.append(bool(item.get("max_steps_hit", False)) or item.get("stage1_stop_reason") in {"max_steps", "max_steps_reached"})
            if s > K: problems.append("support exceeds K")
            if item.get("valid_lbi_step") is not True: problems.append("invalid LBI batch")
        except ValueError as exc: problems.append(str(exc))
    if any(maxhits): problems.append("stage1 max_steps hit")
    transfer = (int(entry["source"]), int(entry["target"]))
    batch = {k: [num(x.get(k), k) for x in online] for k in ("online_runtime_sec", "adapt_runtime_sec", "pu_runtime_sec", "lbi_stage1_runtime_sec", "lbi_stage2_runtime_sec", "peak_gpu_memory_allocated_mb", "peak_gpu_memory_reserved_mb")}
    return {
        "source": transfer[0], "target": transfer[1], "transfer": NAMES[transfer], "experiment_key": entry["experiment_key"],
        "experiment_config_sha256": entry["experiment_config_sha256"], "summary_path": str(summary_path), "metrics_path": str(summary_path.with_name("metrics.jsonl")),
        "PU": num(summary.get("PU-Acc"), "PU-Acc"), "FO": num(summary.get("FO-Acc"), "FO-Acc"),
        "support_utilization_min": min(support), "support_utilization_p05": percentile(support, .05), "support_utilization_mean": mean(support),
        "under_95pct_batch_count": sum(x < .95 for x in support), "under_90pct_batch_count": sum(x < .90 for x in support),
        "mean_budget_gap": mean(gaps), "max_budget_gap": max(gaps), "rollback_count": sum(rollbacks), "rollback_rate": mean(rollbacks),
        "stage1_steps_mean": mean(stages), "stage1_steps_max": max(stages), "stage1_max_steps_hit_count": sum(maxhits),
        "batch_count": len(online), "scientific_valid": not problems, "validation_errors": sorted(set(problems)),
        "runtime_comparable": summary.get("runtime_comparable"), "gpu_name": summary.get("gpu_name"), "torch_version": summary.get("torch_version"), "cuda_version": summary.get("cuda_version"),
        "runtime_resume_used": summary.get("runtime_resume_used"), "runtime_segment_count": summary.get("runtime_segment_count"),
        "fo_eval_runtime_sec": num(summary.get("fo_eval_runtime_sec"), "fo_eval_runtime_sec"), "wall_runtime_sec": num(summary.get("wall_runtime_sec"), "wall_runtime_sec"),
        "batch_metrics": batch, "summary": summary,
        "support_utilizations": support, "budget_gaps": gaps, "stage1_steps": stages,
    }


def baseline_rows():
    plans = [BASELINE_ROOT / "plans/baseline_machine1/plan.json", BASELINE_ROOT / "plans/baseline_machine2/plan.json", BASELINE_ROOT / "plans/machine3/baseline/plan.json"]
    index, invalid = scan_index(BASELINE_ROOT / "runs")
    rows = []
    for path in plans:
        for entry in read(path)["experiments"]:
            exact = [x for x in index.get(entry["experiment_key"], []) if x["experiment_config_sha256"] == entry["experiment_config_sha256"] and x["status"] == "completed"]
            if len(exact) != 1: raise ValueError(f"baseline identity incomplete: {entry['experiment_key']} matches={len(exact)}")
            s = exact[0]["summary"]
            rows.append({"source": entry["source"], "target": entry["target"], "variant": entry["variant"], "budget": entry.get("requested_budget"),
                         "PU": num(s.get("PU-Acc"), "baseline PU"), "FO": num(s.get("FO-Acc"), "baseline FO"),
                         "online_batch_runtime_mean_sec": num(s.get("online_batch_runtime_mean_sec"), "baseline runtime"),
                         "online_compute_runtime_sec": num(s.get("online_compute_runtime_sec"), "baseline compute"),
                         "gpu_peak_allocated_max_mb": num(s.get("gpu_peak_allocated_max_mb"), "baseline memory"),
                         "fo_eval_runtime_sec": num(s.get("fo_eval_runtime_sec"), "baseline FO time"), "gpu_name": s.get("gpu_name"),
                         "torch_version": s.get("torch_version"), "cuda_version": s.get("cuda_version"), "runtime_comparable": s.get("runtime_comparable")})
    if len(rows) != 72 or len(invalid): raise ValueError(f"baseline set invalid: rows={len(rows)} invalid={len(invalid)}")
    return rows


def baseline_comparison(rows, bases):
    grouped = defaultdict(dict)
    for row in bases:
        if row["variant"] in {"module_random", "module_magnitude", "module_saliency"} and math.isclose(float(row["budget"]), .002): grouped[(row["source"], row["target"])][row["variant"]] = row
        elif row["variant"] in {"module_dense", "full_dense", "source_only"}: grouped[(row["source"], row["target"])][row["variant"]] = row
    for row in rows:
        b = grouped[(row["source"], row["target"])]
        if set(("module_random", "module_magnitude", "module_saliency", "module_dense", "full_dense", "source_only")) - set(b): raise ValueError(f"baseline comparison incomplete: {row['transfer']}")
        sparse = max((b[x] for x in ("module_random", "module_magnitude", "module_saliency")), key=lambda x:x["FO"])
        for label, value in (("random_3mask_mean", b["module_random"]), ("magnitude", b["module_magnitude"]), ("saliency", b["module_saliency"]), ("best_sparse", sparse), ("module_dense", b["module_dense"]), ("full_dense", b["full_dense"]), ("source_only", b["source_only"])):
            row[f"{label}_FO"] = value["FO"]; row[f"{label}_PU"] = value["PU"]
        row["best_sparse_variant"] = sparse["variant"]
        for label in ("best_sparse", "module_dense", "full_dense", "source_only"):
            row[f"LBI_minus_{label}_FO"] = row["FO"] - row[f"{label}_FO"]
            row[f"LBI_minus_{label}_PU"] = row["PU"] - row[f"{label}_PU"]


def aggregate(rows):
    batch = defaultdict(list)
    for row in rows:
        for key, values in row["batch_metrics"].items(): batch[key].extend(values)
    support = [x for row in rows for x in row["support_utilizations"]]
    gaps = [x for row in rows for x in row["budget_gaps"]]
    stages = [x for row in rows for x in row["stage1_steps"]]
    result = {"budget": .002, "alpha": .15, "kappa": 1.0, "nu": .50, "omega": .30, "stage2_lr": .005,
              "completed_transfer_count": len(rows), "Office_mean_PU": mean([r["PU"] for r in rows]), "Office_mean_FO": mean([r["FO"] for r in rows]),
              "support_utilization_min": min(support), "support_utilization_p05": percentile(support, .05), "support_utilization_mean": mean(support),
              "under_95pct_batch_count": sum(r["under_95pct_batch_count"] for r in rows), "under_90pct_batch_count": sum(r["under_90pct_batch_count"] for r in rows),
              "mean_budget_gap": mean(gaps), "max_budget_gap": max(gaps),
              "rollback_rate": sum(r["rollback_count"] for r in rows) / sum(r["batch_count"] for r in rows),
              "stage1_steps_mean": mean(stages), "stage1_steps_max": max(stages),
              "stage1_max_steps_hit_count": sum(r["stage1_max_steps_hit_count"] for r in rows),
              "runtime_resume_used": any(r["runtime_resume_used"] for r in rows), "runtime_segment_count": sum(r["runtime_segment_count"] for r in rows),
              "fo_eval_runtime_sec": sum(r["fo_eval_runtime_sec"] for r in rows), "wall_runtime_sec": sum(r["wall_runtime_sec"] for r in rows)}
    for source, prefix in (("online_runtime_sec", "online_batch_runtime"), ("adapt_runtime_sec", "adapt_runtime"), ("pu_runtime_sec", "pu_runtime"), ("lbi_stage1_runtime_sec", "lbi_stage1_runtime"), ("lbi_stage2_runtime_sec", "lbi_stage2_runtime")):
        result[f"{prefix}_mean_sec"], result[f"{prefix}_std_sec"], result[f"{prefix}_total_sec"] = mean(batch[source]), std(batch[source]), sum(batch[source])
    result["online_batch_runtime_median_sec"] = percentile(batch["online_runtime_sec"], .50); result["online_batch_runtime_p95_sec"] = percentile(batch["online_runtime_sec"], .95)
    result["online_compute_runtime_sec"] = sum(batch["online_runtime_sec"])
    for source, prefix in (("peak_gpu_memory_allocated_mb", "gpu_peak_allocated"), ("peak_gpu_memory_reserved_mb", "gpu_peak_reserved")):
        result[f"{prefix}_mean_mb"], result[f"{prefix}_max_mb"] = mean(batch[source]), max(batch[source])
    for label in ("best_sparse", "module_dense", "full_dense", "source_only"):
        result[f"{label}_mean_FO"] = mean([r[f"{label}_FO"] for r in rows]); result[f"LBI_minus_{label}_mean_FO"] = result["Office_mean_FO"] - result[f"{label}_mean_FO"]
        result[f"{label}_mean_PU"] = mean([r[f"{label}_PU"] for r in rows]); result[f"LBI_minus_{label}_mean_PU"] = result["Office_mean_PU"] - result[f"{label}_mean_PU"]
    return result


def main():
    plan_path = ROOT / "plans/machine3/plan.json"; plan = read(plan_path)
    if plan.get("experiment_count") != 6 or len(plan.get("experiments", [])) != 6: raise ValueError("plan is not exact six-entry plan")
    index, invalid = scan_index(ROOT / "runs"); statuses = plan_status(plan, index, invalid)
    counts = Counter(x["state"] for x in statuses)
    counts = {x: counts[x] for x in ("completed", "missing", "failed", "duplicate", "hash_mismatch", "invalid_summary")}
    if counts != {"completed": 6, "missing": 0, "failed": 0, "duplicate": 0, "hash_mismatch": 0, "invalid_summary": 0}:
        raise RuntimeError(json.dumps({"completion": counts, "entries": [{"key": x["entry"]["experiment_key"], "state": x["state"]} for x in statuses]}, indent=2))
    rows = [diagnose(x["entry"], x["record"]) for x in statuses]
    if { (r["source"], r["target"]) for r in rows } != set(TRANSFERS) or not all(r["scientific_valid"] for r in rows):
        raise RuntimeError("completed formal runs failed scientific/frozen validation: " + json.dumps([{r["transfer"]: r["validation_errors"]} for r in rows if not r["scientific_valid"]]))
    bases = baseline_rows(); baseline_comparison(rows, bases)
    baseline_models = sorted(set(r["gpu_name"] for r in bases)); baseline_torch = sorted(set(r["torch_version"] for r in bases)); baseline_cuda = sorted(set(r["cuda_version"] for r in bases))
    new_models = sorted(set(r["gpu_name"] for r in rows)); new_torch = sorted(set(r["torch_version"] for r in rows)); new_cuda = sorted(set(r["cuda_version"] for r in rows))
    comparable = all(r["runtime_comparable"] is True for r in rows) and new_models == baseline_models and new_torch == baseline_torch and new_cuda == baseline_cuda
    hardware_reason = "same GPU model and PyTorch/CUDA environment as formal seed-2026 baselines" if comparable else "hardware/environment provenance differs from or is incomplete relative to formal seed-2026 baselines"
    summary = aggregate(rows)
    reports = ROOT / "reports/machine3"; phase = ROOT / "phase_records/machine3"
    public = [{k:v for k,v in row.items() if k not in {"summary", "batch_metrics", "support_utilizations", "budget_gaps", "stage1_steps"}} for row in rows]
    write_csv(reports / "per_transfer.csv", public); write_json(reports / "per_transfer.json", {"completion": counts, "transfers": public})
    write_csv(reports / "final_budget_summary.csv", [summary]); write_json(reports / "final_budget_summary.json", {"completion": counts, "frozen_tuple": FROZEN, "summary": summary, "hardware": {"runtime_comparable": comparable, "reason": hardware_reason}})
    efficiency = {k:v for k,v in summary.items() if "runtime" in k or "gpu_peak" in k or k in {"fo_eval_runtime_sec", "wall_runtime_sec", "runtime_resume_used", "runtime_segment_count"}}
    write_json(reports / "efficiency_summary.json", {"runtime_comparable": comparable, "reason": hardware_reason, "hardware": {"formal_lbi": {"gpu_models": new_models, "torch_versions": new_torch, "cuda_versions": new_cuda}, "baselines": {"gpu_models": baseline_models, "torch_versions": baseline_torch, "cuda_versions": baseline_cuda}}, "efficiency": efficiency})
    md = ["# Office final formal LBI — machine3 / budget 0.002", "", "No training, retry, resume, or rerun was launched by FINALIZE.", "", "## Completion", "", "| completed | missing | failed | duplicate | hash mismatch | invalid summary |", "|---:|---:|---:|---:|---:|---:|", f"| {counts['completed']} | 0 | 0 | 0 | 0 | 0 |", "", "## Accuracy and formal budget diagnostics", "", "| transfer | PU | FO | best sparse FO | LBI − best sparse | LBI − module dense | LBI − full dense |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in public: md.append(f"| {r['transfer']} | {r['PU']:.6f} | {r['FO']:.6f} | {r['best_sparse_FO']:.6f} ({r['best_sparse_variant']}) | {r['LBI_minus_best_sparse_FO']:.6f} | {r['LBI_minus_module_dense_FO']:.6f} | {r['LBI_minus_full_dense_FO']:.6f} |")
    md += ["", f"Office mean PU = {summary['Office_mean_PU']:.6f}; Office mean FO = {summary['Office_mean_FO']:.6f}.", f"Support utilization: min={summary['support_utilization_min']:.6f}, p05={summary['support_utilization_p05']:.6f}, mean={summary['support_utilization_mean']:.6f}; under-95% batches={summary['under_95pct_batch_count']}, under-90% batches={summary['under_90pct_batch_count']}.", f"GPU peak allocated (primary): mean={summary['gpu_peak_allocated_mean_mb']:.3f} MB, max={summary['gpu_peak_allocated_max_mb']:.3f} MB.", "", "## Baseline and hardware audit", "", "Only new seed-2026 formal Office baselines were used. Best sparse is max(Random 3-mask mean, Magnitude, Saliency) at budget 0.002 per transfer.", f"runtime_comparable = {str(comparable).lower()}: {hardware_reason}.", "", "## Scope", "", "The Office LBI hyperparameter search is closed. These are final formal evaluation runs. No further Office LBI tuning is recommended."]
    (reports / "final_budget_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    ready = {"machine_tag":"machine3", "mode":"FINALIZE", "budget":.002, "timestamp":timestamp, "plan_path":str(plan_path), "completion":counts, "frozen_tuple":FROZEN, "runtime_comparable":comparable, "runtime_comparability_reason":hardware_reason, "training_launched_by_finalize":False, "reports":{x:str(reports/x) for x in ("per_transfer.csv", "per_transfer.json", "final_budget_summary.csv", "final_budget_summary.json", "final_budget_summary.md", "efficiency_summary.json")}}
    write_json(phase / "FINALIZE_READY.json", ready)
    finalize_md = ["# FINALIZE — Office formal LBI machine3 / budget 0.002", "", "Mode: `FINALIZE`", "", "No training was launched by FINALIZE.", "", f"Completion: completed={counts['completed']}, missing=0, failed=0, duplicate=0, hash mismatch=0, invalid summary=0.", "", "Frozen tuple verified for all six runs: budget=0.002, alpha=0.15, kappa=1.0, nu=0.50, omega=0.30, stage2_lr=0.005, stage1_max_steps=3000, stage2_steps=1, support_threshold=1e-4, seed=2026, batch_size=64, workers=4.", f"", f"runtime_comparable = {str(comparable).lower()}: {hardware_reason}.", "", f"Office mean PU={summary['Office_mean_PU']:.6f}; mean FO={summary['Office_mean_FO']:.6f}.", "", "Global aggregation pending: " + ", ".join(str(ROOT / f"phase_records/{machine}/FINALIZE_READY.json") for machine in ("machine1", "machine2") if not (ROOT / f"phase_records/{machine}/FINALIZE_READY.json").exists()), "", "FINALIZE complete for machine3 / budget 0.002.", "No training was launched by FINALIZE."]
    (phase / "FINALIZE.md").write_text("\n".join(finalize_md) + "\n", encoding="utf-8")
    print(json.dumps({"completion":counts, "runtime_comparable":comparable, "mean_PU":summary["Office_mean_PU"], "mean_FO":summary["Office_mean_FO"], "global_pending":[m for m in ("machine1","machine2") if not (ROOT / f"phase_records/{m}/FINALIZE_READY.json").exists()]}))


if __name__ == "__main__": main()
