#!/usr/bin/env python3
"""Artifact-only FINALIZE for the refined VisDA-C M4 Stage-1 grid."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCH = ROOT / "experiment_logs/visda_fc_lbi_seed2026_search_20260819"
PLAN = SEARCH / "plans/wave1_m4_stage1_budget001_omega_0100/plan.json"
MANIFEST = SEARCH / "launcher_logs/wave1_m4_stage1_budget001_omega_0100/launcher_manifest.json"
RUNS = SEARCH / "runs"
REPORT = SEARCH / "reports/wave1_m4_stage1_budget001_omega_0100"
PHASE = SEARCH / "phase_records/wave1_m4"
K = 524
EPS = 1e-12


def finite(value):
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return True


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * p
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (pos - low)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics(path):
    records = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "online_step":
            record["_line"] = line_no
            records.append(record)
    return records


def same(left, right):
    try:
        return abs(float(left) - float(right)) <= EPS
    except (TypeError, ValueError):
        return left == right


def row_for(entry, record, summary_path):
    summary_path = Path(summary_path).resolve()
    metric_path = summary_path.with_name("metrics.jsonl")
    errors = []
    try:
        summary = load(summary_path)
    except Exception as exc:  # noqa: BLE001 - report invalid artifact
        return {"candidate": entry["candidate"], "experiment_key": entry["experiment_key"], "experiment_config_sha256": entry["experiment_config_sha256"], "summary_path": str(summary_path), "metrics_path": str(metric_path), "scientific_valid": False, "hard90_valid": False, "validity_reason": f"invalid summary: {exc}"}
    try:
        online = metrics(metric_path)
    except Exception as exc:  # noqa: BLE001 - report invalid artifact
        online = []
        errors.append(f"invalid metrics: {exc}")
    identity_checks = {
        "experiment_key": summary.get("experiment_key") == entry["experiment_key"],
        "experiment_config_sha256": summary.get("experiment_config_sha256") == entry["experiment_config_sha256"],
        "status": summary.get("status") == "completed",
        "dataset": summary.get("dataset") == "VISDA-C",
        "seed": summary.get("seed") == 2026,
        "batch_size": summary.get("batch_size") == 256,
        "variant": summary.get("variant") == "module_lbi",
        "budget": same(summary.get("requested_budget"), 0.001),
        "alpha": same(summary.get("alpha"), entry["alpha"]),
        "kappa": same(summary.get("kappa"), 1.0),
        "nu": same(summary.get("nu"), entry["nu"]),
        "omega": same(summary.get("omega"), 0.10),
        "stage2_lr": same(summary.get("stage2_lr"), 0.005),
        "stage2_steps": summary.get("stage2_steps_requested") == 1,
        "stage1_max_steps": summary.get("stage1_max_steps") == 3000,
        "support_threshold": same(summary.get("lbi_support_threshold"), 1e-4),
    }
    errors.extend(name for name, ok in identity_checks.items() if not ok)
    if not finite(summary):
        errors.append("nonfinite summary")
    if len(online) != summary.get("online_steps"):
        errors.append("online step count mismatch")
    if any(not finite(item) for item in online):
        errors.append("nonfinite online record")
    if any(item.get("event") == "error" or item.get("error") or item.get("exception") or item.get("traceback") for item in online):
        errors.append("online error record")
    support = [item.get("selected_param_count", item.get("support_param_count")) for item in online]
    steps = [item.get("stage1_steps_completed") for item in online]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in support):
        errors.append("missing support count")
        support = [float(value) for value in support if isinstance(value, (int, float)) and math.isfinite(value)]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in steps):
        errors.append("missing Stage-1 steps")
        steps = [float(value) for value in steps if isinstance(value, (int, float)) and math.isfinite(value)]
    utilization = [value / K for value in support]
    max_steps_hit_count = sum(bool(item.get("max_steps_hit")) or item.get("stage1_stop_reason") == "max_steps" for item in online)
    budget_violation_count = sum(value > K for value in support)
    budget_reached_all = summary.get("budget_reached_all_steps") is True
    valid_lbi_run = summary.get("valid_lbi_run") is True and all(item.get("valid_lbi_step") is True for item in online)
    if not budget_reached_all:
        errors.append("budget not reached on all batches")
    if max_steps_hit_count:
        errors.append("Stage-1 max_steps hit")
    if budget_violation_count:
        errors.append("support budget violation")
    scientific_valid = not errors and valid_lbi_run and bool(online)
    hard90_valid = scientific_valid and bool(utilization) and min(utilization) >= 0.90
    row = {
        "candidate": entry["candidate"], "alpha": entry["alpha"], "kappa": 1.0, "nu": entry["nu"],
        "omega": 0.10, "stage2_lr": 0.005, "budget": 0.001, "K": K,
        "scientific_valid": scientific_valid, "hard90_valid": hard90_valid,
        "support_utilization_min": min(utilization) if utilization else None,
        "support_utilization_p05": percentile(utilization, 0.05),
        "support_utilization_mean": statistics.fmean(utilization) if utilization else None,
        "under_95pct_batch_count": sum(value < 0.95 for value in utilization),
        "under_90pct_batch_count": sum(value < 0.90 for value in utilization),
        "max_steps_hit_count": max_steps_hit_count,
        "stage1_steps_completed_mean": statistics.fmean(steps) if steps else None,
        "stage1_steps_completed_max": max(steps) if steps else None,
        "FO_mean_per_class": summary.get("FO-mean-class-Acc"),
        "FO_worst_class": summary.get("FO-worst-class-Acc"),
        "FO_classwise_std": summary.get("FO-class-std"),
        "FO_overall": summary.get("FO-overall-Acc"),
        "PU_mean_per_class": summary.get("PU-mean-class-Acc"),
        "PU_overall": summary.get("PU-overall-Acc"),
        "PU_worst_class": summary.get("PU-worst-class-Acc"),
        "PU_classwise_std": summary.get("PU-class-std"),
        "runtime_sec": summary.get("runtime"),
        "online_compute_runtime_sec": summary.get("online_compute_runtime_sec"),
        "online_batch_runtime_mean_sec": summary.get("online_batch_runtime_mean_sec"),
        "online_batch_runtime_p95_sec": summary.get("online_batch_runtime_p95_sec"),
        "peak_gpu_memory_allocated_mb": summary.get("peak_gpu_memory_allocated_mb"),
        "online_steps": len(online), "summary_status": summary.get("status"),
        "valid_lbi_run": valid_lbi_run, "budget_reached_all_steps": budget_reached_all,
        "experiment_key": entry["experiment_key"], "experiment_config_sha256": entry["experiment_config_sha256"],
        "summary_path": str(summary_path), "metrics_path": str(metric_path),
        "validity_reason": "ok" if scientific_valid else "; ".join(dict.fromkeys(errors)),
    }
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    plan = load(PLAN)
    manifest = load(MANIFEST)
    entries = plan.get("experiments", [])
    if plan.get("experiment_count") != 8 or len(entries) != 8:
        raise RuntimeError("M4 plan is not exactly 8 entries")
    records = manifest.get("records", [])
    by_key = {}
    for record in records:
        by_key.setdefault(record.get("experiment_key"), []).append(record)
    counts = {"completed": 0, "missing": 0, "failed": 0, "duplicate": 0, "hash mismatch": 0, "invalid summary": 0}
    rows = []
    for index, entry in enumerate(entries, 1):
        entry = dict(entry)
        entry["candidate"] = f"S{index}"
        matches = by_key.get(entry["experiment_key"], [])
        if not matches:
            counts["missing"] += 1
            continue
        if len(matches) > 1:
            counts["duplicate"] += len(matches) - 1
        record = matches[-1]
        if record.get("experiment_config_sha256") != entry["experiment_config_sha256"]:
            counts["hash mismatch"] += 1
            continue
        if record.get("status") != "completed" or record.get("post_run_status") != "completed" or int(record.get("return_code", -1)) != 0:
            counts["failed"] += 1
            continue
        summary_path = record.get("summary_path")
        try:
            expected = Path(entry["expected_output_root"]).resolve()
            if not summary_path or not Path(summary_path).resolve().is_relative_to(expected):
                raise RuntimeError("summary outside expected output root")
            row = row_for(entry, record, summary_path)
            if not row.get("summary_status") or not Path(row["summary_path"]).is_file() or not Path(row["metrics_path"]).is_file():
                raise RuntimeError("missing summary or metrics artifact")
            rows.append(row)
            counts["completed"] += 1
        except Exception as exc:  # noqa: BLE001 - report invalid artifact
            counts["invalid summary"] += 1
            rows.append({"candidate": entry["candidate"], "alpha": entry["alpha"], "kappa": 1.0, "nu": entry["nu"], "omega": 0.10, "stage2_lr": 0.005, "scientific_valid": False, "hard90_valid": False, "validity_reason": f"invalid summary: {exc}", "experiment_key": entry["experiment_key"], "experiment_config_sha256": entry["experiment_config_sha256"], "summary_path": summary_path})
    if len(rows) != 8:
        raise RuntimeError(f"unexpected row count: {len(rows)}")
    ranking = sorted(rows, key=lambda row: (not bool(row.get("scientific_valid")), not bool(row.get("hard90_valid")), -(row.get("FO_mean_per_class") or float("-inf")), -(row.get("FO_worst_class") or float("-inf"))))
    for rank, row in enumerate(ranking, 1):
        row["eligibility_rank"] = rank
    plan_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    summary = {
        "mode": "FINALIZE", "machine": "M4", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(PLAN), "plan_sha256": plan_sha, "launcher_manifest": str(MANIFEST),
        "actual_finalize_command": "python 260817_iclr2027-refined/tools/finalize_visda_wave1_m4_stage1_omega_0100.py",
        "completion": counts, "experiment_count": 8, "rows": len(rows),
        "eligibility_order": ["scientific_valid", "hard90_valid", "FO_mean_per_class_descending", "FO_worst_class_descending"],
        "training_launched_by_finalize": False,
        "reports": [str(REPORT / "per_run.csv"), str(REPORT / "ranking.csv"), str(REPORT / "FINALIZE_SUMMARY.json"), str(PHASE / "FINALIZE.md")],
    }
    write_csv(REPORT / "per_run.csv", rows)
    write_csv(REPORT / "ranking.csv", ranking)
    (REPORT / "FINALIZE_SUMMARY.json").parent.mkdir(parents=True, exist_ok=True)
    (REPORT / "FINALIZE_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# M4 VisDA-C Stage-1 budget=0.001 omega=0.10 — FINALIZE", "",
        "Mode: `FINALIZE`. No training, retry, resume, launcher, or rerun was performed by FINALIZE.", "",
        f"Plan SHA256: `{plan_sha}`", "",
        "## Completion", "",
        "| completed | missing | failed | duplicate | hash mismatch | invalid summary |", "|---:|---:|---:|---:|---:|---:|",
        "| {completed} | {missing} | {failed} | {duplicate} | {hash mismatch} | {invalid summary} |".format(**counts), "",
        "## Aggregation", "",
        "Eligibility order: scientific validity, hard90 utilization, FO mean-per-class descending, FO worst-class descending. PU is report-only; validity is checked before accuracy.", "",
        "| rank | candidate | alpha | kappa | nu | omega | stage2_lr | scientific_valid | hard90_valid | support min / P05 / mean | <95% batches | max-step hits | Stage-1 mean / max | FO mean / worst / std / overall | PU mean | runtime (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in ranking:
        def fmt(value):
            return "NA" if value is None else (f"{value:.6f}" if isinstance(value, float) else str(value))
        md.append("| {eligibility_rank} | {candidate} | {alpha:.2f} | {kappa:.1f} | {nu:.1f} | {omega:.2f} | {stage2_lr:.3f} | {scientific_valid} | {hard90_valid} | {minv} / {p05} / {meanv} | {under95} | {maxhit} | {stepmean} / {stepmax} | {fomean} / {foworst} / {fostd} / {fooverall} | {pumean} | {runtime} |".format(eligibility_rank=row.get("eligibility_rank"), candidate=row.get("candidate"), alpha=row.get("alpha", 0), kappa=row.get("kappa", 0), nu=row.get("nu", 0), omega=row.get("omega", 0), stage2_lr=row.get("stage2_lr", 0), scientific_valid=row.get("scientific_valid"), hard90_valid=row.get("hard90_valid"), minv=fmt(row.get("support_utilization_min")), p05=fmt(row.get("support_utilization_p05")), meanv=fmt(row.get("support_utilization_mean")), under95=row.get("under_95pct_batch_count", "NA"), maxhit=row.get("max_steps_hit_count", "NA"), stepmean=fmt(row.get("stage1_steps_completed_mean")), stepmax=fmt(row.get("stage1_steps_completed_max")), fomean=fmt(row.get("FO_mean_per_class")), foworst=fmt(row.get("FO_worst_class")), fostd=fmt(row.get("FO_classwise_std")), fooverall=fmt(row.get("FO_overall")), pumean=fmt(row.get("PU_mean_per_class")), runtime=fmt(row.get("runtime_sec"))))
    md += ["", "## Runtime diagnostics", "", "| candidate | online compute (s) | online batch mean (s) | online batch P95 (s) | peak GPU allocated (MB) |", "|---|---:|---:|---:|---:|"]
    for row in ranking:
        md.append(f"| {row['candidate']} | {fmt(row.get('online_compute_runtime_sec'))} | {fmt(row.get('online_batch_runtime_mean_sec'))} | {fmt(row.get('online_batch_runtime_p95_sec'))} | {fmt(row.get('peak_gpu_memory_allocated_mb'))} |")
    md += ["", "## Validity decisions", ""]
    for row in ranking:
        md.append(f"- `{row['candidate']}`: `scientific_valid={row.get('scientific_valid')}`, `hard90_valid={row.get('hard90_valid')}` — {row.get('validity_reason', 'NA')}.")
    md += ["", "## Provenance", "", "- Actual FINALIZE command: `python 260817_iclr2027-refined/tools/finalize_visda_wave1_m4_stage1_omega_0100.py`.", f"- Plan: `{PLAN}`", f"- Runs root: `{RUNS}`", f"- Launcher manifest: `{MANIFEST}`", f"- Launcher command history: `{SEARCH / 'command_history/wave1_m4_stage1_budget001_omega_0100_commands.sh'}`", "- Launcher script: `260817_iclr2027-refined/tools/run_visda_wave1_m4_stage1_omega_0100_noninteractive.sh`.", "- Requested environment/GPU allocation: `SHOT_TTA`; GPUs `0,1,2,3,4,5,6,7`; max-workers 8; workers-per-GPU 1.", "- All eight launcher records are completed with return code 0; no retry records were present.", "- Current protocol settings: VisDA-C / ResNet-101 / batch 256 / workers 4 / seed 2026 / one target pass / controlled-FC BN frozen / candidate weight+bias / K=524 / support threshold 1e-4 / stage1 max 3000 / stage2 steps 1.", "- Scientific validity requires completed finite artifacts, valid LBI steps, budget reached on every online batch, no support budget violation, and no Stage-1 max-step hit. Hard90 additionally requires support utilization >= 0.90 for every batch.", "", "FINALIZE complete; no training was launched by FINALIZE."]
    PHASE.mkdir(parents=True, exist_ok=True)
    (PHASE / "FINALIZE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"completion": counts, "ranking": [{"candidate": row["candidate"], "scientific_valid": row.get("scientific_valid"), "hard90_valid": row.get("hard90_valid"), "FO_mean_per_class": row.get("FO_mean_per_class")} for row in ranking], "reports": summary["reports"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
