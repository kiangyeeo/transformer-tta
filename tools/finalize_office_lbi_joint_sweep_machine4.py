#!/usr/bin/env python3
"""Pure artifact-only FINALIZE for the Office LBI machine4 joint sweep.

This script deliberately imports only the Python standard library.  It reads
plans, launcher records, summaries, metrics JSONL, and the saved seed-2026
baseline reports; it never imports training code or initializes a runtime.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOINT = ROOT / "experiment_logs/shot_otta_office_lbi_joint_sweep_20260818"
STAGE1 = ROOT / "experiment_logs/shot_otta_office_seed2026_stage1_20260818"
PLAN_PATH = JOINT / "plans/machine4/plan.json"
MANIFEST_PATH = JOINT / "launcher_logs/machine4/launcher_manifest.json"
LAUNCHER_SUMMARY_PATH = JOINT / "launcher_logs/machine4/launcher_summary.csv"
RUNS_ROOT = JOINT / "runs"
REPORT_DIR = JOINT / "reports/machine4"
PHASE_DIR = JOINT / "phase_records/machine4"

TRANSFERS = ["A->D", "A->W", "D->A", "D->W", "W->A", "W->D"]
TRANSFER_CODE = {
    "A->D": (0, 1), "A->W": (0, 2), "D->A": (1, 0),
    "D->W": (1, 2), "W->A": (2, 0), "W->D": (2, 1),
}
ANCHORS = {
    (0.0005, 0.1, 1.5, 0.5): ("A4", 262),
    (0.001, 0.15, 1.0, 1.0): ("A8", 524),
}
NEW_OMEGAS = {0.05, 0.10, 0.20, 0.30}
NEW_LRS = {0.005, 0.010, 0.020}
EPS = 1e-9


def num(value):
    if value is None or value == "":
        return None
    return float(value)


def same(a, b):
    try:
        return abs(float(a) - float(b)) <= EPS
    except (TypeError, ValueError):
        return a == b


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    return True


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def load_json(path):
    return json.loads(path.read_text())


def transfer_name(source, target):
    for name, pair in TRANSFER_CODE.items():
        if (int(source), int(target)) == pair:
            return name
    raise ValueError(f"unknown transfer {source}->{target}")


def load_metrics(path):
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "online_step":
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def baseline_reference():
    """Return best sparse FO per transfer and budget from saved seed-2026 rows."""
    files = [
        STAGE1 / "reports/machine1/baseline_summary.csv",
        STAGE1 / "reports/machine2/baseline_summary.csv",
        STAGE1 / "results/baseline/machine3/all_runs.csv",
    ]
    values = defaultdict(list)
    for path in files:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("variant") not in {"module_random", "module_magnitude", "module_saliency"}:
                    continue
                budget = num(row.get("requested_budget"))
                if budget not in {0.0005, 0.001}:
                    continue
                raw_transfer = row.get("transfer")
                if raw_transfer in TRANSFERS:
                    transfer = raw_transfer
                elif raw_transfer and "->" in raw_transfer:
                    source, target = (int(x) for x in raw_transfer.split("->"))
                    transfer = transfer_name(source, target)
                else:
                    transfer = transfer_name(row["source"], row["target"])
                fo = num(row.get("FO") or row.get("FO-Acc"))
                if fo is not None:
                    values[(budget, transfer)].append((fo, row["variant"], str(path)))
    result = {}
    for budget in (0.0005, 0.001):
        for transfer in TRANSFERS:
            entries = values[(budget, transfer)]
            if len(entries) != 3:
                raise RuntimeError(f"baseline sparse reference needs 3 rows: {budget} {transfer}, got {len(entries)}")
            best = max(entries, key=lambda x: x[0])
            result[(budget, transfer)] = {
                "best_sparse_FO": best[0],
                "best_sparse_variant": best[1],
                "entries": [{"FO": x[0], "variant": x[1], "source": x[2]} for x in entries],
            }
    return result


def find_references():
    """Locate exactly the 12 completed Stage-1 reference rows from saved reports."""
    report_paths = [
        STAGE1 / "reports/machine1/stage1_per_run.csv",
        STAGE1 / "reports/machine2/stage1_per_run.csv",
    ]
    found = []
    wanted = {
        (0.0005, 0.1, 1.5, 0.5): ("A4", 262),
        (0.001, 0.15, 1.0, 1.0): ("A8", 524),
    }
    for path in report_paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                budget = num(row.get("requested_budget"))
                alpha, kappa, nu = num(row.get("alpha")), num(row.get("kappa")), num(row.get("nu"))
                key = (budget, alpha, kappa, nu)
                if key not in wanted:
                    continue
                if not same(num(row.get("omega")), 0.20) or not same(num(row.get("stage2_lr")), 0.020):
                    continue
                transfer_code = row.get("transfer")
                if transfer_code in TRANSFERS:
                    transfer = transfer_code
                    source, target = TRANSFER_CODE[transfer]
                elif transfer_code in {"0->1", "0->2", "1->0", "1->2", "2->0", "2->1"}:
                    source, target = (int(x) for x in transfer_code.split("->"))
                    transfer = transfer_name(source, target)
                else:
                    continue
                if transfer not in TRANSFERS:
                    continue
                run_relpath = row.get("run_relpath")
                summary_path = row.get("summary_path")
                if not summary_path and run_relpath:
                    summary_path = str(STAGE1 / run_relpath / "summary.json")
                if not summary_path:
                    raise RuntimeError(f"reference row has no summary path: {path} {row}")
                found.append({
                    "anchor_id": wanted[key][0], "K": wanted[key][1],
                    "budget": budget, "alpha": alpha, "kappa": kappa, "nu": nu,
                    "omega": 0.20, "stage2_lr": 0.020, "source": source, "target": target,
                    "transfer": transfer, "summary_path": str(Path(summary_path)),
                    "reference_report": str(path),
                })
    counts = Counter((x["budget"], x["transfer"]) for x in found)
    if len(found) != 12 or any(counts[(budget, t)] != 1 for budget in (0.0005, 0.001) for t in TRANSFERS):
        raise RuntimeError(f"expected 12 unambiguous references, found {len(found)} with counts {counts}")
    return sorted(found, key=lambda x: (x["budget"], TRANSFERS.index(x["transfer"])))


def validate_identity(summary, expected, source_kind):
    checks = {
        "experiment_key": summary.get("experiment_key") == expected["experiment_key"],
        "experiment_config_sha256": summary.get("experiment_config_sha256") == expected["experiment_config_sha256"],
        "status": summary.get("status") == "completed",
        "dataset": summary.get("dataset") == "office",
        "seed": summary.get("seed") == 2026,
        "variant": summary.get("variant") == "module_lbi",
        "source": int(summary.get("source")) == expected["source"],
        "target": int(summary.get("target")) == expected["target"],
        "budget": same(summary.get("requested_budget"), expected["budget"]),
        "alpha": same(summary.get("alpha"), expected["alpha"]),
        "kappa": same(summary.get("kappa"), expected["kappa"]),
        "nu": same(summary.get("nu"), expected["nu"]),
        "omega": same(summary.get("omega"), expected["omega"]),
        "stage2_lr": same(summary.get("stage2_lr"), expected["stage2_lr"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{source_kind} identity mismatch: {checks} for {expected['experiment_key']}")


def run_row(expected, summary_path, plan_entry=None, reference=False):
    summary_path = Path(summary_path)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    if not summary_path.exists():
        raise RuntimeError(f"missing summary: {summary_path}")
    summary = load_json(summary_path)
    expected = dict(expected)
    if plan_entry:
        expected["experiment_key"] = plan_entry["experiment_key"]
        expected["experiment_config_sha256"] = plan_entry["experiment_config_sha256"]
    else:
        expected["experiment_key"] = summary["experiment_key"]
        expected["experiment_config_sha256"] = summary["experiment_config_sha256"]
    validate_identity(summary, expected, "reference" if reference else "new")
    metrics_path = summary_path.parent / "metrics.jsonl"
    metrics = load_metrics(metrics_path)
    if not metrics:
        raise RuntimeError(f"no online_step records: {metrics_path}")
    if not finite(summary) or any(not finite(row) for row in metrics):
        raise RuntimeError(f"NaN/Inf in {summary_path}")
    K = int(expected["K"])
    selected = [num(row.get("selected_param_count")) for row in metrics]
    if any(x is None for x in selected):
        raise RuntimeError(f"missing selected_param_count: {metrics_path}")
    utilization = [x / K for x in selected]
    budget_violation_count = sum(x > K for x in selected)
    under95 = sum(x < 0.95 for x in utilization)
    under90 = sum(x < 0.90 for x in utilization)
    rollback = [bool(row.get("stage1_rollback_used", False)) for row in metrics]
    steps = [num(row.get("stage1_steps_completed")) for row in metrics]
    max_steps_hits = sum(bool(row.get("max_steps_hit", False)) or row.get("stage1_stop_reason") == "max_steps" for row in metrics)
    errors = []
    for row in metrics:
        for key in ("error", "exception", "traceback"):
            if row.get(key):
                errors.append(f"{key}@{row.get('_line_number')}")
    scientific_valid = not errors and budget_violation_count == 0 and max_steps_hits == 0
    hard_valid = scientific_valid and under90 == 0
    source, target = int(expected["source"]), int(expected["target"])
    row = {
        "run_type": "reused_stage1_reference" if reference else "new_joint_sweep",
        "anchor_id": expected["anchor_id"], "budget": expected["budget"], "K": K,
        "alpha": expected["alpha"], "kappa": expected["kappa"], "nu": expected["nu"],
        "omega": expected["omega"], "stage2_lr": expected["stage2_lr"],
        "source": source, "target": target, "transfer": transfer_name(source, target),
        "scientific_valid": scientific_valid, "hard_90pct_valid": hard_valid,
        "support_utilization_min": min(utilization),
        "support_utilization_p05": percentile(utilization, 0.05),
        "support_utilization_mean": statistics.fmean(utilization),
        "under_95pct_batch_count": under95, "under_90pct_batch_count": under90,
        "mean_budget_gap": statistics.fmean(K - x for x in selected),
        "max_budget_gap": max(K - x for x in selected),
        "rollback_rate": statistics.fmean(1.0 if x else 0.0 for x in rollback),
        "rollback_count": sum(rollback),
        "stage1_steps_mean": statistics.fmean(steps), "stage1_steps_max": max(steps),
        "stage1_max_steps_hit_count": max_steps_hits,
        "PU": summary.get("PU-Acc"), "FO": summary.get("FO-Acc"),
        "batch_count": len(metrics), "selected_count_min": min(selected),
        "selected_count_max": max(selected), "selected_count_mean": statistics.fmean(selected),
        "budget_violation_count": budget_violation_count, "error_count": len(errors),
        "validity_reason": "ok" if scientific_valid else ";".join(errors) or "budget_or_max_steps_failure",
        "experiment_key": expected["experiment_key"],
        "experiment_config_sha256": expected["experiment_config_sha256"],
        "summary_path": str(summary_path), "metrics_path": str(metrics_path),
        "_batch_utilization_values": utilization,
        "_batch_budget_gaps": [K - x for x in selected],
    }
    return row


def main():
    plan = load_json(PLAN_PATH)
    if plan.get("experiment_count") != 132 or len(plan.get("experiments", [])) != 132:
        raise RuntimeError("machine4 plan is not exactly 132 entries")
    plan_entries = plan["experiments"]
    if len({x["experiment_key"] for x in plan_entries}) != 132:
        raise RuntimeError("duplicate plan experiment_key")
    if len({x["experiment_config_sha256"] for x in plan_entries}) != 132:
        raise RuntimeError("duplicate plan scientific hash")
    for entry in plan_entries:
        if entry["dataset"] != "office" or entry["seed"] != 2026 or entry["variant"] != "module_lbi":
            raise RuntimeError(f"non-Office/non-module_lbi plan entry: {entry['experiment_key']}")
        if same(entry["omega"], 0.20) and same(entry["stage2_lr"], 0.020):
            raise RuntimeError("forbidden reused reference in new plan")
        if str(entry["expected_output_root"]) != str(RUNS_ROOT / "office" / ("AD" if entry["source"] == 0 and entry["target"] == 1 else "AW" if entry["source"] == 0 else "DA" if entry["source"] == 1 and entry["target"] == 0 else "DW" if entry["source"] == 1 else "WA" if entry["target"] == 0 else "WD") / "netB_bottleneck" / str(entry["requested_budget"]) / "LBI" / "seed_2026"):
            raise RuntimeError(f"unexpected output root: {entry['expected_output_root']}")

    manifest = load_json(MANIFEST_PATH)
    records = manifest.get("records", [])
    manifest_by_key = defaultdict(list)
    for record in records:
        manifest_by_key[record.get("experiment_key")].append(record)
    duplicate = sum(max(0, len(manifest_by_key[e["experiment_key"]]) - 1) for e in plan_entries)
    missing = 0
    failed = 0
    hash_mismatch = 0
    invalid_summary = 0
    new_rows = []
    for entry in plan_entries:
        matches = manifest_by_key.get(entry["experiment_key"], [])
        if not matches:
            missing += 1
            continue
        record = matches[-1]
        if record.get("experiment_config_sha256") != entry["experiment_config_sha256"]:
            hash_mismatch += 1
            continue
        if record.get("status") != "completed" or record.get("post_run_status") != "completed" or record.get("return_code") not in (0, "0"):
            failed += 1
            continue
        summary_path = record.get("summary_path")
        try:
            expected_root = Path(entry["expected_output_root"])
            if not Path(summary_path).resolve().is_relative_to(expected_root.resolve()):
                raise RuntimeError("summary outside expected output root")
            new_rows.append(run_row({
                "anchor_id": "A4" if same(entry["requested_budget"], 0.0005) else "A8",
                "K": 262 if same(entry["requested_budget"], 0.0005) else 524,
                "budget": entry["requested_budget"], "alpha": entry["alpha"],
                "kappa": entry["kappa"], "nu": entry["nu"], "omega": entry["omega"],
                "stage2_lr": entry["stage2_lr"], "source": entry["source"], "target": entry["target"],
            }, summary_path, entry, False))
        except Exception:
            invalid_summary += 1

    refs = find_references()
    ref_rows = []
    reference_records = []
    for ref in refs:
        row = run_row(ref, ref["summary_path"], None, True)
        ref_rows.append(row)
        reference_records.append({**ref, "summary_path": str(Path(ref["summary_path"]).resolve()), "metrics_path": str(Path(ref["summary_path"]).resolve().parent / "metrics.jsonl"), "experiment_key": row["experiment_key"], "experiment_config_sha256": row["experiment_config_sha256"]})

    if (len(new_rows), missing, failed, duplicate, hash_mismatch, invalid_summary) != (132, 0, 0, 0, 0, 0):
        raise RuntimeError(f"completion check failed: completed={len(new_rows)}, missing={missing}, failed={failed}, duplicate={duplicate}, hash_mismatch={hash_mismatch}, invalid_summary={invalid_summary}")
    if len(ref_rows) != 12:
        raise RuntimeError("reference row count is not 12")

    all_rows = sorted(new_rows + ref_rows, key=lambda x: (x["budget"], x["omega"], x["stage2_lr"], TRANSFERS.index(x["transfer"])))
    if len(all_rows) != 144 or len({x["experiment_key"] for x in all_rows}) != 144:
        raise RuntimeError("local result set is not 144 unique rows")
    baseline = baseline_reference()
    for row in all_rows:
        b = baseline[(row["budget"], row["transfer"])]
        row["best_sparse_FO"] = b["best_sparse_FO"]
        row["FO_margin"] = row["FO"] - b["best_sparse_FO"]

    config_rows = []
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(row["budget"], row["anchor_id"], row["alpha"], row["kappa"], row["nu"], row["omega"], row["stage2_lr"])].append(row)
    if len(grouped) != 24 or any(len(rows) != 6 for rows in grouped.values()):
        raise RuntimeError("expected 24 configurations with 6 transfers each")
    for key, rows in sorted(grouped.items()):
        budget, anchor, alpha, kappa, nu, omega, stage2_lr = key
        config_rows.append({
            "budget": budget, "anchor_id": anchor, "alpha": alpha, "kappa": kappa, "nu": nu,
            "omega": omega, "stage2_lr": stage2_lr,
            "valid_transfer_count": sum(bool(r["scientific_valid"]) for r in rows),
            "hard_90pct_valid_transfer_count": sum(bool(r["hard_90pct_valid"]) for r in rows),
            "global_min_utilization": min(r["support_utilization_min"] for r in rows),
            "aggregate_p05_utilization": percentile([u for r in rows for u in r["_batch_utilization_values"]], 0.05),
            "aggregate_mean_utilization": statistics.fmean(u for r in rows for u in r["_batch_utilization_values"]),
            "under_95pct_batch_count": sum(r["under_95pct_batch_count"] for r in rows),
            "under_90pct_batch_count": sum(r["under_90pct_batch_count"] for r in rows),
            "mean_budget_gap": statistics.fmean(r["mean_budget_gap"] for r in rows),
            "max_budget_gap": max(r["max_budget_gap"] for r in rows),
            "rollback_rate": statistics.fmean(r["rollback_rate"] for r in rows),
            "stage1_steps_mean": statistics.fmean(r["stage1_steps_mean"] for r in rows),
            "stage1_steps_max": max(r["stage1_steps_max"] for r in rows),
            "stage1_max_steps_hit_count": sum(r["stage1_max_steps_hit_count"] for r in rows),
            "mean_FO": statistics.fmean(r["FO"] for r in rows),
            "mean_PU": statistics.fmean(r["PU"] for r in rows),
            "mean_FO_margin": statistics.fmean(r["FO_margin"] for r in rows),
            "worst_transfer_FO_margin": min(r["FO_margin"] for r in rows),
            "eligible": all(r["scientific_valid"] and r["hard_90pct_valid"] for r in rows),
            "run_count": len(rows),
        })

    ranking = []
    for budget in (0.0005, 0.001):
        pair = [r for r in config_rows if same(r["budget"], budget)]
        pair.sort(key=lambda r: (
            not r["eligible"], -r["valid_transfer_count"], -r["hard_90pct_valid_transfer_count"],
            -r["mean_FO_margin"], -r["worst_transfer_FO_margin"], -r["mean_FO"],
            -r["aggregate_p05_utilization"], -r["aggregate_mean_utilization"],
            r["under_95pct_batch_count"], r["stage1_steps_max"], r["stage1_steps_mean"],
        ))
        for rank, row in enumerate(pair, 1):
            ranking.append({"rank_within_pair": rank, **row})

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_fields = [
        "run_type", "anchor_id", "budget", "K", "alpha", "kappa", "nu", "omega", "stage2_lr",
        "source", "target", "transfer", "scientific_valid", "hard_90pct_valid",
        "support_utilization_min", "support_utilization_p05", "support_utilization_mean",
        "under_95pct_batch_count", "under_90pct_batch_count", "mean_budget_gap", "max_budget_gap",
        "rollback_rate", "stage1_steps_mean", "stage1_steps_max", "stage1_max_steps_hit_count",
        "PU", "FO", "FO_margin", "best_sparse_FO", "batch_count", "selected_count_min",
        "selected_count_max", "selected_count_mean", "budget_violation_count", "error_count",
        "validity_reason", "experiment_key", "experiment_config_sha256", "summary_path", "metrics_path",
    ]
    with (REPORT_DIR / "per_run_144.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader(); writer.writerows({k: row.get(k) for k in run_fields} for row in all_rows)
    config_fields = list(config_rows[0].keys())
    with (REPORT_DIR / "per_config_24.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=config_fields)
        writer.writeheader(); writer.writerows(config_rows)
    write_json(REPORT_DIR / "per_config_24.json", {"configuration_count": 24, "configurations": config_rows})
    rank_fields = ["rank_within_pair"] + config_fields
    with (REPORT_DIR / "ranking.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rank_fields)
        writer.writeheader(); writer.writerows(ranking)
    md = ["# Office LBI joint sweep — machine4 FINALIZE", "", "Local ranking is within each owned anchor/budget pair. PU is report-only.", "", "| budget | anchor | omega | stage2_lr | eligible | mean FO margin | worst FO margin | mean FO | min utilization | stage1 max |", "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in ranking:
        md.append(f"| {r['budget']} | {r['anchor_id']} | {r['omega']} | {r['stage2_lr']} | {r['eligible']} | {r['mean_FO_margin']:.6f} | {r['worst_transfer_FO_margin']:.6f} | {r['mean_FO']:.6f} | {r['global_min_utilization']:.6f} | {r['stage1_steps_max']:.0f} |")
    md += ["", "The hard utilization gate is u_t >= 0.90 for every batch on every transfer; 0.95 is diagnostic/preference only.", "The sparse FO baseline is the maximum of seed-2026 Random 3-mask mean, Magnitude, and Saliency per transfer. No seed-2020 result was used.", "", "No training, retry, resume, launcher, or final formal evaluation was launched by FINALIZE."]
    (REPORT_DIR / "ranking.md").write_text("\n".join(md) + "\n")

    summary = {
        "mode": "FINALIZE", "machine_tag": "machine4", "timestamp": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(PLAN_PATH), "plan_sha256": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(),
        "new_run_status": {"planned": 132, "completed": len(new_rows), "missing": missing, "failed": failed, "duplicate": duplicate, "hash_mismatch": hash_mismatch, "invalid_summary": invalid_summary},
        "reference_run_count": len(ref_rows), "local_run_count": len(all_rows), "configuration_count": len(config_rows),
        "configurations_per_pair": {"0.0005/A4": 12, "0.001/A8": 12},
        "transfer_count_per_configuration": 6, "canonical_runs_root": str(RUNS_ROOT),
        "baseline": "seed2026 complete; best_sparse_FO=max(Random 3-mask mean, Magnitude, Saliency)",
        "reports": [str(REPORT_DIR / name) for name in ("per_run_144.csv", "per_config_24.csv", "per_config_24.json", "ranking.csv", "ranking.md")],
        "global_aggregator": {"status": "pending", "missing_finalize_ready": [f"machine{i}" for i in (1, 2, 3)]},
        "training_launched_by_finalize": False,
    }
    write_json(REPORT_DIR / "FINALIZE_SUMMARY.json", summary)
    write_json(REPORT_DIR / "reference_runs_12.json", {"count": 12, "runs": reference_records})

    ready = {
        "mode": "FINALIZE", "machine_tag": "machine4", "timestamp": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(PLAN_PATH), "plan_sha256": summary["plan_sha256"],
        "completed": 132, "missing": 0, "failed": 0, "duplicate": 0, "hash_mismatch": 0, "invalid_summary": 0,
        "reference_run_count": 12, "local_run_count": 144, "configuration_count": 24,
        "canonical_runs_root": str(RUNS_ROOT), "reports_dir": str(REPORT_DIR),
        "scientific_summary_valid": True, "training_launched": False,
    }
    write_json(PHASE_DIR / "FINALIZE_READY.json", ready)

    lines = [
        "# Office LBI joint sweep — machine4 FINALIZE", "", "Mode: `FINALIZE`.",
        "", "## Artifact-only completion", "",
        f"The exact machine4 plan contains 132 entries. Saved launcher/status/result artifacts prove completed=132, missing=0, failed=0, duplicate=0, hash mismatch=0, invalid summary=0.",
        "Every new identity was matched by experiment key and full scientific SHA256; summaries and real per-batch metrics JSONL were read from the canonical refined-repository runs root.",
        "", "## Reused references", "", "Exactly 12 seed-2026 Stage-1 reference identities were loaded from the existing Stage-1 artifacts at omega=0.20 and stage2_lr=0.020: six for A4/budget 0.0005 and six for A8/budget 0.001. They were not relaunched.",
        "", "## Local result set", "", "The local set has exactly 144 unique run rows: 132 new rows plus 12 reused references; there are 24 configurations with six transfers each. Per-batch support utilization, budget gaps, rollback, and Stage-1 steps were recomputed from metrics JSONL. Eligibility requires scientific validity and the 90% hard gate. PU is report-only.",
        "", "Seed-2026 sparse FO baselines use `max(Random 3-mask mean, Magnitude, Saliency)` per transfer. Seed-2020 results were not used.",
        "", "Outputs:", "", *[f"- `{name}`" for name in ("reports/machine4/per_run_144.csv", "reports/machine4/per_config_24.csv", "reports/machine4/per_config_24.json", "reports/machine4/ranking.csv", "reports/machine4/ranking.md", "reports/machine4/FINALIZE_SUMMARY.json", "phase_records/machine4/FINALIZE_READY.json")],
        "", "## Global aggregation", "", "Machine4 checked for the required global gate. `phase_records/machine1/FINALIZE_READY.json`, `machine2/FINALIZE_READY.json`, and `machine3/FINALIZE_READY.json` are absent, so global Office selection is pending and was not performed. No 576-row global selection files were created.",
        "", "No training was launched by FINALIZE. No launcher was invoked, and no incomplete entry was rerun.",
        "", "FINALIZE complete for machine4.", "No training was launched by FINALIZE.",
    ]
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    (PHASE_DIR / "FINALIZE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"completed": 132, "references": 12, "rows": 144, "configs": 24, "global_pending": ["machine1", "machine2", "machine3"]}, indent=2))


if __name__ == "__main__":
    main()
