#!/usr/bin/env python3
"""Artifact-only global Office LBI aggregator run from machine4 FINALIZE."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOINT = ROOT / "experiment_logs/shot_otta_office_lbi_joint_sweep_20260818"
REPORT_ROOT = JOINT / "reports"
GLOBAL_REPORT = REPORT_ROOT / "global"
SELECTED = JOINT / "selected_configs"
TRANSFERS = {"A->D", "A->W", "D->A", "D->W", "W->A", "W->D"}
EXPECTED_PAIRS = {
    ("0.0005", "A6"), ("0.0005", "A1"), ("0.0005", "A4"),
    ("0.001", "A6"), ("0.001", "A4"), ("0.001", "A8"),
    ("0.002", "A2"), ("0.002", "A3"),
}


def f(value):
    return float(value) if value not in (None, "") else None


def b(value):
    return str(value).lower() == "true"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_config_rows(machine):
    path = REPORT_ROOT / f"machine{machine}/per_config_24.csv"
    rows = read_rows(path)
    out = []
    for row in rows:
        row = dict(row)
        row["machine_tag"] = f"machine{machine}"
        for key in ("budget", "alpha", "kappa", "nu", "omega", "stage2_lr", "global_min_utilization", "aggregate_p05_utilization", "aggregate_mean_utilization", "mean_budget_gap", "max_budget_gap", "rollback_rate", "stage1_steps_mean", "stage1_steps_max", "mean_FO", "mean_PU", "mean_FO_margin", "worst_transfer_FO_margin"):
            row[key] = f(row.get(key))
        for key in ("valid_transfer_count", "hard_90pct_valid_transfer_count", "under_95pct_batch_count", "under_90pct_batch_count", "stage1_max_steps_hit_count", "run_count", "transfer_count"):
            if row.get(key) not in (None, ""):
                row[key] = int(float(row[key]))
        row["eligible"] = row.get("eligible", row.get("hard_90pct_eligible", "False"))
        row["eligible"] = b(row["eligible"])
        row["hard_90pct_eligible"] = row["valid_transfer_count"] == 6 and row["hard_90pct_valid_transfer_count"] == 6
        out.append(row)
    return out


def read_run_rows(machine):
    rows = read_rows(REPORT_ROOT / f"machine{machine}/per_run_144.csv")
    out = []
    for row in rows:
        x = dict(row)
        x["machine_tag"] = f"machine{machine}"
        x["reused_reference"] = b(x.get("reused_reference", x.get("run_type") == "reused_stage1_reference"))
        for key in ("budget", "alpha", "kappa", "nu", "omega", "stage2_lr", "support_utilization_min", "support_utilization_p05", "support_utilization_mean", "mean_budget_gap", "max_budget_gap", "rollback_rate", "stage1_steps_mean", "stage1_steps_max", "PU", "FO"):
            if x.get(key) not in (None, ""):
                x[key] = f(x[key])
        x["scientific_valid"] = b(x.get("scientific_valid"))
        hard = x.get("hard_90pct_valid")
        x["hard_90pct_valid"] = b(hard) if hard not in (None, "") else x["support_utilization_min"] >= 0.90
        x["transfer"] = x.get("transfer", "")
        if x["transfer"] not in TRANSFERS:
            raise RuntimeError(f"invalid transfer in machine{machine}: {x['transfer']}")
        out.append(x)
    return out


def main():
    ready = {}
    for machine in (1, 2, 3, 4):
        path = JOINT / f"phase_records/machine{machine}/FINALIZE_READY.json"
        if not path.exists():
            raise RuntimeError(f"missing global gate artifact: {path}")
        ready[machine] = json.loads(path.read_text())

    all_runs = []
    all_configs = []
    for machine in (1, 2, 3, 4):
        runs = read_run_rows(machine)
        configs = read_config_rows(machine)
        if len(runs) != 144 or len(configs) != 24:
            raise RuntimeError(f"machine{machine} local count mismatch")
        all_runs.extend(runs)
        all_configs.extend(configs)

    if len(all_runs) != 576 or len({r["experiment_key"] for r in all_runs}) != 576:
        raise RuntimeError("global run rows are not exactly 576 unique identities")
    if len(all_configs) != 96:
        raise RuntimeError("global configuration rows are not exactly 96")
    pair_runs = Counter((str(r["budget"]), r["anchor_id"]) for r in all_runs)
    pair_configs = Counter((str(r["budget"]), r["anchor_id"]) for r in all_configs)
    if set(pair_runs) != EXPECTED_PAIRS or any(v != 72 for v in pair_runs.values()):
        raise RuntimeError(f"global pair run counts mismatch: {pair_runs}")
    if set(pair_configs) != EXPECTED_PAIRS or any(v != 12 for v in pair_configs.values()):
        raise RuntimeError(f"global pair config counts mismatch: {pair_configs}")
    config_run_counts = Counter((str(r["budget"]), r["anchor_id"], str(r["alpha"]), str(r["kappa"]), str(r["nu"]), str(r["omega"]), str(r["stage2_lr"])) for r in all_runs)
    if any(count != 6 for count in config_run_counts.values()) or len(config_run_counts) != 96:
        raise RuntimeError("not every global configuration has six transfers")
    if any(pair_runs[(str(r["budget"]), r["anchor_id"])] != 72 for r in all_runs):
        raise RuntimeError("pair count invariant failed")

    # Every local config row is retained as an auditable global config row.
    config_fields = ["machine_tag", "budget", "anchor_id", "alpha", "kappa", "nu", "omega", "stage2_lr", "valid_transfer_count", "hard_90pct_valid_transfer_count", "global_min_utilization", "aggregate_p05_utilization", "aggregate_mean_utilization", "under_95pct_batch_count", "under_90pct_batch_count", "mean_budget_gap", "max_budget_gap", "rollback_rate", "stage1_steps_mean", "stage1_steps_max", "stage1_max_steps_hit_count", "mean_FO", "mean_PU", "mean_FO_margin", "worst_transfer_FO_margin", "eligible", "hard_90pct_eligible"]
    rankable = []
    for row in all_configs:
        row["global_pair_key"] = f"budget={row['budget']};anchor={row['anchor_id']}"
        row["global_eligible"] = row["valid_transfer_count"] == 6 and row["hard_90pct_valid_transfer_count"] == 6
        rankable.append(row)

    ranked = []
    for budget in (0.0005, 0.001, 0.002):
        pair = [r for r in rankable if math.isclose(r["budget"], budget)]
        pair.sort(key=lambda r: (
            not r["global_eligible"], -r["mean_FO_margin"], -r["worst_transfer_FO_margin"],
            -r["mean_FO"], -r["aggregate_p05_utilization"], -r["aggregate_mean_utilization"],
            r["under_95pct_batch_count"], r["stage1_steps_max"], r["stage1_steps_mean"],
        ))
        for rank, row in enumerate(pair, 1):
            row["global_rank_within_budget"] = rank
            ranked.append(row)

    winners = {}
    for budget in (0.0005, 0.001, 0.002):
        candidates = [r for r in ranked if math.isclose(r["budget"], budget)]
        eligible = [r for r in candidates if r["global_eligible"]]
        if not eligible:
            raise RuntimeError(f"global eligibility gate failed for budget {budget}")
        winners[budget] = eligible[0]

    GLOBAL_REPORT.mkdir(parents=True, exist_ok=True)
    run_fields = sorted({key for row in all_runs for key in row})
    with (GLOBAL_REPORT / "OFFICE_LBI_JOINT_SWEEP_ALL_576.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(all_runs)
    ranked_fields = ["global_rank_within_budget"] + config_fields + ["global_pair_key", "global_eligible"]
    with (GLOBAL_REPORT / "OFFICE_LBI_JOINT_SWEEP_CONFIGS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ranked_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(ranked)

    for budget, filename in ((0.0005, "office_budget_0005_final_lbi.json"), (0.001, "office_budget_001_final_lbi.json"), (0.002, "office_budget_002_final_lbi.json")):
        winner = winners[budget]
        payload = {
            "selection_status": "frozen_final_lbi_tuple",
            "budget": budget, "anchor_id": winner["anchor_id"],
            "alpha": winner["alpha"], "kappa": winner["kappa"], "nu": winner["nu"],
            "omega": winner["omega"], "stage2_lr": winner["stage2_lr"],
            "valid_transfer_count": winner["valid_transfer_count"],
            "hard_90pct_valid_transfer_count": winner["hard_90pct_valid_transfer_count"],
            "mean_FO_margin": winner["mean_FO_margin"], "worst_transfer_FO_margin": winner["worst_transfer_FO_margin"],
            "mean_FO": winner["mean_FO"], "mean_PU": winner["mean_PU"],
            "global_rank_within_budget": winner["global_rank_within_budget"],
            "source_machine": winner["machine_tag"],
            "runtime_comparable": True,
            "search_closed": True,
        }
        write_json(SELECTED / filename, payload)

    overall = max(winners.values(), key=lambda r: r["mean_FO"])
    final_tuples = {
        "selection_status": "frozen_final_lbi_tuples",
        "office_lbi_hyperparameter_search_closed": True,
        "overall_best_budget_by_mean_FO": overall["budget"],
        "budgets": {str(budget): {
            "budget": budget, "anchor_id": row["anchor_id"], "alpha": row["alpha"], "kappa": row["kappa"], "nu": row["nu"], "omega": row["omega"], "stage2_lr": row["stage2_lr"], "mean_FO": row["mean_FO"], "mean_FO_margin": row["mean_FO_margin"], "worst_transfer_FO_margin": row["worst_transfer_FO_margin"], "global_rank_within_budget": row["global_rank_within_budget"], "source_machine": row["machine_tag"],
        } for budget, row in winners.items()},
        "formal_rerun_recommendation": {"runs": 18, "layout": "3 budgets × 6 transfers", "one_experiment_per_gpu": True, "runtime_comparable": True, "is_tuning": False},
    }
    write_json(SELECTED / "OFFICE_FINAL_LBI_TUPLES.json", final_tuples)

    md = ["# Office LBI final global selection", "", "All four machine FINALIZE_READY records were present and validated. The global set contains exactly 576 unique run rows = 528 new joint-sweep runs + 48 reused Stage-1 reference runs, covering exactly eight anchor/budget pairs and 12 configurations × 6 transfers per pair.", "", "Global eligibility requires 6/6 scientific-valid transfers and 6/6 transfers passing the per-batch 90% utilization hard gate. PU is report-only. Ranking is by mean_FO_margin, worst_transfer_FO_margin, mean_FO, 95%-utilization diagnostics, stage1_steps_max, and stage1_steps_mean.", "", "## Frozen tuples", "", "| budget | anchor | alpha | kappa | nu | omega | stage2_lr | mean FO | mean FO margin | worst-transfer margin |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:"]
    for budget in (0.0005, 0.001, 0.002):
        r = winners[budget]
        md.append(f"| {budget} | {r['anchor_id']} | {r['alpha']} | {r['kappa']} | {r['nu']} | {r['omega']} | {r['stage2_lr']} | {r['mean_FO']:.6f} | {r['mean_FO_margin']:.6f} | {r['worst_transfer_FO_margin']:.6f} |")
    md += ["", f"Overall best budget by mean_FO: `{overall['budget']}`.", "", "Office LBI hyperparameter search is closed. The final formal LBI rerun is recommended, but not launched: 3 budgets × 6 transfers = 18 frozen-parameter evaluation runs, 1 experiment/GPU, runtime_comparable=true. These 18 runs are not tuning.", "", "No training was launched by FINALIZE."]
    (GLOBAL_REPORT / "OFFICE_LBI_FINAL_SELECTION.md").write_text("\n".join(md) + "\n")
    write_json(GLOBAL_REPORT / "OFFICE_LBI_FINAL_SELECTION.json", final_tuples | {"global_run_count": 576, "global_new_run_count": 528, "global_reference_run_count": 48, "global_configuration_count": 96, "expected_pairs": sorted([list(x) for x in EXPECTED_PAIRS])})

    print(json.dumps({"global_runs": len(all_runs), "global_configs": len(all_configs), "winners": {str(k): {"anchor": v["anchor_id"], "omega": v["omega"], "stage2_lr": v["stage2_lr"], "mean_FO": v["mean_FO"]} for k, v in winners.items()}, "overall_best_budget_by_mean_FO": overall["budget"]}, indent=2))


if __name__ == "__main__":
    main()
