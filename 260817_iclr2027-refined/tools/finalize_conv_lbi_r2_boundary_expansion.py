#!/usr/bin/env python3
"""Artifact-only audit and closure for Office out-channel R2 expansion."""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tools.check_experiment_status import check_status, load_plan

ROOT = PROJECT_ROOT / "experiment_logs/office_conv_lbi_r2_out_boundary_expansion_seed2026_20260828"
R2_ROOT = PROJECT_ROOT / "experiment_logs/office_conv_lbi_r2_out_joint_sweep_seed2026_20260828"
BASELINE_ROOT = PROJECT_ROOT / "experiment_logs/conv_baseline_formal_global_20260827"
SELECTION_PATH = R2_ROOT / "selected_configs/OFFICE_OUT_R2_SELECTION.json"
INITIAL_FINALIZE = R2_ROOT / "phase_records/R2/FINALIZE.md"
TRANSFERS = ("AD", "AW", "DA", "DW", "WA", "WD")
CELLS = (
    ("E0", "E0_p0005_omega_0p30_lr_0p010", 0.0005, 4, 0.05, 0.50, 0.30, 0.010),
    ("E1", "E1_p0005_omega_0p10_lr_0p020", 0.0005, 4, 0.05, 0.50, 0.10, 0.020),
    ("E2", "E2_p0005_omega_0p30_lr_0p020", 0.0005, 4, 0.05, 0.50, 0.30, 0.020),
    ("E3", "E3_p001_omega_0p30_lr_0p0025", 0.001, 9, 0.05, 0.50, 0.30, 0.0025),
)
EXPECTED_BASELINES = {
    0.0005: (77.71390853145846, 77.82129825807982),
    0.001: (77.78220455871212, 77.96646360748584),
    0.002: (77.83612269319094, 77.93517719210904),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def p05(values):
    values = sorted(float(value) for value in values)
    require(values, "cannot calculate percentile from no values")
    index = 0.05 * (len(values) - 1)
    low, high = int(index), math.ceil(index)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (index - low)


def protocol_checks():
    for name, heading in (
        ("OTTA_CONV_LBI_PROTOCOL_20260826_v1.md", "# OTTA_CONV_LBI_PROTOCOL_20260826_v1"),
        ("OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1.md", "# OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v1"),
        ("OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2.md", "# OTTA_CONV_LBI_SEARCH_PROTOCOL_20260827_v2"),
    ):
        path = PROJECT_ROOT / "protocol/shot-otta_conv" / name
        require(path.is_file() and path.read_text(encoding="utf-8").startswith(heading), f"normative protocol drifted: {path}")
    selection = load(SELECTION_PATH)
    require(selection["budgets"]["0.002"]["status"] == "FROZEN_NO_EXPANSION", ".002 is not frozen")
    require(selection["budgets"]["0.0005"]["upper_boundary_dimensions"] == ["omega", "stage2_lr"], ".0005 boundary state drifted")
    require(selection["budgets"]["0.001"]["upper_boundary_dimensions"] == ["omega"], ".001 boundary state drifted")
    return selection


def expected_cell(entry):
    for cell in CELLS:
        cell_id, _, rho, kg, alpha, nu, omega, stage2_lr = cell
        if entry["cell_id"] == cell_id:
            return cell
    raise RuntimeError(f"unexpected cell id: {entry.get('cell_id')}")


def validate_metrics(entry, summary_path):
    metric_path = summary_path.parent / "metrics.jsonl"
    require(metric_path.is_file(), f"missing metrics: {metric_path}")
    metrics = [json.loads(line) for line in metric_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    online = [row for row in metrics if row.get("event") == "online_step"]
    require(len(online) == int(load(summary_path)["online_steps"]), f"online batch count mismatch: {summary_path}")
    _, _, rho, kg, alpha, nu, omega, stage2_lr = expected_cell(entry)
    for row in online:
        require(finite(row), f"non-finite online record: {summary_path}")
        require((row.get("experiment_key"), row.get("experiment_config_sha256")) == (entry["experiment_key"], entry["experiment_config_sha256"]), f"metrics identity mismatch: {summary_path}")
        require(float(row.get("selected_group_count", math.inf)) <= kg and float(row.get("stage1_support_count", math.inf)) <= kg, f"budget violation: {summary_path}")
        require(row.get("valid_lbi_step") is True and row.get("max_steps_hit") is False, f"invalid Stage-1 batch: {summary_path}")
        require(row.get("strict_budget_boundary_stop") is True, f"strict rollback/budget contract failure: {summary_path}")
        require(row.get("bn_stats_frozen") is True and row.get("bn_stats_policy") == "frozen", f"controlled Conv BN drifted: {summary_path}")
        require((row.get("variant"), row.get("group_mode"), row.get("requested_budget")) == ("conv_out_lbi", "out_channel", rho), f"scientific config drifted: {summary_path}")
        require((row.get("alpha"), row.get("kappa"), row.get("nu"), row.get("omega"), row.get("stage2_lr")) == (alpha, 1.0, nu, omega, stage2_lr), f"LBI tuple drifted: {summary_path}")
        require(row.get("lbi_initialization") == "masked_delta" and row.get("lbi_state_lifecycle") == "reset_every_online_step", f"LBI state contract drifted: {summary_path}")
    return online


def audit_expansion():
    candidates = []
    all_keys, all_hashes, all_roots = set(), set(), set()
    for cell_id, plan_name, rho, kg, alpha, nu, omega, stage2_lr in CELLS:
        plan_path = ROOT / "plans" / f"{plan_name}.jsonl"
        runs_root = ROOT / "runs" / plan_name
        plan = load_plan(plan_path)
        entries = plan["experiments"]
        require(len(entries) == 6 and [entry["transfer"] for entry in entries] == list(TRANSFERS), f"plan transfer scope drifted: {cell_id}")
        status = check_status(plan, str(runs_root))
        require(not status["unassociated_invalid_summaries"], f"unassociated invalid artifacts: {cell_id}")
        states = {row["experiment_key"]: row for row in status["experiments"]}
        rows = []
        for entry in entries:
            require((entry["cell_id"], entry["requested_budget"], entry["max_group_count"], entry["alpha"], entry["kappa"], entry["nu"], entry["omega"], entry["stage2_lr"]) == (cell_id, rho, kg, alpha, 1.0, nu, omega, stage2_lr), f"unauthorized planned scientific cell: {entry['experiment_key']}")
            scientific_lbi = entry["scientific_config"].get("lbi", {})
            require((scientific_lbi.get("omega"), scientific_lbi.get("stage2_lr")) == (omega, stage2_lr), f"omega/LR absent from identity: {entry['experiment_key']}")
            state = states.get(entry["experiment_key"], {})
            paths = state.get("matching_summary_paths", [])
            require(state.get("plan_status") == "completed" and len(paths) == 1, f"planned row is not uniquely complete: {entry['experiment_key']}")
            summary_path = Path(paths[0])
            summary = load(summary_path)
            require(finite(summary) and summary.get("status") == "completed", f"incomplete/non-finite summary: {summary_path}")
            require((summary.get("experiment_key"), summary.get("experiment_config_sha256")) == (entry["experiment_key"], entry["experiment_config_sha256"]), f"summary identity mismatch: {summary_path}")
            online = validate_metrics(entry, summary_path)
            require(summary.get("max_steps_hit_count") == 0 and summary.get("valid_lbi_run") is True, f"scientific validity failure: {summary_path}")
            require(summary.get("bn_stats_frozen") is True and summary.get("bn_stats_policy") == "frozen", f"BN summary contract failure: {summary_path}")
            util_mean = float(summary["group_utilization_mean"])
            util_p05 = float(summary["group_utilization_p05"])
            observed_util = [float(row["selected_group_count"]) / kg for row in online]
            require(abs(util_mean - statistics.fmean(observed_util)) < 1e-12 and abs(util_p05 - p05(observed_util)) < 1e-12, f"utilization summary drifted: {summary_path}")
            rows.append({
                "transfer": entry["transfer"], "experiment_key": entry["experiment_key"], "experiment_config_sha256": entry["experiment_config_sha256"],
                "summary_path": str(summary_path), "metrics_path": str(summary_path.parent / "metrics.jsonl"),
                "FO": float(summary["FO-Acc"]), "PU": float(summary["PU-Acc"]), "scientific_valid": True,
                "utilization_eligible": util_mean >= 0.90 and util_p05 >= 0.75,
                "utilization_mean": util_mean, "utilization_p05": util_p05,
                "stage1_steps_mean": float(summary["stage1_steps_mean"]), "stage1_steps_max": int(summary["stage1_steps_completed_max"]),
                "stage1_runtime_sec": sum(float(row["lbi_stage1_runtime_sec"]) for row in online),
                "online_runtime_sec": float(summary["online_compute_runtime_sec"]),
            })
            all_keys.add(entry["experiment_key"]); all_hashes.add(entry["experiment_config_sha256"]); all_roots.add(entry["expected_output_root"])
        require(len(rows) == 6, f"cell does not contain six rows: {cell_id}")
        candidates.append({
            "candidate_id": cell_id, "origin": "boundary_expansion", "rho": rho, "K_G": kg, "anchor": "A1", "alpha": alpha, "kappa": 1.0, "nu": nu, "omega": omega, "stage2_lr": stage2_lr,
            "transfer_rows": rows, "scientific_valid_all_six": all(row["scientific_valid"] for row in rows),
            "utilization_eligible_all_six": all(row["utilization_eligible"] for row in rows),
            "FO": {row["transfer"]: row["FO"] for row in rows}, "FO_six_transfer_macro": statistics.fmean(row["FO"] for row in rows),
            "worst_transfer_FO": min(row["FO"] for row in rows), "PU_six_transfer_macro": statistics.fmean(row["PU"] for row in rows),
            "worst_stage1_max_steps": max(row["stage1_steps_max"] for row in rows),
            "six_transfer_mean_stage1_steps": statistics.fmean(row["stage1_steps_mean"] for row in rows),
            "mean_stage1_runtime_sec": statistics.fmean(row["stage1_runtime_sec"] for row in rows),
            "mean_online_runtime_sec": statistics.fmean(row["online_runtime_sec"] for row in rows),
        })
    require(len(all_keys) == len(all_hashes) == len(all_roots) == 24, "24-row expansion identity/root audit failed")
    return candidates


def initial_candidates():
    rows = []
    for line in INITIAL_FINALIZE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| .0005 / 4 |") and not line.startswith("| .001 / 9 |"):
            continue
        columns = [column.strip() for column in line.split("|")[1:-1]]
        rho, kg = (float(item.strip()) for item in columns[0].split("/"))
        alpha, nu = (float(item.strip()) for item in columns[1].split("/"))
        omega_field = columns[2].replace(" reused", "")
        omega, stage2_lr = (float(item.strip()) for item in omega_field.split("/"))
        fo = [float(item) for item in columns[5].split("/")]
        require(len(fo) == 6, f"bad initial FO row: {line}")
        rows.append({
            "candidate_id": f"I_{rho:g}_{omega:g}_{stage2_lr:g}", "origin": "initial_R2", "rho": rho, "K_G": int(kg), "anchor": "A1", "alpha": alpha, "kappa": 1.0, "nu": nu, "omega": omega, "stage2_lr": stage2_lr,
            "transfer_rows": [], "scientific_valid_all_six": columns[3] == "YES", "utilization_eligible_all_six": columns[4] == "YES",
            "FO": dict(zip(TRANSFERS, fo)), "FO_six_transfer_macro": float(columns[6]), "worst_transfer_FO": float(columns[7]), "PU_six_transfer_macro": float(columns[8]),
            "worst_stage1_max_steps": int(columns[9]), "six_transfer_mean_stage1_steps": float(columns[10]), "mean_stage1_runtime_sec": None, "mean_online_runtime_sec": float(columns[11]),
        })
    require(sum(row["rho"] == 0.0005 for row in rows) == 9 and sum(row["rho"] == 0.001 for row in rows) == 9, "initial R2 ranking universe drifted")
    return rows


def baseline_comparison():
    sparse = {"conv_out_random", "conv_out_magnitude", "conv_out_saliency", "conv_filter_random", "conv_filter_magnitude", "conv_filter_saliency"}
    by_budget = {}
    with (BASELINE_ROOT / "reports/OFFICE_ALL_CONDITIONS.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for rho in EXPECTED_BASELINES:
        selected = [row for row in rows if row["variant"] in sparse and float(row["rho"]) == rho]
        require(len(selected) == 36, f"frozen baseline six-method/six-transfer coverage drifted at rho={rho}")
        by_method = {}
        by_transfer = {}
        for row in selected:
            by_method.setdefault(row["variant"], []).append(float(row["FO"]))
            by_transfer.setdefault(row["code"], []).append(float(row["FO"]))
        require(set(by_method) == sparse and set(by_transfer) == set(TRANSFERS), f"baseline method/transfer coverage drifted at rho={rho}")
        fixed = max(statistics.fmean(values) for values in by_method.values())
        oracle = statistics.fmean(max(values) for values in by_transfer.values())
        expected_fixed, expected_oracle = EXPECTED_BASELINES[rho]
        require(abs(fixed - expected_fixed) < 1e-10 and abs(oracle - expected_oracle) < 1e-10, f"baseline reference drifted at rho={rho}")
        by_budget[str(rho)] = {"best_fixed_sparse_macro": fixed, "same_budget_per_transfer_sparse_oracle": oracle, "method_macros": {name: statistics.fmean(values) for name, values in sorted(by_method.items())}, "per_transfer_oracle": {name: max(values) for name, values in sorted(by_transfer.items())}}
    return by_budget


def rank(candidates):
    eligible = [item for item in candidates if item["scientific_valid_all_six"] and item["utilization_eligible_all_six"]]
    eligible.sort(key=lambda item: (-item["FO_six_transfer_macro"], -item["worst_transfer_FO"], item["worst_stage1_max_steps"], item["six_transfer_mean_stage1_steps"], item["mean_online_runtime_sec"]))
    for position, item in enumerate(eligible, start=1):
        item["rank"] = position
    for item in candidates:
        if item not in eligible:
            item["rank"] = None
    return eligible


def markdown_table(rows):
    head = "| rank | origin | omega / stage2_lr | valid | util | FO macro | worst FO | max S1 | mean S1 | mean online s |\n|---:|---|---|:---:|:---:|---:|---:|---:|---:|---:|"
    body = []
    for row in rows:
        body.append("| {rank} | {origin} | {omega:.5g} / {stage2_lr:.5g} | {valid} | {util} | {fo:.6f} | {worst:.6f} | {maxs} | {means:.6f} | {runtime:.6f} |".format(rank=row["rank"] if row["rank"] is not None else "—", origin=row["origin"], omega=row["omega"], stage2_lr=row["stage2_lr"], valid="YES" if row["scientific_valid_all_six"] else "NO", util="YES" if row["utilization_eligible_all_six"] else "NO", fo=row["FO_six_transfer_macro"], worst=row["worst_transfer_FO"], maxs=row["worst_stage1_max_steps"], means=row["six_transfer_mean_stage1_steps"], runtime=row["mean_online_runtime_sec"]))
    return "\n".join([head, *body])


def write_outputs(selection, expansion, initial, baselines):
    grouped = {}
    winners = {}
    for rho in (0.0005, 0.001):
        candidates = [item for item in initial + expansion if item["rho"] == rho]
        require(len(candidates) == (12 if rho == 0.0005 else 10), f"bad final ranking universe at rho={rho}")
        ranked = rank(candidates)
        require(ranked, f"no eligible candidates at rho={rho}")
        grouped[str(rho)] = candidates
        winners[str(rho)] = ranked[0]
    p002 = selection["budgets"]["0.002"]["selected_tuple"]
    winners["0.002"] = {"candidate_id": "FROZEN_INITIAL_R2", "origin": "initial_R2_frozen", "rho": 0.002, "K_G": 18, "anchor": "A4", "alpha": 0.1, "kappa": 1.0, "nu": 1.0, "omega": 0.025, "stage2_lr": 0.0025, "FO_six_transfer_macro": p002["FO_six_transfer_macro"], "worst_transfer_FO": p002["worst_transfer_FO"], "worst_stage1_max_steps": p002["worst_stage1_max_steps"], "six_transfer_mean_stage1_steps": p002["six_transfer_mean_stage1_steps"], "mean_online_runtime_sec": p002["mean_online_runtime_sec"], "PU_six_transfer_macro": p002["PU_six_transfer_macro"]}
    for rho, winner in winners.items():
        baseline = baselines[rho]
        winner["delta_vs_same_budget_best_fixed_sparse"] = winner["FO_six_transfer_macro"] - baseline["best_fixed_sparse_macro"]
        winner["delta_vs_same_budget_per_transfer_sparse_oracle"] = winner["FO_six_transfer_macro"] - baseline["same_budget_per_transfer_sparse_oracle"]
    tuples = {"schema_version": 1, "dataset": "office", "method": "shot", "variant": "conv_out_lbi", "group_mode": "out_channel", "seed": 2026, "filter_connection_status": "BLOCKED_V2", "filter_tuple_fabricated": False, "boundary_expansion": "one_time_closed", "tuples": [winners[str(rho)] for rho in (0.0005, 0.001, 0.002)]}
    comparison = {"schema_version": 1, "phase": "R2_BOUNDARY_EXPANSION", "new_rows": 24, "valid_cells": {"p0005": sum(item["scientific_valid_all_six"] and item["utilization_eligible_all_six"] for item in expansion if item["rho"] == 0.0005), "p001": sum(item["scientific_valid_all_six"] and item["utilization_eligible_all_six"] for item in expansion if item["rho"] == 0.001)}, "candidate_ranking_by_budget": grouped, "final_winners": winners, "same_budget_sparse_baselines": baselines, "filter_connection_status": "BLOCKED_V2", "filter_tuple_fabricated": False, "second_boundary_expansion_allowed": False, "office_out_tuning_closed": True}
    selected = ROOT / "selected_configs"; reports = ROOT / "reports"; records = ROOT / "phase_records/R2_BOUNDARY_EXPANSION"
    selected.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True); records.mkdir(parents=True, exist_ok=True)
    (selected / "OFFICE_OUT_LBI_FINAL_TUPLES.json").write_text(json.dumps(tuples, indent=2) + "\n", encoding="utf-8")
    tuple_lines = ["# Office Out-channel Conv-LBI final tuples", "", "`filter_connection_status = BLOCKED_V2`; `filter_tuple_fabricated = false`.", "", "| rho | alpha | nu | omega | stage2_lr | FO macro |", "|---:|---:|---:|---:|---:|---:|"]
    tuple_lines += [f"| {item['rho']:.4g} | {item['alpha']:.3g} | {item['nu']:.3g} | {item['omega']:.4g} | {item['stage2_lr']:.4g} | {item['FO_six_transfer_macro']:.6f} |" for item in tuples["tuples"]]
    (selected / "OFFICE_OUT_LBI_FINAL_TUPLES.md").write_text("\n".join(tuple_lines) + "\n", encoding="utf-8")
    flat = [item for rho in ("0.0005", "0.001") for item in grouped[rho]]
    fields = ["rho", "candidate_id", "origin", "K_G", "alpha", "kappa", "nu", "omega", "stage2_lr", "scientific_valid_all_six", "utilization_eligible_all_six", "rank", "FO_six_transfer_macro", "worst_transfer_FO", "PU_six_transfer_macro", "worst_stage1_max_steps", "six_transfer_mean_stage1_steps", "mean_stage1_runtime_sec", "mean_online_runtime_sec"] + [f"FO_{transfer}" for transfer in TRANSFERS]
    with (reports / "OFFICE_OUT_R2_FINAL_COMPARISON.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for item in flat:
            writer.writerow({**{field: item.get(field) for field in fields}, **{f"FO_{transfer}": item["FO"][transfer] for transfer in TRANSFERS}})
    (reports / "OFFICE_OUT_R2_FINAL_COMPARISON.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    report_lines = ["# Office out-channel R2 final comparison", "", "## .0005 ranking", "", markdown_table(grouped["0.0005"]), "", "## .001 ranking", "", markdown_table(grouped["0.001"]), "", "## Final winners and same-budget sparse references", "", "| rho | final FO | delta best fixed sparse | delta per-transfer oracle |", "|---:|---:|---:|---:|"]
    report_lines += [f"| {rho} | {winners[rho]['FO_six_transfer_macro']:.6f} | {winners[rho]['delta_vs_same_budget_best_fixed_sparse']:+.6f} | {winners[rho]['delta_vs_same_budget_per_transfer_sparse_oracle']:+.6f} |" for rho in ("0.0005", "0.001", "0.002")]
    report_lines += ["", "PU is report-only and was not used for selection. `.002` remains the frozen initial-R2 tuple; no expansion row was created for it."]
    (reports / "OFFICE_OUT_R2_FINAL_COMPARISON.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    exp_lines = ["# Office out-channel R2 one-time boundary expansion — FINALIZE", "", "Mode: artifact-only. No launch, retry, resume, rerun, grid change, or protocol change was performed.", "", "## Expansion audit", "", "All 24/24 planned expansion rows are uniquely completed with matching plan identities and scientific SHA256 values. Every online batch passed the finite-value, `selected_group_count <= K_G`, Stage-1 cap, support/mask/budget contract, and frozen controlled-Conv BN checks.", "", "| cell | scientific valid all six | utilization eligible all six | FO AD/AW/DA/DW/WA/WD | FO macro | worst FO | PU macro | worst S1 max | mean S1 | mean S1 runtime s | mean online runtime s |", "|---|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in expansion:
        fo_values = "/".join(f"{item['FO'][transfer]:.3f}" for transfer in TRANSFERS)
        exp_lines.append(f"| {item['candidate_id']} | {'YES' if item['scientific_valid_all_six'] else 'NO'} | {'YES' if item['utilization_eligible_all_six'] else 'NO'} | {fo_values} | {item['FO_six_transfer_macro']:.6f} | {item['worst_transfer_FO']:.6f} | {item['PU_six_transfer_macro']:.6f} | {item['worst_stage1_max_steps']} | {item['six_transfer_mean_stage1_steps']:.6f} | {item['mean_stage1_runtime_sec']:.6f} | {item['mean_online_runtime_sec']:.6f} |")
    exp_lines += ["", "## Ranking and closure", "", "The `.0005` universe contains the nine initial candidates plus E0/E1/E2; the `.001` universe contains the nine initial candidates plus E3. Ranking used only the frozen FO/worst-FO/Stage-1/runtime order among all-six eligible candidates. `.002` was not reopened.", "", "R2_BOUNDARY_EXPANSION_COMPLETE: YES", "NEW_ROWS: 24/24", f"VALID_CELLS_p0005: {comparison['valid_cells']['p0005']}/3", f"VALID_CELLS_p001: {comparison['valid_cells']['p001']}/1", f"FINAL_p0005: alpha={winners['0.0005']['alpha']:.2f} nu={winners['0.0005']['nu']:.2f} omega={winners['0.0005']['omega']:.4g} stage2_lr={winners['0.0005']['stage2_lr']:.4g} FO={winners['0.0005']['FO_six_transfer_macro']:.6f}", f"FINAL_p001: alpha={winners['0.001']['alpha']:.2f} nu={winners['0.001']['nu']:.2f} omega={winners['0.001']['omega']:.4g} stage2_lr={winners['0.001']['stage2_lr']:.4g} FO={winners['0.001']['FO_six_transfer_macro']:.6f}", f"FINAL_p002: alpha=.10 nu=1.00 omega=.025 stage2_lr=.0025 FO={winners['0.002']['FO_six_transfer_macro']:.6f}", f"SAME_BUDGET_ORACLE_MARGIN_p0005: {winners['0.0005']['delta_vs_same_budget_per_transfer_sparse_oracle']:+.6f}", f"SAME_BUDGET_ORACLE_MARGIN_p001: {winners['0.001']['delta_vs_same_budget_per_transfer_sparse_oracle']:+.6f}", f"SAME_BUDGET_ORACLE_MARGIN_p002: {winners['0.002']['delta_vs_same_budget_per_transfer_sparse_oracle']:+.6f}", "SECOND_BOUNDARY_EXPANSION_ALLOWED: NO", "FILTER_STATUS: BLOCKED_V2", "OFFICE_OUT_TUNING_CLOSED: YES", "NEXT_PHASE: OUT_VISDA_V0"]
    (records / "FINALIZE.md").write_text("\n".join(exp_lines) + "\n", encoding="utf-8")
    return winners, comparison


def main():
    selection = protocol_checks()
    expansion = audit_expansion()
    initial = initial_candidates()
    baselines = baseline_comparison()
    winners, comparison = write_outputs(selection, expansion, initial, baselines)
    print(json.dumps({"new_rows": 24, "valid_cells_p0005": comparison["valid_cells"]["p0005"], "valid_cells_p001": comparison["valid_cells"]["p001"], "final_fo": {rho: winners[rho]["FO_six_transfer_macro"] for rho in ("0.0005", "0.001", "0.002")}, "complete": True}, sort_keys=True))


if __name__ == "__main__":
    main()
