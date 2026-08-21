#!/usr/bin/env python3
"""Aggregate and finalize the VisDA-C Stage-1 budget-0.001 targeted retune."""

import argparse
import csv
import json
from pathlib import Path


SEGMENTS = (("early", 1, 50), ("middle-1", 51, 100),
            ("middle-2", 101, 150), ("late", 151, None))


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summaries_by_key(runs_root):
    found = {}
    for path in Path(runs_root).rglob("summary.json"):
        summary = read_json(path)
        key = summary.get("experiment_key")
        if key:
            if key in found:
                raise ValueError(f"duplicate summary for {key}")
            found[key] = (path, summary)
    return found


def segment_rates(metrics_path):
    events = [json.loads(line) for line in Path(metrics_path).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    events = [event for event in events if event.get("event") == "online_step"]
    rows = []
    for name, first, last in SEGMENTS:
        subset = [event for event in events if event.get("iteration", 0) >= first
                  and (last is None or event.get("iteration", 0) <= last)]
        hits = sum(event.get("budget_reached") is True for event in subset)
        max_steps = sum(event.get("max_steps_hit") is True for event in subset)
        rows.append({"segment": name, "first_batch": first,
                     "last_batch": last or len(events), "online_steps": len(subset),
                     "budget_hit_count": hits,
                     "budget_hit_rate": hits / len(subset) if subset else None,
                     "max_steps_hit_count": max_steps})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-config", required=True)
    parser.add_argument("--next-action", required=True)
    args = parser.parse_args()

    plan = read_json(args.plan)
    summaries = summaries_by_key(args.runs_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, segment_rows = [], []
    for candidate_index, exp in enumerate(plan["experiments"], 1):
        summary_path, summary = summaries[exp["experiment_key"]]
        metrics_path = summary_path.with_name("metrics.jsonl")
        fully_valid = bool(summary.get("status") == "completed" and
                           summary.get("budget_reached_all_steps") is True and
                           summary.get("max_steps_hit_count", 0) == 0 and
                           summary.get("valid_lbi_run") is True)
        segment_data = segment_rates(metrics_path)
        for row in segment_data:
            segment_rows.append({"candidate": f"R{candidate_index}",
                                 "experiment_key": exp["experiment_key"],
                                 "alpha": exp["alpha"], "kappa": exp["kappa"],
                                 "nu": exp["nu"], **row})
        by_segment = {row["segment"]: row["budget_hit_rate"] for row in segment_data}
        pu = {key: summary.get(key) for key in ("PU-Acc", "PU-overall-Acc",
              "PU-mean-class-Acc", "PU-worst-class-Acc", "PU-class-std",
              "PU-Acc-per-class")}
        rows.append({
            "candidate": f"R{candidate_index}",
            "experiment_key": exp["experiment_key"],
            "experiment_config_sha256": exp["experiment_config_sha256"],
            "alpha": exp["alpha"], "kappa": exp["kappa"], "nu": exp["nu"],
            "omega": exp["omega"], "valid_lbi_run": summary.get("valid_lbi_run"),
            "fully_valid": fully_valid,
            "budget_reached_all_steps": summary.get("budget_reached_all_steps"),
            "budget_hit_rate": summary.get("budget_hit_rate"),
            "max_steps_hit_count": summary.get("max_steps_hit_count"),
            "stage1_steps_completed_mean": summary.get("stage1_steps_completed_mean"),
            "stage1_steps_completed_max": summary.get("stage1_steps_completed_max"),
            "support_min": summary.get("stage1_support_count_min"),
            "support_max": summary.get("stage1_support_count_max"),
            "FO_overall": summary.get("FO-overall-Acc"),
            "FO_mean_per_class": summary.get("FO-mean-class-Acc"),
            "FO_worst_class": summary.get("FO-worst-class-Acc"),
            "FO_classwise_std": summary.get("FO-class-std"),
            "PU_metrics": pu, "runtime_seconds": summary.get("runtime"),
            "early_budget_hit_rate": by_segment["early"],
            "middle_1_budget_hit_rate": by_segment["middle-1"],
            "middle_2_budget_hit_rate": by_segment["middle-2"],
            "late_budget_hit_rate": by_segment["late"],
            "summary_path": str(summary_path), "metrics_path": str(metrics_path),
        })

    csv_rows = [{**row, "PU_metrics": json.dumps(row["PU_metrics"])} for row in rows]
    write_csv(output / "candidate_diagnostics.csv", csv_rows)
    dump_json(output / "candidate_diagnostics.json", rows)
    write_csv(output / "segmented_budget_hit_rate.csv", segment_rows)
    dump_json(output / "segmented_budget_hit_rate.json", segment_rows)
    candidate_md = ["# Candidate diagnostics", "",
        "| Candidate | a/k/n/o | Valid | All-hit | Hit rate | Max hits | Steps mean/max | Support min/max | FO overall/mean/worst/std | PU overall/mean/worst/std | Runtime (h) |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: |"]
    for row in rows:
        pu = row["PU_metrics"]
        candidate_md.append(
            "| {candidate} | {alpha:.2f}/{kappa:.2f}/{nu:.2f}/{omega:.2f} | "
            "{valid_lbi_run} | {budget_reached_all_steps} | {budget_hit_rate:.4f} | "
            "{max_steps_hit_count} | {stage1_steps_completed_mean:.2f}/{stage1_steps_completed_max} | "
            "{support_min}/{support_max} | {FO_overall:.4f}/{FO_mean_per_class:.4f}/"
            "{FO_worst_class:.4f}/{FO_classwise_std:.4f} | "
            "{pu_overall:.4f}/{pu_mean:.4f}/{pu_worst:.4f}/{pu_std:.4f} | {hours:.2f} |".format(
                **row, pu_overall=pu["PU-overall-Acc"], pu_mean=pu["PU-mean-class-Acc"],
                pu_worst=pu["PU-worst-class-Acc"], pu_std=pu["PU-class-std"],
                hours=row["runtime_seconds"] / 3600))
    (output / "candidate_diagnostics.md").write_text(
        "\n".join(candidate_md) + "\n", encoding="utf-8")
    segment_md = ["# Segmented budget-hit rate", "",
        "| Candidate | Segment | Batches | Hit count / steps | Hit rate | Max-step hits |",
        "| --- | --- | --- | --- | ---: | ---: |"]
    for row in segment_rows:
        segment_md.append(
            "| {candidate} | {segment} | {first_batch}-{last_batch} | "
            "{budget_hit_count}/{online_steps} | {budget_hit_rate:.4f} | "
            "{max_steps_hit_count} |".format(**row))
    (output / "segmented_budget_hit_rate.md").write_text(
        "\n".join(segment_md) + "\n", encoding="utf-8")

    def diagnostic_key(row):
        null_low = lambda value: value if value is not None else float("-inf")
        null_high = lambda value: value if value is not None else float("inf")
        return (-null_low(row["budget_hit_rate"]), -null_low(row["late_budget_hit_rate"]),
                -null_low(row["support_min"]), null_high(row["max_steps_hit_count"]),
                -null_low(row["FO_mean_per_class"]), -null_low(row["FO_worst_class"]),
                null_high(row["FO_classwise_std"]), row["candidate"])

    ranking = sorted(rows, key=diagnostic_key)
    for rank, row in enumerate(ranking, 1):
        row["diagnostic_rank"] = rank
    ranking_csv = [{**row, "PU_metrics": json.dumps(row["PU_metrics"])} for row in ranking]
    write_csv(output / "diagnostic_ranking.csv", ranking_csv)
    dump_json(output / "diagnostic_ranking.json", ranking)

    valid = [row for row in rows if row["fully_valid"]]
    def selection_key(row):
        low = lambda value: value if value is not None else float("-inf")
        high = lambda value: value if value is not None else float("inf")
        return (-low(row["FO_mean_per_class"]), -low(row["FO_worst_class"]),
                high(row["FO_classwise_std"]), -low(row["FO_overall"]),
                high(row["stage1_steps_completed_max"]),
                high(row["stage1_steps_completed_mean"]), high(row["runtime_seconds"]),
                row["candidate"])
    selected = sorted(valid, key=selection_key)[:2]
    if len(valid) >= 2:
        dump_json(args.selected_config, {"budget": 0.001, "provenance":
            "stage1_budget_001_targeted_retune", "selection_rule":
            "validity-first; FO mean/worst/std/overall; Stage-1 cost; runtime",
            "selected": selected})
    else:
        failure = "Stage-1 reachability"
        late_values = [row["late_budget_hit_rate"] for row in rows]
        overall_values = [row["budget_hit_rate"] for row in rows]
        if max(overall_values) >= 0.9 and min(late_values) < 0.5:
            failure = "late-stream accumulation"
        Path(args.next_action).write_text(
            "# Next action required\n\n"
            f"No formal selection was made: only {len(valid)}/8 candidates are fully valid.\n\n"
            f"Primary remaining failure: **{failure}**. All candidates had at least one "
            "Stage-1 max-step failure, so no Stage-1 budget-0.001 configuration "
            "can advance to phases 05/06. `omega` remains fixed at 0.20 and must "
            "not be changed automatically.\n\n"
            "Diagnostic ranking is report-only; review it before authorizing a new phase.\n",
            encoding="utf-8")

    md = ["# VisDA-C Stage-1 budget 0.001 targeted retune", "",
          f"Completed planned identities: {len(rows)}/8. Fully valid: {len(valid)}/8.", "",
          "## Diagnostic ranking (report-only)", "",
          "| Rank | Candidate | Hit rate | Late hit rate | Max-step hits | FO mean | FO worst | FO std | Runtime (h) |",
          "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in ranking:
        md.append("| {diagnostic_rank} | {candidate} | {budget_hit_rate:.4f} | "
                  "{late_budget_hit_rate:.4f} | {max_steps_hit_count} | "
                  "{FO_mean_per_class:.4f} | {FO_worst_class:.4f} | "
                  "{FO_classwise_std:.4f} | {hours:.2f} |".format(
                      **row, hours=row["runtime_seconds"] / 3600))
    md.extend(["", "## Budget-hit rate by target-stream segment", "",
               "| Candidate | Early 1–50 | Middle-1 51–100 | Middle-2 101–150 | Late 151–217 |",
               "| --- | ---: | ---: | ---: | ---: |"])
    for row in rows:
        md.append("| {candidate} | {early_budget_hit_rate:.4f} | {middle_1_budget_hit_rate:.4f} | "
                  "{middle_2_budget_hit_rate:.4f} | {late_budget_hit_rate:.4f} |".format(**row))
    md.extend(["", "Formal validity requires budget reached at every online step and no "
               "Stage-1 max-step failure. PU metrics are report-only."])
    (output / "FINALIZE_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
