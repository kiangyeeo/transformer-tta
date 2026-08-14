#!/usr/bin/env python3
"""Rank VisDA LBI candidates strictly by FO evidence and validity."""

import argparse
import csv
import json
from pathlib import Path


def _number(value):
    return None if value in (None, "") else float(value)


def _load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _scan_summaries(runs_root):
    result = {}
    for path in Path(runs_root).rglob("summary.json"):
        summary = _load_json(path)
        key = summary.get("experiment_key")
        if key:
            if key in result:
                raise ValueError(f"duplicate experiment key: {key}")
            result[key] = (path, summary)
    return result


def _valid(summary):
    return bool(
        summary
        and summary.get("status") == "completed"
        and summary.get("valid_lbi_run") is True
        and summary.get("budget_diagnostics_available") is True
    )


def _sort_number(value, descending=False):
    if value is None:
        return float("inf")
    return -value if descending else value


def analyze(plan_path, runs_root, output_dir):
    plan = _load_json(plan_path)
    summaries = _scan_summaries(runs_root)
    rows = []
    for experiment in plan["experiments"]:
        if experiment.get("dataset") != "VISDA-C" or experiment.get("variant") != "module_lbi":
            continue
        path, summary = summaries.get(experiment["experiment_key"], (None, None))
        row = {
            "experiment_key": experiment["experiment_key"],
            "candidate_tuple": "|".join(f"{name}={experiment.get(name)}" for name in ("requested_budget", "alpha", "kappa", "nu", "omega", "stage1_max_steps", "budget_tolerance", "stage2_lr", "stage2_steps_requested", "delta_nonzero_tolerance")),
            "summary_path": str(path) if path else "",
            "validity": _valid(summary),
            "FO-Acc": _number(summary.get("FO-Acc")) if summary else None,
            "FO-worst-class-Acc": _number(summary.get("FO-worst-class-Acc")) if summary else None,
            "FO-class-std": _number(summary.get("FO-class-std")) if summary else None,
            "FO-overall-Acc": _number(summary.get("FO-overall-Acc")) if summary else None,
            "stage1_steps_max": _number(summary.get("stage1_steps_completed_max")) if summary else None,
            "stage1_steps_mean": _number(summary.get("stage1_steps_completed_mean")) if summary else None,
            "runtime": _number(summary.get("runtime")) if summary else None,
            "PU-Acc": _number(summary.get("PU-Acc")) if summary else None,
            "PU-overall-Acc": _number(summary.get("PU-overall-Acc")) if summary else None,
            "PU-worst-class-Acc": _number(summary.get("PU-worst-class-Acc")) if summary else None,
            "PU-class-std": _number(summary.get("PU-class-std")) if summary else None,
        }
        rows.append(row)
    rows.sort(key=lambda row: (
        not row["validity"],
        _sort_number(row["FO-Acc"], True),
        _sort_number(row["FO-worst-class-Acc"], True),
        _sort_number(row["FO-class-std"]),
        _sort_number(row["FO-overall-Acc"], True),
        _sort_number(row["stage1_steps_max"]),
        _sort_number(row["stage1_steps_mean"]),
        _sort_number(row["runtime"]),
        row["candidate_tuple"],
    ))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["rank", "experiment_key"]
    with (output / "candidate_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "candidate_ranking.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    analyze(args.plan, args.runs_root, args.output_dir)


if __name__ == "__main__":
    main()
