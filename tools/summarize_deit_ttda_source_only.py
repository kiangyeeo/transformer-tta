#!/usr/bin/env python3
"""Validate and summarize all seven DeiT TTDA source-only runs."""

import argparse
import csv
import json
import os
import os.path as osp
import statistics
import sys


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.check_experiment_status import check_status, load_plan  # noqa: E402


METRIC_FIELDS = (
    "Acc",
    "overall-Acc",
)
ROW_FIELDS = (
    "dataset",
    "transfer",
    "source",
    "target",
    "source_name",
    "target_name",
    *METRIC_FIELDS,
    "processed_sample_count",
    "runtime",
    "peak_gpu_memory_bytes",
    "source_checkpoint_sha256",
    "experiment_key",
    "summary_path",
)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _validate_summary(summary, experiment, path):
    for field in ("dataset", "source", "target", "seed", "variant", "task"):
        if summary.get(field) != experiment.get(field):
            raise ValueError(f"Summary {field} mismatch: {path}")
    if summary.get("adaptation_steps") != 0:
        raise ValueError(f"Summary contains adaptation steps: {path}")
    if summary.get("optimizer_created") is not False:
        raise ValueError(f"Summary created an optimizer: {path}")
    if summary.get("loss_computed") is not False:
        raise ValueError(f"Summary computed an adaptation loss: {path}")
    if summary.get("backward_calls") != 0:
        raise ValueError(f"Summary contains backward calls: {path}")
    if summary.get("target_labels_usage") != "evaluation_only":
        raise ValueError(f"Summary target-label policy mismatch: {path}")
    if summary.get("prediction_passes") != 1:
        raise ValueError(f"Summary did not use one evaluation pass: {path}")
    if summary.get("single_evaluation_pass") is not True:
        raise ValueError(f"Summary evaluation-pass invariant failed: {path}")
    if summary.get("model_state_unchanged") is not True:
        raise ValueError(f"Summary model state changed: {path}")
    if summary.get("model_state_sha256_before") != summary.get(
        "model_state_sha256_after"
    ):
        raise ValueError(f"Summary model hashes differ: {path}")
    if summary.get("source_checkpoint_sha256") != experiment.get(
        "source_checkpoint_sha256"
    ):
        raise ValueError(f"Summary source checkpoint hash mismatch: {path}")
    expected_samples = experiment["scientific_config"]["target_data"][
        "sample_count"
    ]
    if summary.get("processed_sample_count") != expected_samples:
        raise ValueError(f"Summary processed sample count mismatch: {path}")
    for metric in METRIC_FIELDS:
        value = summary.get(metric)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Summary metric {metric} is missing: {path}")


def build_report(plan, runs_root):
    status = check_status(plan, runs_root)
    if status["unassociated_invalid_summaries"]:
        raise ValueError("Runs root contains unassociated invalid summaries")
    incomplete = [
        row
        for row in status["experiments"]
        if row["plan_status"] != "completed"
    ]
    if incomplete:
        details = [
            f"{row['experiment_key']}={row['plan_status']}" for row in incomplete
        ]
        raise ValueError("Plan is not complete: " + ", ".join(details))
    by_key = {
        row["experiment_key"]: row for row in status["experiments"]
    }
    rows = []
    for experiment in plan["experiments"]:
        path = by_key[experiment["experiment_key"]][
            "matching_summary_paths"
        ][0]
        summary = _load_json(path)
        _validate_summary(summary, experiment, path)
        rows.append(
            {
                "dataset": summary["dataset"],
                "transfer": (
                    f"{summary['source_name']}→{summary['target_name']}"
                ),
                "source": summary["source"],
                "target": summary["target"],
                "source_name": summary["source_name"],
                "target_name": summary["target_name"],
                **{field: summary[field] for field in METRIC_FIELDS},
                "processed_sample_count": summary["processed_sample_count"],
                "runtime": summary["runtime"],
                "peak_gpu_memory_bytes": summary["peak_gpu_memory_bytes"],
                "source_checkpoint_sha256": summary[
                    "source_checkpoint_sha256"
                ],
                "experiment_key": summary["experiment_key"],
                "summary_path": path,
            }
        )
    if len(rows) != 7:
        raise RuntimeError(f"Expected seven completed rows, got {len(rows)}")
    office = [row for row in rows if row["dataset"] == "office31"]
    if len(office) != 6:
        raise RuntimeError("Office-31 summary must contain six directions")
    office_average = {
        "dataset": "office31",
        "transfer": "six-direction average",
        "source": None,
        "target": None,
        "source_name": None,
        "target_name": None,
        **{
            field: float(statistics.fmean(row[field] for row in office))
            for field in METRIC_FIELDS
        },
        "processed_sample_count": sum(
            row["processed_sample_count"] for row in office
        ),
        "runtime": float(statistics.fmean(row["runtime"] for row in office)),
        "peak_gpu_memory_bytes": max(
            row["peak_gpu_memory_bytes"] for row in office
        ),
        "source_checkpoint_sha256": None,
        "experiment_key": None,
        "summary_path": None,
    }
    return {
        "schema_version": 2,
        "primary_metric": "macro_class_accuracy",
        "experiment_count": len(rows),
        "rows": rows,
        "office31_six_direction_average": office_average,
        "report_rows": [*office, office_average, *[
            row for row in rows if row["dataset"] == "visda-c"
        ]],
    }


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report_rows):
    lines = [
        "# DeiT-S TTDA source-only results",
        "",
        "Acc is fixed-class macro accuracy. Overall accuracy is reported "
        "alongside it.",
        "",
        "| Dataset | Transfer | Acc | Overall Acc | Samples |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report_rows:
        lines.append(
            f"| {row['dataset']} | {row['transfer']} | "
            f"{row['Acc']:.4f} | {row['overall-Acc']:.4f} | "
            f"{row['processed_sample_count']} |"
        )
    return "\n".join(lines) + "\n"


def write_report(report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(
        osp.join(output_dir, "report.json"), "w", encoding="utf-8"
    ) as file_obj:
        json.dump(report, file_obj, indent=2, ensure_ascii=False)
    _write_csv(osp.join(output_dir, "per_transfer.csv"), report["rows"])
    _write_csv(osp.join(output_dir, "report.csv"), report["report_rows"])
    with open(
        osp.join(output_dir, "report.md"), "w", encoding="utf-8"
    ) as file_obj:
        file_obj.write(_markdown(report["report_rows"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = build_report(load_plan(args.plan), args.runs_root)
    write_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "experiment_count": report["experiment_count"],
                "output_dir": osp.abspath(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
