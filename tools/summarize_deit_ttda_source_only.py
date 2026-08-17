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
from shot_otta.deit_source_only.config import VISDA_CLASS_NAMES  # noqa: E402


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
VISDA_CLASSWISE_FIELDS = ("class_id", "class_name", "Acc", "sample_count")


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


def _visda_classwise_rows(summary, path):
    class_names = summary.get("class-names")
    per_class = summary.get("Acc-per-class")
    class_counts = summary.get("class-count")
    if class_names != list(VISDA_CLASS_NAMES):
        raise ValueError(f"VisDA class names/order mismatch: {path}")
    if not isinstance(per_class, list) or len(per_class) != 12:
        raise ValueError(f"VisDA per-class accuracy must have 12 values: {path}")
    if not isinstance(class_counts, list) or len(class_counts) != 12:
        raise ValueError(f"VisDA class counts must have 12 values: {path}")
    if not all(isinstance(value, (int, float)) for value in per_class):
        raise ValueError(f"VisDA per-class accuracy is not numeric: {path}")
    if not all(isinstance(value, int) and value > 0 for value in class_counts):
        raise ValueError(f"VisDA class counts are invalid: {path}")
    macro = float(statistics.fmean(per_class))
    if abs(macro - float(summary["Acc"])) > 1.0e-10:
        raise ValueError(f"VisDA macro Acc does not equal class mean: {path}")
    if sum(class_counts) != summary["processed_sample_count"]:
        raise ValueError(f"VisDA class counts do not cover all samples: {path}")
    return [
        {
            "class_id": class_id,
            "class_name": class_name,
            "Acc": float(per_class[class_id]),
            "sample_count": int(class_counts[class_id]),
        }
        for class_id, class_name in enumerate(class_names)
    ]


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
    visda_classwise = None
    visda_metrics = None
    for experiment in plan["experiments"]:
        path = by_key[experiment["experiment_key"]][
            "matching_summary_paths"
        ][0]
        summary = _load_json(path)
        _validate_summary(summary, experiment, path)
        if summary["dataset"] == "visda-c":
            if visda_classwise is not None:
                raise ValueError("More than one VisDA-C summary was found")
            visda_classwise = _visda_classwise_rows(summary, path)
            visda_metrics = {
                "Acc": float(summary["Acc"]),
                "overall-Acc": float(summary["overall-Acc"]),
            }
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
    if visda_classwise is None or visda_metrics is None:
        raise RuntimeError("VisDA-C classwise summary is missing")
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
        "schema_version": 3,
        "primary_metric": "macro_class_accuracy",
        "experiment_count": len(rows),
        "rows": rows,
        "office31_six_direction_average": office_average,
        "visda_classwise": visda_classwise,
        "visda_metrics": visda_metrics,
        "report_rows": [*office, office_average, *[
            row for row in rows if row["dataset"] == "visda-c"
        ]],
    }


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report_rows, visda_classwise):
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
    lines.extend(
        [
            "",
            "## VisDA-C per-class accuracy",
            "",
            "| ID | Class | Acc | Samples |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in visda_classwise:
        lines.append(
            f"| {row['class_id']} | {row['class_name']} | "
            f"{row['Acc']:.4f} | {row['sample_count']} |"
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
        osp.join(output_dir, "visda_classwise.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=VISDA_CLASSWISE_FIELDS)
        writer.writeheader()
        writer.writerows(report["visda_classwise"])
    wide_row = {
        row["class_name"]: row["Acc"] for row in report["visda_classwise"]
    }
    wide_row.update(
        {
            "mean_class_Acc": report["visda_metrics"]["Acc"],
            "overall_Acc": report["visda_metrics"]["overall-Acc"],
        }
    )
    with open(
        osp.join(output_dir, "visda_classwise_wide.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as file_obj:
        fieldnames = [*VISDA_CLASS_NAMES, "mean_class_Acc", "overall_Acc"]
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(wide_row)
    with open(
        osp.join(output_dir, "report.md"), "w", encoding="utf-8"
    ) as file_obj:
        file_obj.write(
            _markdown(report["report_rows"], report["visda_classwise"])
        )


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
