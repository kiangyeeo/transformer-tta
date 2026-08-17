#!/usr/bin/env python3
"""Validate and summarize all seven DeiT OTTA source-only runs."""

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

from shot_otta.deit_source_only.config import VISDA_CLASS_NAMES  # noqa: E402
from tools.check_experiment_status import check_status, load_plan  # noqa: E402


TRANSFER_FIELDS = [
    "dataset",
    "source",
    "target",
    "transfer",
    "PU-Acc",
    "FO-Acc",
    "PU-overall-Acc",
    "FO-overall-Acc",
    "samples",
    "batches",
    "runtime",
    "peak_gpu_memory_bytes",
]
VISDA_CLASSWISE_FIELDS = [
    "class_id",
    "class_name",
    "PU-Acc",
    "FO-Acc",
    "sample_count",
]


def _load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _require_equal(summary, pu_name, fo_name, path):
    if summary.get(pu_name) != summary.get(fo_name):
        raise RuntimeError(
            f"PU/FO mismatch in {path}: {pu_name} != {fo_name}"
        )


def _validate_summary(experiment, summary, path):
    expected = {
        "status": "completed",
        "experiment_key": experiment["experiment_key"],
        "experiment_config_sha256": experiment["experiment_config_sha256"],
        "method": "no_tta",
        "variant": "source_only",
        "task": "otta",
        "dataset": experiment["dataset"],
        "source": experiment["source"],
        "target": experiment["target"],
        "seed": experiment["seed"],
        "source_checkpoint_sha256": experiment["source_checkpoint_sha256"],
        "adaptation_steps": 0,
        "adaptation_steps_per_batch": 0,
        "optimizer_created": False,
        "loss_computed": False,
        "backward_calls": 0,
        "target_labels_usage": "evaluation_only",
        "model_state_unchanged": True,
        "prediction_passes": 2,
        "stream_prediction_passes": 1,
        "fo_prediction_passes": 1,
        "fo_predictions_reused": False,
        "fo_evaluation_scope": "full_target_dataset",
        "PU-equals-FO": True,
        "PU-predictions-equal-FO": True,
        "drop_last": False,
        "tail_batch_size_one_policy": "kept",
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise RuntimeError(
                f"OTTA summary mismatch for {field} in {path}: "
                f"expected {value!r}, got {summary.get(field)!r}"
            )
    expected_samples = experiment["scientific_config"]["target_stream"][
        "sample_count"
    ]
    if summary.get("processed_sample_count") != expected_samples:
        raise RuntimeError(f"Processed sample count mismatch in {path}")
    if summary.get("model_state_sha256_before") != summary.get(
        "model_state_sha256_after"
    ):
        raise RuntimeError(f"Model state hash changed in {path}")
    if summary.get("model_state_sha256_before") != summary.get(
        "model_state_sha256_after_stream"
    ):
        raise RuntimeError(f"Model state changed during PU stream in {path}")
    if summary.get("model_state_sha256_after_stream") != summary.get(
        "model_state_sha256_after_fo"
    ):
        raise RuntimeError(f"Model state changed during FO evaluation in {path}")
    if summary.get("PU-prediction-sha256") != summary.get(
        "FO-prediction-sha256"
    ):
        raise RuntimeError(f"PU/FO prediction hash mismatch in {path}")
    for suffix in (
        "Acc",
        "mean-class-Acc",
        "overall-Acc",
        "Acc-per-class",
        "class-count",
        "worst-class-Acc",
        "worst-class-id",
        "worst-class-name",
        "class-std",
    ):
        _require_equal(summary, f"PU-{suffix}", f"FO-{suffix}", path)


def _completed_summaries(plan, runs_root):
    status = check_status(plan, runs_root)
    counts = status["status_counts"]
    if counts.get("completed") != len(plan["experiments"]):
        raise RuntimeError(
            "OTTA result set is not complete and unique: "
            f"{json.dumps(counts, sort_keys=True)}"
        )
    if status["unassociated_invalid_summaries"]:
        raise RuntimeError("OTTA runs root contains invalid summaries")
    result = {}
    for row in status["experiments"]:
        paths = row["matching_summary_paths"]
        if len(paths) != 1:
            raise RuntimeError(
                f"Expected one summary for {row['experiment_key']}"
            )
        result[row["experiment_key"]] = (paths[0], _load_json(paths[0]))
    return result


def build_report(plan, runs_root):
    if (
        plan.get("task") != "otta"
        or plan.get("variant") != "source_only"
        or len(plan.get("experiments", [])) != 7
    ):
        raise ValueError("Expected the fixed seven-task DeiT OTTA source-only plan")
    summaries = _completed_summaries(plan, runs_root)
    rows = []
    visda_summary = None
    for experiment in plan["experiments"]:
        path, summary = summaries[experiment["experiment_key"]]
        _validate_summary(experiment, summary, path)
        row = {
            "dataset": experiment["dataset"],
            "source": experiment["source_name"],
            "target": experiment["target_name"],
            "transfer": (
                f"{experiment['source_name']}->{experiment['target_name']}"
            ),
            "PU-Acc": float(summary["PU-Acc"]),
            "FO-Acc": float(summary["FO-Acc"]),
            "PU-overall-Acc": float(summary["PU-overall-Acc"]),
            "FO-overall-Acc": float(summary["FO-overall-Acc"]),
            "samples": int(summary["processed_sample_count"]),
            "batches": int(summary["target_batch_count"]),
            "runtime": float(summary["runtime"]),
            "peak_gpu_memory_bytes": int(summary["peak_gpu_memory_bytes"]),
            "experiment_key": experiment["experiment_key"],
            "source_checkpoint_sha256": experiment[
                "source_checkpoint_sha256"
            ],
            "summary_path": osp.abspath(path),
        }
        rows.append(row)
        if experiment["dataset"] == "visda-c":
            visda_summary = summary

    office_rows = [row for row in rows if row["dataset"] == "office31"]
    if len(office_rows) != 6 or visda_summary is None:
        raise RuntimeError("Expected six Office rows and one VisDA row")
    office_average = {
        "dataset": "office31",
        "source": "-",
        "target": "-",
        "transfer": "six-direction-average",
        **{
            metric: statistics.fmean(row[metric] for row in office_rows)
            for metric in (
                "PU-Acc",
                "FO-Acc",
                "PU-overall-Acc",
                "FO-overall-Acc",
            )
        },
        "samples": sum(row["samples"] for row in office_rows),
        "batches": sum(row["batches"] for row in office_rows),
        "runtime": sum(row["runtime"] for row in office_rows),
        "peak_gpu_memory_bytes": max(
            row["peak_gpu_memory_bytes"] for row in office_rows
        ),
    }
    class_names = visda_summary.get("class-names")
    pu_per_class = visda_summary.get("PU-Acc-per-class")
    fo_per_class = visda_summary.get("FO-Acc-per-class")
    class_counts = visda_summary.get("PU-class-count")
    if (
        class_names != list(VISDA_CLASS_NAMES)
        or not all(
            isinstance(values, list) and len(values) == 12
            for values in (pu_per_class, fo_per_class, class_counts)
        )
    ):
        raise RuntimeError("VisDA-C classwise metrics are incomplete")
    visda_classwise = [
        {
            "class_id": class_id,
            "class_name": class_name,
            "PU-Acc": float(pu_per_class[class_id]),
            "FO-Acc": float(fo_per_class[class_id]),
            "sample_count": int(class_counts[class_id]),
        }
        for class_id, class_name in enumerate(class_names)
    ]
    return {
        "schema_version": 1,
        "task": "otta",
        "variant": "source_only",
        "primary_metrics": ["PU-Acc", "FO-Acc"],
        "experiment_count": 7,
        "rows": rows,
        "office31_six_direction_average": office_average,
        "report_rows": [*rows, office_average],
        "visda_classwise": visda_classwise,
        "PU-equals-FO-for-all-runs": True,
    }


def _write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report_rows, visda_classwise):
    lines = [
        "# DeiT OTTA source-only results",
        "",
        "| Dataset | Transfer | PU-Acc | FO-Acc | PU overall | FO overall | Samples |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report_rows:
        lines.append(
            f"| {row['dataset']} | {row['transfer']} | {row['PU-Acc']} | "
            f"{row['FO-Acc']} | {row['PU-overall-Acc']} | "
            f"{row['FO-overall-Acc']} | {row['samples']} |"
        )
    lines.extend(
        [
            "",
            (
                "PU comes from the ordered online stream; FO comes from an "
                "independent full-target pass with the frozen final model. "
                "Source-only has zero adaptation steps, so they must match."
            ),
            "",
            "## VisDA-C per-class PU/FO accuracy",
            "",
            "| Class | PU-Acc | FO-Acc | Samples |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in visda_classwise:
        lines.append(
            f"| {row['class_name']} | {row['PU-Acc']} | "
            f"{row['FO-Acc']} | {row['sample_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(
        osp.join(output_dir, "report.json"), "w", encoding="utf-8"
    ) as file_obj:
        json.dump(report, file_obj, indent=2, ensure_ascii=False)
    _write_csv(
        osp.join(output_dir, "per_transfer.csv"),
        report["rows"],
        TRANSFER_FIELDS,
    )
    _write_csv(
        osp.join(output_dir, "report.csv"),
        report["report_rows"],
        TRANSFER_FIELDS,
    )
    _write_csv(
        osp.join(output_dir, "visda_classwise.csv"),
        report["visda_classwise"],
        VISDA_CLASSWISE_FIELDS,
    )
    wide_row = {}
    for row in report["visda_classwise"]:
        wide_row[f"PU-{row['class_name']}"] = row["PU-Acc"]
        wide_row[f"FO-{row['class_name']}"] = row["FO-Acc"]
    visda_row = next(
        row for row in report["rows"] if row["dataset"] == "visda-c"
    )
    wide_row.update(
        {
            "PU-mean-class-Acc": visda_row["PU-Acc"],
            "FO-mean-class-Acc": visda_row["FO-Acc"],
            "PU-overall-Acc": visda_row["PU-overall-Acc"],
            "FO-overall-Acc": visda_row["FO-overall-Acc"],
        }
    )
    _write_csv(
        osp.join(output_dir, "visda_classwise_wide.csv"),
        [wide_row],
        list(wide_row),
    )
    with open(
        osp.join(output_dir, "report.md"), "w", encoding="utf-8"
    ) as file_obj:
        file_obj.write(_markdown(report["report_rows"], report["visda_classwise"]))


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
