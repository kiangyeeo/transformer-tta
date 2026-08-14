#!/usr/bin/env python3
"""Match planned experiments to structured run summaries."""

import argparse
import csv
import json
import os
import os.path as osp


REQUIRED_SUMMARY_FIELDS = {
    "experiment_key",
    "experiment_config_sha256",
    "status",
}
LEGAL_RUN_STATUSES = {"completed", "failed", "incomplete"}
STATUS_CSV_FIELDS = [
    "experiment_key",
    "experiment_config_sha256",
    "method",
    "task",
    "dataset",
    "source",
    "target",
    "seed",
    "variant",
    "requested_budget",
    "selection_seed",
    "plan_status",
    "matching_completed_count",
    "matching_summary_paths",
    "hash_mismatch_paths",
    "invalid_summary_paths",
]


def load_plan(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        plan = json.load(file_obj)
    experiments = plan.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("plan.json must contain an experiments list")
    return plan


def scan_run_summaries(runs_root):
    records = []
    invalid_records = []
    if not osp.isdir(runs_root):
        return records, invalid_records
    for directory, _, filenames in os.walk(runs_root):
        if "summary.json" not in filenames:
            continue
        path = osp.join(directory, "summary.json")
        try:
            with open(path, "r", encoding="utf-8") as file_obj:
                summary = json.load(file_obj)
        except (OSError, json.JSONDecodeError) as error:
            invalid_records.append(
                {
                    "summary_path": osp.abspath(path),
                    "reason": f"invalid_json: {error}",
                    "experiment_key": None,
                }
            )
            continue
        if not isinstance(summary, dict):
            invalid_records.append(
                {
                    "summary_path": osp.abspath(path),
                    "reason": "summary_not_object",
                    "experiment_key": None,
                }
            )
            continue
        missing = sorted(REQUIRED_SUMMARY_FIELDS - set(summary))
        status = summary.get("status")
        if missing or status not in LEGAL_RUN_STATUSES:
            reason = (
                f"missing_fields: {missing}"
                if missing
                else f"illegal_status: {status}"
            )
            invalid_records.append(
                {
                    "summary_path": osp.abspath(path),
                    "reason": reason,
                    "experiment_key": summary.get("experiment_key"),
                    "experiment_config_sha256": summary.get(
                        "experiment_config_sha256"
                    ),
                }
            )
            continue
        records.append(
            {
                "summary_path": osp.abspath(path),
                "experiment_key": summary["experiment_key"],
                "experiment_config_sha256": summary[
                    "experiment_config_sha256"
                ],
                "status": status,
                "summary": summary,
            }
        )
    return records, invalid_records


def classify_plan(plan, records, invalid_records):
    by_key = {}
    for record in records:
        by_key.setdefault(record["experiment_key"], []).append(record)
    invalid_by_key = {}
    for record in invalid_records:
        key = record.get("experiment_key")
        if key is not None:
            invalid_by_key.setdefault(key, []).append(record)

    statuses = []
    for experiment in plan["experiments"]:
        key = experiment["experiment_key"]
        expected_hash = experiment["experiment_config_sha256"]
        key_records = by_key.get(key, [])
        exact = [
            record
            for record in key_records
            if record["experiment_config_sha256"] == expected_hash
        ]
        mismatched = [
            record
            for record in key_records
            if record["experiment_config_sha256"] != expected_hash
        ]
        completed = [
            record
            for record in exact
            if record["status"] == "completed"
        ]
        invalid = invalid_by_key.get(key, [])

        if invalid:
            status = "invalid_summary"
        elif mismatched:
            status = "hash_mismatch"
        elif len(completed) > 1:
            status = "duplicate_completed"
        elif len(completed) == 1:
            status = "completed"
        else:
            status = "missing"

        statuses.append(
            {
                **{
                    field: experiment.get(field)
                    for field in (
                        "experiment_key",
                        "experiment_config_sha256",
                        "method",
                        "task",
                        "dataset",
                        "source",
                        "target",
                        "seed",
                        "variant",
                        "requested_budget",
                        "selection_seed",
                    )
                },
                "plan_status": status,
                "matching_completed_count": len(completed),
                "matching_summary_paths": [
                    record["summary_path"] for record in completed
                ],
                "hash_mismatch_paths": [
                    record["summary_path"] for record in mismatched
                ],
                "invalid_summary_paths": [
                    record["summary_path"] for record in invalid
                ],
            }
        )
    return statuses


def _write_json(path, payload):
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def _csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path, rows, fieldnames=None):
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    resolved_fieldnames = list(fieldnames or [])
    for row in rows:
        for field in row:
            if field not in resolved_fieldnames:
                resolved_fieldnames.append(field)
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        if not resolved_fieldnames:
            file_obj.write("")
            return
        writer = csv.DictWriter(
            file_obj,
            fieldnames=resolved_fieldnames,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _csv_value(row.get(field))
                    for field in resolved_fieldnames
                }
            )


def check_status(plan, runs_root):
    records, invalid_records = scan_run_summaries(runs_root)
    statuses = classify_plan(plan, records, invalid_records)
    return {
        "status_schema_version": 1,
        "plan_experiment_count": len(plan["experiments"]),
        "status_counts": {
            status: sum(
                row["plan_status"] == status for row in statuses
            )
            for status in (
                "completed",
                "missing",
                "duplicate_completed",
                "hash_mismatch",
                "invalid_summary",
            )
        },
        "experiments": statuses,
        "unassociated_invalid_summaries": [
            record
            for record in invalid_records
            if record.get("experiment_key") is None
        ],
    }


def write_status_outputs(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    statuses = result["experiments"]
    _write_json(
        osp.join(output_dir, "experiment_status.json"),
        result,
    )
    write_csv(
        osp.join(output_dir, "experiment_status.csv"),
        statuses,
        fieldnames=STATUS_CSV_FIELDS,
    )
    write_csv(
        osp.join(output_dir, "missing_runs.csv"),
        [
            row
            for row in statuses
            if row["plan_status"] == "missing"
        ],
        fieldnames=STATUS_CSV_FIELDS,
    )
    write_csv(
        osp.join(output_dir, "duplicate_runs.csv"),
        [
            row
            for row in statuses
            if row["plan_status"] == "duplicate_completed"
        ],
        fieldnames=STATUS_CSV_FIELDS,
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("runs_root")
    parser.add_argument("--output-dir", default="iclr2027/status")
    return parser


def main():
    args = build_parser().parse_args()
    plan = load_plan(args.plan)
    result = check_status(plan, args.runs_root)
    write_status_outputs(result, args.output_dir)
    print(
        json.dumps(
            result["status_counts"],
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
