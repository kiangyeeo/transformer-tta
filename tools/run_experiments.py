#!/usr/bin/env python3
"""Serial, status-aware executor for generated experiment plans."""

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.check_experiment_status import (  # noqa: E402
    check_status,
    load_plan,
)
from tools.summarize_runs import (  # noqa: E402
    build_summary_outputs,
    write_summary_outputs,
)


BLOCKING_PRE_RUN_STATUSES = {
    "duplicate_completed",
    "hash_mismatch",
    "invalid_summary",
}
SUMMARY_FIELDS = [
    "experiment_key",
    "experiment_config_sha256",
    "command_args",
    "command_display",
    "pre_run_status",
    "post_run_status",
    "launcher_status",
    "return_code",
    "started_at_utc",
    "completed_at_utc",
    "runtime_seconds",
    "stdout_log_path",
    "stderr_log_path",
    "launch_record_path",
]


class TerminationRequested(Exception):
    """Raised by the SIGTERM handler in the launcher main thread."""

    def __init__(self, signum=signal.SIGTERM):
        super().__init__(f"Termination requested by signal {signum}")
        self.signum = signum


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _safe_key_path(experiment_key):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", experiment_key)
    return sanitized.strip("._") or "experiment"


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def _csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_launcher_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SUMMARY_FIELDS)
    for record in records:
        for field in record:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    field: _csv_value(record.get(field))
                    for field in fieldnames
                }
            )


def _stream_pipe(pipe, terminal, log_path):
    with Path(log_path).open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as log_file:
        for chunk in iter(pipe.readline, ""):
            terminal.write(chunk)
            terminal.flush()
            log_file.write(chunk)
    pipe.close()


def _terminate_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _run_process(command_args, cwd, stdout_log_path, stderr_log_path):
    process = subprocess.Popen(
        command_args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        shell=False,
    )
    stdout_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stdout, sys.stdout, stdout_log_path),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stderr, sys.stderr, stderr_log_path),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        return_code = process.wait()
    except (KeyboardInterrupt, TerminationRequested):
        _terminate_process(process)
        raise
    finally:
        stdout_thread.join(timeout=10)
        stderr_thread.join(timeout=10)
    return return_code


def _base_record(experiment, pre_run_status):
    command_args = experiment.get("command_args")
    if (
        not isinstance(command_args, list)
        or not command_args
        or not all(isinstance(value, str) for value in command_args)
    ):
        raise ValueError(
            f"Invalid command_args for {experiment['experiment_key']}"
        )
    return {
        "experiment_key": experiment["experiment_key"],
        "experiment_config_sha256": experiment[
            "experiment_config_sha256"
        ],
        "command_args": command_args,
        "command_display": shlex.join(command_args),
        "pre_run_status": pre_run_status,
        "post_run_status": pre_run_status,
        "launcher_status": None,
        "return_code": None,
        "started_at_utc": None,
        "completed_at_utc": None,
        "runtime_seconds": None,
        "stdout_log_path": None,
        "stderr_log_path": None,
        "launch_record_path": None,
    }


def _status_by_key(status_result):
    return {
        row["experiment_key"]: row
        for row in status_result["experiments"]
    }


def _matches_filters(experiment, filters):
    for field in (
        "variant",
        "dataset",
        "source",
        "target",
        "seed",
        "requested_budget",
    ):
        accepted = filters.get(field)
        if accepted is not None and experiment.get(field) not in accepted:
            return False
    return True


def _attempt_directory(logs_root, experiment_key):
    experiment_dir = Path(logs_root) / _safe_key_path(experiment_key)
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    attempt_dir = experiment_dir / timestamp
    duplicate_index = 0
    while attempt_dir.exists():
        duplicate_index += 1
        attempt_dir = experiment_dir / (
            f"{timestamp}__dup{duplicate_index}"
        )
    attempt_dir.mkdir(parents=True)
    return attempt_dir


def _post_run_status(plan, runs_root, experiment_key):
    result = check_status(plan, str(runs_root))
    if result["unassociated_invalid_summaries"]:
        return "invalid_summary"
    status_row = _status_by_key(result).get(experiment_key)
    if status_row is None:
        raise RuntimeError(
            f"Experiment missing from status result: {experiment_key}"
        )
    return status_row["plan_status"]


def _execute_one(
    experiment,
    pre_run_status,
    plan,
    runs_root,
    workdir,
    logs_root,
    process_executor,
):
    record = _base_record(experiment, pre_run_status)
    attempt_dir = _attempt_directory(
        logs_root,
        experiment["experiment_key"],
    )
    stdout_log = attempt_dir / "stdout.log"
    stderr_log = attempt_dir / "stderr.log"
    launch_record = attempt_dir / "launch_record.json"
    record.update(
        {
            "started_at_utc": _utc_now(),
            "stdout_log_path": str(stdout_log.resolve()),
            "stderr_log_path": str(stderr_log.resolve()),
            "launch_record_path": str(launch_record.resolve()),
        }
    )
    started = time.perf_counter()
    try:
        return_code = process_executor(
            record["command_args"],
            workdir,
            stdout_log,
            stderr_log,
        )
        record["return_code"] = int(return_code)
        record["post_run_status"] = _post_run_status(
            plan,
            runs_root,
            experiment["experiment_key"],
        )
        if return_code != 0:
            record["launcher_status"] = "failed"
        elif record["post_run_status"] == "completed":
            record["launcher_status"] = "completed"
        elif record["post_run_status"] == "missing":
            record["launcher_status"] = "missing_completed_summary"
        else:
            record["launcher_status"] = record["post_run_status"]
    except KeyboardInterrupt:
        record["launcher_status"] = "interrupted"
        record["post_run_status"] = _post_run_status(
            plan,
            runs_root,
            experiment["experiment_key"],
        )
    except TerminationRequested as error:
        record["launcher_status"] = "interrupted"
        record["termination_signal"] = error.signum
        record["post_run_status"] = _post_run_status(
            plan,
            runs_root,
            experiment["experiment_key"],
        )
    except Exception as error:
        record["launcher_status"] = "failed"
        record["error_type"] = type(error).__name__
        record["error_message"] = str(error)
        record["post_run_status"] = _post_run_status(
            plan,
            runs_root,
            experiment["experiment_key"],
        )
    finally:
        record["completed_at_utc"] = _utc_now()
        record["runtime_seconds"] = (
            time.perf_counter() - started
        )
        stdout_log.touch(exist_ok=True)
        stderr_log.touch(exist_ok=True)
        _write_json(launch_record, record)
    return record


def _write_launcher_summary(
    logs_root,
    records,
    preflight_errors=None,
):
    payload = {
        "launcher_schema_version": 1,
        "created_at_utc": _utc_now(),
        "preflight_errors": preflight_errors or [],
        "records": records,
    }
    logs_root = Path(logs_root)
    logs_root.mkdir(parents=True, exist_ok=True)
    _write_json(logs_root / "launcher_summary.json", payload)
    _write_launcher_csv(
        logs_root / "launcher_summary.csv",
        records,
    )
    return payload


def _dry_run_report(plan, status_result, filters, max_experiments):
    status_map = _status_by_key(status_result)
    completed = []
    missing = []
    selected = []
    filtered_out = []
    for experiment in plan["experiments"]:
        status = status_map[experiment["experiment_key"]][
            "plan_status"
        ]
        if status == "completed":
            completed.append(experiment)
        elif status == "missing":
            missing.append(experiment)
            if _matches_filters(experiment, filters):
                selected.append(experiment)
            else:
                filtered_out.append(experiment)
    deferred_by_max = []
    if max_experiments is not None:
        deferred_by_max = selected[max_experiments:]
        selected = selected[:max_experiments]
    report = {
        "completed_count": len(completed),
        "missing_count": len(missing),
        "will_execute": [
            {
                "experiment_key": experiment["experiment_key"],
                "command_args": experiment["command_args"],
                "command_display": shlex.join(
                    experiment["command_args"]
                ),
            }
            for experiment in selected
        ],
        "will_skip_completed": [
            experiment["experiment_key"] for experiment in completed
        ],
        "filtered_out_missing": [
            experiment["experiment_key"]
            for experiment in filtered_out
        ],
        "deferred_by_max_experiments": [
            experiment["experiment_key"]
            for experiment in deferred_by_max
        ],
    }
    return report


def run_launcher(
    plan,
    runs_root,
    workdir=None,
    logs_root=None,
    filters=None,
    max_experiments=None,
    continue_on_error=False,
    dry_run=False,
    summarize_after_run=False,
    process_executor=None,
):
    if max_experiments is not None and max_experiments <= 0:
        raise ValueError("max_experiments must be greater than 0")
    runs_root = Path(runs_root).resolve()
    workdir = Path(workdir or Path.cwd()).resolve()
    logs_root = Path(
        logs_root or PROJECT_DIR / "launcher_logs"
    ).resolve()
    filters = filters or {}
    process_executor = process_executor or _run_process

    status_result = check_status(plan, str(runs_root))
    status_map = _status_by_key(status_result)
    blocking = [
        row
        for row in status_result["experiments"]
        if row["plan_status"] in BLOCKING_PRE_RUN_STATUSES
    ]
    unassociated_invalid = status_result[
        "unassociated_invalid_summaries"
    ]
    if dry_run:
        report = _dry_run_report(
            plan,
            status_result,
            filters,
            max_experiments,
        )
        report["blocking_statuses"] = [
            {
                "experiment_key": row["experiment_key"],
                "plan_status": row["plan_status"],
            }
            for row in blocking
        ]
        report["unassociated_invalid_summaries"] = (
            unassociated_invalid
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return {
            "blocked": bool(blocking or unassociated_invalid),
            "interrupted": False,
            "had_failures": False,
            "records": [],
            "dry_run": True,
            "report": report,
        }

    records_by_key = {}
    if blocking or unassociated_invalid:
        blocking_keys = {
            row["experiment_key"]: row["plan_status"]
            for row in blocking
        }
        for experiment in plan["experiments"]:
            pre_status = status_map[experiment["experiment_key"]][
                "plan_status"
            ]
            record = _base_record(experiment, pre_status)
            if experiment["experiment_key"] in blocking_keys:
                record["launcher_status"] = (
                    f"blocked_{blocking_keys[experiment['experiment_key']]}"
                )
            elif pre_status == "completed":
                record["launcher_status"] = "skipped_completed"
            else:
                record["launcher_status"] = (
                    "not_started_preflight_error"
                )
            records_by_key[experiment["experiment_key"]] = record
        records = [
            records_by_key[experiment["experiment_key"]]
            for experiment in plan["experiments"]
        ]
        errors = [
            {
                "experiment_key": row["experiment_key"],
                "plan_status": row["plan_status"],
            }
            for row in blocking
        ] + unassociated_invalid
        _write_launcher_summary(logs_root, records, errors)
        return {
            "blocked": True,
            "interrupted": False,
            "had_failures": True,
            "records": records,
            "dry_run": False,
        }

    filtered_missing = []
    for experiment in plan["experiments"]:
        pre_status = status_map[experiment["experiment_key"]][
            "plan_status"
        ]
        if pre_status == "completed":
            record = _base_record(experiment, pre_status)
            record["launcher_status"] = "skipped_completed"
            records_by_key[experiment["experiment_key"]] = record
        elif _matches_filters(experiment, filters):
            filtered_missing.append(experiment)
        else:
            record = _base_record(experiment, pre_status)
            record["launcher_status"] = "skipped_by_filter"
            records_by_key[experiment["experiment_key"]] = record

    selected = filtered_missing
    deferred = []
    if max_experiments is not None:
        selected = filtered_missing[:max_experiments]
        deferred = filtered_missing[max_experiments:]
    for experiment in deferred:
        record = _base_record(experiment, "missing")
        record["launcher_status"] = "deferred_by_max_experiments"
        records_by_key[experiment["experiment_key"]] = record

    stopped = False
    interrupted = False
    had_failures = False
    for index, experiment in enumerate(selected):
        if stopped:
            record = _base_record(experiment, "missing")
            record["launcher_status"] = (
                "not_started_after_interrupt"
                if interrupted
                else "not_started_after_failure"
            )
            records_by_key[experiment["experiment_key"]] = record
            continue
        print(f"[launcher] {experiment['experiment_key']}")
        print(
            f"[command] {shlex.join(experiment['command_args'])}"
        )
        record = _execute_one(
            experiment,
            pre_run_status="missing",
            plan=plan,
            runs_root=runs_root,
            workdir=workdir,
            logs_root=logs_root,
            process_executor=process_executor,
        )
        records_by_key[experiment["experiment_key"]] = record
        if record["launcher_status"] == "interrupted":
            interrupted = True
            had_failures = True
            stopped = True
        elif record["launcher_status"] != "completed":
            had_failures = True
            if not continue_on_error:
                stopped = True

    records = [
        records_by_key[experiment["experiment_key"]]
        for experiment in plan["experiments"]
    ]
    _write_launcher_summary(logs_root, records)

    scope_fully_processed = not any(
        record["launcher_status"]
        in {
            "not_started_after_failure",
            "not_started_after_interrupt",
        }
        for record in records
    )
    if summarize_after_run and scope_fully_processed:
        summary_outputs = build_summary_outputs(
            str(runs_root),
            plan=plan,
        )
        write_summary_outputs(
            summary_outputs,
            logs_root / "summary",
        )

    return {
        "blocked": False,
        "interrupted": interrupted,
        "had_failures": had_failures,
        "records": records,
        "dry_run": False,
    }


def _signal_handler(signum, _frame):
    raise TerminationRequested(signum)


def _filters_from_args(args):
    return {
        "variant": set(args.variant) if args.variant else None,
        "dataset": set(args.dataset) if args.dataset else None,
        "source": set(args.source) if args.source else None,
        "target": set(args.target) if args.target else None,
        "seed": set(args.seed) if args.seed else None,
        "requested_budget": (
            set(args.requested_budget)
            if args.requested_budget
            else None
        ),
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--logs-root", default=None)
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--variant", action="append")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--source", action="append", type=int)
    parser.add_argument("--target", action="append", type=int)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument(
        "--requested-budget",
        action="append",
        type=float,
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summarize-after-run",
        action="store_true",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if (
        args.max_experiments is not None
        and args.max_experiments <= 0
    ):
        parser.error("--max-experiments must be greater than 0")
    plan = load_plan(args.plan)
    previous_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _signal_handler)
    try:
        result = run_launcher(
            plan=plan,
            runs_root=args.runs_root,
            workdir=args.workdir,
            logs_root=args.logs_root,
            filters=_filters_from_args(args),
            max_experiments=args.max_experiments,
            continue_on_error=args.continue_on_error,
            dry_run=args.dry_run,
            summarize_after_run=args.summarize_after_run,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    if result["blocked"] or result["interrupted"]:
        return 2
    if result["had_failures"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
