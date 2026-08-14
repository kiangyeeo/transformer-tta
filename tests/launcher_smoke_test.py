#!/usr/bin/env python3
"""Temporary-directory checks for the serial plan launcher."""

import csv
import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.run_experiments import (  # noqa: E402
    TerminationRequested,
    _matches_filters,
    build_parser,
    run_launcher,
)


def _experiment(
    key,
    variant="source_only",
    dataset="office",
    source=0,
    target=1,
    seed=2020,
    budget=None,
    command_args=None,
):
    sha = (key.encode("utf-8").hex().ljust(64, "0"))[:64]
    return {
        "experiment_key": key,
        "experiment_config_sha256": sha,
        "method": "shot",
        "task": "otta",
        "dataset": dataset,
        "source": source,
        "target": target,
        "seed": seed,
        "variant": variant,
        "requested_budget": budget,
        "selection_seed": None,
        "selection_seed_mode": None,
        "command_args": command_args or ["fake-command", key, sha],
    }


def _plan(*experiments):
    return {"experiments": list(experiments)}


def _write_summary(runs_root, experiment, suffix="", status="completed"):
    run_dir = Path(runs_root) / f"{experiment['experiment_key']}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "experiment_key": experiment["experiment_key"],
        "experiment_config_sha256": experiment[
            "experiment_config_sha256"
        ],
        "method": experiment["method"],
        "task": experiment["task"],
        "dataset": experiment["dataset"],
        "source": experiment["source"],
        "target": experiment["target"],
        "source-target": (
            f"{experiment['source']}-{experiment['target']}"
        ),
        "seed": experiment["seed"],
        "variant": experiment["variant"],
        "requested_budget": experiment["requested_budget"],
        "selection_seed": experiment["selection_seed"],
        "PU-Acc": 0.12345678901234568,
        "FO-Acc": 0.9876543210987654,
        "runtime": 1.2345678901234567,
        "run_id": run_dir.name,
    }
    with (run_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file_obj:
        json.dump(payload, file_obj)


class RecordingExecutor:
    def __init__(self, runs_root, outcomes=None):
        self.runs_root = Path(runs_root)
        self.outcomes = outcomes or {}
        self.calls = []

    def __call__(
        self,
        command_args,
        _cwd,
        stdout_log,
        stderr_log,
    ):
        key = command_args[1]
        sha = command_args[2]
        self.calls.append(key)
        Path(stdout_log).write_text(
            f"stdout {key}\n",
            encoding="utf-8",
        )
        Path(stderr_log).write_text(
            f"stderr {key}\n",
            encoding="utf-8",
        )
        outcome = self.outcomes.get(key, "success")
        if outcome == "keyboard":
            raise KeyboardInterrupt
        if outcome == "sigterm":
            raise TerminationRequested(15)
        if outcome == "no_summary":
            return 0
        if isinstance(outcome, int):
            return outcome
        experiment = {
            "experiment_key": key,
            "experiment_config_sha256": sha,
            "method": "shot",
            "task": "otta",
            "dataset": "office",
            "source": 0,
            "target": 1,
            "seed": 2020,
            "variant": "source_only",
            "requested_budget": None,
            "selection_seed": None,
        }
        _write_summary(self.runs_root, experiment)
        return 0


def _record(result, key):
    return next(
        record
        for record in result["records"]
        if record["experiment_key"] == key
    )


def _check_completed_missing_and_resume(temp_root):
    runs = temp_root / "runs"
    logs = temp_root / "logs"
    completed = _experiment("completed")
    missing = _experiment("missing")
    plan = _plan(completed, missing)
    _write_summary(runs, completed)
    executor = RecordingExecutor(runs)
    first = run_launcher(
        plan,
        runs,
        logs_root=logs,
        process_executor=executor,
    )
    assert executor.calls == ["missing"]
    assert _record(first, "completed")["launcher_status"] == (
        "skipped_completed"
    )
    assert _record(first, "missing")["launcher_status"] == "completed"

    def must_not_run(*_args):
        raise AssertionError("completed experiment was executed again")

    second = run_launcher(
        plan,
        runs,
        logs_root=logs,
        process_executor=must_not_run,
    )
    assert all(
        record["launcher_status"] == "skipped_completed"
        for record in second["records"]
    )


def _check_fake_lbi_plan(temp_root):
    runs = temp_root / "runs"
    logs = temp_root / "logs"
    experiment = _experiment(
        "module-lbi-fake",
        variant="module_lbi",
        budget=0.001,
    )
    executor = RecordingExecutor(runs)
    result = run_launcher(
        _plan(experiment),
        runs,
        logs_root=logs,
        process_executor=executor,
    )
    assert executor.calls == ["module-lbi-fake"]
    assert (
        _record(result, "module-lbi-fake")["launcher_status"]
        == "completed"
    )


def _check_preflight_blocking(temp_root):
    cases = (
        ("duplicate_completed", "duplicate"),
        ("hash_mismatch", "mismatch"),
        ("invalid_summary", "invalid"),
    )
    for expected_status, key in cases:
        case_root = temp_root / expected_status
        runs = case_root / "runs"
        logs = case_root / "logs"
        experiment = _experiment(key)
        if expected_status == "duplicate_completed":
            _write_summary(runs, experiment, "-one")
            _write_summary(runs, experiment, "-two")
        elif expected_status == "hash_mismatch":
            mismatched = dict(experiment)
            mismatched["experiment_config_sha256"] = "f" * 64
            _write_summary(runs, mismatched)
        else:
            run_dir = runs / "invalid"
            run_dir.mkdir(parents=True)
            with (run_dir / "summary.json").open(
                "w",
                encoding="utf-8",
            ) as file_obj:
                json.dump(
                    {
                        "experiment_key": key,
                        "experiment_config_sha256": experiment[
                            "experiment_config_sha256"
                        ],
                    },
                    file_obj,
                )
        executor = RecordingExecutor(runs)
        result = run_launcher(
            _plan(experiment),
            runs,
            logs_root=logs,
            process_executor=executor,
        )
        assert result["blocked"] is True
        assert executor.calls == []
        assert _record(result, key)["launcher_status"] == (
            f"blocked_{expected_status}"
        )


def _check_dry_run_filters_and_max(temp_root):
    experiments = [
        _experiment(
            "first",
            variant="source_only",
            dataset="office",
            source=0,
            target=1,
            seed=1,
        ),
        _experiment(
            "second",
            variant="full_dense",
            dataset="office",
            source=1,
            target=0,
            seed=2,
            budget=1.0,
        ),
        _experiment(
            "third",
            variant="module_random",
            dataset="office-home",
            source=2,
            target=3,
            seed=3,
            budget=0.001,
        ),
    ]
    selected = experiments[2]
    assert _matches_filters(selected, {"variant": {"module_random"}})
    assert _matches_filters(selected, {"dataset": {"office-home"}})
    assert _matches_filters(selected, {"source": {2}})
    assert _matches_filters(selected, {"target": {3}})
    assert _matches_filters(selected, {"seed": {3}})
    assert _matches_filters(selected, {"requested_budget": {0.001}})
    assert not _matches_filters(selected, {"variant": {"full_dense"}})

    called = []

    def no_subprocess(*_args):
        called.append(True)
        return 0

    result = run_launcher(
        _plan(*experiments),
        temp_root / "dry-runs",
        logs_root=temp_root / "dry-logs",
        filters={"variant": {"source_only", "full_dense"}},
        max_experiments=1,
        dry_run=True,
        process_executor=no_subprocess,
    )
    assert result["dry_run"] is True
    assert called == []
    assert not (temp_root / "dry-logs").exists()
    assert [
        item["experiment_key"]
        for item in result["report"]["will_execute"]
    ] == ["first"]

    parser = build_parser()
    parsed = parser.parse_args(
        [
            "plan.json",
            "--runs-root",
            "runs",
            "--variant",
            "source_only",
            "--variant",
            "full_dense",
            "--source",
            "0",
            "--target",
            "1",
            "--seed",
            "2020",
            "--requested-budget",
            "0.001",
        ]
    )
    assert parsed.variant == ["source_only", "full_dense"]
    assert parsed.source == [0]
    assert parsed.target == [1]
    assert parsed.seed == [2020]
    assert parsed.requested_budget == [0.001]
    try:
        run_launcher(
            _plan(*experiments),
            temp_root / "bad-max",
            max_experiments=0,
            dry_run=True,
        )
    except ValueError as error:
        assert "greater than 0" in str(error)
    else:
        raise AssertionError("max_experiments=0 was accepted")


def _check_order_limit_and_failure_semantics(temp_root):
    experiments = [
        _experiment("order-one"),
        _experiment("order-two"),
        _experiment("order-three"),
    ]
    runs = temp_root / "order-runs"
    executor = RecordingExecutor(runs)
    limited = run_launcher(
        _plan(*experiments),
        runs,
        logs_root=temp_root / "order-logs",
        max_experiments=2,
        process_executor=executor,
    )
    assert executor.calls == ["order-one", "order-two"]
    assert _record(
        limited,
        "order-three",
    )["launcher_status"] == "deferred_by_max_experiments"

    fail_runs = temp_root / "fail-runs"
    fail_executor = RecordingExecutor(
        fail_runs,
        outcomes={"fail-first": 7},
    )
    default_stop = run_launcher(
        _plan(_experiment("fail-first"), _experiment("after-fail")),
        fail_runs,
        logs_root=temp_root / "fail-logs",
        process_executor=fail_executor,
    )
    assert fail_executor.calls == ["fail-first"]
    assert _record(default_stop, "fail-first")["launcher_status"] == (
        "failed"
    )
    assert _record(
        default_stop,
        "after-fail",
    )["launcher_status"] == "not_started_after_failure"

    continue_runs = temp_root / "continue-runs"
    continue_executor = RecordingExecutor(
        continue_runs,
        outcomes={"continue-fail": 9},
    )
    continued = run_launcher(
        _plan(
            _experiment("continue-fail"),
            _experiment("continue-success"),
        ),
        continue_runs,
        logs_root=temp_root / "continue-logs",
        continue_on_error=True,
        process_executor=continue_executor,
    )
    assert continue_executor.calls == [
        "continue-fail",
        "continue-success",
    ]
    assert _record(
        continued,
        "continue-success",
    )["launcher_status"] == "completed"

    no_summary_runs = temp_root / "no-summary-runs"
    no_summary = run_launcher(
        _plan(_experiment("no-summary")),
        no_summary_runs,
        logs_root=temp_root / "no-summary-logs",
        process_executor=RecordingExecutor(
            no_summary_runs,
            outcomes={"no-summary": "no_summary"},
        ),
    )
    assert _record(
        no_summary,
        "no-summary",
    )["launcher_status"] == "missing_completed_summary"


def _check_interrupt_and_retry(temp_root):
    for kind in ("keyboard", "sigterm", "failed"):
        runs = temp_root / f"{kind}-retry-runs"
        key = f"{kind}-retry"
        outcome = 5 if kind == "failed" else kind
        first = run_launcher(
            _plan(_experiment(key)),
            runs,
            logs_root=temp_root / f"{kind}-first-logs",
            process_executor=RecordingExecutor(
                runs,
                outcomes={key: outcome},
            ),
        )
        expected = "failed" if kind == "failed" else "interrupted"
        assert _record(first, key)["launcher_status"] == expected
        retry_executor = RecordingExecutor(runs)
        second = run_launcher(
            _plan(_experiment(key)),
            runs,
            logs_root=temp_root / f"{kind}-second-logs",
            process_executor=retry_executor,
        )
        assert retry_executor.calls == [key]
        assert _record(second, key)["launcher_status"] == "completed"
        if kind == "sigterm":
            assert _record(first, key)["termination_signal"] == 15


def _check_real_subprocess_spaces_and_summary(temp_root):
    workdir = temp_root / "working directory with spaces"
    runs = temp_root / "runs directory with spaces"
    logs = temp_root / "logs directory with spaces"
    workdir.mkdir(parents=True)
    script = workdir / "fake worker with spaces.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import pathlib",
                "import sys",
                "runs = pathlib.Path(sys.argv[1])",
                "key, sha, spaced = sys.argv[2:5]",
                "assert spaced == 'argument with spaces'",
                "out = runs / 'fake completed run'",
                "out.mkdir(parents=True)",
                "payload = {",
                "  'status': 'completed',",
                "  'experiment_key': key,",
                "  'experiment_config_sha256': sha,",
                "  'method': 'shot', 'task': 'otta',",
                "  'dataset': 'office', 'source': 0, 'target': 1,",
                "  'source-target': '0-1', 'seed': 2020,",
                "  'variant': 'source_only',",
                "  'requested_budget': None, 'selection_seed': None,",
                "  'PU-Acc': 0.12345678901234568,",
                "  'FO-Acc': 0.9876543210987654,",
                "  'runtime': 1.2345678901234567, 'run_id': 'fake'",
                "}",
                "(out / 'summary.json').write_text(",
                "  json.dumps(payload), encoding='utf-8')",
                "print('stdout with spaces')",
                "print('stderr with spaces', file=sys.stderr)",
            ]
        ),
        encoding="utf-8",
    )
    experiment = _experiment("real-process")
    experiment["command_args"] = [
        sys.executable,
        str(script),
        str(runs),
        experiment["experiment_key"],
        experiment["experiment_config_sha256"],
        "argument with spaces",
    ]
    result = run_launcher(
        _plan(experiment),
        runs,
        workdir=workdir,
        logs_root=logs,
        summarize_after_run=True,
    )
    record = _record(result, "real-process")
    assert record["launcher_status"] == "completed"
    assert record["return_code"] == 0
    assert "stdout with spaces" in Path(
        record["stdout_log_path"]
    ).read_text(encoding="utf-8")
    assert "stderr with spaces" in Path(
        record["stderr_log_path"]
    ).read_text(encoding="utf-8")
    assert (logs / "summary" / "all_runs.json").is_file()

    with (logs / "launcher_summary.json").open(
        "r",
        encoding="utf-8",
    ) as file_obj:
        launcher_json = json.load(file_obj)
    with (logs / "launcher_summary.csv").open(
        "r",
        encoding="utf-8",
    ) as file_obj:
        launcher_csv = next(csv.DictReader(file_obj))
    json_runtime = launcher_json["records"][0]["runtime_seconds"]
    assert launcher_csv["runtime_seconds"] == str(json_runtime)


def main():
    with tempfile.TemporaryDirectory(
        prefix="iclr2027_launcher_smoke_"
    ) as temp_dir:
        root = Path(temp_dir)
        _check_completed_missing_and_resume(root / "resume")
        _check_fake_lbi_plan(root / "lbi")
        _check_preflight_blocking(root / "preflight")
        _check_dry_run_filters_and_max(root / "selection")
        _check_order_limit_and_failure_semantics(root / "semantics")
        _check_interrupt_and_retry(root / "interrupt")
        _check_real_subprocess_spaces_and_summary(root / "real")
    print("iclr2027 launcher smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
