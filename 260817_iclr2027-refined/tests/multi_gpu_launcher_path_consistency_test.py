#!/usr/bin/env python3
"""Regression coverage for launcher plan/output-root path consistency."""

import json
import sys
import tempfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from tools.run_experiments_multi_gpu import (
    execute,
    validate_expected_output_roots,
    validate_visda_batch_size,
)


RUNS_ROOT = (
    PROJECT
    / "experiment_logs"
    / "shot_otta_office_seed2026_stage1_20260818"
    / "runs"
).resolve()
PLAN_PATHS = (
    PROJECT / "experiment_logs/shot_otta_office_seed2026_stage1_20260818/plans/baseline_machine1/plan.json",
    PROJECT / "experiment_logs/shot_otta_office_seed2026_stage1_20260818/plans/baseline_machine2/plan.json",
    PROJECT / "experiment_logs/shot_otta_office_seed2026_stage1_20260818/plans/machine3/baseline/plan.json",
    PROJECT / "experiment_logs/shot_otta_office_seed2026_stage1_20260818/plans/lbi_stage1_machine1_budget_0005/plan.json",
    PROJECT / "experiment_logs/shot_otta_office_seed2026_stage1_20260818/plans/lbi_stage1_machine2/plan.json",
    PROJECT / "experiment_logs/shot_otta_office_seed2026_stage1_20260818/plans/machine3/stage1/plan.json",
)


def test_existing_plans_are_under_runs_root():
    for plan_path in PLAN_PATHS:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        validate_expected_output_roots(plan, RUNS_ROOT)


def test_mismatch_fails_before_execution():
    experiment = {
        "experiment_key": "path_guard_fixture",
        "expected_output_root": str(RUNS_ROOT.parent / "outside"),
        "command_args": ["fake", "--gpu-id", "0"],
    }
    plan = {"experiments": [experiment]}
    called = False

    def process_executor(*_args):
        nonlocal called
        called = True
        raise AssertionError("child process executor must not be called")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        try:
            execute(
                plan,
                root / "runs",
                root / "logs",
                ["0"],
                root,
                dry_run=True,
                process_executor=process_executor,
            )
        except RuntimeError as error:
            assert "outside --runs-root" in str(error)
        else:
            raise AssertionError("mismatched expected_output_root was accepted")
    assert not called


def test_visda_shard_plan_uses_full_stream_batch_size():
    validate_visda_batch_size(
        {
            "experiments": [
                {
                    "experiment_key": "visda_shard_fixture",
                    "dataset": "VISDA-C",
                    "full_stream": {"batch_size": 256},
                }
            ]
        }
    )


def test_visda_conflicting_batch_size_metadata_fails():
    plan = {
        "experiments": [
            {
                "experiment_key": "visda_conflict_fixture",
                "dataset": "VISDA-C",
                "scientific_config": {"data": {"batch_size": 256}},
                "full_stream": {"batch_size": 64},
            }
        ]
    }
    try:
        validate_visda_batch_size(plan)
    except RuntimeError as error:
        assert "conflicting" in str(error)
    else:
        raise AssertionError("conflicting VISDA-C batch metadata was accepted")


if __name__ == "__main__":
    test_existing_plans_are_under_runs_root()
    test_mismatch_fails_before_execution()
    test_visda_shard_plan_uses_full_stream_batch_size()
    test_visda_conflicting_batch_size_metadata_fails()
    print("multi-GPU launcher path consistency regression passed")
