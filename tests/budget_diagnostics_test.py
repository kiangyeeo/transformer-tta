#!/usr/bin/env python3
"""Unit tests for Split-LBI budget diagnostics and legacy backfill."""

import csv
import json
import math
import os
import os.path as osp
import sys
import tempfile


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from core.lbi import (  # noqa: E402
    compute_lbi_run_budget_diagnostics,
    compute_lbi_step_budget_diagnostics,
    target_support_count,
)
from tools.summarize_runs import (  # noqa: E402
    aggregate_across_seeds,
    aggregate_dataset_macro,
    build_summary_outputs,
    write_summary_outputs,
)


def _step(
    support,
    reason="rollback_feasible",
    requested_budget=0.1,
    candidate_count=100,
    stage1_steps=12,
):
    return {
        "event": "online_step",
        "requested_budget": requested_budget,
        "candidate_scope_param_count": candidate_count,
        "stage1_support_count": support,
        "stage1_support_ratio": support / candidate_count,
        "stage1_steps_completed": stage1_steps,
        "stage1_stop_reason": reason,
        "stage1_rollback_used": reason == "rollback_feasible",
    }


def _check_run_diagnostics():
    all_hit = compute_lbi_run_budget_diagnostics(
        [_step(10) for _ in range(8)]
    )
    assert all_hit["budget_hit_count"] == 8
    assert all_hit["budget_hit_rate"] == 1.0
    assert all_hit["budget_reached_all_steps"] is True
    assert all_hit["max_steps_hit_count"] == 0
    assert all_hit["valid_lbi_run"] is True
    assert all_hit["stage1_stop_reason_counts"][
        "rollback_feasible"
    ] == 8
    assert all_hit["stage1_stop_reason_counts"]["max_steps"] == 0

    partial = compute_lbi_run_budget_diagnostics(
        [_step(10) for _ in range(6)]
        + [_step(9) for _ in range(2)]
    )
    assert partial["budget_hit_count"] == 6
    assert partial["budget_hit_rate"] == 0.75
    assert partial["underfilled_step_count"] == 2
    assert partial["budget_reached_all_steps"] is False
    assert partial["valid_lbi_run"] is False

    max_steps = compute_lbi_run_budget_diagnostics(
        [_step(10), _step(10, reason="max_steps")]
    )
    assert max_steps["budget_hit_count"] == 2
    assert max_steps["max_steps_hit_count"] == 1
    assert max_steps["valid_lbi_step_count"] == 1
    assert max_steps["valid_lbi_run"] is False

    legacy_max_steps = compute_lbi_run_budget_diagnostics(
        [_step(10, reason="max_steps_reached")]
    )
    assert legacy_max_steps["max_steps_hit_count"] == 1
    assert legacy_max_steps["stage1_stop_reason_counts"][
        "max_steps"
    ] == 1

    rollback_hit = compute_lbi_step_budget_diagnostics(
        0.1, 100, 10, "rollback_feasible"
    )
    rollback_underfilled = compute_lbi_step_budget_diagnostics(
        0.1, 100, 9, "rollback_feasible"
    )
    assert rollback_hit["budget_reached"] is True
    assert rollback_hit["valid_lbi_step"] is True
    assert rollback_underfilled["budget_reached"] is False
    assert rollback_underfilled["valid_lbi_step"] is False

    assert target_support_count(0.002, 524544) == 1050


def _summary_payload(key, seed, diagnostics=None):
    payload = {
        "schema_version": 8,
        "status": "completed",
        "experiment_key": key,
        "experiment_config_sha256": (key * 64)[:64],
        "method": "shot",
        "task": "otta",
        "dataset": "office",
        "source": 0,
        "target": 1,
        "source-target": "amazon-dslr",
        "seed": seed,
        "variant": "module_lbi",
        "selection": "lbi",
        "requested_budget": 0.1,
        "candidate_scope_param_count": 100,
        "alpha": 0.1,
        "kappa": 1.0,
        "nu": 1.0,
        "omega": 0.1,
        "stage1_max_steps": 100,
        "budget_tolerance": 0.0001,
        "stage2_lr": 0.01,
        "stage2_steps_requested": 1,
        "delta_nonzero_tolerance": 1.0e-12,
        "online_steps": 8,
        "PU-Acc": 12.345678901234567,
        "FO-Acc": 23.456789012345677,
        "runtime": 34.56789012345679,
        "run_id": key,
    }
    if diagnostics is not None:
        payload.update(diagnostics)
    return payload


def _write_legacy_run(root, dirname, summary, steps=None):
    run_dir = osp.join(root, dirname)
    os.makedirs(run_dir)
    with open(
        osp.join(run_dir, "summary.json"), "w", encoding="utf-8"
    ) as file_obj:
        json.dump(summary, file_obj)
    if steps is not None:
        with open(
            osp.join(run_dir, "metrics.jsonl"),
            "w",
            encoding="utf-8",
        ) as file_obj:
            for step in steps:
                file_obj.write(json.dumps(step) + "\n")
            file_obj.write(
                json.dumps({"event": "final", "status": "completed"})
                + "\n"
            )
    return run_dir


def _check_legacy_backfill_and_missing():
    with tempfile.TemporaryDirectory(
        prefix="iclr2027_budget_backfill_"
    ) as temp_dir:
        runs_root = osp.join(temp_dir, "runs")
        os.makedirs(runs_root)
        _write_legacy_run(
            runs_root,
            "legacy-with-metrics",
            _summary_payload("a", 1),
            [_step(10) for _ in range(8)],
        )
        _write_legacy_run(
            runs_root,
            "legacy-without-metrics",
            _summary_payload("b", 2),
            None,
        )
        outputs = build_summary_outputs(runs_root)
        by_key = {
            row["experiment_key"]: row
            for row in outputs["all_runs"]
        }
        backfilled = by_key["a"]
        unavailable = by_key["b"]
        assert backfilled["budget_diagnostics_available"] is True
        assert (
            backfilled["budget_diagnostics_source"]
            == "metrics_jsonl"
        )
        assert backfilled["budget_reached_all_steps"] is True
        assert backfilled["target_support_count"] == 10
        assert unavailable["budget_diagnostics_available"] is False
        assert unavailable["budget_reached_all_steps"] is None
        assert unavailable[
            "budget_diagnostics_unavailable_reason"
        ].startswith("metrics_jsonl_not_found")

        output_dir = osp.join(temp_dir, "tables")
        write_summary_outputs(outputs, output_dir)
        for extension in ("json", "csv", "md"):
            assert osp.isfile(
                osp.join(
                    output_dir,
                    f"lbi_budget_diagnostics.{extension}",
                )
            )
        with open(
            osp.join(output_dir, "all_runs.csv"),
            "r",
            encoding="utf-8",
        ) as file_obj:
            csv_text = file_obj.read()
        with open(
            osp.join(output_dir, "all_runs.md"),
            "r",
            encoding="utf-8",
        ) as file_obj:
            markdown_text = file_obj.read()
        assert str(backfilled["FO-Acc"]) in csv_text
        assert str(backfilled["FO-Acc"]) in markdown_text
        with open(
            osp.join(output_dir, "lbi_budget_diagnostics.csv"),
            "r",
            encoding="utf-8",
        ) as file_obj:
            diagnostic_rows = list(csv.DictReader(file_obj))
        assert len(diagnostic_rows) == 2
        assert "PU-Acc-per-class" not in diagnostic_rows[0]


def _aggregate_row(seed, diagnostics):
    row = _summary_payload(f"seed{seed}", seed, diagnostics)
    row["budget_diagnostics_available"] = True
    return row


def _check_aggregation():
    valid = compute_lbi_run_budget_diagnostics(
        [_step(10) for _ in range(8)]
    )
    invalid = compute_lbi_run_budget_diagnostics(
        [_step(10) for _ in range(6)]
        + [_step(9) for _ in range(2)]
    )
    rows = [_aggregate_row(1, valid), _aggregate_row(2, invalid)]
    transfer = aggregate_across_seeds(rows)
    assert len(transfer) == 1
    aggregate = transfer[0]
    assert aggregate["budget_diagnostics_run_count"] == 2
    assert aggregate["budget_valid_run_count"] == 1
    assert aggregate["budget_valid_run_ratio"] == 0.5
    assert aggregate["budget_reached_all_runs"] is False
    assert aggregate["budget_hit_rate_mean"] == 0.875
    assert aggregate["budget_hit_rate_min"] == 0.75
    assert aggregate["underfilled_step_count_sum"] == 2
    assert aggregate["max_steps_hit_count_sum"] == 0

    macro = aggregate_dataset_macro(
        transfer, completed_rows=rows
    )
    assert len(macro) == 1
    assert macro[0]["budget_valid_run_count"] == 1
    assert macro[0]["budget_valid_run_ratio"] == 0.5
    assert macro[0]["budget_reached_all_runs"] is False

    unavailable = _summary_payload("unknown", 3)
    unavailable.update(
        {
            "budget_diagnostics_available": False,
            "budget_reached_all_steps": None,
            "valid_lbi_run": None,
        }
    )
    with_unknown = aggregate_across_seeds([rows[0], unavailable])[0]
    assert with_unknown["budget_diagnostics_run_count"] == 1
    assert with_unknown["budget_reached_all_runs"] is None


def main():
    _check_run_diagnostics()
    _check_legacy_backfill_and_missing()
    _check_aggregation()
    print("iclr2027 budget diagnostics tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
