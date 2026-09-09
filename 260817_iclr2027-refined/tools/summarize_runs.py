#!/usr/bin/env python3
"""Produce lossless run, seed, and dataset-macro result tables."""

import argparse
import csv
import json
import os
import os.path as osp
import statistics
import sys


PROJECT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from tools.check_experiment_status import (  # noqa: E402
    check_status,
    load_plan,
    scan_run_summaries,
)
from core.lbi import (  # noqa: E402
    RUN_BUDGET_DIAGNOSTIC_FIELDS,
    compute_lbi_run_budget_diagnostics,
    unavailable_lbi_budget_diagnostics,
)


LBI_VARIANTS = {"module_lbi", "conv_out_lbi", "conv_filter_lbi"}
CONV_LBI_VARIANTS = {"conv_out_lbi", "conv_filter_lbi"}
RANDOM_VARIANTS = {"module_random", "conv_out_random", "conv_filter_random"}


COLUMNS = [
    "status",
    "implementation_revision",
    "experiment_key",
    "experiment_config_sha256",
    "method",
    "variant",
    "task",
    "dataset",
    "source",
    "target",
    "source-target",
    "seed",
    "selection",
    "selection_seed",
    "mask_static",
    "candidate_scope",
    "candidate_scope_type",
    "requested_budget",
    "max_support_count",
    "target_support_count",
    "exact_budget_reached_all_steps",
    "strict_budget_boundary_all_steps",
    "exact_budget_hit_count",
    "budget_reached_all_steps",
    "budget_hit_count",
    "budget_hit_rate",
    "valid_lbi_run",
    "valid_lbi_step_count",
    "valid_lbi_step_rate",
    "underfilled_step_count",
    "max_steps_hit_count",
    "max_steps_hit_rate",
    "stage1_stop_reason_counts",
    "support_gap_count_first",
    "support_gap_count_last",
    "support_gap_count_min",
    "support_gap_count_max",
    "support_gap_count_mean",
    "stage1_support_count_first",
    "stage1_support_count_last",
    "stage1_support_count_min",
    "stage1_support_count_max",
    "stage1_support_count_mean",
    "stage1_steps_completed_min",
    "stage1_steps_completed_max",
    "stage1_steps_completed_mean",
    "budget_diagnostics_available",
    "budget_diagnostics_source",
    "budget_diagnostics_unavailable_reason",
    "ranking_source",
    "mask_refresh_policy",
    "saliency_score",
    "lbi_state_lifecycle",
    "lbi_initialization",
    "stage3_mode",
    "alpha",
    "kappa",
    "nu",
    "omega",
    "lbi_alpha",
    "lbi_kappa",
    "lbi_nu",
    "lbi_omega",
    "stage1_max_steps",
    "budget_tolerance",
    "stage2_lr",
    "stage2_steps_requested",
    "stage2_optimizer",
    "delta_nonzero_tolerance",
    "expected_selected_param_count_per_step",
    "selected_param_count",
    "candidate_scope_param_count",
    "total_model_param_count",
    "selected_over_scope_ratio",
    "selected_over_model_ratio",
    "selected_param_count_first",
    "selected_param_count_last",
    "selected_param_count_min",
    "selected_param_count_max",
    "selected_param_count_mean",
    "selected_over_scope_ratio_first",
    "selected_over_scope_ratio_last",
    "selected_over_scope_ratio_min",
    "selected_over_scope_ratio_max",
    "selected_over_scope_ratio_mean",
    "selected_over_model_ratio_first",
    "selected_over_model_ratio_last",
    "selected_over_model_ratio_min",
    "selected_over_model_ratio_max",
    "selected_over_model_ratio_mean",
    "support_param_count",
    "support_over_scope_ratio",
    "support_over_model_ratio",
    "effective_delta_nonzero_count",
    "effective_delta_over_scope_ratio",
    "effective_delta_over_model_ratio",
    "effective_delta_l1",
    "effective_delta_l2",
    "applied_update_nonzero_count",
    "applied_update_over_scope_ratio",
    "applied_update_over_model_ratio",
    "applied_update_l1",
    "applied_update_l2",
    "stage1_steps_mean",
    "stage2_steps_mean",
    "bn_stats_policy",
    "bn_stats_frozen",
    "bn_module_count",
    "PU-Acc",
    "FO-Acc",
    "runtime",
    "efficiency_protocol_revision",
    "online_batch_runtime_mean_sec",
    "online_batch_runtime_std_sec",
    "online_batch_runtime_median_sec",
    "online_batch_runtime_p95_sec",
    "online_compute_runtime_sec",
    "online_batch_runtime_mask_std_sec",
    "online_compute_runtime_mask_std_sec",
    "random_total_online_compute_runtime_sec",
    "adapt_batch_runtime_mean_sec",
    "adapt_batch_runtime_std_sec",
    "adapt_batch_runtime_median_sec",
    "adapt_batch_runtime_p95_sec",
    "adapt_runtime_total_sec",
    "pu_batch_runtime_mean_sec",
    "pu_batch_runtime_std_sec",
    "pu_batch_runtime_median_sec",
    "pu_batch_runtime_p95_sec",
    "pu_runtime_total_sec",
    "gpu_peak_allocated_mean_mb",
    "gpu_peak_allocated_max_mb",
    "gpu_peak_reserved_mean_mb",
    "gpu_peak_reserved_max_mb",
    "fo_eval_runtime_sec",
    "wall_runtime_sec",
    "peak_gpu_memory_allocated_mb",
    "peak_gpu_memory_reserved_mb",
    "gpu_device_index",
    "torch_version",
    "cuda_version",
    "runtime_resume_used",
    "runtime_segment_count",
    "gpu_name",
    "runtime_comparable",
    "started_at_utc",
    "completed_at_utc",
    "run_id",
]

SEED_GROUP_FIELDS = [
    "implementation_revision",
    "method",
    "task",
    "dataset",
    "source",
    "target",
    "variant",
    "requested_budget",
    "alpha",
    "kappa",
    "nu",
    "omega",
    "stage1_max_steps",
    "budget_tolerance",
    "stage2_lr",
    "stage2_steps_requested",
    "delta_nonzero_tolerance",
]
MACRO_GROUP_FIELDS = [
    "implementation_revision",
    "method",
    "task",
    "dataset",
    "variant",
    "requested_budget",
    "alpha",
    "kappa",
    "nu",
    "omega",
    "stage1_max_steps",
    "budget_tolerance",
    "stage2_lr",
    "stage2_steps_requested",
    "delta_nonzero_tolerance",
]
AGGREGATED_METRICS = (
    "PU-Acc",
    "FO-Acc",
    "runtime",
    "online_batch_runtime_mean_sec",
    "online_batch_runtime_std_sec",
    "online_batch_runtime_median_sec",
    "online_batch_runtime_p95_sec",
    "online_compute_runtime_sec",
    "online_batch_runtime_mask_std_sec",
    "online_compute_runtime_mask_std_sec",
    "random_total_online_compute_runtime_sec",
    "adapt_batch_runtime_mean_sec",
    "adapt_batch_runtime_std_sec",
    "adapt_batch_runtime_median_sec",
    "adapt_batch_runtime_p95_sec",
    "adapt_runtime_total_sec",
    "pu_batch_runtime_mean_sec",
    "pu_batch_runtime_std_sec",
    "pu_batch_runtime_median_sec",
    "pu_batch_runtime_p95_sec",
    "pu_runtime_total_sec",
    "gpu_peak_allocated_mean_mb",
    "gpu_peak_allocated_max_mb",
    "gpu_peak_reserved_mean_mb",
    "gpu_peak_reserved_max_mb",
    "fo_eval_runtime_sec",
    "wall_runtime_sec",
    "peak_gpu_memory_allocated_mb",
    "peak_gpu_memory_reserved_mb",
)


def _aggregate_runtime_metadata(group_rows):
    comparable_values = [
        row.get("runtime_comparable") is True for row in group_rows
    ]
    gpu_names = sorted(
        {row.get("gpu_name") for row in group_rows if row.get("gpu_name")}
    )
    protocols = sorted(
        {
            row.get("efficiency_protocol_revision")
            for row in group_rows
            if row.get("efficiency_protocol_revision")
        }
    )
    torch_versions = sorted(
        {row.get("torch_version") for row in group_rows if row.get("torch_version")}
    )
    cuda_versions = sorted(
        {row.get("cuda_version") for row in group_rows if row.get("cuda_version")}
    )
    return {
        "runtime_comparable": bool(group_rows) and all(comparable_values),
        "gpu_name": (
            gpu_names[0]
            if len(gpu_names) == 1
            else ("mixed" if gpu_names else None)
        ),
        "efficiency_protocol_revision": (
            protocols[0]
            if len(protocols) == 1
            else ("mixed" if protocols else None)
        ),
        "torch_version": (
            torch_versions[0]
            if len(torch_versions) == 1
            else ("mixed" if torch_versions else None)
        ),
        "cuda_version": (
            cuda_versions[0]
            if len(cuda_versions) == 1
            else ("mixed" if cuda_versions else None)
        ),
        "runtime_resume_used": any(
            row.get("runtime_resume_used") is True for row in group_rows
        ),
        "runtime_segment_count": max(
            [int(row.get("runtime_segment_count", 1)) for row in group_rows]
            or [1]
        ),
    }

LBI_BUDGET_DIAGNOSTIC_COLUMNS = [
    "implementation_revision",
    "dataset",
    "source",
    "target",
    "seed",
    "requested_budget",
    "max_support_count",
    "target_support_count",
    "exact_budget_reached_all_steps",
    "strict_budget_boundary_all_steps",
    "exact_budget_hit_count",
    "alpha",
    "kappa",
    "nu",
    "omega",
    "stage1_max_steps",
    "budget_reached_all_steps",
    "budget_hit_count",
    "online_steps",
    "budget_hit_rate",
    "valid_lbi_run",
    "underfilled_step_count",
    "max_steps_hit_count",
    "stage1_stop_reason_counts",
    "stage1_support_count_min",
    "stage1_support_count_max",
    "support_gap_count_min",
    "support_gap_count_max",
    "stage1_steps_completed_min",
    "stage1_steps_completed_max",
    "stage1_steps_completed_mean",
    "FO-Acc",
    "runtime",
    "efficiency_protocol_revision",
    "online_batch_runtime_mean_sec",
    "online_batch_runtime_std_sec",
    "online_batch_runtime_median_sec",
    "online_batch_runtime_p95_sec",
    "online_compute_runtime_sec",
    "adapt_batch_runtime_mean_sec",
    "adapt_batch_runtime_std_sec",
    "adapt_runtime_total_sec",
    "pu_batch_runtime_mean_sec",
    "pu_batch_runtime_std_sec",
    "pu_runtime_total_sec",
    "gpu_peak_allocated_mean_mb",
    "gpu_peak_allocated_max_mb",
    "gpu_peak_reserved_mean_mb",
    "gpu_peak_reserved_max_mb",
    "fo_eval_runtime_sec",
    "wall_runtime_sec",
    "peak_gpu_memory_allocated_mb",
    "peak_gpu_memory_reserved_mb",
    "gpu_name",
    "runtime_comparable",
    "budget_diagnostics_available",
    "budget_diagnostics_source",
    "budget_diagnostics_unavailable_reason",
    "experiment_key",
    "summary_path",
]


CONV_LBI_BUDGET_DIAGNOSTIC_COLUMNS = list(dict.fromkeys([
    "variant",
    "protocol_revision",
    "conv_protocol_revision",
    "group_mode",
    "total_group_count",
    "max_group_count",
    "selected_group_count_first",
    "selected_group_count_last",
    "selected_group_count_min",
    "selected_group_count_max",
    "selected_group_count_mean",
    "selected_scalar_count_first",
    "selected_scalar_count_last",
    "selected_scalar_count_min",
    "selected_scalar_count_max",
    "selected_scalar_count_mean",
    "realized_group_ratio_first",
    "realized_group_ratio_last",
    "realized_group_ratio_min",
    "realized_group_ratio_max",
    "realized_group_ratio_mean",
    "realized_scalar_ratio_first",
    "realized_scalar_ratio_last",
    "realized_scalar_ratio_min",
    "realized_scalar_ratio_max",
    "realized_scalar_ratio_mean",
    "group_utilization_min",
    "group_utilization_mean",
    "group_utilization_p05",
    "group_utilization_below_95_count",
    "group_utilization_below_95_fraction",
    *LBI_BUDGET_DIAGNOSTIC_COLUMNS,
]))


def _read_online_step_metrics(summary_path):
    metrics_path = osp.join(
        osp.dirname(osp.abspath(summary_path)), "metrics.jsonl"
    )
    if not osp.isfile(metrics_path):
        return None, f"metrics_jsonl_not_found: {metrics_path}"
    online_steps = []
    try:
        with open(metrics_path, "r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    return (
                        None,
                        "invalid_metrics_jsonl_at_line_"
                        f"{line_number}: {error}",
                    )
                if (
                    isinstance(record, dict)
                    and record.get("event") == "online_step"
                ):
                    online_steps.append(record)
    except OSError as error:
        return None, f"metrics_jsonl_read_error: {error}"
    if not online_steps:
        return None, "metrics_jsonl_has_no_online_step_records"
    return online_steps, None


def _enrich_lbi_budget_diagnostics(summary, summary_path):
    enriched = dict(summary)
    if summary.get("variant") not in LBI_VARIANTS:
        return enriched

    has_availability = "budget_diagnostics_available" in summary
    has_all_fields = all(
        field in summary for field in RUN_BUDGET_DIAGNOSTIC_FIELDS
    )
    if has_availability and (
        summary.get("budget_diagnostics_available") is False
        or has_all_fields
    ):
        enriched["budget_diagnostics_source"] = "summary_json"
        return enriched

    online_steps, read_error = _read_online_step_metrics(summary_path)
    if read_error is not None:
        diagnostics = unavailable_lbi_budget_diagnostics(read_error)
    else:
        diagnostics = compute_lbi_run_budget_diagnostics(online_steps)
        if diagnostics["budget_diagnostics_available"]:
            diagnostics["budget_diagnostics_source"] = "metrics_jsonl"
    enriched.update(diagnostics)
    return enriched


def _summary_to_row(summary, summary_path):
    summary = _enrich_lbi_budget_diagnostics(
        summary, summary_path
    )
    row = {
        column: summary.get(column)
        for column in COLUMNS
        if column in summary
    }
    for field, value in summary.items():
        if field not in row:
            row[field] = value
    row["summary_path"] = osp.abspath(summary_path)
    return row


def collect_summaries(runs_root):
    records, _ = scan_run_summaries(runs_root)
    rows = [
        _summary_to_row(record["summary"], record["summary_path"])
        for record in records
        if record["status"] == "completed"
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("dataset")),
            str(row.get("source")),
            str(row.get("target")),
            str(row.get("seed")),
            str(row.get("variant")),
            str(row.get("requested_budget")),
            str(row.get("run_id")),
        )
    )
    return rows


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
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _csv_value(row.get(field))
                    for field in resolved_fieldnames
                }
            )


def _markdown_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(path, rows):
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    columns = []
    for row in rows:
        for field in row:
            if field not in columns:
                columns.append(field)
    with open(path, "w", encoding="utf-8") as file_obj:
        if not columns:
            file_obj.write("")
            return
        file_obj.write("| " + " | ".join(columns) + " |\n")
        file_obj.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            file_obj.write(
                "| "
                + " | ".join(
                    _markdown_value(row.get(column))
                    for column in columns
                )
                + " |\n"
            )


def _selection_seed_mode(row, plan_by_identity):
    identity = (
        row.get("experiment_key"),
        row.get("experiment_config_sha256"),
    )
    planned = plan_by_identity.get(identity)
    if planned is not None:
        return planned.get("selection_seed_mode")
    if row.get("variant") not in RANDOM_VARIANTS:
        return None
    if row.get("selection_seed") == row.get("seed"):
        return "same_as_run_seed"
    return "explicit"


def _aggregation_selection_seed(row, mode):
    if row.get("variant") not in RANDOM_VARIANTS:
        return None
    if mode == "same_as_run_seed":
        return None
    return row.get("selection_seed")


def _metric_statistics(values):
    usable = [value for value in values if value is not None]
    return {
        "count": len(usable),
        "mean": (
            sum(usable) / len(usable) if usable else None
        ),
        "sample_std": (
            statistics.stdev(usable) if len(usable) > 1 else None
        ),
        "min": min(usable) if usable else None,
        "max": max(usable) if usable else None,
    }


def _aggregate_lbi_budget_diagnostics(group_rows):
    if not group_rows or group_rows[0].get("variant") not in LBI_VARIANTS:
        return {}
    available_rows = [
        row
        for row in group_rows
        if row.get("budget_diagnostics_available") is True
    ]
    available_count = len(available_rows)
    valid_count = sum(
        row.get("valid_lbi_run") is True for row in available_rows
    )
    diagnostics = {
        "budget_diagnostics_run_count": available_count,
        "budget_valid_run_count": valid_count,
        "budget_valid_run_ratio": (
            valid_count / available_count
            if available_count > 0
            else None
        ),
        "budget_reached_all_runs": (
            all(
                row.get("budget_reached_all_steps") is True
                for row in available_rows
            )
            if available_count == len(group_rows)
            and available_count > 0
            else None
        ),
    }

    def values(field):
        return [
            row.get(field)
            for row in available_rows
            if row.get(field) is not None
        ]

    budget_hit_rates = values("budget_hit_rate")
    diagnostics["budget_hit_rate_mean"] = (
        sum(budget_hit_rates) / len(budget_hit_rates)
        if budget_hit_rates
        else None
    )
    diagnostics["budget_hit_rate_min"] = (
        min(budget_hit_rates) if budget_hit_rates else None
    )
    diagnostics["underfilled_step_count_sum"] = (
        sum(values("underfilled_step_count"))
        if values("underfilled_step_count")
        else None
    )
    diagnostics["max_steps_hit_count_sum"] = (
        sum(values("max_steps_hit_count"))
        if values("max_steps_hit_count")
        else None
    )
    support_minimums = values("stage1_support_count_min")
    diagnostics["stage1_support_count_min_across_runs"] = (
        min(support_minimums) if support_minimums else None
    )
    gap_minimums = values("support_gap_count_min")
    diagnostics["support_gap_count_min_across_runs"] = (
        min(gap_minimums) if gap_minimums else None
    )
    step_maximums = values("stage1_steps_completed_max")
    diagnostics["stage1_steps_completed_max_across_runs"] = (
        max(step_maximums) if step_maximums else None
    )
    if group_rows[0].get("variant") in CONV_LBI_VARIANTS:
        utilization_mins = values("group_utilization_min")
        utilization_means = values("group_utilization_mean")
        utilization_p05s = values("group_utilization_p05")
        below_95_counts = values("group_utilization_below_95_count")
        diagnostics.update(
            {
                "group_utilization_min_across_runs": (
                    min(utilization_mins) if utilization_mins else None
                ),
                "group_utilization_mean_across_runs": (
                    sum(utilization_means) / len(utilization_means)
                    if utilization_means
                    else None
                ),
                "group_utilization_p05_min_across_runs": (
                    min(utilization_p05s) if utilization_p05s else None
                ),
                "group_utilization_below_95_count_sum": (
                    int(sum(below_95_counts)) if below_95_counts else None
                ),
            }
        )
    return diagnostics


def aggregate_across_seeds(rows, plan=None):
    plan_by_identity = {}
    if plan is not None:
        plan_by_identity = {
            (
                experiment["experiment_key"],
                experiment["experiment_config_sha256"],
            ): experiment
            for experiment in plan["experiments"]
        }
    groups = {}
    for row in rows:
        mode = _selection_seed_mode(row, plan_by_identity)
        aggregation_selection_seed = _aggregation_selection_seed(
            row,
            mode,
        )
        key = tuple(row.get(field) for field in SEED_GROUP_FIELDS) + (
            mode,
            aggregation_selection_seed,
        )
        groups.setdefault(key, []).append(row)

    aggregated = []
    for key, group_rows in groups.items():
        output = {
            field: key[index]
            for index, field in enumerate(SEED_GROUP_FIELDS)
        }
        output["selection_seed_mode"] = key[-2]
        output["selection_seed"] = key[-1]
        output["seed_count"] = len(
            {row.get("seed") for row in group_rows}
        )
        output["seeds"] = sorted(
            {row.get("seed") for row in group_rows}
        )
        for metric in AGGREGATED_METRICS:
            stats = _metric_statistics(
                [row.get(metric) for row in group_rows]
            )
            for statistic, value in stats.items():
                output[f"{metric}_{statistic}"] = value
        output.update(_aggregate_runtime_metadata(group_rows))
        output.update(_aggregate_lbi_budget_diagnostics(group_rows))
        aggregated.append(output)
    aggregated.sort(
        key=lambda row: tuple(
            str(row.get(field))
            for field in (
                *SEED_GROUP_FIELDS,
                "selection_seed_mode",
                "selection_seed",
            )
        )
    )
    return aggregated


def _macro_key_from_row(row):
    return tuple(row.get(field) for field in MACRO_GROUP_FIELDS) + (
        row.get("selection_seed_mode"),
        row.get("selection_seed"),
    )


def _combine_lbi_transfer_diagnostics(transfer_rows):
    if not transfer_rows or transfer_rows[0].get("variant") not in LBI_VARIANTS:
        return {}
    diagnostics_count = sum(
        row.get("budget_diagnostics_run_count", 0)
        for row in transfer_rows
    )
    valid_count = sum(
        row.get("budget_valid_run_count", 0)
        for row in transfer_rows
    )
    all_runs_values = [
        row.get("budget_reached_all_runs") for row in transfer_rows
    ]
    weighted_hit_rate_sum = sum(
        row["budget_hit_rate_mean"]
        * row.get("budget_diagnostics_run_count", 0)
        for row in transfer_rows
        if row.get("budget_hit_rate_mean") is not None
    )
    hit_rate_weight = sum(
        row.get("budget_diagnostics_run_count", 0)
        for row in transfer_rows
        if row.get("budget_hit_rate_mean") is not None
    )

    def non_null(field):
        return [
            row.get(field)
            for row in transfer_rows
            if row.get(field) is not None
        ]

    hit_rate_minimums = non_null("budget_hit_rate_min")
    support_minimums = non_null(
        "stage1_support_count_min_across_runs"
    )
    gap_minimums = non_null("support_gap_count_min_across_runs")
    step_maximums = non_null(
        "stage1_steps_completed_max_across_runs"
    )
    return {
        "budget_diagnostics_run_count": diagnostics_count,
        "budget_valid_run_count": valid_count,
        "budget_valid_run_ratio": (
            valid_count / diagnostics_count
            if diagnostics_count > 0
            else None
        ),
        "budget_reached_all_runs": (
            all(value is True for value in all_runs_values)
            if all_runs_values
            and all(value is not None for value in all_runs_values)
            else None
        ),
        "budget_hit_rate_mean": (
            weighted_hit_rate_sum / hit_rate_weight
            if hit_rate_weight > 0
            else None
        ),
        "budget_hit_rate_min": (
            min(hit_rate_minimums) if hit_rate_minimums else None
        ),
        "underfilled_step_count_sum": sum(
            non_null("underfilled_step_count_sum")
        )
        if non_null("underfilled_step_count_sum")
        else None,
        "max_steps_hit_count_sum": sum(
            non_null("max_steps_hit_count_sum")
        )
        if non_null("max_steps_hit_count_sum")
        else None,
        "stage1_support_count_min_across_runs": (
            min(support_minimums) if support_minimums else None
        ),
        "support_gap_count_min_across_runs": (
            min(gap_minimums) if gap_minimums else None
        ),
        "stage1_steps_completed_max_across_runs": (
            max(step_maximums) if step_maximums else None
        ),
    }


def aggregate_dataset_macro(
    transfer_aggregates,
    plan=None,
    completed_rows=None,
):
    observed = {}
    for row in transfer_aggregates:
        observed.setdefault(_macro_key_from_row(row), []).append(row)

    expected_transfers = {}
    expected_experiments = {}
    if plan is not None:
        for experiment in plan["experiments"]:
            mode = experiment.get("selection_seed_mode")
            selection_seed = (
                None
                if mode == "same_as_run_seed"
                else experiment.get("selection_seed")
            )
            key = tuple(
                experiment.get(field)
                for field in MACRO_GROUP_FIELDS
            ) + (mode, selection_seed)
            expected_transfers.setdefault(key, set()).add(
                (experiment["source"], experiment["target"])
            )
            expected_experiments[key] = (
                expected_experiments.get(key, 0) + 1
            )

    completed_experiments = {}
    completed_rows_by_key = {}
    if completed_rows is not None:
        plan_by_identity = {}
        if plan is not None:
            plan_by_identity = {
                (
                    experiment["experiment_key"],
                    experiment["experiment_config_sha256"],
                ): experiment
                for experiment in plan["experiments"]
            }
        for row in completed_rows:
            mode = _selection_seed_mode(row, plan_by_identity)
            selection_seed = _aggregation_selection_seed(row, mode)
            key = tuple(
                row.get(field) for field in MACRO_GROUP_FIELDS
            ) + (mode, selection_seed)
            completed_experiments[key] = (
                completed_experiments.get(key, 0) + 1
            )
            completed_rows_by_key.setdefault(key, []).append(row)

    keys = set(observed) | set(expected_transfers)
    outputs = []
    for key in keys:
        transfer_rows = observed.get(key, [])
        output = {
            field: key[index]
            for index, field in enumerate(MACRO_GROUP_FIELDS)
        }
        output["selection_seed_mode"] = key[-2]
        output["selection_seed"] = key[-1]
        completed_transfer_count = len(transfer_rows)
        expected_transfer_count = len(
            expected_transfers.get(
                key,
                {
                    (row["source"], row["target"])
                    for row in transfer_rows
                },
            )
        )
        output["completed_transfer_count"] = completed_transfer_count
        output["expected_transfer_count"] = expected_transfer_count
        output["coverage_ratio"] = (
            completed_transfer_count / expected_transfer_count
            if expected_transfer_count > 0
            else 0.0
        )
        completed_experiment_count = completed_experiments.get(
            key,
            sum(row.get("PU-Acc_count", 0) for row in transfer_rows),
        )
        expected_experiment_count = expected_experiments.get(
            key,
            completed_experiment_count,
        )
        output["completed_experiment_count"] = (
            completed_experiment_count
        )
        output["expected_experiment_count"] = expected_experiment_count
        output["is_complete"] = (
            completed_transfer_count == expected_transfer_count
            and completed_experiment_count
            == expected_experiment_count
        )
        for metric in AGGREGATED_METRICS:
            transfer_means = [
                row[f"{metric}_mean"]
                for row in transfer_rows
                if row.get(f"{metric}_mean") is not None
            ]
            output[f"{metric}_macro_mean"] = (
                sum(transfer_means) / len(transfer_means)
                if transfer_means
                else None
            )
        output.update(_aggregate_runtime_metadata(transfer_rows))
        if output.get("variant") in LBI_VARIANTS:
            if key in completed_rows_by_key:
                output.update(
                    _aggregate_lbi_budget_diagnostics(
                        completed_rows_by_key[key]
                    )
                )
            else:
                output.update(
                    _combine_lbi_transfer_diagnostics(transfer_rows)
                )
        outputs.append(output)
    outputs.sort(
        key=lambda row: tuple(
            str(row.get(field))
            for field in (
                *MACRO_GROUP_FIELDS,
                "selection_seed_mode",
                "selection_seed",
            )
        )
    )
    return outputs


def _identity_groups(records):
    groups = {}
    for record in records:
        if record["status"] != "completed":
            continue
        identity = (
            record["experiment_key"],
            record["experiment_config_sha256"],
        )
        groups.setdefault(identity, []).append(record)
    return groups


def build_summary_outputs(runs_root, plan=None):
    records, invalid_records = scan_run_summaries(runs_root)
    planned_identities = (
        {
            (
                experiment["experiment_key"],
                experiment["experiment_config_sha256"],
            )
            for experiment in plan["experiments"]
        }
        if plan is not None
        else None
    )
    completed_records = [
        record for record in records if record["status"] == "completed"
        and (
            planned_identities is None
            or (
                record["experiment_key"],
                record["experiment_config_sha256"],
            )
            in planned_identities
        )
    ]
    all_rows = [
        _summary_to_row(record["summary"], record["summary_path"])
        for record in completed_records
    ]
    all_rows.sort(key=lambda row: str(row["summary_path"]))

    identity_groups = _identity_groups(records)
    if plan is not None:
        status_result = check_status(plan, runs_root)
        aggregatable_identities = {
            (
                row["experiment_key"],
                row["experiment_config_sha256"],
            )
            for row in status_result["experiments"]
            if row["plan_status"] == "completed"
        }
    else:
        status_result = None
        aggregatable_identities = {
            identity
            for identity, group in identity_groups.items()
            if len(group) == 1
        }
    aggregate_records = [
        group[0]
        for identity, group in identity_groups.items()
        if identity in aggregatable_identities and len(group) == 1
    ]
    aggregate_rows = [
        _summary_to_row(record["summary"], record["summary_path"])
        for record in aggregate_records
    ]

    transfer_aggregates = aggregate_across_seeds(
        aggregate_rows,
        plan=plan,
    )
    macro_aggregates = aggregate_dataset_macro(
        transfer_aggregates,
        plan=plan,
        completed_rows=aggregate_rows,
    )
    lbi_budget_diagnostics = [
        {
            field: row.get(field)
            for field in LBI_BUDGET_DIAGNOSTIC_COLUMNS
        }
        for row in all_rows
        if row.get("variant") == "module_lbi"
    ]
    conv_lbi_budget_diagnostics = [
        {
            field: row.get(field)
            for field in CONV_LBI_BUDGET_DIAGNOSTIC_COLUMNS
        }
        for row in all_rows
        if row.get("variant") in CONV_LBI_VARIANTS
    ]
    classwise_long, pu_classwise_wide, fo_classwise_wide = (
        _build_visda_classwise_outputs(all_rows)
    )

    invalid_runs = list(invalid_records)
    invalid_runs.extend(
        {
            "summary_path": record["summary_path"],
            "experiment_key": record["experiment_key"],
            "experiment_config_sha256": record[
                "experiment_config_sha256"
            ],
            "reason": f"non_completed_status: {record['status']}",
        }
        for record in records
        if record["status"] != "completed"
    )

    if plan is not None:
        missing_runs = [
            row
            for row in status_result["experiments"]
            if row["plan_status"] == "missing"
        ]
        duplicate_runs = [
            row
            for row in status_result["experiments"]
            if row["plan_status"] == "duplicate_completed"
        ]
        invalid_runs.extend(
            {
                **row,
                "reason": f"plan_status: {row['plan_status']}",
            }
            for row in status_result["experiments"]
            if row["plan_status"]
            in {"hash_mismatch", "invalid_summary"}
        )
    else:
        missing_runs = []
        duplicate_runs = [
            {
                "experiment_key": identity[0],
                "experiment_config_sha256": identity[1],
                "completed_count": len(group),
                "summary_paths": [
                    record["summary_path"] for record in group
                ],
            }
            for identity, group in identity_groups.items()
            if len(group) > 1
        ]

    return {
        "all_runs": all_rows,
        "per_transfer_seed_aggregated": transfer_aggregates,
        "dataset_macro_aggregated": macro_aggregates,
        "lbi_budget_diagnostics": lbi_budget_diagnostics,
        "conv_lbi_budget_diagnostics": conv_lbi_budget_diagnostics,
        "classwise_long": classwise_long,
        "PU_classwise_wide": pu_classwise_wide,
        "FO_classwise_wide": fo_classwise_wide,
        "missing_runs": missing_runs,
        "duplicate_runs": duplicate_runs,
        "invalid_runs": invalid_runs,
    }


def _build_visda_classwise_outputs(rows):
    """Return lossless long/wide class metrics for completed VisDA summaries."""
    identity_fields = (
        "experiment_key", "experiment_config_sha256", "variant", "seed",
        "requested_budget", "source", "target", "summary_path",
    )
    long_rows, pu_wide, fo_wide = [], [], []
    for row in rows:
        if row.get("dataset") != "VISDA-C":
            continue
        names = row.get("class-names") or []
        for prefix, wide_rows in (("PU", pu_wide), ("FO", fo_wide)):
            values = row.get(f"{prefix}-Acc-per-class")
            if not isinstance(values, list) or len(values) != len(names):
                continue
            wide = {field: row.get(field) for field in identity_fields}
            for class_id, (name, value) in enumerate(zip(names, values)):
                wide[f"class_{class_id}_{name}"] = value
                long_rows.append(
                    {
                        **{field: row.get(field) for field in identity_fields},
                        "metric": prefix,
                        "class_id": class_id,
                        "class_name": name,
                        "accuracy": value,
                    }
                )
            wide_rows.append(wide)
    return long_rows, pu_wide, fo_wide


def _write_table_set(output_dir, basename, rows):
    _write_json(osp.join(output_dir, f"{basename}.json"), rows)
    write_csv(osp.join(output_dir, f"{basename}.csv"), rows)
    write_markdown(osp.join(output_dir, f"{basename}.md"), rows)


def write_summary_outputs(outputs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    _write_table_set(
        output_dir,
        "all_runs",
        outputs["all_runs"],
    )
    _write_table_set(
        output_dir,
        "per_transfer_seed_aggregated",
        outputs["per_transfer_seed_aggregated"],
    )
    _write_table_set(
        output_dir,
        "dataset_macro_aggregated",
        outputs["dataset_macro_aggregated"],
    )
    _write_table_set(
        output_dir,
        "lbi_budget_diagnostics",
        outputs["lbi_budget_diagnostics"],
    )
    _write_table_set(
        output_dir,
        "conv_lbi_budget_diagnostics",
        outputs["conv_lbi_budget_diagnostics"],
    )
    write_csv(
        osp.join(output_dir, "classwise_long.csv"), outputs["classwise_long"]
    )
    write_csv(
        osp.join(output_dir, "PU_classwise_wide.csv"),
        outputs["PU_classwise_wide"],
    )
    write_csv(
        osp.join(output_dir, "FO_classwise_wide.csv"),
        outputs["FO_classwise_wide"],
    )
    write_csv(
        osp.join(output_dir, "missing_runs.csv"),
        outputs["missing_runs"],
        fieldnames=[
            "experiment_key",
            "experiment_config_sha256",
            "plan_status",
        ],
    )
    write_csv(
        osp.join(output_dir, "duplicate_runs.csv"),
        outputs["duplicate_runs"],
        fieldnames=[
            "experiment_key",
            "experiment_config_sha256",
            "plan_status",
        ],
    )
    write_csv(
        osp.join(output_dir, "invalid_runs.csv"),
        outputs["invalid_runs"],
        fieldnames=[
            "summary_path",
            "experiment_key",
            "experiment_config_sha256",
            "reason",
        ],
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_root")
    parser.add_argument("--plan", default=None)
    parser.add_argument("--output-dir", default="iclr2027/summary")
    return parser


def main():
    args = build_parser().parse_args()
    plan = load_plan(args.plan) if args.plan else None
    outputs = build_summary_outputs(args.runs_root, plan=plan)
    write_summary_outputs(outputs, args.output_dir)
    print(
        json.dumps(
            {
                "completed_runs": len(outputs["all_runs"]),
                "transfer_groups": len(
                    outputs["per_transfer_seed_aggregated"]
                ),
                "dataset_macro_groups": len(
                    outputs["dataset_macro_aggregated"]
                ),
                "lbi_budget_diagnostic_runs": len(
                    outputs["lbi_budget_diagnostics"]
                ),
                "missing": len(outputs["missing_runs"]),
                "duplicates": len(outputs["duplicate_runs"]),
                "invalid": len(outputs["invalid_runs"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
