"""Count-based Split-LBI budget diagnostics shared by training and reports."""

import math


MAX_STEPS_REASON_ALIASES = {"max_steps", "max_steps_reached"}
STANDARD_STOP_REASONS = (
    "branch_disabled",
    "budget_reached",
    "strict_budget_rollback",
    "rollback_feasible",
    "cross_no_feasible",
    "max_steps",
)

STEP_BUDGET_DIAGNOSTIC_FIELDS = (
    "max_support_count",
    "target_support_count",
    "support_gap_count",
    "exact_budget_reached",
    "budget_reached",
    "strict_budget_boundary_stop",
    "max_steps_hit",
    "valid_lbi_step",
)

RUN_BUDGET_DIAGNOSTIC_FIELDS = (
    "max_support_count",
    "target_support_count",
    "exact_budget_hit_count",
    "exact_budget_reached_all_steps",
    "strict_budget_boundary_all_steps",
    "budget_hit_count",
    "budget_hit_rate",
    "budget_reached_all_steps",
    "valid_lbi_step_count",
    "valid_lbi_step_rate",
    "valid_lbi_run",
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
    "stage1_steps_completed_first",
    "stage1_steps_completed_last",
    "stage1_steps_completed_min",
    "stage1_steps_completed_max",
    "stage1_steps_completed_mean",
)


def max_support_count(requested_budget, candidate_param_count):
    """Return the strict global integer support upper bound.

    This is the single source of truth for every refined sparse budget:
    ``floor(requested_budget * candidate_param_count)``.
    """
    return math.floor(
        float(requested_budget) * int(candidate_param_count)
    )


def target_support_count(requested_budget, candidate_scope_param_count):
    """Compatibility alias for the strict maximum legal support count."""
    return max_support_count(requested_budget, candidate_scope_param_count)


def _canonical_stop_reason(reason):
    reason = str(reason)
    if reason in MAX_STEPS_REASON_ALIASES:
        return "max_steps"
    return reason


def compute_lbi_step_budget_diagnostics(
    requested_budget,
    candidate_scope_param_count,
    stage1_support_count,
    stage1_stop_reason,
):
    target_count = max_support_count(
        requested_budget, candidate_scope_param_count
    )
    support_count = int(stage1_support_count)
    gap_count = support_count - target_count
    stop_reason = str(stage1_stop_reason)
    exact_budget_reached = (
        (
            stop_reason == "budget_reached"
            and support_count == target_count
        )
        or (stop_reason == "branch_disabled" and target_count == 0)
    )
    max_steps_hit = stop_reason in MAX_STEPS_REASON_ALIASES
    strict_budget_boundary_stop = (
        not max_steps_hit
        and support_count <= target_count
        and (
            (
                stop_reason == "budget_reached"
                and support_count == target_count
            )
            or (
                stop_reason == "strict_budget_rollback"
                and support_count <= target_count
            )
            or (
                stop_reason == "rollback_feasible"
                and support_count == target_count
            )
            or (stop_reason == "branch_disabled" and target_count == 0)
        )
    )
    return {
        "max_support_count": target_count,
        "target_support_count": target_count,
        "support_gap_count": gap_count,
        "exact_budget_reached": exact_budget_reached,
        # Keep the field name for downstream compatibility.  It now means
        # exact integer equality, never a floating-point ratio comparison or
        # an over-budget state.
        "budget_reached": exact_budget_reached,
        "strict_budget_boundary_stop": strict_budget_boundary_stop,
        "max_steps_hit": max_steps_hit,
        "valid_lbi_step": strict_budget_boundary_stop,
    }


def unavailable_lbi_budget_diagnostics(reason):
    diagnostics = {
        field: None for field in RUN_BUDGET_DIAGNOSTIC_FIELDS
    }
    diagnostics.update(
        {
            "budget_diagnostics_available": False,
            "budget_diagnostics_source": None,
            "budget_diagnostics_unavailable_reason": str(reason),
        }
    )
    return diagnostics


def _series_statistics(values, field):
    return {
        f"{field}_first": values[0],
        f"{field}_last": values[-1],
        f"{field}_min": min(values),
        f"{field}_max": max(values),
        f"{field}_mean": sum(values) / len(values),
    }


def compute_lbi_run_budget_diagnostics(step_records):
    """Summarize exact per-step support without inferring from means."""
    steps = list(step_records)
    if not steps:
        return unavailable_lbi_budget_diagnostics(
            "no_online_step_records"
        )

    required = {
        "requested_budget",
        "candidate_scope_param_count",
        "stage1_support_count",
        "stage1_stop_reason",
        "stage1_steps_completed",
    }
    annotated = []
    requested_budgets = set()
    candidate_counts = set()
    for index, step in enumerate(steps):
        missing = sorted(required - set(step))
        if missing:
            return unavailable_lbi_budget_diagnostics(
                f"missing_step_fields_at_{index}: {missing}"
            )
        try:
            requested_budget = float(step["requested_budget"])
            candidate_count = int(step["candidate_scope_param_count"])
            support_count = int(step["stage1_support_count"])
            steps_completed = int(step["stage1_steps_completed"])
        except (TypeError, ValueError) as error:
            return unavailable_lbi_budget_diagnostics(
                f"invalid_step_values_at_{index}: {error}"
            )
        requested_budgets.add(requested_budget)
        candidate_counts.add(candidate_count)
        step_diagnostics = compute_lbi_step_budget_diagnostics(
            requested_budget,
            candidate_count,
            support_count,
            step["stage1_stop_reason"],
        )
        annotated.append(
            {
                **step_diagnostics,
                "stage1_support_count": support_count,
                "stage1_steps_completed": steps_completed,
                "stage1_stop_reason": str(
                    step["stage1_stop_reason"]
                ),
            }
        )

    if len(requested_budgets) != 1:
        return unavailable_lbi_budget_diagnostics(
            "inconsistent_requested_budget_across_steps"
        )
    if len(candidate_counts) != 1:
        return unavailable_lbi_budget_diagnostics(
            "inconsistent_candidate_scope_param_count_across_steps"
        )

    online_steps = len(annotated)
    exact_budget_hit_count = sum(
        int(step["budget_reached"]) for step in annotated
    )
    valid_step_count = sum(
        int(step["valid_lbi_step"]) for step in annotated
    )
    max_steps_hit_count = sum(
        int(step["max_steps_hit"]) for step in annotated
    )
    underfilled_step_count = sum(
        step["stage1_support_count"]
        < step["target_support_count"]
        for step in annotated
    )
    stop_reason_counts = {
        reason: 0 for reason in STANDARD_STOP_REASONS
    }
    for step in annotated:
        reason = _canonical_stop_reason(
            step["stage1_stop_reason"]
        )
        stop_reason_counts[reason] = (
            stop_reason_counts.get(reason, 0) + 1
        )

    exact_budget_reached_all_steps = (
        exact_budget_hit_count == online_steps
    )
    strict_budget_boundary_all_steps = all(
        step["strict_budget_boundary_stop"] for step in annotated
    )
    # This legacy aggregate is used by search/finalize tools.  Preserve its
    # role as an all-steps budget-constrained termination flag, including a
    # legal strict-budget rollback whose final support is below K.
    budget_reached_all_steps = all(
        step["budget_reached"] or step["strict_budget_boundary_stop"]
        for step in annotated
    )
    diagnostics = {
        "budget_diagnostics_available": True,
        "budget_diagnostics_source": None,
        "budget_diagnostics_unavailable_reason": None,
        "max_support_count": annotated[0]["max_support_count"],
        "target_support_count": annotated[0][
            "target_support_count"
        ],
        "exact_budget_hit_count": exact_budget_hit_count,
        "exact_budget_reached_all_steps": exact_budget_reached_all_steps,
        "strict_budget_boundary_all_steps": strict_budget_boundary_all_steps,
        "budget_hit_count": exact_budget_hit_count,
        "budget_hit_rate": exact_budget_hit_count / online_steps,
        "budget_reached_all_steps": budget_reached_all_steps,
        "valid_lbi_step_count": valid_step_count,
        "valid_lbi_step_rate": valid_step_count / online_steps,
        "valid_lbi_run": (
            online_steps > 0
            and budget_reached_all_steps
            and max_steps_hit_count == 0
        ),
        "underfilled_step_count": int(underfilled_step_count),
        "max_steps_hit_count": max_steps_hit_count,
        "max_steps_hit_rate": max_steps_hit_count / online_steps,
        "stage1_stop_reason_counts": stop_reason_counts,
    }
    diagnostics.update(
        _series_statistics(
            [step["support_gap_count"] for step in annotated],
            "support_gap_count",
        )
    )
    diagnostics.update(
        _series_statistics(
            [step["stage1_support_count"] for step in annotated],
            "stage1_support_count",
        )
    )
    diagnostics.update(
        _series_statistics(
            [step["stage1_steps_completed"] for step in annotated],
            "stage1_steps_completed",
        )
    )
    return diagnostics
