"""Shared refined Split-LBI core for SHOT-OTTA FC and Conv variants."""

from .engine import SplitLBIEngine
from .diagnostics import (
    RUN_BUDGET_DIAGNOSTIC_FIELDS,
    STEP_BUDGET_DIAGNOSTIC_FIELDS,
    compute_lbi_run_budget_diagnostics,
    compute_lbi_step_budget_diagnostics,
    max_support_count,
    target_support_count,
    unavailable_lbi_budget_diagnostics,
)
from .groups import CONV_GROUP_MODES
from .state import LBIResult, LBIState

__all__ = [
    "LBIResult",
    "LBIState",
    "RUN_BUDGET_DIAGNOSTIC_FIELDS",
    "STEP_BUDGET_DIAGNOSTIC_FIELDS",
    "SplitLBIEngine",
    "compute_lbi_run_budget_diagnostics",
    "compute_lbi_step_budget_diagnostics",
    "CONV_GROUP_MODES",
    "max_support_count",
    "target_support_count",
    "unavailable_lbi_budget_diagnostics",
]
