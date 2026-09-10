"""Structural-group budget helpers, imported from the SHOT substrate.

Re-exported rather than reimplemented so that ``floor(rho * 6912)`` -> 3/6/13
and the rho tags used in artifact paths are literally the same code objects as
the SHOT-Transformer ones (no slack, no ceil, no per-block quota).
"""

from transformer.group_random.config import (
    BUDGET_TO_K,
    FORMAL_BUDGETS,
    GROUP_SIZE,
    MASK_SEEDS,
    NUM_RANDOM_MASKS,
    TOTAL_GROUPS,
    budget_group_count,
    budget_key,
    budget_tag,
    normalize_budget,
    parse_budgets,
)

__all__ = [
    "BUDGET_TO_K",
    "FORMAL_BUDGETS",
    "GROUP_SIZE",
    "MASK_SEEDS",
    "NUM_RANDOM_MASKS",
    "TOTAL_GROUPS",
    "budget_group_count",
    "budget_key",
    "budget_tag",
    "normalize_budget",
    "parse_budgets",
]
