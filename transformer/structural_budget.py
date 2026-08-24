"""Shared formatting helpers for structural-group budget ratios."""

from __future__ import annotations


def format_structural_budget(value: float) -> str:
    """Format rho exactly enough for artifacts, with at least three decimals."""
    text = f"{float(value):.12f}".rstrip("0").rstrip(".")
    whole, separator, fraction = text.partition(".")
    if not separator:
        fraction = ""
    return f"{whole}.{fraction.ljust(3, '0')}"
