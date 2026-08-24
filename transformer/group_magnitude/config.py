"""Frozen configuration, budgets, and identities for Group-Magnitude SHOT-OTTA."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any, Iterable

from transformer.candidate_dense.config import (
    PROTOCOL_REVISION as CANDIDATE_DENSE_PROTOCOL_REVISION,
    _validate_frozen_fields as _validate_candidate_dense_fields,
    load_config,
    resolve_transfer_config as resolve_candidate_dense_transfer,
)
from transformer.source_only.config import FORMAL_SEED, TRANSFERS, canonical_sha256
from transformer.structural_budget import format_structural_budget


PROTOCOL_REVISION = "transformer_group_magnitude_otta_20260824_v2"
FORMAL_BUDGETS = (0.0005, 0.001, 0.002)
TOTAL_GROUPS = 6_912
GROUP_SIZE = 768
BUDGET_TO_K = {budget: math.floor(budget * TOTAL_GROUPS) for budget in FORMAL_BUDGETS}


def budget_tag(budget: float) -> str:
    value = normalize_budget(budget)
    return f"rho-{format_structural_budget(value)}"


def budget_key(budget: float | str) -> str:
    return format_structural_budget(normalize_budget(budget))


def normalize_budget(value: float | str) -> float:
    parsed = float(value)
    for budget in FORMAL_BUDGETS:
        if math.isclose(parsed, budget, rel_tol=0.0, abs_tol=1.0e-12):
            return budget
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed > 1.0:
        raise ValueError("rho must be finite and in the interval (0, 1]")
    if math.floor(parsed * TOTAL_GROUPS) < 1:
        raise ValueError(
            f"rho={parsed:.12g} selects zero groups; rho must be at least "
            f"1/{TOTAL_GROUPS}"
        )
    return parsed


def parse_budgets(selection: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(selection, str):
        if selection.strip().lower() == "all":
            return FORMAL_BUDGETS
        values = [item.strip() for item in selection.split(",") if item.strip()]
    else:
        values = list(selection)
    budgets = tuple(normalize_budget(item) for item in values)
    if not budgets:
        raise ValueError("At least one formal budget is required")
    if len(set(budgets)) != len(budgets):
        raise ValueError("Budgets must be unique")
    return budgets


def budget_group_count(budget: float | str) -> int:
    return math.floor(normalize_budget(budget) * TOTAL_GROUPS)


def _candidate_compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    candidate = copy.deepcopy(config)
    candidate["protocol_revision"] = CANDIDATE_DENSE_PROTOCOL_REVISION
    candidate["variant"] = "candidate_dense"
    candidate.pop("selection", None)
    return candidate


def _validate_frozen_fields(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("group-magnitude config schema_version must be 1")
    if config.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError(f"protocol_revision must be {PROTOCOL_REVISION}")
    if config.get("formal_seed") != FORMAL_SEED:
        raise ValueError(f"formal_seed must be {FORMAL_SEED}")
    if config.get("method") != "shot" or config.get("variant") != "group_magnitude":
        raise ValueError("method/variant must be shot/group_magnitude")
    expected_selection = {
        "type": "source_w0_structural_group_l2",
        "score": "paired_group_l2_norm",
        "score_device": "cpu",
        "score_dtype": "float64",
        "tie_break": "ascending_canonical_group_id",
        "group_order": "block_then_qk_vo_ffn_then_coordinate",
        "total_groups": TOTAL_GROUPS,
        "group_size": GROUP_SIZE,
        "budgets": list(FORMAL_BUDGETS),
        "integer_rule": "floor",
        "integer_budgets": [BUDGET_TO_K[item] for item in FORMAL_BUDGETS],
        "ranking_source": "source_checkpoint_pre_adaptation",
        "mask_refresh_policy": "once_before_adaptation",
    }
    if config.get("selection") != expected_selection:
        raise ValueError(f"selection must be exactly {expected_selection}")
    _validate_candidate_dense_fields(_candidate_compatible_config(config))


def resolve_transfer_config(
    raw_config: dict[str, Any],
    *,
    project_root: Path,
    dataset: str,
    source: str,
    target: str,
    budget: float | str,
    device: str,
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Resolve one Magnitude condition and verify all local assets."""
    _validate_frozen_fields(raw_config)
    normalized_budget = normalize_budget(budget)
    resolved = resolve_candidate_dense_transfer(
        _candidate_compatible_config(raw_config),
        project_root=project_root,
        dataset=dataset,
        source=source,
        target=target,
        device=device,
        output_dir=output_dir,
    )
    resolved["protocol_revision"] = PROTOCOL_REVISION
    resolved["variant"] = "group_magnitude"
    resolved["selection"] = {
        **copy.deepcopy(raw_config["selection"]),
        "requested_budget": normalized_budget,
        "requested_group_count": budget_group_count(normalized_budget),
        "active_candidate_scalars": budget_group_count(normalized_budget) * GROUP_SIZE,
    }
    resolved["output_dir"] = str(Path(output_dir).resolve())
    scientific = {
        key: value
        for key, value in resolved.items()
        if key
        not in {
            "output_dir",
            "checkpoint_manifest_path",
            "scientific_config_sha256",
            "experiment_key",
        }
    }
    resolved["scientific_config_sha256"] = canonical_sha256(scientific)
    resolved["experiment_key"] = (
        f"{dataset}_{source}-{target}_group-magnitude_{budget_tag(normalized_budget)}_"
        f"seed-{FORMAL_SEED}_{resolved['scientific_config_sha256'][:12]}"
    )
    return resolved


def select_transfers(selection: str) -> tuple[tuple[str, str, str], ...]:
    if selection == "all":
        return TRANSFERS
    if selection == "office31":
        return tuple(item for item in TRANSFERS if item[0] == "office31")
    if selection == "visda-c":
        return tuple(item for item in TRANSFERS if item[0] == "visda-c")
    raise ValueError("selection must be all, office31, or visda-c")
