"""Configuration, profiles, budgets, and identities for Group Split-LBI."""

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


PROTOCOL_REVISION = "transformer_group_lbi_otta_20260824_v2"
FORMAL_BUDGETS = (0.0005, 0.001, 0.002)
TOTAL_GROUPS = 6_912
GROUP_SIZE = 768
BUDGET_TO_K = {budget: math.floor(budget * TOTAL_GROUPS) for budget in FORMAL_BUDGETS}
PROFILE_FIELDS = (
    "alpha",
    "kappa",
    "nu",
    "omega",
    "prox_lambda",
    "tau_g",
    "stage1_max_steps",
    "stage2_lr",
)


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


def budget_tag(budget: float | str) -> str:
    value = normalize_budget(budget)
    return f"rho-{format_structural_budget(value)}"


def budget_key(budget: float | str) -> str:
    return format_structural_budget(normalize_budget(budget))


def profile_budget_key(budget: float | str) -> str:
    return budget_key(budget)


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


def lbi_cli_overrides(args) -> dict[str, Any]:
    mapping = {
        "alpha": "lbi_alpha",
        "kappa": "lbi_kappa",
        "nu": "lbi_nu",
        "omega": "lbi_omega",
        "prox_lambda": "lbi_prox_lambda",
        "tau_g": "lbi_tau_g",
        "stage1_max_steps": "lbi_stage1_max_steps",
        "stage2_lr": "lbi_stage2_lr",
    }
    return {
        key: getattr(args, argument, None)
        for key, argument in mapping.items()
        if getattr(args, argument, None) is not None
    }


def validate_lbi_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise ValueError("LBI profile must be a mapping")
    missing = sorted(set(PROFILE_FIELDS) - set(profile))
    if missing:
        raise ValueError(f"LBI profile is missing fields: {missing}")
    extra = sorted(set(profile) - set(PROFILE_FIELDS))
    if extra:
        raise ValueError(f"LBI profile has unsupported fields: {extra}")
    resolved = copy.deepcopy(profile)
    for key in (
        "alpha",
        "kappa",
        "nu",
        "omega",
        "prox_lambda",
        "tau_g",
        "stage2_lr",
    ):
        value = float(resolved[key])
        if not math.isfinite(value):
            raise ValueError(f"lbi.{key} must be finite")
        resolved[key] = value
    if resolved["alpha"] <= 0 or resolved["kappa"] <= 0:
        raise ValueError("lbi.alpha and lbi.kappa must be > 0")
    if resolved["nu"] <= 0 or resolved["prox_lambda"] <= 0:
        raise ValueError("lbi.nu and lbi.prox_lambda must be > 0")
    if not 0.0 <= resolved["omega"] <= 1.0:
        raise ValueError("lbi.omega must be in [0, 1]")
    if resolved["tau_g"] <= 0:
        raise ValueError("lbi.tau_g must be > 0")
    if resolved["stage2_lr"] <= 0:
        raise ValueError("lbi.stage2_lr must be > 0")
    steps = resolved["stage1_max_steps"]
    if isinstance(steps, bool) or int(steps) != steps or int(steps) <= 0:
        raise ValueError("lbi.stage1_max_steps must be a positive integer")
    resolved["stage1_max_steps"] = int(steps)
    return resolved


def resolve_lbi_profile(
    config: dict[str, Any],
    dataset: str,
    budget: float | str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profiles = config.get("lbi_profiles", {})
    dataset_profiles = profiles.get(dataset)
    if not isinstance(dataset_profiles, dict):
        raise ValueError(f"lbi_profiles.{dataset} must be a mapping")
    key = profile_budget_key(budget)
    profile = dataset_profiles.get(key)
    resolved_profile_key = key
    if not isinstance(profile, dict):
        profile = dataset_profiles.get("default")
        resolved_profile_key = "default"
    if not isinstance(profile, dict):
        raise ValueError(
            f"Missing LBI profile for {dataset} budget {key} and no default profile exists"
        )
    values = copy.deepcopy(profile.get("parameters"))
    if not isinstance(values, dict):
        raise ValueError(f"LBI profile {dataset}/{key} is missing parameters")
    values.update(overrides or {})
    resolved = {
        "status": str(profile.get("status", "unresolved")),
        "profile_key": f"{dataset}/{resolved_profile_key}",
        **validate_lbi_profile(values),
    }
    if resolved_profile_key == "default":
        resolved["requested_rho_key"] = key
    return resolved


def _candidate_compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    candidate = copy.deepcopy(config)
    candidate["protocol_revision"] = CANDIDATE_DENSE_PROTOCOL_REVISION
    candidate["variant"] = "candidate_dense"
    candidate.pop("selection", None)
    candidate.pop("lbi_profiles", None)
    candidate.pop("stage2_optimization", None)
    candidate.pop("checkpointing", None)
    return candidate


def _validate_frozen_fields(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("group-lbi config schema_version must be 1")
    if config.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError(f"protocol_revision must be {PROTOCOL_REVISION}")
    if config.get("formal_seed") != FORMAL_SEED:
        raise ValueError(f"formal_seed must be {FORMAL_SEED}")
    if config.get("method") != "shot" or config.get("variant") != "group_lbi":
        raise ValueError("method/variant must be shot/group_lbi")
    expected_selection = {
        "type": "group_split_lbi",
        "group_order": "block_then_qk_vo_ffn_then_coordinate",
        "total_groups": TOTAL_GROUPS,
        "group_size": GROUP_SIZE,
        "budgets": list(FORMAL_BUDGETS),
        "integer_rule": "floor",
        "integer_budgets": [BUDGET_TO_K[item] for item in FORMAL_BUDGETS],
        "support_measure": "gamma_group_l2_div_sqrt_group_size",
        "strict_rollback": True,
        "topk_trim": False,
        "budget_slack": False,
        "state_lifecycle": "theta_delta_gamma_z_reset_every_online_batch",
        "stage2_initialization": "base_plus_masked_delta",
    }
    if config.get("selection") != expected_selection:
        raise ValueError(f"selection must be exactly {expected_selection}")
    expected_stage2 = {
        "optimizer": "adamw",
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.01,
        "steps": 1,
        "lr_schedule": "none",
        "strict_off_mask_value_freezing": True,
        "clear_off_mask_coordinate_state": ["exp_avg", "exp_avg_sq"],
    }
    if config.get("stage2_optimization") != expected_stage2:
        raise ValueError(f"stage2_optimization must be exactly {expected_stage2}")
    expected_checkpointing = {
        "enabled_by_default": True,
        "granularity": "completed_online_batch",
        "save_adapted_model": False,
    }
    if config.get("checkpointing") != expected_checkpointing:
        raise ValueError(f"checkpointing must be exactly {expected_checkpointing}")
    for dataset in ("office31", "visda-c"):
        for budget in FORMAL_BUDGETS:
            resolve_lbi_profile(config, dataset, budget)
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
    lbi_overrides: dict[str, Any] | None = None,
    stream_checkpoint: bool | None = None,
) -> dict[str, Any]:
    """Resolve one Group-LBI condition and verify its local assets."""
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
    lbi = resolve_lbi_profile(raw_config, dataset, normalized_budget, lbi_overrides)
    checkpoint_enabled = (
        raw_config["checkpointing"]["enabled_by_default"]
        if stream_checkpoint is None
        else bool(stream_checkpoint)
    )
    resolved["protocol_revision"] = PROTOCOL_REVISION
    resolved["variant"] = "group_lbi"
    resolved["selection"] = {
        **copy.deepcopy(raw_config["selection"]),
        "requested_budget": normalized_budget,
        "requested_group_count": budget_group_count(normalized_budget),
        "maximum_active_candidate_scalars": budget_group_count(normalized_budget)
        * GROUP_SIZE,
    }
    resolved["lbi"] = lbi
    resolved["stage2_optimization"] = copy.deepcopy(raw_config["stage2_optimization"])
    resolved["checkpointing"] = {
        **copy.deepcopy(raw_config["checkpointing"]),
        "enabled": checkpoint_enabled,
    }
    resolved["diagnostics"] = copy.deepcopy(raw_config["diagnostics"])
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
            "checkpointing",
            "diagnostics",
        }
    }
    resolved["scientific_config_sha256"] = canonical_sha256(scientific)
    resolved["experiment_key"] = (
        f"{dataset}_{source}-{target}_group-lbi_{budget_tag(normalized_budget)}_"
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
