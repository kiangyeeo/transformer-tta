"""Fail-closed configuration for COME sparse/LBI variants."""

import copy
import math

from experiment_identity import resolve_experiment_identity
from protocol_constants import (
    COME_IMPLEMENTATION_REVISION,
    COME_LBI_FROZEN_CONSTANTS,
    COME_LBI_IMPLEMENTATION_REVISION,
    COME_LBI_PROTOCOL_REVISION,
    COME_PROTOCOL_REVISION,
    FORMAL_BUDGETS,
)

from .sparse import LBI_VARIANTS, RANDOM_VARIANTS, SPARSE_VARIANTS, variant_track


LBI_TUNABLES = {"alpha", "kappa", "nu", "omega", "stage2_lr"}


def apply_sparse_overrides(config, args, dense_apply_overrides):
    candidate_variant = getattr(args, "variant", None) or config.get("variant")
    if candidate_variant not in SPARSE_VARIANTS:
        return dense_apply_overrides(config, args)
    effective = copy.deepcopy(config)
    direct = {
        ("data", "dataset"): getattr(args, "dataset", None),
        ("data", "source"): getattr(args, "source", None),
        ("data", "target"): getattr(args, "target", None),
        ("data", "batch_size"): getattr(args, "batch_size", None),
        ("data", "workers"): getattr(args, "workers", None),
        ("data", "root"): getattr(args, "data_root", None),
        ("source_checkpoint", "root"): getattr(args, "source_checkpoint_root", None),
        ("output", "root"): getattr(args, "output_root", None),
        ("output", "run_name"): getattr(args, "run_name", None),
        ("device", "gpu_id"): getattr(args, "gpu_id", None),
    }
    for (section, key), value in direct.items():
        if value is not None:
            effective[section][key] = value
    for key, value in {
        "seed": getattr(args, "seed", None),
        "variant": getattr(args, "variant", None),
        "requested_budget": getattr(args, "requested_budget", None),
        "selection_seed": getattr(args, "selection_seed", None),
        "num_random_masks": getattr(args, "num_random_masks", None),
        "experiment_key": getattr(args, "experiment_key", None),
        "experiment_config_sha256": getattr(args, "experiment_config_sha256", None),
    }.items():
        if value is not None:
            effective[key] = value
    supplied_group_mode = getattr(args, "group_mode", None)
    if supplied_group_mode == "filter_connection":
        raise ValueError("COME sparse family makes filter_connection inaccessible")
    if supplied_group_mode is not None:
        effective["group_mode"] = supplied_group_mode
    runtime = effective.setdefault("runtime", {})
    if getattr(args, "workers_per_gpu", None) is not None:
        runtime["workers_per_gpu"] = args.workers_per_gpu
    if getattr(args, "runtime_comparable", False):
        runtime["runtime_comparable"] = True
    if getattr(args, "debug_max_outer_batches", None) is not None:
        runtime["debug_max_outer_batches"] = args.debug_max_outer_batches
    if getattr(args, "save_model", False):
        effective["output"]["save_model"] = True
    elif getattr(args, "no_save_model", False):
        effective["output"]["save_model"] = False
    lbi_values = {
        "alpha": getattr(args, "lbi_alpha", None),
        "kappa": getattr(args, "lbi_kappa", None),
        "nu": getattr(args, "lbi_nu", None),
        "omega": getattr(args, "lbi_omega", None),
        "stage2_lr": getattr(args, "lbi_stage2_lr", None),
    }
    if any(value is not None for value in lbi_values.values()):
        effective.setdefault("lbi", {})
        for key, value in lbi_values.items():
            if value is not None:
                effective["lbi"][key] = value
    frozen_overrides = {
        "lbi_stage1_max_steps": getattr(args, "lbi_stage1_max_steps", None),
        "lbi_budget_tolerance": getattr(args, "lbi_budget_tolerance", None),
        "lbi_stage2_steps": getattr(args, "lbi_stage2_steps", None),
        "lbi_delta_nonzero_tolerance": getattr(
            args, "lbi_delta_nonzero_tolerance", None
        ),
        "lbi_support_threshold": getattr(args, "lbi_support_threshold", None),
    }
    supplied = sorted(
        key for key, value in frozen_overrides.items() if value is not None
    )
    if supplied:
        raise ValueError(
            "COME frozen LBI constants reject overrides: " + ", ".join(supplied)
        )
    return effective


def _budget(value):
    if value is None or isinstance(value, bool):
        raise ValueError("COME sparse requested_budget must be explicit")
    value = float(value)
    if not math.isfinite(value) or not any(
        abs(value - item) <= 1.0e-12 for item in FORMAL_BUDGETS
    ):
        raise ValueError("COME sparse requested_budget must be 0.0005, 0.001, or 0.002")
    return value


def _lbi(lbi):
    if not isinstance(lbi, dict) or set(lbi) != LBI_TUNABLES:
        raise ValueError("COME-LBI requires only lbi.alpha/kappa/nu/omega/stage2_lr")
    result = {}
    for key, value in lbi.items():
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"lbi.{key} must be finite")
        result[key] = float(value)
    for key in ("alpha", "kappa", "nu", "stage2_lr"):
        if result[key] <= 0:
            raise ValueError(f"lbi.{key} must be > 0")
    if not 0.0 <= result["omega"] <= 1.0:
        raise ValueError("lbi.omega must be in [0, 1]")
    return result


def resolve_sparse_config(config, workspace_root, dense_resolver):
    if config.get("variant") not in SPARSE_VARIANTS:
        return dense_resolver(config, workspace_root)
    variant = config["variant"]
    formal = config.get("formal_protocol")
    if not isinstance(formal, bool):
        raise ValueError("COME sparse formal_protocol must be boolean")
    if formal and variant in LBI_VARIANTS:
        raise ValueError(
            "formal COME-LBI launch remains blocked until search protocol "
            "and tuned tuples are frozen"
        )
    track = variant_track(variant)
    budget = _budget(config.get("requested_budget"))
    group_mode = config.get("group_mode")
    if track == "conv_out_channel":
        if group_mode not in {None, "out_channel"}:
            raise ValueError("COME Conv sparse family supports out_channel only")
        group_mode = "out_channel"
    elif group_mode is not None:
        raise ValueError("COME FC sparse family rejects group_mode")

    if variant in RANDOM_VARIANTS:
        selection_seed = config.get("selection_seed", 202600)
        selection_seed = 202600 if selection_seed is None else selection_seed
        if isinstance(selection_seed, bool) or int(selection_seed) != 202600:
            raise ValueError("COME Random child seeds are 202600/202601/202602")
        if config.get("num_random_masks") not in {None, 3}:
            raise ValueError("COME Random requires exactly 3 child trajectories")
        num_random_masks = 3
    else:
        selection_seed = None
        num_random_masks = None

    if variant in LBI_VARIANTS:
        lbi = _lbi(config.get("lbi"))
        lbi_runtime = {
            **lbi,
            **COME_LBI_FROZEN_CONSTANTS,
            "requested_budget": budget,
            **({"group_mode": "out_channel"} if track == "conv_out_channel" else {}),
        }
    else:
        if config.get("lbi") is not None:
            raise ValueError("non-LBI COME variants reject the lbi namespace")
        lbi = None
        lbi_runtime = None

    surrogate = copy.deepcopy(config)
    surrogate.update(
        {
            "protocol_track": "come_baseline",
            "protocol_revision": COME_PROTOCOL_REVISION,
            "implementation_revision": COME_IMPLEMENTATION_REVISION,
            "variant": "come_conv_module_dense"
            if track == "conv_out_channel"
            else "come_fc_module_dense",
            "requested_budget": 1.0,
            "group_mode": None,
            "lbi": None,
            "selection_seed": None,
            "num_random_masks": None,
            "experiment_key": None,
            "experiment_config_sha256": None,
        }
    )
    effective = dense_resolver(surrogate, workspace_root)
    effective.update(
        {
            "protocol_track": "come_sparse_lbi",
            "variant": variant,
            "requested_budget": budget,
            "group_mode": group_mode,
            "selection_seed": selection_seed,
            "num_random_masks": num_random_masks,
            "lbi": lbi,
            "lbi_runtime": lbi_runtime,
            "protocol_revision": COME_LBI_PROTOCOL_REVISION,
            "implementation_revision": COME_LBI_IMPLEMENTATION_REVISION,
        }
    )
    identity = resolve_experiment_identity(
        effective,
        provided_key=config.get("experiment_key"),
        provided_sha256=config.get("experiment_config_sha256"),
    )
    effective["experiment_key"] = identity["experiment_key"]
    effective["experiment_config_sha256"] = identity["experiment_config_sha256"]
    effective["scientific_config"] = identity["scientific_config"]
    return effective
