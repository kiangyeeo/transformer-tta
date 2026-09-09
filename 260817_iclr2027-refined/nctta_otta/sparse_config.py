"""Fail-closed configuration layer for NCTTA sparse/LBI variants."""

import copy
import math

from experiment_identity import resolve_experiment_identity
from protocol_constants import (
    FORMAL_BUDGETS,
    NCTTA_LBI_FROZEN_CONSTANTS,
    NCTTA_LBI_IMPLEMENTATION_REVISION,
    NCTTA_LBI_PROTOCOL_REVISION,
)

from .sparse import LBI_VARIANTS, RANDOM_VARIANTS, SPARSE_VARIANTS, variant_track


LBI_TUNABLES = {"alpha", "kappa", "nu", "omega", "stage2_lr"}


def apply_sparse_overrides(config, args, dense_apply_overrides):
    if (
        config.get("variant") not in SPARSE_VARIANTS
        and getattr(args, "variant", None) not in SPARSE_VARIANTS
    ):
        return dense_apply_overrides(config, args)
    effective = copy.deepcopy(config)
    direct = {
        ("data", "dataset"): args.dataset,
        ("data", "source"): args.source,
        ("data", "target"): args.target,
        ("data", "batch_size"): args.batch_size,
        ("data", "workers"): args.workers,
        ("data", "root"): args.data_root,
        ("source_checkpoint", "root"): args.source_checkpoint_root,
        ("output", "root"): args.output_root,
        ("output", "run_name"): args.run_name,
        ("device", "gpu_id"): args.gpu_id,
    }
    for (section, key), value in direct.items():
        if value is not None:
            effective[section][key] = value
    for key, value in {
        "seed": args.seed,
        "variant": args.variant,
        "group_mode": args.group_mode,
        "requested_budget": args.requested_budget,
        "selection_seed": args.selection_seed,
        "num_random_masks": args.num_random_masks,
        "experiment_key": args.experiment_key,
        "experiment_config_sha256": args.experiment_config_sha256,
    }.items():
        if value is not None:
            effective[key] = value
    runtime = effective.setdefault("runtime", {})
    if getattr(args, "workers_per_gpu", None) is not None:
        runtime["workers_per_gpu"] = args.workers_per_gpu
    if getattr(args, "runtime_comparable", False):
        runtime["runtime_comparable"] = True
    if getattr(args, "debug_max_outer_batches", None) is not None:
        runtime["debug_max_outer_batches"] = args.debug_max_outer_batches
    if args.save_model:
        effective["output"]["save_model"] = True
    elif args.no_save_model:
        effective["output"]["save_model"] = False
    lbi_values = {
        "alpha": args.lbi_alpha,
        "kappa": args.lbi_kappa,
        "nu": args.lbi_nu,
        "omega": args.lbi_omega,
        "stage2_lr": args.lbi_stage2_lr,
    }
    if any(value is not None for value in lbi_values.values()):
        if effective.get("lbi") is None:
            effective["lbi"] = {}
        for key, value in lbi_values.items():
            if value is not None:
                effective["lbi"][key] = value
    frozen_overrides = {
        "lbi_stage1_max_steps": args.lbi_stage1_max_steps,
        "lbi_budget_tolerance": args.lbi_budget_tolerance,
        "lbi_stage2_steps": args.lbi_stage2_steps,
        "lbi_delta_nonzero_tolerance": args.lbi_delta_nonzero_tolerance,
        "lbi_support_threshold": args.lbi_support_threshold,
    }
    supplied = sorted(
        key for key, value in frozen_overrides.items() if value is not None
    )
    if supplied:
        raise ValueError(
            "NCTTA frozen LBI constants reject overrides: " + ", ".join(supplied)
        )
    return effective


def _budget(value):
    if value is None or isinstance(value, bool):
        raise ValueError("NCTTA sparse requested_budget must be explicit")
    value = float(value)
    if not any(abs(value - item) <= 1.0e-12 for item in FORMAL_BUDGETS):
        raise ValueError(
            "NCTTA sparse requested_budget must be 0.0005, 0.001, or 0.002"
        )
    return value


def _lbi(lbi):
    if not isinstance(lbi, dict) or set(lbi) != LBI_TUNABLES:
        raise ValueError("NCTTA-LBI requires only lbi.alpha/kappa/nu/omega/stage2_lr")
    out = {}
    for key, value in lbi.items():
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"lbi.{key} must be finite")
        out[key] = float(value)
    for key in ("alpha", "kappa", "nu", "stage2_lr"):
        if out[key] <= 0:
            raise ValueError(f"lbi.{key} must be > 0")
    if not 0 <= out["omega"] <= 1:
        raise ValueError("lbi.omega must be in [0, 1]")
    return out


def resolve_sparse_config(config, workspace_root, dense_resolver):
    if config.get("variant") not in SPARSE_VARIANTS:
        return dense_resolver(config, workspace_root)
    if config.get("formal_protocol"):
        raise ValueError("formal NCTTA sparse/LBI launch remains blocked")
    variant = config["variant"]
    track = variant_track(variant)
    budget = _budget(config.get("requested_budget"))
    group_mode = config.get("group_mode")
    if track == "conv_out_channel":
        if group_mode not in {None, "out_channel"}:
            raise ValueError("NCTTA Conv sparse family supports out_channel only")
        group_mode = "out_channel"
    elif group_mode is not None:
        raise ValueError("NCTTA FC sparse family rejects group_mode")
    if variant in RANDOM_VARIANTS:
        selection_seed = config.get("selection_seed")
        selection_seed = 202600 if selection_seed is None else selection_seed
        if isinstance(selection_seed, bool) or int(selection_seed) != 202600:
            raise ValueError(
                "NCTTA Random child seeds are frozen to 202600/202601/202602"
            )
        if config.get("num_random_masks") not in {None, 3}:
            raise ValueError("NCTTA Random requires exactly 3 child trajectories")
        num_random_masks = 3
    else:
        selection_seed = None
        num_random_masks = None
    if variant in LBI_VARIANTS:
        lbi = _lbi(config.get("lbi"))
        lbi_runtime = {
            **lbi,
            **NCTTA_LBI_FROZEN_CONSTANTS,
            "requested_budget": budget,
            **({"group_mode": "out_channel"} if track == "conv_out_channel" else {}),
        }
    else:
        if config.get("lbi") is not None:
            raise ValueError("non-LBI NCTTA variants reject lbi namespace")
        lbi = None
        lbi_runtime = None

    # Reuse the audited dense resolver for all common NCTTA/data/optimizer
    # invariants, then restore the sparse scientific fields and re-hash them.
    surrogate = copy.deepcopy(config)
    surrogate.update(
        {
            "variant": "nctta_conv_module_dense"
            if track == "conv_out_channel"
            else "nctta_fc_module_dense",
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
            "variant": variant,
            "requested_budget": budget,
            "group_mode": group_mode,
            "selection_seed": selection_seed,
            "num_random_masks": num_random_masks,
            "lbi": lbi,
            "lbi_runtime": lbi_runtime,
            "protocol_revision": NCTTA_LBI_PROTOCOL_REVISION,
            "implementation_revision": NCTTA_LBI_IMPLEMENTATION_REVISION,
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
