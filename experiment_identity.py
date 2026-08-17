"""Stable identity for result-affecting SHOT-OTTA experiment settings."""

import copy
import hashlib
import json

from shot_otta.backbones.deit import candidate_parameter_names


GROUP_RANDOM_PROTOCOL = "sequential_target_stream_group_random_adaptation"


SUPPORTED_VARIANTS = {
    "source_only",
    "full_dense",
    "module_dense",
    "module_random",
    "module_magnitude",
    "module_saliency",
    "module_lbi",
}


VARIANT_METADATA = {
    "source_only": {
        "selection": "none",
        "candidate_scope": "none",
        "bn_stats_policy": "frozen",
        "bn_stats_frozen": True,
        "mask_static": False,
        "mask_refresh_policy": None,
    },
    "full_dense": {
        "selection": "dense",
        "candidate_scope": "netF+netB",
        "bn_stats_policy": "adaptive",
        "bn_stats_frozen": False,
        "mask_static": False,
        "mask_refresh_policy": None,
    },
    "module_dense": {
        "selection": "dense",
        "candidate_scope": "netB.bottleneck",
        "bn_stats_policy": "frozen",
        "bn_stats_frozen": True,
        "mask_static": False,
        "mask_refresh_policy": None,
    },
    "module_random": {
        "selection": "random",
        "candidate_scope": "netB.bottleneck",
        "bn_stats_policy": "frozen",
        "bn_stats_frozen": True,
        "mask_static": True,
        "mask_refresh_policy": "once_before_adaptation",
        "ranking_source": "independent_random_generator",
    },
    "module_magnitude": {
        "selection": "magnitude",
        "candidate_scope": "netB.bottleneck",
        "bn_stats_policy": "frozen",
        "bn_stats_frozen": True,
        "mask_static": True,
        "mask_refresh_policy": "once_before_adaptation",
        "ranking_source": "source_checkpoint_pre_adaptation",
    },
    "module_saliency": {
        "selection": "saliency",
        "candidate_scope": "netB.bottleneck",
        "bn_stats_policy": "frozen",
        "bn_stats_frozen": True,
        "mask_static": False,
        "mask_refresh_policy": "every_online_step_after_backward",
        "ranking_source": "current_parameter_times_current_gradient",
        "saliency_score": "abs_parameter_times_gradient",
    },
    "module_lbi": {
        "selection": "lbi",
        "candidate_scope": "netB.bottleneck",
        "bn_stats_policy": "frozen",
        "bn_stats_frozen": True,
        "mask_static": False,
        "mask_refresh_policy": "every_online_step_via_split_lbi",
        "ranking_source": "split_lbi_gamma_support",
        "lbi_initialization": "dense",
        "stage3_mode": "accumulation",
        "lbi_state_lifecycle": "reset_every_online_step",
        "support_threshold": 1.0e-4,
        "stage2_optimizer": "sgd",
        "stage2_momentum": 0.9,
        "stage2_weight_decay": 1.0e-3,
        "stage2_nesterov": True,
        "stage2_lr_policy": "shot",
        "stage2_lr_gamma": 10.0,
        "stage2_lr_power": 0.75,
    },
}


def _normalized_budget(config):
    variant = config["variant"]
    if variant == "source_only":
        return None
    if variant in {"full_dense", "module_dense"}:
        return 1.0
    return float(config["requested_budget"])


def _normalized_selection_seed(config):
    if config["variant"] != "module_random":
        return None
    return int(config["selection_seed"])


def _normalized_num_random_masks(config):
    if config["variant"] != "module_random":
        return None
    return int(config["num_random_masks"])


def build_scientific_config(config):
    """Return only settings that define the scientific experiment."""
    variant = config["variant"]
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported variant for identity: {variant}")
    data = config["data"]
    scientific_config = {
        "method": config["method"],
        "task": config["task"],
        "dataset": data["dataset"],
        "source": int(data["source"]),
        "target": int(data["target"]),
        "seed": int(config["seed"]),
        "variant": variant,
        "requested_budget": _normalized_budget(config),
        "selection_seed": _normalized_selection_seed(config),
        "num_random_masks": _normalized_num_random_masks(config),
        "model": copy.deepcopy(config["model"]),
        "optimization": copy.deepcopy(config["optimization"]),
        "loss": copy.deepcopy(config["loss"]),
        "data": {
            "batch_size": int(data["batch_size"]),
            "da": data["da"],
        },
        "variant_policy": copy.deepcopy(VARIANT_METADATA[variant]),
    }
    if variant == "module_lbi":
        lbi = config["lbi"]
        scientific_config["lbi"] = {
            "alpha": float(lbi["alpha"]),
            "kappa": float(lbi["kappa"]),
            "nu": float(lbi["nu"]),
            "omega": float(lbi["omega"]),
            "stage1_max_steps": int(lbi["stage1_max_steps"]),
            "budget_tolerance": float(lbi["budget_tolerance"]),
            "stage2_lr": float(lbi["stage2_lr"]),
            "stage2_steps": int(lbi["stage2_steps"]),
            "delta_nonzero_tolerance": float(
                lbi["delta_nonzero_tolerance"]
            ),
            "lbi_initialization": "dense",
            "stage3_mode": "accumulation",
            "lbi_state_lifecycle": "reset_every_online_step",
        }
    return scientific_config


def canonical_scientific_json(scientific_config):
    return json.dumps(
        scientific_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _budget_token(value):
    if value is None:
        return None
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _experiment_key(scientific_config, full_sha256):
    parts = [
        scientific_config["method"],
        scientific_config["task"],
        scientific_config["dataset"],
        (
            f"s{scientific_config['source']}-"
            f"t{scientific_config['target']}"
        ),
        f"seed{scientific_config['seed']}",
        scientific_config["variant"],
    ]
    budget = _budget_token(scientific_config["requested_budget"])
    if budget is not None:
        parts.append(f"budget{budget}")
    selection_seed = scientific_config["selection_seed"]
    if selection_seed is not None:
        parts.append(f"sel{selection_seed}")
    parts.append(full_sha256[:12])
    return "__".join(parts)


def build_experiment_identity(config):
    scientific_config = build_scientific_config(config)
    canonical_json = canonical_scientific_json(scientific_config)
    full_sha256 = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()
    return {
        "experiment_key": _experiment_key(
            scientific_config,
            full_sha256,
        ),
        "experiment_config_sha256": full_sha256,
        "scientific_config": scientific_config,
    }


def resolve_experiment_identity(
    config,
    provided_key=None,
    provided_sha256=None,
):
    computed = build_experiment_identity(config)
    if (provided_key is None) != (provided_sha256 is None):
        raise ValueError(
            "experiment_key and experiment_config_sha256 must be "
            "provided together"
        )
    if provided_key is not None:
        if provided_key != computed["experiment_key"]:
            raise ValueError(
                "Provided experiment_key does not match the effective "
                "scientific configuration"
            )
        if provided_sha256 != computed["experiment_config_sha256"]:
            raise ValueError(
                "Provided experiment_config_sha256 does not match the "
                "effective scientific configuration"
            )
    return computed


def build_deit_ttda_source_only_identity(config):
    """Build an identity for the no-adaptation DeiT TTDA control.

    This is intentionally separate from ``build_experiment_identity`` so the
    legacy FC/SHOT identity remains byte-for-byte stable.
    """
    data = config["data"]
    checkpoint = config["source_checkpoint"]
    evaluation = config["evaluation"]
    scientific_config = {
        "method": "no_tta",
        "task": "ttda",
        "protocol": "full_target_dataset_no_adaptation",
        "variant": "source_only",
        "dataset": data["dataset"],
        "source": int(data["source"]),
        "target": int(data["target"]),
        "source_name": data["source_name"],
        "target_name": data["target_name"],
        "seed": int(config["seed"]),
        "model": {
            "name": config["model"]["name"],
            "implementation": "timm",
            "non_distilled": True,
            "head_schema": config["model"]["head_schema"],
            "drop_rate": 0.0,
            "drop_path_rate": 0.0,
        },
        "source_checkpoint": {
            "sha256": checkpoint["sha256"],
            "schema_version": checkpoint["schema_version"],
            "kind": checkpoint["kind"],
            "source_training_seed": checkpoint["source_training_seed"],
        },
        "target_data": {
            "list_sha256": data["target_list_sha256"],
            "class_mapping_sha256": data["class_mapping_sha256"],
            "sample_count": int(data["target_sample_count"]),
            "num_classes": int(data["num_classes"]),
            "order": "sequential",
            "drop_last": False,
        },
        "preprocessing": copy.deepcopy(config["preprocessing"]),
        "evaluation": {
            "batch_size": int(evaluation["batch_size"]),
            "amp": bool(evaluation["amp"]),
            "deterministic": bool(evaluation["deterministic"]),
            "prediction_passes": 1,
        },
        "metrics": copy.deepcopy(config["metrics"]),
        "adaptation": {
            "steps": 0,
            "optimizer": None,
            "loss": None,
            "backward": False,
            "target_labels_usage": "evaluation_only",
        },
    }
    canonical_json = canonical_scientific_json(scientific_config)
    full_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    experiment_key = "__".join(
        [
            "no_tta",
            "ttda",
            data["dataset"],
            f"s{int(data['source'])}-t{int(data['target'])}",
            f"seed{int(config['seed'])}",
            "source_only",
            full_sha256[:12],
        ]
    )
    return {
        "experiment_key": experiment_key,
        "experiment_config_sha256": full_sha256,
        "scientific_config": scientific_config,
    }


def resolve_deit_ttda_source_only_identity(
    config,
    provided_key=None,
    provided_sha256=None,
):
    computed = build_deit_ttda_source_only_identity(config)
    if (provided_key is None) != (provided_sha256 is None):
        raise ValueError(
            "experiment_key and experiment_config_sha256 must be "
            "provided together"
        )
    if provided_key is not None:
        if provided_key != computed["experiment_key"]:
            raise ValueError(
                "Provided experiment_key does not match the effective DeiT "
                "TTDA source-only configuration"
            )
        if provided_sha256 != computed["experiment_config_sha256"]:
            raise ValueError(
                "Provided experiment_config_sha256 does not match the "
                "effective DeiT TTDA source-only configuration"
            )
    return computed


def build_deit_otta_source_only_identity(config):
    """Build an identity for the no-adaptation DeiT OTTA control.

    The identity is deliberately separate from both the legacy FC/SHOT
    identity and the DeiT TTDA identity.  OTTA is a sequential target stream,
    even though the source-only control performs zero parameter updates.
    """
    data = config["data"]
    checkpoint = config["source_checkpoint"]
    evaluation = config["evaluation"]
    scientific_config = {
        "method": "no_tta",
        "task": "otta",
        "protocol": "sequential_target_stream_no_adaptation",
        "variant": "source_only",
        "dataset": data["dataset"],
        "source": int(data["source"]),
        "target": int(data["target"]),
        "source_name": data["source_name"],
        "target_name": data["target_name"],
        "seed": int(config["seed"]),
        "model": {
            "name": config["model"]["name"],
            "implementation": "timm",
            "non_distilled": True,
            "head_schema": config["model"]["head_schema"],
            "drop_rate": 0.0,
            "drop_path_rate": 0.0,
        },
        "source_checkpoint": {
            "sha256": checkpoint["sha256"],
            "schema_version": checkpoint["schema_version"],
            "kind": checkpoint["kind"],
            "source_training_seed": checkpoint["source_training_seed"],
        },
        "target_stream": {
            "list_sha256": data["target_list_sha256"],
            "class_mapping_sha256": data["class_mapping_sha256"],
            "sample_count": int(data["target_sample_count"]),
            "num_classes": int(data["num_classes"]),
            "order": "sequential",
            "drop_last": False,
            "tail_batch_size_one_policy": "kept",
        },
        "preprocessing": copy.deepcopy(config["preprocessing"]),
        "evaluation": {
            "batch_size": int(evaluation["batch_size"]),
            "amp": bool(evaluation["amp"]),
            "deterministic": bool(evaluation["deterministic"]),
            "stream_prediction_passes": 1,
            "fo_prediction_passes": 1,
            "final_prediction_policy": "independent_full_target_pass",
        },
        "metrics": copy.deepcopy(config["metrics"]),
        "adaptation": {
            "steps_per_batch": 0,
            "optimizer": None,
            "loss": None,
            "backward": False,
            "state_carried_between_batches": True,
            "target_labels_usage": "evaluation_only",
        },
    }
    canonical_json = canonical_scientific_json(scientific_config)
    full_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    experiment_key = "__".join(
        [
            "no_tta",
            "otta",
            data["dataset"],
            f"s{int(data['source'])}-t{int(data['target'])}",
            f"seed{int(config['seed'])}",
            "source_only",
            full_sha256[:12],
        ]
    )
    return {
        "experiment_key": experiment_key,
        "experiment_config_sha256": full_sha256,
        "scientific_config": scientific_config,
    }


def resolve_deit_otta_source_only_identity(
    config,
    provided_key=None,
    provided_sha256=None,
):
    computed = build_deit_otta_source_only_identity(config)
    if (provided_key is None) != (provided_sha256 is None):
        raise ValueError(
            "experiment_key and experiment_config_sha256 must be "
            "provided together"
        )
    if provided_key is not None:
        if provided_key != computed["experiment_key"]:
            raise ValueError(
                "Provided experiment_key does not match the effective DeiT "
                "OTTA source-only configuration"
            )
        if provided_sha256 != computed["experiment_config_sha256"]:
            raise ValueError(
                "Provided experiment_config_sha256 does not match the "
                "effective DeiT OTTA source-only configuration"
            )
    return computed


ADAPTATION_PROTOCOLS = {
    "full_dense": "sequential_target_stream_full_dense_adaptation",
    "candidate_dense": "sequential_target_stream_candidate_dense_adaptation",
}


def build_deit_otta_adaptation_identity(config):
    """Build an identity for a DeiT OTTA adaptation baseline.

    Supports the ``full_dense`` and ``candidate_dense`` variants.  The SHOT
    objective plus AdamW updates form a new protocol, so the identity is
    separate from both the legacy FC identity and the DeiT source-only
    identity.  Variant, update scope, candidate blocks, optimizer, loss, and
    adaptation settings are identity-bound: changing any of them produces a
    different experiment key.
    """
    variant = config["variant"]
    data = config["data"]
    checkpoint = config["source_checkpoint"]
    evaluation = config["evaluation"]
    adaptation = config["adaptation"]
    candidate_blocks = adaptation.get("candidate_blocks")
    adaptation_record = {
        "update_scope": adaptation["update_scope"],
        "model_mode": adaptation["model_mode"],
        "steps_per_batch": int(adaptation["steps_per_batch"]),
        "delta_semantics": "unrestricted_accumulation",
        "budget": "full",
        "target_labels_usage": "evaluation_only",
        "state_carried_between_batches": True,
    }
    if variant == "candidate_dense":
        adaptation_record["candidate_blocks"] = sorted(
            int(block) for block in candidate_blocks
        )
        adaptation_record["candidate_parameter_names"] = sorted(
            candidate_parameter_names(candidate_blocks)
        )
    scientific_config = {
        "method": "shot",
        "task": "otta",
        "protocol": ADAPTATION_PROTOCOLS[variant],
        "variant": variant,
        "dataset": data["dataset"],
        "source": int(data["source"]),
        "target": int(data["target"]),
        "source_name": data["source_name"],
        "target_name": data["target_name"],
        "seed": int(config["seed"]),
        "model": {
            "name": config["model"]["name"],
            "implementation": "timm",
            "non_distilled": True,
            "head_schema": config["model"]["head_schema"],
            "drop_rate": 0.0,
            "drop_path_rate": 0.0,
        },
        "source_checkpoint": {
            "sha256": checkpoint["sha256"],
            "schema_version": checkpoint["schema_version"],
            "kind": checkpoint["kind"],
            "source_training_seed": checkpoint["source_training_seed"],
        },
        "target_stream": {
            "list_sha256": data["target_list_sha256"],
            "class_mapping_sha256": data["class_mapping_sha256"],
            "sample_count": int(data["target_sample_count"]),
            "num_classes": int(data["num_classes"]),
            "order": "sequential",
            "drop_last": False,
            "tail_batch_size_one_policy": "kept",
        },
        "preprocessing": copy.deepcopy(config["preprocessing"]),
        "evaluation": {
            "batch_size": int(evaluation["batch_size"]),
            "amp": bool(evaluation["amp"]),
            "deterministic": bool(evaluation["deterministic"]),
            "stream_prediction_passes": 1,
            "fo_prediction_passes": 1,
            "final_prediction_policy": "independent_full_target_pass",
        },
        "metrics": copy.deepcopy(config["metrics"]),
        "adaptation": adaptation_record,
        "optimization": copy.deepcopy(config["optimization"]),
        "loss": copy.deepcopy(config["loss"]),
    }
    canonical_json = canonical_scientific_json(scientific_config)
    full_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    experiment_key = "__".join(
        [
            "shot",
            "otta",
            data["dataset"],
            f"s{int(data['source'])}-t{int(data['target'])}",
            f"seed{int(config['seed'])}",
            variant,
            full_sha256[:12],
        ]
    )
    return {
        "experiment_key": experiment_key,
        "experiment_config_sha256": full_sha256,
        "scientific_config": scientific_config,
    }


def resolve_deit_otta_adaptation_identity(
    config,
    provided_key=None,
    provided_sha256=None,
):
    computed = build_deit_otta_adaptation_identity(config)
    if (provided_key is None) != (provided_sha256 is None):
        raise ValueError(
            "experiment_key and experiment_config_sha256 must be "
            "provided together"
        )
    if provided_key is not None:
        if provided_key != computed["experiment_key"]:
            raise ValueError(
                "Provided experiment_key does not match the effective DeiT "
                "OTTA adaptation configuration"
            )
        if provided_sha256 != computed["experiment_config_sha256"]:
            raise ValueError(
                "Provided experiment_config_sha256 does not match the "
                "effective DeiT OTTA adaptation configuration"
            )
    return computed


def build_deit_otta_group_random_identity(config):
    """Build an identity for the DeiT OTTA random structural-group baseline.

    Budget, grouping, candidate blocks, mask count/seeds, and the strict
    masked delta semantics are identity-bound: changing any of them produces a
    different experiment key.
    """
    variant = config["variant"]
    data = config["data"]
    checkpoint = config["source_checkpoint"]
    evaluation = config["evaluation"]
    adaptation = config["adaptation"]
    candidate_blocks = adaptation["candidate_blocks"]
    adaptation_record = {
        "update_scope": adaptation["update_scope"],
        "model_mode": adaptation["model_mode"],
        "steps_per_batch": int(adaptation["steps_per_batch"]),
        "delta_semantics": adaptation["delta_semantics"],
        "grouping": adaptation["grouping"],
        "selection": "random",
        "mask_static": True,
        "mask_refresh_policy": "once_before_adaptation",
        "ranking_source": "independent_random_group_generator",
        "budget": float(adaptation["budget"]),
        "budget_unit": "structural_groups",
        "num_random_masks": int(adaptation["num_random_masks"]),
        "mask_seeds": [int(seed) for seed in adaptation["mask_seeds"]],
        "candidate_blocks": sorted(int(block) for block in candidate_blocks),
        "candidate_parameter_names": sorted(
            candidate_parameter_names(candidate_blocks)
        ),
        "total_groups": int(adaptation["total_groups"]),
        "active_groups": int(adaptation["active_groups"]),
        "target_labels_usage": "evaluation_only",
        "state_carried_between_batches": True,
    }
    scientific_config = {
        "method": "shot",
        "task": "otta",
        "protocol": GROUP_RANDOM_PROTOCOL,
        "variant": variant,
        "dataset": data["dataset"],
        "source": int(data["source"]),
        "target": int(data["target"]),
        "source_name": data["source_name"],
        "target_name": data["target_name"],
        "seed": int(config["seed"]),
        "model": {
            "name": config["model"]["name"],
            "implementation": "timm",
            "non_distilled": True,
            "head_schema": config["model"]["head_schema"],
            "drop_rate": 0.0,
            "drop_path_rate": 0.0,
        },
        "source_checkpoint": {
            "sha256": checkpoint["sha256"],
            "schema_version": checkpoint["schema_version"],
            "kind": checkpoint["kind"],
            "source_training_seed": checkpoint["source_training_seed"],
        },
        "target_stream": {
            "list_sha256": data["target_list_sha256"],
            "class_mapping_sha256": data["class_mapping_sha256"],
            "sample_count": int(data["target_sample_count"]),
            "num_classes": int(data["num_classes"]),
            "order": "sequential",
            "drop_last": False,
            "tail_batch_size_one_policy": "kept",
        },
        "preprocessing": copy.deepcopy(config["preprocessing"]),
        "evaluation": {
            "batch_size": int(evaluation["batch_size"]),
            "amp": bool(evaluation["amp"]),
            "deterministic": bool(evaluation["deterministic"]),
            "stream_prediction_passes": 1,
            "fo_prediction_passes": 1,
            "final_prediction_policy": "independent_full_target_pass",
        },
        "metrics": copy.deepcopy(config["metrics"]),
        "adaptation": adaptation_record,
        "optimization": copy.deepcopy(config["optimization"]),
        "loss": copy.deepcopy(config["loss"]),
    }
    canonical_json = canonical_scientific_json(scientific_config)
    full_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    experiment_key = "__".join(
        [
            "shot",
            "otta",
            data["dataset"],
            f"s{int(data['source'])}-t{int(data['target'])}",
            f"seed{int(config['seed'])}",
            "group_random",
            f"budget{float(adaptation['budget'])}",
            full_sha256[:12],
        ]
    )
    return {
        "experiment_key": experiment_key,
        "experiment_config_sha256": full_sha256,
        "scientific_config": scientific_config,
    }


def resolve_deit_otta_group_random_identity(
    config,
    provided_key=None,
    provided_sha256=None,
):
    computed = build_deit_otta_group_random_identity(config)
    if (provided_key is None) != (provided_sha256 is None):
        raise ValueError(
            "experiment_key and experiment_config_sha256 must be "
            "provided together"
        )
    if provided_key is not None:
        if provided_key != computed["experiment_key"]:
            raise ValueError(
                "Provided experiment_key does not match the effective DeiT "
                "OTTA group-random configuration"
            )
        if provided_sha256 != computed["experiment_config_sha256"]:
            raise ValueError(
                "Provided experiment_config_sha256 does not match the "
                "effective DeiT OTTA group-random configuration"
            )
    return computed

