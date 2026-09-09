"""Stable identity for result-affecting SHOT-OTTA experiment settings."""

import copy
import hashlib
import json

from protocol_constants import (
    COME_IMPLEMENTATION_REVISION,
    COME_LBI_IMPLEMENTATION_REVISION,
    COME_LBI_PROTOCOL_REVISION,
    CONV_IMPLEMENTATION_REVISION,
    EFFICIENCY_PROTOCOL_REVISION as EFFICIENCY_PROTOCOL_REVISION,
    IMPLEMENTATION_REVISION,
    IST_IMPLEMENTATION_REVISION,
    IST_LBI_IMPLEMENTATION_REVISION,
    NCTTA_IMPLEMENTATION_REVISION,
    SOURCE_CHECKPOINT_REVISION,
)
from shot_otta.candidates import CONV_CANDIDATE_ORDER, MODULE_CANDIDATE_ORDER


SUPPORTED_VARIANTS = {
    "source_only",
    "full_dense",
    "module_dense",
    "module_random",
    "module_magnitude",
    "module_saliency",
    "module_lbi",
    "conv_module_dense",
    "conv_out_random",
    "conv_out_magnitude",
    "conv_out_saliency",
    "conv_out_lbi",
    "conv_filter_random",
    "conv_filter_magnitude",
    "conv_filter_saliency",
    "conv_filter_lbi",
    "ist_full_dense",
    "ist_fc_module_dense",
    "ist_fc_random",
    "ist_fc_magnitude",
    "ist_fc_saliency",
    "ist_fc_lbi",
    "ist_conv_module_dense",
    "ist_conv_out_random",
    "ist_conv_out_magnitude",
    "ist_conv_out_saliency",
    "ist_conv_out_lbi",
    "nctta_native_norm",
    "nctta_full_dense",
    "nctta_fc_module_dense",
    "nctta_conv_module_dense",
    "nctta_fc_random",
    "nctta_fc_magnitude",
    "nctta_fc_saliency",
    "nctta_fc_lbi",
    "nctta_conv_out_random",
    "nctta_conv_out_magnitude",
    "nctta_conv_out_saliency",
    "nctta_conv_out_lbi",
    "come_fc_random",
    "come_fc_magnitude",
    "come_fc_saliency",
    "come_fc_lbi",
    "come_conv_out_random",
    "come_conv_out_magnitude",
    "come_conv_out_saliency",
    "come_conv_out_lbi",
}

CONV_VARIANTS = {
    "conv_module_dense",
    "conv_out_random",
    "conv_out_magnitude",
    "conv_out_saliency",
    "conv_out_lbi",
    "conv_filter_random",
    "conv_filter_magnitude",
    "conv_filter_saliency",
    "conv_filter_lbi",
}
CONV_RANDOM_VARIANTS = {"conv_out_random", "conv_filter_random"}
CONV_LBI_VARIANTS = {"conv_out_lbi", "conv_filter_lbi"}
IST_FC_VARIANTS = {
    "ist_fc_module_dense",
    "ist_fc_random",
    "ist_fc_magnitude",
    "ist_fc_saliency",
    "ist_fc_lbi",
}
IST_CONV_VARIANTS = {
    "ist_conv_module_dense",
    "ist_conv_out_random",
    "ist_conv_out_magnitude",
    "ist_conv_out_saliency",
    "ist_conv_out_lbi",
}
IST_RANDOM_VARIANTS = {"ist_fc_random", "ist_conv_out_random"}
IST_LBI_VARIANTS = {"ist_fc_lbi", "ist_conv_out_lbi"}
IST_DENSE_VARIANTS = {
    "ist_full_dense",
    "ist_fc_module_dense",
    "ist_conv_module_dense",
}
NCTTA_DENSE_VARIANTS = {
    "nctta_native_norm",
    "nctta_full_dense",
    "nctta_fc_module_dense",
    "nctta_conv_module_dense",
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
        "budget_semantics": "global_fc_floor_integer",
    },
    "module_random": {
        "selection": "random",
        "candidate_scope": "netB.bottleneck",
        "bn_stats_policy": "frozen",
        "bn_stats_frozen": True,
        "mask_static": True,
        "mask_refresh_policy": "once_before_adaptation",
        "ranking_source": "independent_random_generator",
        "budget_semantics": "global_fc_floor_integer",
        "masked_optimizer_state": "mask_out_momentum_zeroed",
    },
    "module_magnitude": {
        "selection": "magnitude",
        "candidate_scope": "netB.bottleneck",
        "bn_stats_policy": "frozen",
        "bn_stats_frozen": True,
        "mask_static": True,
        "mask_refresh_policy": "once_before_adaptation",
        "ranking_source": "source_checkpoint_pre_adaptation",
        "budget_semantics": "global_fc_floor_integer",
        "masked_optimizer_state": "mask_out_momentum_zeroed",
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
        "budget_semantics": "global_fc_floor_integer",
        "masked_optimizer_state": "mask_out_momentum_zeroed",
    },
    "module_lbi": {
        "selection": "lbi",
        "candidate_scope": "netB.bottleneck",
        "bn_stats_policy": "frozen",
        "bn_stats_frozen": True,
        "mask_static": False,
        "mask_refresh_policy": "every_online_step_via_split_lbi",
        "ranking_source": "split_lbi_gamma_support",
        "lbi_initialization": "masked_delta",
        "stage3_mode": "accumulation",
        "lbi_state_lifecycle": "reset_every_online_step",
        "stage2_optimizer": "sgd",
        "stage2_momentum": 0.9,
        "stage2_weight_decay": 1.0e-3,
        "stage2_nesterov": True,
        "budget_semantics": "global_fc_floor_integer_strict_rollback",
        "masked_optimizer_state": "lbi_stage2_local_optimizer_unchanged",
    },
    "conv_module_dense": {
        "selection": "dense",
        "candidate_scope": "netF.layer4_conv",
        "bn_stats_policy": "frozen",
        "bn_stats_frozen": True,
        "mask_static": False,
        "mask_refresh_policy": None,
        "budget_semantics": "not_applicable_dense",
    },
}

VARIANT_METADATA.update(
    {
        "ist_full_dense": {
            "selection": "dense",
            "candidate_scope": "netF+netB",
            "bn_stats_policy": "native_full_train",
            "bn_stats_frozen": False,
            "mask_static": False,
            "mask_refresh_policy": None,
        },
        "ist_fc_module_dense": {
            "selection": "dense",
            "candidate_scope": "netB.bottleneck",
            "bn_stats_policy": "frozen",
            "bn_stats_frozen": True,
            "mask_static": False,
            "mask_refresh_policy": None,
        },
        "ist_conv_module_dense": {
            "selection": "dense",
            "candidate_scope": "netF.layer4_conv",
            "bn_stats_policy": "frozen",
            "bn_stats_frozen": True,
            "mask_static": False,
            "mask_refresh_policy": None,
        },
    }
)
VARIANT_METADATA["nctta_native_norm"] = {
    "selection": "native",
    "candidate_scope": "official_nctta_normalization_selector",
    "normalization_module_types": ["BatchNorm2d", "LayerNorm", "GroupNorm"],
    "bn_stats_policy": "native_batch_statistics_no_running_stats",
    "bn_stats_frozen": False,
    "mask_static": False,
    "mask_refresh_policy": None,
    "native_nctta_norm_only_scope": True,
    "netB_batchnorm1d_frozen": True,
    "native_norm_fo_uses_batch_statistics": True,
}
VARIANT_METADATA.update(
    {
        "nctta_full_dense": {
            "selection": "dense",
            "candidate_scope": "netF+netB",
            "bn_stats_policy": "adaptive",
            "bn_stats_frozen": False,
            "mask_static": False,
            "mask_refresh_policy": None,
            "native_nctta_norm_only_scope": False,
        },
        "nctta_fc_module_dense": {
            "selection": "dense",
            "candidate_scope": "netB.bottleneck",
            "bn_stats_policy": "frozen",
            "bn_stats_frozen": True,
            "mask_static": False,
            "mask_refresh_policy": None,
            "native_nctta_norm_only_scope": False,
        },
        "nctta_conv_module_dense": {
            "selection": "dense",
            "candidate_scope": "netF.layer4_conv",
            "bn_stats_policy": "frozen",
            "bn_stats_frozen": True,
            "mask_static": False,
            "mask_refresh_policy": None,
            "native_nctta_norm_only_scope": False,
        },
    }
)


for _variant in sorted(
    (IST_FC_VARIANTS | IST_CONV_VARIANTS)
    - {
        "ist_fc_module_dense",
        "ist_conv_module_dense",
    }
):
    _selection = _variant.rsplit("_", 1)[-1]
    _is_conv = _variant.startswith("ist_conv_")
    VARIANT_METADATA[_variant] = {
        "selection": _selection,
        "candidate_scope": ("netF.layer4_conv" if _is_conv else "netB.bottleneck"),
        "group_mode": "out_channel" if _is_conv else None,
        "group_partition_semantics": "out_channel" if _is_conv else None,
        "bn_stats_policy": "frozen",
        "bn_stats_frozen": True,
        "mask_static": _selection in {"random", "magnitude"},
        "mask_refresh_policy": (
            "once_before_target_stream"
            if _selection in {"random", "magnitude"}
            else "once_per_outer_batch_full_ist_objective"
            if _selection == "saliency"
            else "once_per_outer_batch_via_split_lbi"
        ),
        "ranking_source": {
            "random": "deterministic_independent_child_mask",
            "magnitude": "source_checkpoint_pre_adaptation",
            "saliency": "full_outer_batch_parameter_times_gradient",
            "lbi": "split_lbi_gamma_support",
        }[_selection],
        "budget_semantics": (
            "global_conv_out_channel_floor_integer"
            if _is_conv
            else "global_fc_floor_integer"
        )
        + ("_strict_rollback" if _selection == "lbi" else ""),
        "native_ist_ema": _selection != "lbi",
        "persistent_writeback": (
            "lbi_omega_only" if _selection == "lbi" else "ist_native_ema"
        ),
    }


for _mode in ("out", "filter"):
    _group_mode = "out_channel" if _mode == "out" else "filter_connection"
    for _selection in ("random", "magnitude", "saliency", "lbi"):
        _variant = f"conv_{_mode}_{_selection}"
        VARIANT_METADATA[_variant] = {
            "selection": _selection,
            "candidate_scope": "netF.layer4_conv",
            "group_mode": _group_mode,
            "group_partition_semantics": _group_mode,
            "group_norm": "l2",
            "group_penalty": (
                "unweighted_group_lasso" if _selection == "lbi" else None
            ),
            "group_score": {
                "random": "uniform_global_group_sample",
                "magnitude": "group_l2_weight_norm",
                "saliency": "group_l2_parameter_times_gradient",
                "lbi": "group_lasso_prox",
            }[_selection],
            "bn_stats_policy": "frozen",
            "bn_stats_frozen": True,
            "mask_static": _selection in {"random", "magnitude"},
            "mask_refresh_policy": (
                "once_before_adaptation"
                if _selection in {"random", "magnitude"}
                else "every_online_step_via_split_lbi"
                if _selection == "lbi"
                else "every_online_step_after_backward"
            ),
            "budget_semantics": "global_conv_group_floor_integer"
            + ("_strict_rollback" if _selection == "lbi" else ""),
        }


def _normalized_budget(config):
    variant = config["variant"]
    if variant == "source_only":
        return None
    if variant in {
        "full_dense",
        "module_dense",
        "conv_module_dense",
        "ist_full_dense",
        "ist_fc_module_dense",
        "ist_conv_module_dense",
        "nctta_native_norm",
        "nctta_full_dense",
        "nctta_fc_module_dense",
        "nctta_conv_module_dense",
    }:
        return 1.0
    return float(config["requested_budget"])


def _normalized_selection_seed(config):
    if config["variant"] not in {
        "module_random",
        *CONV_RANDOM_VARIANTS,
        *IST_RANDOM_VARIANTS,
    }:
        return None
    return int(config["selection_seed"])


def _normalized_num_random_masks(config):
    if config["variant"] not in {
        "module_random",
        *CONV_RANDOM_VARIANTS,
        *IST_RANDOM_VARIANTS,
    }:
        return None
    return int(config["num_random_masks"])


def build_scientific_config(config):
    """Return only settings that define the scientific experiment."""
    variant = config["variant"]
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported variant for identity: {variant}")
    data = config["data"]
    if config["method"] == "COME":
        come_sparse_variants = {
            "come_fc_random",
            "come_fc_magnitude",
            "come_fc_saliency",
            "come_fc_lbi",
            "come_conv_out_random",
            "come_conv_out_magnitude",
            "come_conv_out_saliency",
            "come_conv_out_lbi",
        }
        if variant not in come_sparse_variants:
            raise ValueError("shared identity handles only COME sparse/LBI variants")
        scalar = variant.startswith("come_fc_")
        selection = variant.rsplit("_", 1)[-1]
        names = list(MODULE_CANDIDATE_ORDER) if scalar else list(CONV_CANDIDATE_ORDER)
        candidate_count = 524544 if scalar else 12845056
        group_count = None if scalar else 9216
        integer_budget = int(
            float(config["requested_budget"])
            * (candidate_count if scalar else group_count)
        )
        objective = config["come"]
        scientific = {
            "implementation_revision": config.get(
                "implementation_revision", COME_LBI_IMPLEMENTATION_REVISION
            ),
            "protocol_track": "come_sparse_lbi",
            "protocol_revision": config.get(
                "protocol_revision", COME_LBI_PROTOCOL_REVISION
            ),
            "parent_come_baseline_protocol_revision": (
                "OTTA_COME_BASELINE_PROTOCOL_20260907_v1"
            ),
            "come_lbi_protocol_revision": COME_LBI_PROTOCOL_REVISION,
            "come_baseline_implementation_revision": COME_IMPLEMENTATION_REVISION,
            "official_come_commit": objective["official_commit"],
            "source_checkpoint_revision": config.get(
                "source_checkpoint_revision", SOURCE_CHECKPOINT_REVISION
            ),
            "source_checkpoints": copy.deepcopy(
                config.get("source_checkpoint_identity")
            ),
            "method": "COME",
            "task": config["task"],
            "dataset": data["dataset"],
            "source": int(data["source"]),
            "target": int(data["target"]),
            "transfer": config.get("task_name"),
            "backbone": config["model"]["backbone"],
            "seed": int(config["seed"]),
            "outer_batch_size": int(data["batch_size"]),
            "target_stream": copy.deepcopy(config.get("target_stream_identity")),
            "primary_metric_name": (
                "fixed_12_class_mAcc"
                if data["dataset"] == "VISDA-C"
                else "sample_level_overall_accuracy"
            ),
            "variant": variant,
            "requested_budget": float(config["requested_budget"]),
            "integer_budget": integer_budget,
            "selection_seed": (
                int(config["selection_seed"]) if selection == "random" else None
            ),
            "num_random_masks": (
                int(config["num_random_masks"]) if selection == "random" else None
            ),
            "candidate_track": ("fc_scalar" if scalar else "conv_out_channel"),
            "candidate_scope": ("netB.bottleneck" if scalar else "netF.layer4_conv"),
            "candidate_layer_names": names,
            "candidate_tensor_count": len(names),
            "candidate_scalar_count": candidate_count,
            "group_mode": None if scalar else "out_channel",
            "group_count": group_count,
            "selector_definition": {
                "random": "uniform_global_exact_budget_three_child_trajectories",
                "magnitude": "source_checkpoint_global_static_once",
                "saliency": "current_state_parameter_times_come_gradient_once_per_batch",
                "lbi": "thresholded_gamma_strict_budget_rollback",
            }[selection],
            "come": {
                "p": float(objective["p"]),
                "tau": float(objective["tau"]),
                "class_count": int(config["model"]["class_num"]),
                "opinion_eps": float(objective["opinion_eps"]),
                "norm_detach_semantics": "multiplicative_logit_norm_only",
                "numerical_evaluation": "stable_log_domain_logsumexp",
            },
            "optimization": copy.deepcopy(config["optimization"]),
            "data": {
                "batch_size": int(data["batch_size"]),
                "workers": int(data["workers"]),
                "da": data["da"],
                "target_passes": 1,
                "drop_last": False,
                "outer_stream": "fixed_seed_permutation",
                "singleton_policy": "skip_size_1_before_all_state_transitions",
            },
            "persistent_writeback": (
                "lbi_omega_only" if selection == "lbi" else "masked_host_optimizer"
            ),
            "host_optimizer_persistent_step": selection != "lbi",
            "lbi_omega": (
                float(config["lbi"]["omega"]) if selection == "lbi" else None
            ),
        }
        if selection == "lbi":
            scientific["lbi"] = copy.deepcopy(config["lbi"])
            scientific["lbi_frozen"] = {
                key: config["lbi_runtime"][key]
                for key in (
                    "support_threshold",
                    "stage1_max_steps",
                    "stage2_steps",
                    "budget_tolerance",
                    "delta_nonzero_tolerance",
                )
            }
        return scientific
    if config["method"] == "NCTTA":
        if variant not in NCTTA_DENSE_VARIANTS:
            from nctta_otta.identity import build_nctta_scientific_config

            return build_nctta_scientific_config(config, VARIANT_METADATA)
        is_native = variant == "nctta_native_norm"
        is_fc = variant == "nctta_fc_module_dense"
        candidate_scope = (
            "official_nctta_normalization_selector"
            if is_native
            else "netF+netB"
            if variant == "nctta_full_dense"
            else "netB.bottleneck"
            if is_fc
            else "netF.layer4_conv"
        )
        candidate_layer_names = (
            []
            if is_native or variant == "nctta_full_dense"
            else list(MODULE_CANDIDATE_ORDER)
            if is_fc
            else list(CONV_CANDIDATE_ORDER)
        )
        actual_optimizer = (
            config["native_optimizer"] if is_native else config["optimization"]
        )
        return {
            "implementation_revision": config.get(
                "implementation_revision", NCTTA_IMPLEMENTATION_REVISION
            ),
            "protocol_track": "nctta",
            "protocol_revision": config.get("protocol_revision"),
            "method": "NCTTA",
            "task": config["task"],
            "dataset": data["dataset"],
            "source": int(data["source"]),
            "target": int(data["target"]),
            "seed": int(config["seed"]),
            "variant": variant,
            "requested_budget": 1.0,
            "integer_budget": None,
            "selection_seed": None,
            "num_random_masks": None,
            "candidate_scope": candidate_scope,
            "candidate_layer_names": candidate_layer_names,
            "group_mode": None,
            "model": copy.deepcopy(config["model"]),
            "optimization": copy.deepcopy(actual_optimizer),
            "optimizer_provenance": (
                "official_nctta_ttab_defaults"
                if is_native
                else "common_shot_optimizer_substrate"
            ),
            "loss": copy.deepcopy(config["loss"]),
            "nctta": copy.deepcopy(config["nctta"]),
            "data": {
                "batch_size": int(data["batch_size"]),
                "workers": int(data["workers"]),
                "da": data["da"],
                "target_passes": 1,
                "drop_last": False,
                "outer_stream": "fixed_seed_permutation",
                "singleton_policy": "skip_size_1",
            },
            "source_checkpoint_revision": config.get(
                "source_checkpoint_revision", SOURCE_CHECKPOINT_REVISION
            ),
            "variant_policy": copy.deepcopy(VARIANT_METADATA[variant]),
            "feature_source": "post_netB_classifier_input",
            "classifier_reference": "effective_frozen_netC.fc.weight.detach",
            "objective_recomputation": "once_per_current_outer_batch",
            "ttab_runtime_imported": False,
        }
    if config["method"] == "IST":
        is_fc = variant in IST_FC_VARIANTS
        is_conv = variant in IST_CONV_VARIANTS
        is_sparse = variant not in IST_DENSE_VARIANTS
        candidate_scope = (
            "netF+netB"
            if variant == "ist_full_dense"
            else "netB.bottleneck"
            if is_fc
            else "netF.layer4_conv"
        )
        candidate_layer_names = (
            []
            if variant == "ist_full_dense"
            else list(MODULE_CANDIDATE_ORDER)
            if is_fc
            else list(CONV_CANDIDATE_ORDER)
        )
        scientific = {
            "implementation_revision": config.get(
                "implementation_revision",
                IST_LBI_IMPLEMENTATION_REVISION
                if is_sparse
                else IST_IMPLEMENTATION_REVISION,
            ),
            "protocol_track": "ist",
            "protocol_revision": config.get("protocol_revision"),
            "method": "IST",
            "task": config["task"],
            "dataset": data["dataset"],
            "source": int(data["source"]),
            "target": int(data["target"]),
            "seed": int(config["seed"]),
            "variant": variant,
            "requested_budget": _normalized_budget(config),
            "integer_budget": (
                int(float(config["requested_budget"]) * (9216 if is_conv else 524544))
                if is_sparse
                else None
            ),
            "selection_seed": _normalized_selection_seed(config),
            "num_random_masks": _normalized_num_random_masks(config),
            "candidate_scope": candidate_scope,
            "candidate_layer_names": candidate_layer_names,
            "group_mode": "out_channel" if is_conv and is_sparse else None,
            "model": copy.deepcopy(config["model"]),
            "optimization": copy.deepcopy(config["optimization"]),
            "loss": copy.deepcopy(config["loss"]),
            "ist": copy.deepcopy(config["ist"]),
            "data": {
                "batch_size": int(data["batch_size"]),
                "workers": int(data["workers"]),
                "da": data["da"],
                "target_passes": 1,
                "drop_last": False,
                "outer_stream": "fixed_seed_permutation",
            },
            "source_checkpoint_revision": config.get(
                "source_checkpoint_revision", SOURCE_CHECKPOINT_REVISION
            ),
            "variant_policy": copy.deepcopy(VARIANT_METADATA[variant]),
        }
        if variant in IST_LBI_VARIANTS:
            scientific["lbi"] = copy.deepcopy(config["lbi"])
            scientific["lbi_frozen"] = {
                key: config["lbi_runtime"][key]
                for key in (
                    "support_threshold",
                    "stage1_max_steps",
                    "stage2_steps",
                    "budget_tolerance",
                    "delta_nonzero_tolerance",
                )
            }
            scientific["native_ist_ema"] = False
            scientific["persistent_writeback"] = "lbi_omega_only"
        else:
            scientific["native_ist_ema"] = True
            scientific["ema_momentum"] = 0.9
            scientific["persistent_writeback"] = "ist_native_ema"
        return scientific
    is_conv_track = config.get("protocol_track") == "conv"
    if is_conv_track and variant in (CONV_VARIANTS | {"source_only", "full_dense"}):
        is_controlled_conv = variant in CONV_VARIANTS
        is_group_sparse = variant not in {
            "source_only",
            "full_dense",
            "conv_module_dense",
        }
        scientific_config = {
            "implementation_revision": config.get(
                "implementation_revision", CONV_IMPLEMENTATION_REVISION
            ),
            "protocol_track": "conv",
            "protocol_revision": config.get("protocol_revision"),
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
            "candidate_scope": (
                "netF.layer4_conv"
                if is_controlled_conv
                else "netF+netB"
                if variant == "full_dense"
                else "none"
            ),
            "candidate_layer_names": [
                "netF.layer4.0.conv1.weight",
                "netF.layer4.0.conv2.weight",
                "netF.layer4.0.conv3.weight",
                "netF.layer4.1.conv1.weight",
                "netF.layer4.1.conv2.weight",
                "netF.layer4.1.conv3.weight",
                "netF.layer4.2.conv1.weight",
                "netF.layer4.2.conv2.weight",
                "netF.layer4.2.conv3.weight",
            ]
            if is_controlled_conv
            else [],
            "group_mode": (config.get("group_mode") if is_group_sparse else None),
            "group_semantics": (config.get("group_mode") if is_group_sparse else None),
            "model": copy.deepcopy(config["model"]),
            "optimization": copy.deepcopy(config["optimization"]),
            "loss": copy.deepcopy(config["loss"]),
            "data": {
                "batch_size": int(data["batch_size"]),
                "workers": int(data["workers"]),
                "da": data["da"],
            },
            "source_checkpoint_revision": config.get(
                "source_checkpoint_revision", SOURCE_CHECKPOINT_REVISION
            ),
            "variant_policy": copy.deepcopy(VARIANT_METADATA[variant]),
        }
        if variant in CONV_LBI_VARIANTS:
            lbi = config["lbi"]
            if lbi is None:
                scientific_config["lbi"] = {
                    "status": "unresolved",
                    "tuning_granularity": "dataset_method_group_mode_budget",
                }
            else:
                scientific_config["lbi"] = {
                    key: (
                        int(lbi[key])
                        if key in {"stage1_max_steps", "stage2_steps"}
                        else float(lbi[key])
                    )
                    for key in (
                        "alpha",
                        "kappa",
                        "nu",
                        "omega",
                        "stage1_max_steps",
                        "budget_tolerance",
                        "stage2_lr",
                        "stage2_steps",
                        "delta_nonzero_tolerance",
                        "support_threshold",
                    )
                }
                scientific_config["lbi"].update(
                    {
                        "lbi_initialization": "masked_delta",
                        "stage3_mode": "accumulation",
                        "lbi_state_lifecycle": "reset_every_online_step",
                    }
                )
        return scientific_config
    scientific_config = {
        "implementation_revision": IMPLEMENTATION_REVISION,
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
            "workers": int(data["workers"]),
            "da": data["da"],
        },
        "source_checkpoint_revision": config.get(
            "source_checkpoint_revision", SOURCE_CHECKPOINT_REVISION
        ),
        "variant_policy": copy.deepcopy(VARIANT_METADATA[variant]),
    }
    if variant == "module_lbi":
        lbi = config["lbi"]
        if lbi is None:
            scientific_config["lbi"] = {
                "status": "unresolved",
                "tuning_granularity": "dataset_method_budget",
            }
            return scientific_config
        scientific_config["lbi"] = {
            "alpha": float(lbi["alpha"]),
            "kappa": float(lbi["kappa"]),
            "nu": float(lbi["nu"]),
            "omega": float(lbi["omega"]),
            "stage1_max_steps": int(lbi["stage1_max_steps"]),
            "budget_tolerance": float(lbi["budget_tolerance"]),
            "stage2_lr": float(lbi["stage2_lr"]),
            "stage2_steps": int(lbi["stage2_steps"]),
            "delta_nonzero_tolerance": float(lbi["delta_nonzero_tolerance"]),
            "support_threshold": float(lbi["support_threshold"]),
            "lbi_initialization": "masked_delta",
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
        (f"s{scientific_config['source']}-" f"t{scientific_config['target']}"),
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
    full_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
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
            "experiment_key and experiment_config_sha256 must be " "provided together"
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
