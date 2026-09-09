"""Scientific identity fields for dense and sparse NCTTA families."""

import copy
import math

from protocol_constants import (
    CONV_GROUP_COUNTS,
    FC_CANDIDATE_PARAM_COUNT,
    NCTTA_IMPLEMENTATION_REVISION,
    NCTTA_LBI_IMPLEMENTATION_REVISION,
    NCTTA_PROTOCOL_REVISION,
    SOURCE_CHECKPOINT_REVISION,
)
from shot_otta.candidates import CONV_CANDIDATE_ORDER, MODULE_CANDIDATE_ORDER

from .sparse import (
    LBI_VARIANTS,
    RANDOM_VARIANTS,
    SPARSE_VARIANTS,
    variant_selection,
    variant_track,
)


def build_nctta_scientific_config(config, metadata):
    variant = config["variant"]
    data = config["data"]
    sparse = variant in SPARSE_VARIANTS
    native = variant == "nctta_native_norm"
    track = variant_track(variant)
    fc = track == "fc_scalar"
    conv = track == "conv_out_channel"
    candidate_scope = (
        "official_nctta_normalization_selector"
        if native
        else "netF+netB"
        if variant == "nctta_full_dense"
        else "netB.bottleneck"
        if fc
        else "netF.layer4_conv"
    )
    candidate_names = (
        []
        if native or variant == "nctta_full_dense"
        else (list(MODULE_CANDIDATE_ORDER) if fc else list(CONV_CANDIDATE_ORDER))
    )
    budget = float(config["requested_budget"]) if sparse else 1.0
    integer_budget = None
    if sparse:
        integer_budget = math.floor(
            budget
            * (CONV_GROUP_COUNTS["out_channel"] if conv else FC_CANDIDATE_PARAM_COUNT)
        )
    selection = variant_selection(variant)
    policy = copy.deepcopy(metadata.get(variant, {}))
    if sparse:
        policy.update(
            {
                "selection": selection,
                "candidate_scope": candidate_scope,
                "group_mode": "out_channel" if conv else None,
                "bn_stats_policy": "frozen",
                "bn_stats_frozen": True,
                "mask_static": selection in {"random", "magnitude"},
                "mask_refresh_policy": (
                    "once_before_target_stream"
                    if selection in {"random", "magnitude"}
                    else "once_per_outer_batch_current_state_nctta_objective"
                    if selection == "saliency"
                    else "once_per_outer_batch_via_split_lbi"
                ),
                "persistent_writeback": "lbi_omega_only"
                if selection == "lbi"
                else "masked_host_optimizer",
                "native_nctta_norm_only_scope": False,
                "ranking_source": {
                    "random": "deterministic_independent_child_mask",
                    "magnitude": "source_checkpoint_pre_adaptation",
                    "saliency": "current_state_nctta_parameter_times_gradient",
                    "lbi": "split_lbi_thresholded_gamma_support",
                }[selection],
                "selector_definition": {
                    "random": "uniform_global_exact_budget_child_mask",
                    "magnitude": "global_source_magnitude_top_budget",
                    "saliency": "global_current_state_saliency_top_budget",
                    "lbi": "strict_budget_thresholded_gamma_no_topk_repair",
                }[selection],
                "budget_semantics": (
                    "global_conv_out_channel_floor_integer"
                    if conv
                    else "global_fc_scalar_floor_integer"
                )
                + ("_strict_rollback" if selection == "lbi" else ""),
                "group_partition_semantics": "out_channel" if conv else None,
            }
        )
    scientific = {
        "implementation_revision": config.get(
            "implementation_revision",
            NCTTA_LBI_IMPLEMENTATION_REVISION
            if sparse
            else NCTTA_IMPLEMENTATION_REVISION,
        ),
        "protocol_track": "nctta",
        "protocol_revision": config.get("protocol_revision"),
        "parent_nctta_baseline_protocol_revision": NCTTA_PROTOCOL_REVISION,
        "official_nctta_upstream_commit": "b4d442472a36af6b3f4d6e75f5138ca97d7d8eec",
        "method": "NCTTA",
        "task": config["task"],
        "dataset": data["dataset"],
        "source": int(data["source"]),
        "target": int(data["target"]),
        "seed": int(config["seed"]),
        "variant": variant,
        "requested_budget": budget,
        "integer_budget": integer_budget,
        "selection_seed": int(config["selection_seed"])
        if variant in RANDOM_VARIANTS
        else None,
        "num_random_masks": int(config["num_random_masks"])
        if variant in RANDOM_VARIANTS
        else None,
        "candidate_track": track,
        "candidate_scope": candidate_scope,
        "candidate_layer_names": candidate_names,
        "group_mode": "out_channel" if conv and sparse else None,
        "model": copy.deepcopy(config["model"]),
        "optimization": copy.deepcopy(
            config["native_optimizer"] if native else config["optimization"]
        ),
        "optimizer_provenance": "official_nctta_ttab_defaults"
        if native
        else "common_shot_optimizer_substrate",
        "loss": copy.deepcopy(config["loss"]),
        # These nested namespaces intentionally retain two separately owned
        # fields named nu; no unqualified nu is emitted.
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
        "variant_policy": policy,
        "feature_source": "post_netB_classifier_input",
        "classifier_reference": "effective_frozen_netC.fc.weight.detach",
        "primary_metric_name": (
            "fixed_12_class_mean_per_class_accuracy"
            if data["dataset"] == "VISDA-C"
            else "sample_level_overall_accuracy"
        ),
        "visda_fixed_class_count": 12 if data["dataset"] == "VISDA-C" else None,
        "objective_recomputation": (
            "every_stage1_candidate_and_stage2_current_state"
            if variant in LBI_VARIANTS
            else "current_state_once_per_outer_batch"
        ),
        "persistent_writeback": "lbi_omega_only"
        if variant in LBI_VARIANTS
        else ("masked_host_optimizer" if sparse else "host_optimizer"),
    }
    if variant in LBI_VARIANTS:
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
        scientific["host_optimizer_persistent_step"] = False
    return scientific
