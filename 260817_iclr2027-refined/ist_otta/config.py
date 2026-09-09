"""Strict IST configuration for dense and FC/Conv sparse families."""

import copy
import math
import os.path as osp

from experiment_identity import resolve_experiment_identity
from protocol_constants import (
    EFFICIENCY_PROTOCOL_REVISION,
    FORMAL_BUDGETS,
    FORMAL_SEED,
    IST_IMPLEMENTATION_REVISION,
    IST_LBI_FROZEN_CONSTANTS,
    IST_LBI_IMPLEMENTATION_REVISION,
    IST_LBI_PROTOCOL_REVISION,
    IST_PROTOCOL_REVISION,
    SOURCE_CHECKPOINT_REVISION,
)
from shot_otta.config import DATASETS, _absolute

from .sparse import (
    LBI_VARIANTS,
    RANDOM_VARIANTS,
    SALIENCY_VARIANTS,
    SPARSE_VARIANTS,
    validate_ratio,
    variant_track,
)


IST_VARIANTS = {
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
}
DENSE_VARIANTS = {
    "ist_full_dense", "ist_fc_module_dense", "ist_conv_module_dense"
}
LBI_TUNABLES = {"alpha", "kappa", "nu", "omega", "stage2_lr"}


def validate_ist_config(config):
    """Fail closed unless every frozen IST semantic is explicit."""

    ist = config.get("ist")
    if not isinstance(ist, dict):
        raise ValueError("method=IST requires an explicit ist config mapping")
    for key, value in {
        "extend": 8,
        "iters": 1,
        "ema_momentum": 0.9,
    }.items():
        if ist.get(key) != value:
            raise ValueError(f"IST requires ist.{key}={value}")
    nested_expected = {
        "plca": {
            "repeat": 1,
            "k": 50,
            "gamma": 3,
            "mode": "l2",
            "propagation_alpha": 0.99,
            "solver_max_steps": 20,
            "solver_tolerance": 1.0e-6,
        },
        "memory": {"max_len": 10000},
        "augmentation": {
            "source": "raw_pre_normalization",
            "resize_size": 256,
            "crop_size": 224,
            "horizontal_flip_probability": 0.5,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "rng": {
            "derivation": "formal_seed_plus_fixed_offset",
            "reference_seed_offset": 10000,
            "adaptation_seed_offset": 20000,
            "inner_order_seed_offset": 30000,
        },
    }
    for section, values in nested_expected.items():
        actual = ist.get(section)
        if not isinstance(actual, dict):
            raise ValueError(f"IST config requires ist.{section}")
        for key, value in values.items():
            if actual.get(key) != value:
                raise ValueError(
                    f"IST requires ist.{section}.{key}={value}"
                )
    chunk_size = ist["plca"].get("distance_chunk_size")
    if (
        isinstance(chunk_size, bool)
        or chunk_size is None
        or int(chunk_size) <= 0
    ):
        raise ValueError("ist.plca.distance_chunk_size must be positive")
    ist["plca"]["distance_chunk_size"] = int(chunk_size)
    objective_chunk_size = ist.get("objective_chunk_size")
    if objective_chunk_size is not None:
        if (
            isinstance(objective_chunk_size, bool)
            or int(objective_chunk_size) <= 0
        ):
            raise ValueError("ist.objective_chunk_size must be positive")
        ist["objective_chunk_size"] = int(objective_chunk_size)

    optimization = config["optimization"]
    for key, value in {
        "optimizer": "sgd",
        "lr_decay1": 0.1,
        "lr_decay2": 1.0,
        "momentum": 0.9,
        "weight_decay": 0.001,
        "nesterov": True,
        "lr_gamma": 10.0,
        "lr_power": 0.75,
    }.items():
        if optimization.get(key) != value:
            raise ValueError(
                f"IST common optimizer requires optimization.{key}={value}"
            )
    if config.get("loss") != {
        "components": ["hard_ce", "soft_kl"],
        "hard_ce_weight": 1.0,
        "soft_kl_weight": 1.0,
    }:
        raise ValueError("IST loss must be exactly hard CE + soft KL")


def _validate_lbi_tunables(lbi):
    if not isinstance(lbi, dict):
        raise ValueError("IST-LBI requires alpha/kappa/nu/omega/stage2_lr")
    if set(lbi) != LBI_TUNABLES:
        raise ValueError(
            "IST-LBI exposes only alpha, kappa, nu, omega, stage2_lr"
        )
    for key in LBI_TUNABLES:
        value = lbi[key]
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("IST-LBI " + key + " must be finite")
    for key in ("alpha", "kappa", "nu", "stage2_lr"):
        if float(lbi[key]) <= 0:
            raise ValueError("IST-LBI " + key + " must be > 0")
    if not 0.0 <= float(lbi["omega"]) <= 1.0:
        raise ValueError("IST-LBI omega must be in [0, 1]")
    return {key: float(lbi[key]) for key in sorted(LBI_TUNABLES)}


def _resolve_variant_fields(effective):
    variant = effective["variant"]
    if variant in DENSE_VARIANTS:
        if effective.get("lbi") is not None:
            raise ValueError("dense IST variants reject LBI config")
        if effective.get("group_mode") is not None:
            raise ValueError("dense IST variants reject group_mode")
        if effective.get("requested_budget") not in {None, 1, 1.0}:
            raise ValueError("dense IST variants require full budget=1")
        effective.update(
            {
                "requested_budget": 1.0,
                "selection_seed": None,
                "num_random_masks": None,
                "group_mode": None,
                "lbi_runtime": None,
            }
        )
        return

    requested_budget = validate_ratio(
        effective.get("requested_budget"), FORMAL_BUDGETS
    )
    effective["requested_budget"] = requested_budget
    track = variant_track(variant)
    if track == "conv_out_channel":
        group_mode = effective.get("group_mode")
        if group_mode not in {None, "out_channel"}:
            raise ValueError(
                "IST Conv sparse family supports out_channel only"
            )
        effective["group_mode"] = "out_channel"
    else:
        if effective.get("group_mode") is not None:
            raise ValueError("IST FC sparse family rejects group_mode")
        effective["group_mode"] = None

    if variant in RANDOM_VARIANTS:
        seed = effective.get("selection_seed")
        if seed is None:
            seed = effective["seed"]
        if isinstance(seed, bool) or int(seed) != seed:
            raise ValueError("IST Random selection_seed must be an integer")
        if (
            effective.get("formal_protocol")
            and int(seed) != int(effective["seed"])
        ):
            raise ValueError(
                "formal IST Random requires selection_seed == formal seed"
            )
        num_random_masks = effective.get("num_random_masks")
        if num_random_masks is None:
            num_random_masks = 3
        if num_random_masks != 3:
            raise ValueError(
                "IST Random requires exactly 3 independent child masks"
            )
        effective["selection_seed"] = int(seed)
        effective["num_random_masks"] = 3
    else:
        effective["selection_seed"] = None
        effective["num_random_masks"] = None

    if variant in LBI_VARIANTS:
        if effective.get("formal_protocol"):
            raise ValueError(
                "formal IST-LBI launch is blocked until tuned tuples freeze"
            )
        tunables = _validate_lbi_tunables(effective.get("lbi"))
        effective["lbi"] = tunables
        effective["lbi_runtime"] = {
            **tunables,
            **IST_LBI_FROZEN_CONSTANTS,
            "requested_budget": requested_budget,
            **(
                {"group_mode": "out_channel"}
                if track == "conv_out_channel" else {}
            ),
        }
    else:
        if effective.get("lbi") is not None:
            raise ValueError("non-LBI IST variants reject LBI config")
        effective["lbi_runtime"] = None


def resolve_effective_config(config, workspace_root):
    effective = copy.deepcopy(config)
    if effective.get("method") != "IST":
        raise ValueError("IST config resolver requires method=IST")
    if effective.get("task") != "otta":
        raise ValueError("IST config resolver requires task=otta")
    if effective.get("protocol_track") != "ist":
        raise ValueError("method=IST requires protocol_track=ist")
    if effective.get("variant") not in IST_VARIANTS:
        raise ValueError("unsupported IST variant: " + str(effective.get("variant")))

    if effective.get("formal_protocol"):
        dataset_name = effective.get("data", {}).get("dataset")
        settings = effective.get("formal_dataset_settings", {}).get(dataset_name)
        if settings is None:
            raise ValueError(
                f"formal IST protocol has no settings for {dataset_name}"
            )
        effective["data"].update(
            {
                "batch_size": int(settings["batch_size"]),
                "workers": int(settings["workers"]),
            }
        )
        effective["model"]["backbone"] = settings["backbone"]
        effective["optimization"]["lr"] = float(settings["lr"])
        if int(effective.get("seed", FORMAL_SEED)) != FORMAL_SEED:
            raise ValueError(f"formal IST protocol requires seed={FORMAL_SEED}")
        effective["seed"] = FORMAL_SEED

    _resolve_variant_fields(effective)
    is_sparse = effective["variant"] in SPARSE_VARIANTS
    effective["protocol_revision"] = (
        IST_LBI_PROTOCOL_REVISION if is_sparse else IST_PROTOCOL_REVISION
    )
    effective["implementation_revision"] = (
        IST_LBI_IMPLEMENTATION_REVISION
        if is_sparse
        else IST_IMPLEMENTATION_REVISION
    )
    effective["source_checkpoint_revision"] = SOURCE_CHECKPOINT_REVISION

    runtime = effective.setdefault("runtime", {})
    runtime.setdefault("workers_per_gpu", None)
    runtime.setdefault("runtime_comparable", False)
    runtime.setdefault("debug_max_outer_batches", None)
    debug_max_outer_batches = runtime["debug_max_outer_batches"]
    if debug_max_outer_batches is not None:
        if (
            isinstance(debug_max_outer_batches, bool)
            or int(debug_max_outer_batches) != debug_max_outer_batches
            or int(debug_max_outer_batches) < 1
        ):
            raise ValueError(
                "runtime.debug_max_outer_batches must be a positive integer"
            )
        if effective.get("formal_protocol"):
            raise ValueError(
                "formal protocol rejects runtime.debug_max_outer_batches"
            )
        runtime["debug_max_outer_batches"] = int(debug_max_outer_batches)
    if runtime["workers_per_gpu"] is not None:
        value = runtime["workers_per_gpu"]
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError("runtime.workers_per_gpu must be positive")
        runtime["workers_per_gpu"] = int(value)
    runtime["runtime_comparable"] = bool(
        runtime["runtime_comparable"] and runtime["workers_per_gpu"] == 1
    )
    runtime["efficiency_protocol_revision"] = EFFICIENCY_PROTOCOL_REVISION
    if effective["variant"] in SALIENCY_VARIANTS | LBI_VARIANTS:
        effective["ist"].setdefault(
            "objective_chunk_size", effective["data"]["batch_size"]
        )
    validate_ist_config(effective)

    data = effective["data"]
    dataset_name = data["dataset"]
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    dataset_info = DATASETS[dataset_name]
    domains = dataset_info["domains"]
    source = int(data["source"])
    target = int(data["target"])
    if source not in range(len(domains)) or target not in range(len(domains)):
        raise ValueError("invalid source/target domain index")
    if source == target:
        raise ValueError("Source and target domains must be different")
    data_root = _absolute(data["root"], workspace_root)
    source_name = domains[source]
    target_name = domains[target]
    data.update(
        {
            "root": data_root,
            "domains": domains,
            "source_name": source_name,
            "target_name": target_name,
            "source_list": osp.join(
                data_root, dataset_name, f"{source_name}_list.txt"
            ),
            "target_list": osp.join(
                data_root, dataset_name, f"{target_name}_list.txt"
            ),
            "test_list": osp.join(
                data_root, dataset_name, f"{target_name}_list.txt"
            ),
        }
    )
    effective["model"]["class_num"] = dataset_info["class_num"]
    if data["da"] != "uda":
        raise ValueError("IST formal families only define UDA")

    checkpoint_root = _absolute(
        effective["source_checkpoint"]["root"], workspace_root
    )
    effective["source_checkpoint"]["root"] = checkpoint_root
    effective["source_checkpoint"]["resolved_dir"] = osp.join(
        checkpoint_root, data["da"], dataset_name, source_name[0].upper()
    )
    effective["output"]["root"] = _absolute(
        effective["output"]["root"], workspace_root
    )
    effective["task_name"] = f"{source_name[0].upper()}{target_name[0].upper()}"
    identity = resolve_experiment_identity(
        effective,
        provided_key=effective.get("experiment_key"),
        provided_sha256=effective.get("experiment_config_sha256"),
    )
    effective["experiment_key"] = identity["experiment_key"]
    effective["experiment_config_sha256"] = identity[
        "experiment_config_sha256"
    ]
    effective["scientific_config"] = identity["scientific_config"]
    effective["implementation_revision"] = identity["scientific_config"][
        "implementation_revision"
    ]
    return effective
