"""Configuration loading, CLI overrides, validation, and path resolution."""

import copy
import math
import os.path as osp

import yaml

from experiment_identity import (
    EFFICIENCY_PROTOCOL_REVISION,
    resolve_experiment_identity,
)
from protocol_constants import (
    CONV_IMPLEMENTATION_REVISION,
    CONV_PROTOCOL_REVISION,
    FORMAL_SEED,
    LBI_FROZEN_CONSTANTS,
    PROTOCOL_REVISION,
    SOURCE_CHECKPOINT_REVISION,
)


DEFAULT_LBI_SUPPORT_THRESHOLD = 1.0e-4


DATASETS = {
    "office-home": {
        "domains": ["Art", "Clipart", "Product", "RealWorld"],
        "class_num": 65,
    },
    "office": {
        "domains": ["amazon", "dslr", "webcam"],
        "class_num": 31,
    },
    "VISDA-C": {
        "domains": ["train", "validation"],
        "class_num": 12,
    },
    "office-caltech": {
        "domains": ["amazon", "caltech", "dslr", "webcam"],
        "class_num": 10,
    },
}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        loaded = yaml.safe_load(file_obj)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return loaded


def _absolute(path, workspace_root):
    if osp.isabs(path):
        return osp.normpath(path)
    return osp.abspath(osp.join(workspace_root, path))


def apply_overrides(config, args):
    effective = copy.deepcopy(config)
    direct_overrides = {
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
    for (section, key), value in direct_overrides.items():
        if value is not None:
            effective[section][key] = value
    if getattr(args, "workers_per_gpu", None) is not None:
        effective.setdefault("runtime", {})[
            "workers_per_gpu"
        ] = args.workers_per_gpu
    if getattr(args, "runtime_comparable", False):
        effective.setdefault("runtime", {})[
            "runtime_comparable"
        ] = True
    if getattr(args, "debug_max_outer_batches", None) is not None:
        if effective.get("method") != "IST":
            raise ValueError(
                "--debug-max-outer-batches is an IST-only smoke option"
            )
        effective.setdefault("runtime", {})[
            "debug_max_outer_batches"
        ] = args.debug_max_outer_batches
    if args.seed is not None:
        effective["seed"] = args.seed
    if args.variant is not None:
        effective["variant"] = args.variant
    if getattr(args, "group_mode", None) is not None:
        effective["group_mode"] = args.group_mode
    if args.requested_budget is not None:
        effective["requested_budget"] = args.requested_budget
    if args.selection_seed is not None:
        effective["selection_seed"] = args.selection_seed
    if args.num_random_masks is not None:
        effective["num_random_masks"] = args.num_random_masks
    lbi_overrides = {
        "alpha": getattr(args, "lbi_alpha", None),
        "kappa": getattr(args, "lbi_kappa", None),
        "nu": getattr(args, "lbi_nu", None),
        "omega": getattr(args, "lbi_omega", None),
        "stage1_max_steps": getattr(
            args, "lbi_stage1_max_steps", None
        ),
        "budget_tolerance": getattr(
            args, "lbi_budget_tolerance", None
        ),
        "stage2_lr": getattr(args, "lbi_stage2_lr", None),
        "stage2_steps": getattr(args, "lbi_stage2_steps", None),
        "delta_nonzero_tolerance": getattr(
            args, "lbi_delta_nonzero_tolerance", None
        ),
        "support_threshold": getattr(
            args, "lbi_support_threshold", None
        ),
    }
    if any(value is not None for value in lbi_overrides.values()):
        if effective.get("lbi") is None:
            effective["lbi"] = {}
    for key, value in lbi_overrides.items():
        if value is not None:
            effective["lbi"][key] = value
    if args.experiment_key is not None:
        effective["experiment_key"] = args.experiment_key
    if args.experiment_config_sha256 is not None:
        effective["experiment_config_sha256"] = (
            args.experiment_config_sha256
        )
    if args.save_model:
        effective["output"]["save_model"] = True
    elif args.no_save_model:
        effective["output"]["save_model"] = False
    return effective


def resolve_effective_config(config, workspace_root):
    effective = copy.deepcopy(config)
    protocol_track = effective.setdefault("protocol_track", "fc")
    if protocol_track not in {"fc", "conv"}:
        raise ValueError("protocol_track must be either 'fc' or 'conv'")
    if protocol_track == "conv":
        effective.setdefault("conv_protocol_revision", CONV_PROTOCOL_REVISION)
        effective.setdefault(
            "implementation_revision", CONV_IMPLEMENTATION_REVISION
        )
    if effective.get("formal_protocol"):
        dataset_name = effective.get("data", {}).get("dataset")
        dataset_settings = effective.get("formal_dataset_settings", {}).get(
            dataset_name
        )
        if dataset_settings is None:
            raise ValueError(
                f"formal protocol has no dataset settings for {dataset_name}"
            )
        effective["data"].update(
            {
                "batch_size": int(dataset_settings["batch_size"]),
                "workers": int(dataset_settings["workers"]),
            }
        )
        effective["model"]["backbone"] = dataset_settings["backbone"]
        effective["optimization"]["lr"] = float(dataset_settings["lr"])
        if int(effective.get("seed", FORMAL_SEED)) != FORMAL_SEED:
            raise ValueError(f"formal protocol requires seed={FORMAL_SEED}")
        effective["seed"] = FORMAL_SEED
        effective["source_checkpoint_revision"] = SOURCE_CHECKPOINT_REVISION
        if protocol_track == "fc":
            effective["protocol_revision"] = PROTOCOL_REVISION
        else:
            effective["protocol_revision"] = CONV_PROTOCOL_REVISION
            effective["conv_protocol_revision"] = CONV_PROTOCOL_REVISION
            effective["implementation_revision"] = CONV_IMPLEMENTATION_REVISION
        effective["lbi_protocol"] = copy.deepcopy(LBI_FROZEN_CONSTANTS)
    runtime = effective.setdefault("runtime", {})
    runtime.setdefault("workers_per_gpu", None)
    runtime.setdefault("runtime_comparable", False)
    if runtime["workers_per_gpu"] is not None:
        if (
            isinstance(runtime["workers_per_gpu"], bool)
            or int(runtime["workers_per_gpu"]) != runtime["workers_per_gpu"]
            or int(runtime["workers_per_gpu"]) < 1
        ):
            raise ValueError(
                "runtime.workers_per_gpu must be a positive integer"
            )
        runtime["workers_per_gpu"] = int(runtime["workers_per_gpu"])
    runtime["runtime_comparable"] = bool(
        runtime["runtime_comparable"]
        and runtime["workers_per_gpu"] == 1
    )
    runtime["efficiency_protocol_revision"] = (
        EFFICIENCY_PROTOCOL_REVISION
    )
    if protocol_track == "fc":
        effective.setdefault("requested_budget", 0.1)
    effective.setdefault("selection_seed", effective.get("seed", FORMAL_SEED))
    effective.setdefault("num_random_masks", 3)
    if effective.get("method") != "shot":
        raise ValueError("The first-stage refactor only supports method=shot")
    if effective.get("task") != "otta":
        raise ValueError("The first-stage refactor only supports task=otta")
    supported_variants = {
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
    }
    if effective.get("variant") not in supported_variants:
        raise ValueError(
            "variant must be one of: source_only, full_dense, "
            "module_dense, module_random, module_magnitude, "
            "module_saliency, module_lbi, conv_module_dense, "
            "conv_out_random, conv_out_magnitude, conv_out_saliency, "
            "conv_out_lbi, conv_filter_random, conv_filter_magnitude, "
            "conv_filter_saliency, conv_filter_lbi"
        )
    is_conv_variant = effective["variant"].startswith("conv_")
    if is_conv_variant and protocol_track != "conv":
        raise ValueError("Conv variants require protocol_track=conv")
    if protocol_track == "conv" and effective["variant"].startswith("module_"):
        raise ValueError("FC module variants require protocol_track=fc")
    if (
        protocol_track == "fc"
        and effective["variant"].startswith("module_") is False
        and is_conv_variant is False
        and effective["variant"] not in {"source_only", "full_dense"}
    ):
        raise ValueError("unsupported FC protocol variant")
    optimization = effective["optimization"]
    if effective["variant"] == "full_dense" and (
        optimization["lr_decay1"] <= 0
        or optimization["lr_decay2"] <= 0
    ):
        raise ValueError(
            "full_dense requires positive lr_decay1 and lr_decay2"
        )
    if (
        effective["variant"].startswith("module_")
        and optimization["lr_decay2"] <= 0
    ):
        raise ValueError("FC module variants require positive lr_decay2")
    if (
        effective["variant"].startswith("conv_")
        and optimization["lr_decay1"] <= 0
    ):
        raise ValueError("Conv variants require positive lr_decay1")
    conv_sparse_variants = {
        "conv_out_random",
        "conv_out_magnitude",
        "conv_out_saliency",
        "conv_out_lbi",
        "conv_filter_random",
        "conv_filter_magnitude",
        "conv_filter_saliency",
        "conv_filter_lbi",
    }
    if effective["variant"] == "source_only":
        effective["requested_budget"] = None
    elif effective["variant"] in {
        "full_dense",
        "module_dense",
        "conv_module_dense",
    }:
        effective["requested_budget"] = 1.0
    else:
        if (
            effective["variant"] in conv_sparse_variants
            and effective.get("requested_budget") is None
        ):
            raise ValueError(
                "sparse Conv variants require an explicit requested_budget; "
                "the Conv rho grid is TBD"
            )
        requested_budget = float(effective["requested_budget"])
        if not 0.0 <= requested_budget <= 1.0:
            raise ValueError("requested_budget must be in [0, 1]")
        effective["requested_budget"] = requested_budget
    if effective["variant"] in {"module_random", *conv_sparse_variants.intersection({
        "conv_out_random", "conv_filter_random"
    })}:
        effective["selection_seed"] = int(effective["selection_seed"])
        num_random_masks = effective["num_random_masks"]
        if (
            isinstance(num_random_masks, bool)
            or int(num_random_masks) != num_random_masks
            or int(num_random_masks) <= 0
        ):
            raise ValueError("num_random_masks must be a positive integer")
        effective["num_random_masks"] = int(num_random_masks)
    else:
        effective["selection_seed"] = None
        effective["num_random_masks"] = None
    if effective["variant"] in {"module_lbi", "conv_out_lbi", "conv_filter_lbi"}:
        if effective.get("lbi") is None:
            if not effective.get("allow_unresolved_lbi", False):
                raise ValueError(
                    "LBI variants require a resolved tuned tuple; "
                    "formal launch is blocked until it is provided"
                )
        else:
            if effective["variant"] in {"conv_out_lbi", "conv_filter_lbi"}:
                # Only alpha/kappa/nu/omega/stage2_lr are Conv tuning values.
                # Fill every frozen correctness constant from the protocol so
                # direct commands cannot omit or silently redefine them.
                for key, value in LBI_FROZEN_CONSTANTS.items():
                    effective["lbi"].setdefault(key, value)
            validate_lbi_config(effective.get("lbi"))
            if effective["variant"] in {"conv_out_lbi", "conv_filter_lbi"}:
                frozen_conv_lbi = {
                    "support_threshold": DEFAULT_LBI_SUPPORT_THRESHOLD,
                    "stage1_max_steps": 3000,
                    "stage2_steps": 1,
                    "delta_nonzero_tolerance": 1.0e-12,
                    "budget_tolerance": 1.0e-4,
                }
                for key, expected in frozen_conv_lbi.items():
                    if effective["lbi"][key] != expected:
                        raise ValueError(
                            f"Conv LBI requires frozen {key}={expected}"
                        )
    if effective["variant"] in conv_sparse_variants:
        group_mode = effective.get("group_mode")
        expected_mode = (
            "out_channel"
            if effective["variant"].startswith("conv_out_")
            else "filter_connection"
        )
        if group_mode != expected_mode:
            raise ValueError(
                f"{effective['variant']} requires group_mode={expected_mode}"
            )
    elif effective["variant"] == "conv_module_dense":
        effective["group_mode"] = None

    data_config = effective["data"]
    dataset_name = data_config["dataset"]
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_info = DATASETS[dataset_name]
    domains = dataset_info["domains"]
    source = int(data_config["source"])
    target = int(data_config["target"])
    if source < 0 or source >= len(domains):
        raise ValueError(f"Invalid source index {source} for {dataset_name}")
    if target < 0 or target >= len(domains):
        raise ValueError(f"Invalid target index {target} for {dataset_name}")
    if source == target:
        raise ValueError("Source and target domains must be different")

    data_root = _absolute(data_config["root"], workspace_root)
    source_name = domains[source]
    target_name = domains[target]
    data_config.update(
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

    if data_config["da"] == "pda":
        if dataset_name != "office-home":
            raise ValueError("The migrated PDA mapping is defined for office-home only")
        data_config["source_classes"] = list(range(65))
        data_config["target_classes"] = list(range(25))

    checkpoint_root = _absolute(
        effective["source_checkpoint"]["root"], workspace_root
    )
    effective["source_checkpoint"]["root"] = checkpoint_root
    effective["source_checkpoint"]["resolved_dir"] = osp.join(
        checkpoint_root,
        data_config["da"],
        dataset_name,
        source_name[0].upper(),
    )
    effective["output"]["root"] = _absolute(
        effective["output"]["root"], workspace_root
    )
    effective["task_name"] = (
        f"{source_name[0].upper()}{target_name[0].upper()}"
    )
    identity = resolve_experiment_identity(
        effective,
        provided_key=effective.get("experiment_key"),
        provided_sha256=effective.get("experiment_config_sha256"),
    )
    effective["experiment_key"] = identity["experiment_key"]
    effective["experiment_config_sha256"] = identity[
        "experiment_config_sha256"
    ]
    effective["implementation_revision"] = identity[
        "scientific_config"
    ]["implementation_revision"]
    effective["scientific_config"] = identity["scientific_config"]
    return effective


def validate_lbi_config(lbi):
    if not isinstance(lbi, dict):
        raise ValueError("LBI variants require an lbi config mapping")
    required = {
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
    }
    lbi.setdefault("support_threshold", DEFAULT_LBI_SUPPORT_THRESHOLD)
    missing = sorted(required - set(lbi))
    if missing:
        raise ValueError(f"Missing LBI config fields: {missing}")
    forbidden = sorted({"init_mode", "stage3_mode"} & set(lbi))
    if forbidden:
        raise ValueError(
            f"LBI semantics are fixed; forbidden config fields: {forbidden}"
        )
    for key in (
        "alpha",
        "kappa",
        "nu",
        "omega",
        "budget_tolerance",
        "stage2_lr",
        "delta_nonzero_tolerance",
        "support_threshold",
    ):
        value = float(lbi[key])
        if not math.isfinite(value):
            raise ValueError(f"lbi.{key} must be finite")
        lbi[key] = value
    if lbi["alpha"] <= 0:
        raise ValueError("lbi.alpha must be > 0")
    if lbi["kappa"] <= 0:
        raise ValueError("lbi.kappa must be > 0")
    if lbi["nu"] <= 0:
        raise ValueError("lbi.nu must be > 0")
    if not 0 <= lbi["omega"] <= 1:
        raise ValueError("lbi.omega must be in [0, 1]")
    if lbi["budget_tolerance"] < 0:
        raise ValueError("lbi.budget_tolerance must be >= 0")
    if lbi["stage2_lr"] <= 0:
        raise ValueError("lbi.stage2_lr must be > 0")
    if lbi["delta_nonzero_tolerance"] < 0:
        raise ValueError(
            "lbi.delta_nonzero_tolerance must be >= 0"
        )
    if lbi["support_threshold"] < 0:
        raise ValueError("lbi.support_threshold must be >= 0")
    for key in ("stage1_max_steps", "stage2_steps"):
        value = lbi[key]
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError(f"lbi.{key} must be a positive integer")
        lbi[key] = int(value)


def dump_yaml(path, config):
    with open(path, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(
            config,
            file_obj,
            sort_keys=False,
            allow_unicode=True,
        )
