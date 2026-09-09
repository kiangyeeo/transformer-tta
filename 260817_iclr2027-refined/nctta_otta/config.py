"""Strict configuration for controlled and native-scope NCTTA references."""

import copy
import math
import os.path as osp

from experiment_identity import resolve_experiment_identity
from protocol_constants import (
    EFFICIENCY_PROTOCOL_REVISION,
    FORMAL_SEED,
    NCTTA_IMPLEMENTATION_REVISION,
    NCTTA_PROTOCOL_REVISION,
    SOURCE_CHECKPOINT_REVISION,
)
from shot_otta.config import DATASETS, _absolute
from .sparse import SPARSE_VARIANTS
from .sparse_config import apply_sparse_overrides, resolve_sparse_config


NCTTA_VARIANTS = {
    "nctta_full_dense",
    "nctta_native_norm",
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
}
NCTTA_DENSE_VARIANTS = NCTTA_VARIANTS - SPARSE_VARIANTS

OFFICIAL_NATIVE_OPTIMIZER = {
    "optimizer": "sgd",
    "lr": 1.0e-3,
    "momentum": 0.9,
    "dampening": 0.0,
    "weight_decay": 0.0,
    "nesterov": True,
    "scheduler": "none",
}

OFFICIAL_NCTTA_DEFAULTS = {
    "thre_ent": 0.4 * math.log(1000),
    "margin_ent": 0.4 * math.log(1000),
    "reweight_ent": 1.0,
    "nu": 5.0,
    "eta": 1.0,
    "scale": 5.0,
    "top_k": 10,
    "mix_prob_weight": 0.3,
}

H31 = 0.4 * math.log(31)
NCTTA_HOST_OBJECTIVE_DIAGNOSTIC_TUPLES = {
    "A": {
        **OFFICIAL_NCTTA_DEFAULTS,
    },
    "B": {
        **OFFICIAL_NCTTA_DEFAULTS,
        "thre_ent": H31,
        "margin_ent": H31,
    },
    "C": {
        **OFFICIAL_NCTTA_DEFAULTS,
        "thre_ent": H31,
        "margin_ent": H31,
        "scale": 1.0,
    },
    "D": {
        **OFFICIAL_NCTTA_DEFAULTS,
        "thre_ent": H31,
        "margin_ent": H31,
        "scale": 1.0,
        "nu": 1.0,
    },
}

OFFICIAL_LOSS_PATH = {
    "nc_type": "infonce",
    "metric": "cos",
    "tau_align": 1.0,
    "margin": 0.2,
    "nc_reduction": "none",
    "batch_reduction": "weighted_selected_sample_mean",
}


def apply_overrides(config, args):
    """Apply only objective-agnostic CLI overrides accepted by NCTTA P1."""

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
    if args.seed is not None:
        effective["seed"] = args.seed
    if args.variant is not None:
        effective["variant"] = args.variant
    if getattr(args, "workers_per_gpu", None) is not None:
        effective.setdefault("runtime", {})["workers_per_gpu"] = args.workers_per_gpu
    if getattr(args, "runtime_comparable", False):
        effective.setdefault("runtime", {})["runtime_comparable"] = True
    if getattr(args, "debug_max_outer_batches", None) is not None:
        effective.setdefault("runtime", {})["debug_max_outer_batches"] = (
            args.debug_max_outer_batches
        )
    if args.experiment_key is not None:
        effective["experiment_key"] = args.experiment_key
    if args.experiment_config_sha256 is not None:
        effective["experiment_config_sha256"] = args.experiment_config_sha256
    if args.save_model:
        effective["output"]["save_model"] = True
    elif args.no_save_model:
        effective["output"]["save_model"] = False
    forbidden = {
        "group_mode": getattr(args, "group_mode", None),
        "requested_budget": getattr(args, "requested_budget", None),
        "selection_seed": getattr(args, "selection_seed", None),
        "num_random_masks": getattr(args, "num_random_masks", None),
        "lbi_alpha": getattr(args, "lbi_alpha", None),
        "lbi_kappa": getattr(args, "lbi_kappa", None),
        "lbi_nu": getattr(args, "lbi_nu", None),
        "lbi_omega": getattr(args, "lbi_omega", None),
        "lbi_stage1_max_steps": getattr(args, "lbi_stage1_max_steps", None),
        "lbi_budget_tolerance": getattr(args, "lbi_budget_tolerance", None),
        "lbi_stage2_lr": getattr(args, "lbi_stage2_lr", None),
        "lbi_stage2_steps": getattr(args, "lbi_stage2_steps", None),
        "lbi_delta_nonzero_tolerance": getattr(
            args, "lbi_delta_nonzero_tolerance", None
        ),
        "lbi_support_threshold": getattr(args, "lbi_support_threshold", None),
    }
    supplied = sorted(key for key, value in forbidden.items() if value is not None)
    if supplied:
        raise ValueError(
            "NCTTA P1 dense variants reject sparse/LBI overrides: "
            + ", ".join(supplied)
        )
    return effective


def _validate_nctta(config):
    params = config.get("nctta")
    if not isinstance(params, dict) or set(params) != set(OFFICIAL_NCTTA_DEFAULTS):
        raise ValueError(
            "NCTTA P1 requires exactly these explicit method parameters: "
            + ", ".join(OFFICIAL_NCTTA_DEFAULTS)
        )
    diagnostic = config.get("diagnostic")
    if diagnostic is None:
        expected_params = OFFICIAL_NCTTA_DEFAULTS
    else:
        if not isinstance(diagnostic, dict) or set(diagnostic) != {
            "name",
            "debug_only",
            "tuple",
        }:
            raise ValueError(
                "NCTTA diagnostic must contain exactly name/debug_only/tuple"
            )
        tuple_name = diagnostic["tuple"]
        if (
            diagnostic["name"] != "nctta_host_objective"
            or diagnostic["debug_only"] is not True
            or tuple_name not in NCTTA_HOST_OBJECTIVE_DIAGNOSTIC_TUPLES
        ):
            raise ValueError("unsupported NCTTA diagnostic registration")
        expected_params = NCTTA_HOST_OBJECTIVE_DIAGNOSTIC_TUPLES[tuple_name]
    for key, expected in expected_params.items():
        value = params[key]
        if key == "top_k":
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError("nctta.top_k must be a positive integer")
            params[key] = int(value)
        else:
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"nctta.{key} must be finite")
            params[key] = float(value)
        if params[key] != expected:
            raise ValueError(
                "NCTTA objective is frozen to the registered tuple: "
                f"nctta.{key}={expected}; got {params[key]}"
            )
    if config.get("loss") != OFFICIAL_LOSS_PATH:
        raise ValueError("NCTTA loss path must match the official fixed call")


def _validate_optimizer(config):
    optimization = config.get("optimization", {})
    expected = {
        "optimizer": "sgd",
        "lr_decay1": 0.1,
        "lr_decay2": 1.0,
        "momentum": 0.9,
        "weight_decay": 0.001,
        "nesterov": True,
        "lr_gamma": 10.0,
        "lr_power": 0.75,
    }
    for key, value in expected.items():
        if optimization.get(key) != value:
            raise ValueError(
                f"NCTTA common optimizer requires optimization.{key}={value}"
            )
    lr = optimization.get("lr")
    if isinstance(lr, bool) or not math.isfinite(float(lr)) or float(lr) <= 0:
        raise ValueError("optimization.lr must be finite and positive")
    optimization["lr"] = float(lr)

    native = config.get("native_optimizer")
    if not isinstance(native, dict) or set(native) != set(OFFICIAL_NATIVE_OPTIMIZER):
        raise ValueError(
            "NCTTA native_optimizer must explicitly record the official "
            "optimizer fields"
        )
    for key, expected_value in OFFICIAL_NATIVE_OPTIMIZER.items():
        if native.get(key) != expected_value:
            raise ValueError(f"NCTTA native_optimizer.{key} must be {expected_value}")


def optimizer_config_for_variant(config):
    """Return the optimizer semantics that actually drive this variant."""
    return (
        config["native_optimizer"]
        if config["variant"] == "nctta_native_norm"
        else config["optimization"]
    )


def resolve_effective_config(config, workspace_root):
    effective = copy.deepcopy(config)
    if effective.get("method") != "NCTTA":
        raise ValueError("NCTTA config resolver requires method=NCTTA")
    if effective.get("task") != "otta":
        raise ValueError("NCTTA config resolver requires task=otta")
    if effective.get("protocol_track") != "nctta":
        raise ValueError("method=NCTTA requires protocol_track=nctta")
    if effective.get("variant") not in NCTTA_VARIANTS:
        raise ValueError("unsupported NCTTA variant: " + str(effective.get("variant")))
    if effective.get("group_mode") is not None or effective.get("lbi") is not None:
        raise ValueError("NCTTA baseline implements native/dense variants only")
    if effective.get("requested_budget") not in {None, 1, 1.0}:
        raise ValueError("NCTTA dense variants require full budget=1")
    effective.update(
        {
            "requested_budget": 1.0,
            "group_mode": None,
            "selection_seed": None,
            "num_random_masks": None,
            "protocol_revision": NCTTA_PROTOCOL_REVISION,
            "implementation_revision": NCTTA_IMPLEMENTATION_REVISION,
            "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
        }
    )

    if effective.get("formal_protocol"):
        raise ValueError(
            "NCTTA P1 official log(1000) defaults are correctness-smoke only; "
            "formal launch remains blocked until dataset-specific values freeze"
        )
    if int(effective.get("seed", FORMAL_SEED)) != FORMAL_SEED:
        raise ValueError(f"NCTTA P1 requires seed={FORMAL_SEED}")
    effective["seed"] = FORMAL_SEED

    runtime = effective.setdefault("runtime", {})
    runtime.setdefault("workers_per_gpu", None)
    runtime.setdefault("runtime_comparable", False)
    runtime.setdefault("debug_max_outer_batches", None)
    debug_limit = runtime["debug_max_outer_batches"]
    if debug_limit is not None:
        if (
            isinstance(debug_limit, bool)
            or int(debug_limit) != debug_limit
            or int(debug_limit) < 1
        ):
            raise ValueError("runtime.debug_max_outer_batches must be positive")
        runtime["debug_max_outer_batches"] = int(debug_limit)
    workers_per_gpu = runtime["workers_per_gpu"]
    if workers_per_gpu is not None:
        if (
            isinstance(workers_per_gpu, bool)
            or int(workers_per_gpu) != workers_per_gpu
            or int(workers_per_gpu) < 1
        ):
            raise ValueError("runtime.workers_per_gpu must be positive")
        runtime["workers_per_gpu"] = int(workers_per_gpu)
    runtime["runtime_comparable"] = bool(
        runtime["runtime_comparable"] and runtime["workers_per_gpu"] == 1
    )
    runtime["efficiency_protocol_revision"] = EFFICIENCY_PROTOCOL_REVISION

    diagnostic = effective.get("diagnostic")
    if diagnostic is not None:
        if effective.get("formal_protocol"):
            raise ValueError("NCTTA host-objective diagnostic must be non-formal")
        if runtime["debug_max_outer_batches"] != 5:
            raise ValueError(
                "NCTTA host-objective diagnostic requires exactly 5 outer batches"
            )
        if effective["variant"] not in {
            "nctta_native_norm",
            "nctta_fc_module_dense",
            "nctta_conv_module_dense",
        }:
            raise ValueError("unsupported host-objective diagnostic variant")

    _validate_nctta(effective)
    _validate_optimizer(effective)
    data = effective["data"]
    dataset_name = data["dataset"]
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    dataset_info = DATASETS[dataset_name]
    domains = dataset_info["domains"]
    source = int(data["source"])
    target = int(data["target"])
    if diagnostic is not None and (dataset_name, source, target) != (
        "office",
        1,
        0,
    ):
        raise ValueError("NCTTA host-objective diagnostic is fixed to Office D->A")
    if source not in range(len(domains)) or target not in range(len(domains)):
        raise ValueError("invalid source/target domain index")
    if source == target:
        raise ValueError("Source and target domains must be different")
    source_name = domains[source]
    target_name = domains[target]
    data_root = _absolute(data["root"], workspace_root)
    data.update(
        {
            "root": data_root,
            "domains": domains,
            "source_name": source_name,
            "target_name": target_name,
            "source_list": osp.join(data_root, dataset_name, f"{source_name}_list.txt"),
            "target_list": osp.join(data_root, dataset_name, f"{target_name}_list.txt"),
            "test_list": osp.join(data_root, dataset_name, f"{target_name}_list.txt"),
        }
    )
    effective["model"]["class_num"] = dataset_info["class_num"]
    if data.get("da") != "uda":
        raise ValueError("NCTTA P1 controlled families define UDA only")
    expected = {
        "office": ("resnet50", 64, 0.01),
        "VISDA-C": ("resnet101", 256, 0.001),
    }
    if dataset_name not in expected:
        raise ValueError("NCTTA P1 supports only Office-31 and VisDA-C")
    backbone, batch_size, learning_rate = expected[dataset_name]
    if effective["model"].get("backbone") != backbone:
        raise ValueError(f"NCTTA {dataset_name} requires {backbone}")
    if int(data.get("batch_size")) != batch_size:
        raise ValueError(f"NCTTA {dataset_name} requires batch_size={batch_size}")
    if float(effective["optimization"]["lr"]) != learning_rate:
        raise ValueError(f"NCTTA {dataset_name} requires lr={learning_rate}")

    checkpoint_root = _absolute(effective["source_checkpoint"]["root"], workspace_root)
    effective["source_checkpoint"]["root"] = checkpoint_root
    effective["source_checkpoint"]["resolved_dir"] = osp.join(
        checkpoint_root, data["da"], dataset_name, source_name[0].upper()
    )
    effective["output"]["root"] = _absolute(effective["output"]["root"], workspace_root)
    effective["task_name"] = f"{source_name[0].upper()}{target_name[0].upper()}"
    identity = resolve_experiment_identity(
        effective,
        provided_key=effective.get("experiment_key"),
        provided_sha256=effective.get("experiment_config_sha256"),
    )
    effective["experiment_key"] = identity["experiment_key"]
    effective["experiment_config_sha256"] = identity["experiment_config_sha256"]
    effective["scientific_config"] = identity["scientific_config"]
    effective["implementation_revision"] = identity["scientific_config"][
        "implementation_revision"
    ]
    return effective


# Sparse dispatch is layered after the audited dense resolver so dense/native
# semantics remain unchanged.
_dense_apply_overrides = apply_overrides
_dense_resolve_effective_config = resolve_effective_config


def apply_overrides(config, args):
    return apply_sparse_overrides(config, args, _dense_apply_overrides)


def resolve_effective_config(config, workspace_root):
    return resolve_sparse_config(
        config, workspace_root, _dense_resolve_effective_config
    )
