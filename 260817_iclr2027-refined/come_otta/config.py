"""Strict configuration for the formal controlled dense COME baseline."""

import copy
import hashlib
import json
import math
import os.path as osp
from functools import lru_cache

from protocol_constants import EFFICIENCY_PROTOCOL_REVISION
from shot_otta.config import DATASETS, _absolute
from shot_otta.data import resolve_target_order

from .objective import (
    OFFICIAL_COME_COMMIT,
    OFFICIAL_ENTROPY_EPSILON,
    OFFICIAL_P,
    OFFICIAL_TAU,
)


COME_VARIANTS = {
    "come_full_dense",
    "come_fc_module_dense",
    "come_conv_module_dense",
}
COME_PROTOCOL_REVISION = "OTTA_COME_BASELINE_PROTOCOL_20260907_v1"
COME_IMPLEMENTATION_REVISION = "come_otta_baseline_20260908_v2"
SOURCE_CHECKPOINT_REVISION = "nips2026_shot_otta_uda_source_v1"
FORMAL_SEED = 2026

FORMAL_DATASET_SETTINGS = {
    "office": {
        "backbone": "resnet50",
        "batch_size": 64,
        "workers": 4,
        "class_count": 31,
        "base_lr": 0.01,
        "primary_metric_name": "sample_level_overall_accuracy",
    },
    "VISDA-C": {
        "backbone": "resnet101",
        "batch_size": 256,
        "workers": 4,
        "class_count": 12,
        "base_lr": 0.001,
        "primary_metric_name": "fixed_12_class_mAcc",
    },
}

SCOPE_METADATA = {
    "come_full_dense": {
        "candidate_scope": "netF+netB",
        "candidate_layer_names": "all_netF_and_netB_parameters",
        "candidate_tensor_count": {"office": 163, "VISDA-C": 316},
        "candidate_scalar_count": {"office": 24033088, "VISDA-C": 43025216},
        "bn_semantics": "shot_full_dense_native_train_running_buffers_adaptive",
    },
    "come_fc_module_dense": {
        "candidate_scope": "netB.bottleneck.weight+bias",
        "candidate_layer_names": [
            "netB.bottleneck.weight",
            "netB.bottleneck.bias",
        ],
        "candidate_tensor_count": 2,
        "candidate_scalar_count": 524544,
        "bn_semantics": "all_batchnorm_parameters_and_buffers_frozen_eval",
    },
    "come_conv_module_dense": {
        "candidate_scope": "netF.layer4_9_conv_weights",
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
        ],
        "candidate_tensor_count": 9,
        "candidate_scalar_count": 12845056,
        "bn_semantics": "all_batchnorm_parameters_and_buffers_frozen_eval",
    },
}


def apply_overrides(config, args):
    """Apply only overrides that do not create a sparse COME variant."""

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
            "COME dense baselines reject sparse/LBI overrides: " + ", ".join(supplied)
        )
    return effective


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _cached_source_identity(checkpoint_dir):
    paths = {
        "netF": osp.join(checkpoint_dir, "source_F.pt"),
        "netB": osp.join(checkpoint_dir, "source_B.pt"),
        "netC": osp.join(checkpoint_dir, "source_C.pt"),
    }
    missing = [path for path in paths.values() if not osp.isfile(path)]
    if missing:
        raise FileNotFoundError(f"COME source checkpoints are missing: {missing}")
    return {
        name: {"resolved_path": path, "sha256": _sha256_file(path)}
        for name, path in paths.items()
    }


def _source_identity(checkpoint_dir):
    return copy.deepcopy(_cached_source_identity(checkpoint_dir))


def _target_stream_identity(data, seed):
    target_list = data["target_list"]
    with open(target_list, "r", encoding="utf-8") as file_obj:
        target_size = sum(1 for line in file_obj if line.strip())
    order, order_record = resolve_target_order(
        {"seed": seed, "data": {"dataset": data["dataset"]}}, target_size
    )
    order_payload = ",".join(str(index) for index in order).encode("utf-8")
    return {
        "target_list_resolved_path": target_list,
        "target_list_sha256": _sha256_file(target_list),
        "target_sample_count": target_size,
        "target_order_sampler": order_record["sampler"],
        "target_order_sha256": hashlib.sha256(order_payload).hexdigest(),
        "outer_batch_partition": f"contiguous_chunks_bs{int(data['batch_size'])}",
    }


def _scope_identity(variant, dataset):
    scope = copy.deepcopy(SCOPE_METADATA[variant])
    for key in ("candidate_tensor_count", "candidate_scalar_count"):
        if isinstance(scope[key], dict):
            scope[key] = scope[key][dataset]
    return scope


def _identity_payload(config):
    data = config["data"]
    dataset = data["dataset"]
    settings = FORMAL_DATASET_SETTINGS[dataset]
    return {
        "protocol_revision": COME_PROTOCOL_REVISION,
        "implementation_revision": COME_IMPLEMENTATION_REVISION,
        "official_come_commit": OFFICIAL_COME_COMMIT,
        "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
        "source_checkpoints": copy.deepcopy(config["source_checkpoint_identity"]),
        "method": "COME",
        "task": "otta",
        "variant": config["variant"],
        "dataset": dataset,
        "source": int(data["source"]),
        "target": int(data["target"]),
        "transfer": config["task_name"],
        "backbone": config["model"]["backbone"],
        "seed": int(config["seed"]),
        "outer_batch_size": int(data["batch_size"]),
        "target_passes": 1,
        "drop_last": False,
        "target_stream": copy.deepcopy(config["target_stream_identity"]),
        "singleton_outer_batch_policy": (
            "skip_size_1_before_objective_scheduler_optimizer_bn_and_pu"
        ),
        **_scope_identity(config["variant"], dataset),
        "come": {
            "objective": "mean_entropy_of_subjective_opinion",
            "p": OFFICIAL_P,
            "tau": OFFICIAL_TAU,
            "class_count": settings["class_count"],
            "opinion_eps": OFFICIAL_ENTROPY_EPSILON,
            "norm_dim": -1,
            "norm_epsilon": None,
            "norm_detach_semantics": "multiplicative_logit_norm_only",
            "evidence": "exp_constrained_logits",
            "strength": "sum_evidence_plus_dataset_class_count",
            "belief": "evidence_div_strength",
            "uncertainty": "dataset_class_count_div_strength",
            "numerical_evaluation": "stable_log_domain_logsumexp",
        },
        "optimizer": {
            "family": "sgd",
            "base_lr": settings["base_lr"],
            "netF_multiplier": 0.1,
            "netB_multiplier": 1.0,
            "momentum": 0.9,
            "weight_decay": 0.001,
            "nesterov": True,
            "scheduler": "outer_poly_(1+10*t/T)^(-0.75)",
            "scheduler_timeline": "processed_outer_stream_over_full_loader_T",
        },
        "primary_metric_name": settings["primary_metric_name"],
        "fixed_class_count": 12 if dataset == "VISDA-C" else None,
        "formal_no_partial_resume": True,
        "save_model": False,
    }


def resolve_effective_config(config, workspace_root):
    """Resolve the formal baseline or its bounded non-formal reviewed smoke."""

    effective = copy.deepcopy(config)
    if effective.get("method") != "COME" or effective.get("task") != "otta":
        raise ValueError("COME baseline config requires method=COME and task=otta")
    if effective.get("protocol_track") != "come_baseline":
        raise ValueError("COME baseline requires protocol_track=come_baseline")
    if effective.get("variant") not in COME_VARIANTS:
        raise ValueError("unsupported formal COME dense variant")
    if effective.get("requested_budget") not in {None, 1, 1.0}:
        raise ValueError("COME dense variants require full budget=1")
    if effective.get("group_mode") is not None or effective.get("lbi") is not None:
        raise ValueError("COME dense baselines reject sparse/LBI settings")
    if int(effective.get("seed")) != FORMAL_SEED:
        raise ValueError(f"COME baseline requires seed={FORMAL_SEED}")
    effective.update(
        {
            "seed": FORMAL_SEED,
            "requested_budget": 1.0,
            "group_mode": None,
            "selection_seed": None,
            "num_random_masks": None,
            "protocol_revision": COME_PROTOCOL_REVISION,
            "implementation_revision": COME_IMPLEMENTATION_REVISION,
            "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
        }
    )

    formal = effective.get("formal_protocol")
    if not isinstance(formal, bool):
        raise ValueError("COME baseline formal_protocol must be boolean")
    runtime = effective.setdefault("runtime", {})
    runtime.setdefault("workers_per_gpu", None)
    runtime.setdefault("runtime_comparable", False)
    runtime.setdefault("debug_max_outer_batches", None)
    debug_limit = runtime["debug_max_outer_batches"]
    if formal and debug_limit is not None:
        raise ValueError("formal COME baseline rejects debug_max_outer_batches")
    if debug_limit is not None:
        if isinstance(debug_limit, bool) or int(debug_limit) != debug_limit:
            raise ValueError("debug_max_outer_batches must be a positive integer")
        debug_limit = int(debug_limit)
        if debug_limit <= 0:
            raise ValueError("debug_max_outer_batches must be a positive integer")
        runtime["debug_max_outer_batches"] = debug_limit
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

    data = effective.get("data", {})
    dataset = data.get("dataset")
    if dataset not in FORMAL_DATASET_SETTINGS:
        raise ValueError("COME baseline supports only office and VISDA-C")
    settings = FORMAL_DATASET_SETTINGS[dataset]
    if formal:
        data["batch_size"] = settings["batch_size"]
        data["workers"] = settings["workers"]
        effective["model"]["backbone"] = settings["backbone"]
        effective["optimization"]["lr"] = settings["base_lr"]
    domains = DATASETS[dataset]["domains"]
    source, target = int(data.get("source")), int(data.get("target"))
    if source not in range(len(domains)) or target not in range(len(domains)):
        raise ValueError("invalid COME source/target domain index")
    if source == target:
        raise ValueError("COME source and target domains must differ")
    if dataset == "VISDA-C" and (source, target) != (0, 1):
        raise ValueError("formal COME VisDA supports only train-to-validation")
    if data.get("da") != "uda":
        raise ValueError("COME baseline requires UDA source checkpoints")
    if int(data.get("batch_size")) != settings["batch_size"]:
        raise ValueError(f"COME {dataset} requires batch_size={settings['batch_size']}")
    if int(data.get("workers")) != settings["workers"]:
        raise ValueError(f"COME {dataset} requires workers={settings['workers']}")
    data_root = _absolute(data["root"], workspace_root)
    source_name, target_name = domains[source], domains[target]
    data.update(
        {
            "root": data_root,
            "domains": domains,
            "source": source,
            "target": target,
            "source_name": source_name,
            "target_name": target_name,
            "source_list": osp.join(data_root, dataset, f"{source_name}_list.txt"),
            "target_list": osp.join(data_root, dataset, f"{target_name}_list.txt"),
            "test_list": osp.join(data_root, dataset, f"{target_name}_list.txt"),
        }
    )
    if not osp.isfile(data["target_list"]):
        raise FileNotFoundError(f"COME target list is missing: {data['target_list']}")

    model = effective.get("model", {})
    if model.get("backbone") != settings["backbone"]:
        raise ValueError(f"COME {dataset} requires backbone={settings['backbone']}")
    model["class_num"] = settings["class_count"]

    expected_optimization = {
        "optimizer": "sgd",
        "lr": settings["base_lr"],
        "lr_decay1": 0.1,
        "lr_decay2": 1.0,
        "momentum": 0.9,
        "weight_decay": 0.001,
        "nesterov": True,
        "lr_gamma": 10.0,
        "lr_power": 0.75,
    }
    optimization = effective.get("optimization", {})
    for key, expected in expected_optimization.items():
        value = optimization.get(key)
        if isinstance(expected, float) and value is not None:
            value = float(value)
        if value != expected:
            raise ValueError(
                f"COME matched substrate requires optimization.{key}={expected}"
            )

    objective = effective.get("come")
    expected_objective = {
        "p": OFFICIAL_P,
        "tau": OFFICIAL_TAU,
        "opinion_eps": OFFICIAL_ENTROPY_EPSILON,
        "official_commit": OFFICIAL_COME_COMMIT,
    }
    if objective != expected_objective:
        raise ValueError("COME objective parameters must match audited official source")
    for key in ("p", "tau", "opinion_eps"):
        if not math.isfinite(float(objective[key])):
            raise ValueError(f"come.{key} must be finite")

    checkpoint_root = _absolute(effective["source_checkpoint"]["root"], workspace_root)
    checkpoint_dir = osp.join(checkpoint_root, "uda", dataset, source_name[0].upper())
    effective["source_checkpoint"].update(
        {"root": checkpoint_root, "resolved_dir": checkpoint_dir}
    )
    effective["source_checkpoint_identity"] = _source_identity(checkpoint_dir)
    effective["output"]["root"] = _absolute(effective["output"]["root"], workspace_root)
    if effective["output"].get("save_model"):
        raise ValueError(
            "formal COME baseline and reviewed smoke require save_model=false"
        )
    effective["task_name"] = f"{source_name[0].upper()}{target_name[0].upper()}"
    effective["target_stream_identity"] = _target_stream_identity(data, FORMAL_SEED)

    scientific = _identity_payload(effective)
    canonical = json.dumps(
        scientific,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    expected_sha = hashlib.sha256(canonical).hexdigest()
    expected_key = (
        f"COME__otta__{dataset}__s{source}-t{target}__seed{FORMAL_SEED}__"
        f"{effective['variant']}__{expected_sha[:12]}"
    )
    provided_key = effective.get("experiment_key")
    provided_sha = effective.get("experiment_config_sha256")
    if (provided_key is None) != (provided_sha is None):
        raise ValueError("experiment_key and experiment_config_sha256 must be paired")
    if provided_key is not None and provided_key != expected_key:
        raise ValueError("provided COME experiment_key does not match config")
    if provided_sha is not None and provided_sha != expected_sha:
        raise ValueError("provided COME experiment_config_sha256 does not match config")
    effective["experiment_key"] = expected_key
    effective["experiment_config_sha256"] = expected_sha
    effective["scientific_config"] = scientific
    return effective


# Sparse dispatch is layered after the audited frozen dense resolver so the
# three existing COME baseline branches keep byte-for-byte control flow.
_dense_apply_overrides = apply_overrides
_dense_resolve_effective_config = resolve_effective_config


def apply_overrides(config, args):
    from .sparse_config import apply_sparse_overrides

    return apply_sparse_overrides(config, args, _dense_apply_overrides)


def resolve_effective_config(config, workspace_root):
    from .sparse_config import resolve_sparse_config

    return resolve_sparse_config(
        config,
        workspace_root,
        _dense_resolve_effective_config,
    )
