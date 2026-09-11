"""Fail-closed COME-Transformer configuration, budgets and experiment identity.

One module resolves every non-LBI variant.  The substrate - source W0 loading,
the target stream, the 12-tensor candidate universe, the 6912 QK/VO/FFN paired
groups, the floor budgets 3/6/13 and the rho tags - is imported from the SHOT
``transformer`` packages rather than reimplemented, so COME and SHOT run on
literally the same code objects and only the host objective differs.

The frozen YAML carries the blocks that are common to all five variants.  The
per-variant blocks (protocol revision, update scope, structural-group selection
policy, artifact root) are derived here by :func:`variant_config_view`, which
rebuilds exactly the per-variant config a variant used to own.  Deriving them
means a variant cannot acquire a different substrate through a YAML edit;
``tests/transformer_come_substrate_identity_test.py`` still compares every
derived block against the matching SHOT YAML, so drift from SHOT stays caught.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

from transformer.candidate_dense.config import (
    CANDIDATE_BLOCKS,
    CANDIDATE_SCALAR_COUNT,
    CANDIDATE_SUFFIXES,
    CANDIDATE_TENSOR_COUNT,
    candidate_parameter_names,
)
# Re-exported rather than reimplemented so that ``floor(rho * 6912)`` -> 3/6/13
# and the rho tags used in artifact paths are the same code objects as the
# SHOT-Transformer ones: no slack, no ceil, no per-block quota.
from transformer.group_random.config import (
    BUDGET_TO_K,
    FORMAL_BUDGETS,
    GROUP_SIZE,
    MASK_SEEDS,
    NUM_RANDOM_MASKS,
    TOTAL_GROUPS,
    budget_group_count,
    budget_key,
    budget_tag,
    normalize_budget,
    parse_budgets,
)
from transformer.source_only.config import (
    DATASETS,
    FORMAL_SEED,
    MODEL_NAME,
    TRANSFERS,
    _absolute,
    _read_class_mapping,
    canonical_sha256,
    sha256_file,
)

from .objective import (
    OFFICIAL_COME_COMMIT,
    OFFICIAL_ENTROPY_EPSILON,
    OFFICIAL_P,
    OFFICIAL_TAU,
)


FULL_DENSE = "full_dense"
CANDIDATE_DENSE = "candidate_dense"
GROUP_RANDOM = "group_random"
GROUP_MAGNITUDE = "group_magnitude"
GROUP_SALIENCY = "group_saliency"
DENSE_VARIANTS = (FULL_DENSE, CANDIDATE_DENSE)
SPARSE_VARIANTS = (GROUP_RANDOM, GROUP_MAGNITUDE, GROUP_SALIENCY)
SUPPORTED_VARIANTS = DENSE_VARIANTS + SPARSE_VARIANTS


# Protocol section 20: COME carries its own Transformer protocol revision,
# baseline implementation revision and sparse/LBI implementation revision.  It
# must not reuse the SHOT-Transformer revisions or the ResNet/FC COME revisions
# (``come_otta_baseline_20260908_v2`` / ``come_otta_sparse_lbi_20260908_v3``),
# which describe a different substrate.
PROTOCOL_DOCUMENT = "OTTA_COME_TRANSFORMER_LBI_PROTOCOL_20260909_v1"

BASELINE_IMPLEMENTATION_REVISION = "come_transformer_baseline_20260909_v1"
SPARSE_LBI_IMPLEMENTATION_REVISION = "come_transformer_sparse_lbi_20260909_v1"

PROTOCOL_REVISIONS = {
    FULL_DENSE: "come_transformer_full_dense_otta_20260909_v1",
    CANDIDATE_DENSE: "come_transformer_candidate_dense_otta_20260909_v1",
    GROUP_RANDOM: "come_transformer_group_random_otta_20260909_v1",
    GROUP_MAGNITUDE: "come_transformer_group_magnitude_otta_20260909_v1",
    GROUP_SALIENCY: "come_transformer_group_saliency_otta_20260909_v1",
}

IMPLEMENTATION_REVISIONS = {
    FULL_DENSE: BASELINE_IMPLEMENTATION_REVISION,
    CANDIDATE_DENSE: BASELINE_IMPLEMENTATION_REVISION,
    GROUP_RANDOM: SPARSE_LBI_IMPLEMENTATION_REVISION,
    GROUP_MAGNITUDE: SPARSE_LBI_IMPLEMENTATION_REVISION,
    GROUP_SALIENCY: SPARSE_LBI_IMPLEMENTATION_REVISION,
}

VARIANT_KEYS = {
    FULL_DENSE: "full-dense",
    CANDIDATE_DENSE: "candidate-dense",
    GROUP_RANDOM: "group-random",
    GROUP_MAGNITUDE: "group-magnitude",
    GROUP_SALIENCY: "group-saliency",
}

RUN_PREFIXES = {
    variant: f"come_{variant}_seed{FORMAL_SEED}" for variant in SUPPORTED_VARIANTS
}

CHILDREN_PER_CONDITION = {
    **{variant: 1 for variant in SUPPORTED_VARIANTS},
    GROUP_RANDOM: NUM_RANDOM_MASKS,
}


# The frozen ``come:`` config block.  Its own namespace keeps ``come.tau``
# (the logit-constraint parameter, 1.0) separate from the future ``lbi.tau_g``
# (the normalized Gamma support threshold, 1e-4).
EXPECTED_COME_BLOCK = {
    "objective": "come_entropy_of_opinion",
    "official_come_commit": OFFICIAL_COME_COMMIT,
    "p": OFFICIAL_P,
    "tau": OFFICIAL_TAU,
    "entropy_epsilon": OFFICIAL_ENTROPY_EPSILON,
    "class_count_source": "dataset_classifier_output_dimension",
    "norm_epsilon": "none",
    "norm_clamp": "none",
    "stop_gradient_on_norm": True,
    "opinion_implementation": "stable_log_domain",
    "renormalize_after_epsilon": False,
    "tunable_hyperparameters": [],
}

EXPECTED_MODEL = {
    "name": MODEL_NAME,
    "checkpoint_root": "/home/nas3/biod/wangkangyi/checkpoints/source_models",
}
EXPECTED_DATA_BLOCKS = {
    "office31": {
        "batch_size": 64,
        "fo_batch_size": 64,
        "workers": 4,
        "num_classes": 31,
    },
    "visda-c": {
        "batch_size": 256,
        "fo_batch_size": 256,
        "workers": 4,
        "num_classes": 12,
    },
}
EXPECTED_PREPROCESSING = {
    "resize_size": 256,
    "crop_size": 224,
    "interpolation": "bilinear",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "online_random_crop": True,
    "online_random_horizontal_flip": True,
    "fo_center_crop": True,
}
EXPECTED_STREAM = {
    "order": "fixed_random_permutation",
    "one_pass": True,
    "drop_last": False,
}
EXPECTED_OPTIMIZATION = {
    "optimizer": "adamw",
    "lr": 1.0e-5,
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
    "weight_decay": 0.01,
}
EXPECTED_RUNTIME = {"deterministic": True, "amp": False, "pin_memory": True}

FULL_DENSE_ADAPTATION = {
    "update_scope": "all_except_head",
    "model_mode": "eval",
    "steps_per_online_batch": 1,
}
CANDIDATE_ADAPTATION = {
    "update_scope": "last_three_blocks_qkv_proj_mlp_weights",
    "candidate_blocks": list(CANDIDATE_BLOCKS),
    "candidate_suffixes": list(CANDIDATE_SUFFIXES),
    "candidate_tensor_count": CANDIDATE_TENSOR_COUNT,
    "candidate_scalar_count": CANDIDATE_SCALAR_COUNT,
    "model_mode": "eval",
    "steps_per_online_batch": 1,
}
ADAPTATION_BLOCKS = {
    FULL_DENSE: FULL_DENSE_ADAPTATION,
    CANDIDATE_DENSE: CANDIDATE_ADAPTATION,
    GROUP_RANDOM: CANDIDATE_ADAPTATION,
    GROUP_MAGNITUDE: CANDIDATE_ADAPTATION,
    GROUP_SALIENCY: CANDIDATE_ADAPTATION,
}

RANDOM_SELECTION = {
    "type": "uniform_structural_group_random",
    "group_order": "block_then_qk_vo_ffn_then_coordinate",
    "total_groups": TOTAL_GROUPS,
    "group_size": GROUP_SIZE,
    "budgets": list(FORMAL_BUDGETS),
    "integer_rule": "floor",
    "integer_budgets": [BUDGET_TO_K[item] for item in FORMAL_BUDGETS],
    "num_random_masks": NUM_RANDOM_MASKS,
    "mask_seeds": list(MASK_SEEDS),
    "cross_budget_policy": "nested_prefix_same_permutation",
    "mask_refresh_policy": "once_before_adaptation",
}
MAGNITUDE_SELECTION = {
    "type": "source_w0_structural_group_l2",
    "score": "paired_group_l2_norm",
    "score_device": "cpu",
    "score_dtype": "float64",
    "tie_break": "ascending_canonical_group_id",
    "group_order": "block_then_qk_vo_ffn_then_coordinate",
    "total_groups": TOTAL_GROUPS,
    "group_size": GROUP_SIZE,
    "budgets": list(FORMAL_BUDGETS),
    "integer_rule": "floor",
    "integer_budgets": [BUDGET_TO_K[item] for item in FORMAL_BUDGETS],
    "ranking_source": "source_checkpoint_pre_adaptation",
    "mask_refresh_policy": "once_before_adaptation",
}
# Identical to the SHOT saliency selection block except for the objective named
# in ranking_source: the score is |W * grad(L_COME)| instead of
# |W * grad(L_SHOT)|.  Definition, axis, tie-break and refresh policy are
# unchanged.
SALIENCY_SELECTION = {
    "type": "dynamic_abs_weight_times_gradient_group_l2",
    "score": "paired_group_l2_norm_of_abs_weight_times_gradient",
    "score_device": "model_device",
    "score_dtype": "parameter_dtype",
    "tie_break": "ascending_canonical_group_id",
    "group_order": "block_then_qk_vo_ffn_then_coordinate",
    "total_groups": TOTAL_GROUPS,
    "group_size": GROUP_SIZE,
    "budgets": list(FORMAL_BUDGETS),
    "integer_rule": "floor",
    "integer_budgets": [BUDGET_TO_K[item] for item in FORMAL_BUDGETS],
    "ranking_source": "current_pre_update_weight_and_current_come_gradient",
    "mask_refresh_policy": "every_online_batch_after_backward_before_step",
}
SELECTION_BLOCKS = {
    GROUP_RANDOM: RANDOM_SELECTION,
    GROUP_MAGNITUDE: MAGNITUDE_SELECTION,
    GROUP_SALIENCY: SALIENCY_SELECTION,
}

# Saliency records the per-step active scalar count, because its support is
# rebuilt every online batch instead of once before the stream.
PER_STEP_SCALARS = (GROUP_SALIENCY,)

NON_SCIENTIFIC_KEYS = {
    "output_dir",
    "checkpoint_manifest_path",
    "scientific_config_sha256",
    "experiment_key",
}


def come_objective_payload(come_block: dict) -> dict:
    """COME-specific scientific payload recorded in every artifact."""

    return {
        "host_objective": "come",
        "come_official_commit": come_block["official_come_commit"],
        "come_p": float(come_block["p"]),
        "come_tau": float(come_block["tau"]),
        "come_entropy_epsilon": float(come_block["entropy_epsilon"]),
        "come_class_count_source": come_block["class_count_source"],
        "come_norm_epsilon": come_block["norm_epsilon"],
        "come_norm_clamp": come_block["norm_clamp"],
        "come_stop_gradient_on_norm": bool(come_block["stop_gradient_on_norm"]),
        "come_opinion_implementation": come_block["opinion_implementation"],
        "come_renormalize_after_epsilon": bool(
            come_block["renormalize_after_epsilon"]
        ),
        "come_objective_chunking": "none_single_full_batch_objective",
        "come_pseudo_label_or_teacher_or_memory": "none",
        "come_tunable_hyperparameters": list(come_block["tunable_hyperparameters"]),
    }


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def require_supported_variant(variant: str) -> str:
    if variant == "group_lbi":
        raise ValueError(
            "group_lbi is intentionally not implemented in this revision: the "
            "COME-LBI search protocol and its six tuples are not frozen"
        )
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"variant must be one of {SUPPORTED_VARIANTS}")
    return variant


def validate_raw_config(config: dict[str, Any]) -> None:
    """Fail closed on every frozen field of the shared COME YAML."""

    expected_top_level_keys = {
        "schema_version",
        "protocol_document",
        "method",
        "formal_seed",
        "model",
        "data",
        "optimization",
        "come",
        "runtime",
        "output",
    }
    if set(config) != expected_top_level_keys:
        raise ValueError(
            "COME config top-level keys differ from the frozen non-LBI schema: "
            f"{sorted(set(config) ^ expected_top_level_keys)}"
        )
    expected_top = {
        "schema_version": 1,
        "protocol_document": PROTOCOL_DOCUMENT,
        "method": "come",
        "formal_seed": FORMAL_SEED,
    }
    for key, expected in expected_top.items():
        if config.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}")
    if config.get("model") != EXPECTED_MODEL:
        raise ValueError("model config differs from the frozen local DeiT source")
    data = config.get("data", {})
    if set(data) != {"list_root", "office31", "visda-c", "preprocessing", "stream"}:
        raise ValueError("COME data config keys differ from the frozen schema")
    for dataset, expected in EXPECTED_DATA_BLOCKS.items():
        if data.get(dataset) != expected:
            raise ValueError(f"data.{dataset} must be exactly {expected}")
    if data.get("preprocessing") != EXPECTED_PREPROCESSING:
        raise ValueError("data.preprocessing differs from the frozen protocol")
    if data.get("stream") != EXPECTED_STREAM:
        raise ValueError(f"data.stream must be exactly {EXPECTED_STREAM}")
    if config.get("optimization") != EXPECTED_OPTIMIZATION:
        raise ValueError(f"optimization must be exactly {EXPECTED_OPTIMIZATION}")
    if config.get("come") != EXPECTED_COME_BLOCK:
        raise ValueError(f"come must be exactly {EXPECTED_COME_BLOCK}")
    if config.get("runtime") != EXPECTED_RUNTIME:
        raise ValueError(f"runtime must be exactly {EXPECTED_RUNTIME}")
    output = config.get("output")
    if not isinstance(output, dict) or set(output) != {"root", "variant_dir_prefix"}:
        raise ValueError("COME output config must contain only root and variant_dir_prefix")
    if not Path(output["root"]).is_absolute():
        raise ValueError("COME output root must be absolute")


def variant_output_root(raw_config: dict[str, Any], variant: str) -> str:
    """The per-variant artifact root this flat config resolves to."""

    output = raw_config["output"]
    return str(Path(output["root"]) / f"{output['variant_dir_prefix']}_{variant}")


def variant_config_view(raw_config: dict[str, Any], variant: str) -> dict[str, Any]:
    """Rebuild the per-variant config block set from the shared YAML.

    The result is exactly the config a variant used to own: the shared blocks
    verbatim, plus the derived protocol revision, update scope, structural-group
    selection policy and artifact root.  Everything downstream - validation,
    transfer resolution and the SHOT substrate identity test - works on this
    view, so the resolved config of every condition is unchanged.
    """

    require_supported_variant(variant)
    view = {
        "schema_version": raw_config["schema_version"],
        "protocol_revision": PROTOCOL_REVISIONS[variant],
        "formal_seed": raw_config["formal_seed"],
        "method": raw_config["method"],
        "variant": variant,
        "model": copy.deepcopy(raw_config["model"]),
        "data": copy.deepcopy(raw_config["data"]),
        "adaptation": copy.deepcopy(ADAPTATION_BLOCKS[variant]),
    }
    if variant in SPARSE_VARIANTS:
        view["selection"] = copy.deepcopy(SELECTION_BLOCKS[variant])
    view["optimization"] = copy.deepcopy(raw_config["optimization"])
    view["come"] = copy.deepcopy(raw_config["come"])
    view["runtime"] = copy.deepcopy(raw_config["runtime"])
    view["output"] = {"root": variant_output_root(raw_config, variant)}
    return view


def validate_variant_config(config: dict[str, Any], *, variant: str) -> None:
    """Fail closed on every frozen non-selection field of a variant view."""

    require_supported_variant(variant)
    expected_top = {
        "schema_version": 1,
        "protocol_revision": PROTOCOL_REVISIONS[variant],
        "formal_seed": FORMAL_SEED,
        "method": "come",
        "variant": variant,
    }
    for key, expected in expected_top.items():
        if config.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}")
    if config.get("model", {}).get("name") != MODEL_NAME:
        raise ValueError(f"model.name must be {MODEL_NAME}")
    if "loss" in config:
        raise ValueError(
            "COME configs must not carry a SHOT loss block; the come block is "
            "the only host objective definition"
        )
    if config.get("come") != EXPECTED_COME_BLOCK:
        raise ValueError(f"come must be exactly {EXPECTED_COME_BLOCK}")

    for dataset, expected in EXPECTED_DATA_BLOCKS.items():
        if config.get("data", {}).get(dataset) != expected:
            raise ValueError(f"data.{dataset} must be exactly {expected}")
    if config.get("data", {}).get("preprocessing") != EXPECTED_PREPROCESSING:
        raise ValueError("data.preprocessing differs from the frozen protocol")
    if config.get("data", {}).get("stream") != EXPECTED_STREAM:
        raise ValueError(f"data.stream must be exactly {EXPECTED_STREAM}")
    if config.get("adaptation") != ADAPTATION_BLOCKS[variant]:
        raise ValueError(f"adaptation must be exactly {ADAPTATION_BLOCKS[variant]}")
    if config.get("optimization") != EXPECTED_OPTIMIZATION:
        raise ValueError(f"optimization must be exactly {EXPECTED_OPTIMIZATION}")
    if config.get("runtime") != EXPECTED_RUNTIME:
        raise ValueError(f"runtime must be exactly {EXPECTED_RUNTIME}")
    if variant in SPARSE_VARIANTS:
        if config.get("selection") != SELECTION_BLOCKS[variant]:
            raise ValueError(f"selection must be exactly {SELECTION_BLOCKS[variant]}")
    elif "selection" in config:
        raise ValueError("dense COME variants must not carry a selection block")


def _resolve_substrate(
    config: dict[str, Any],
    *,
    variant: str,
    project_root: Path,
    dataset: str,
    source: str,
    target: str,
    device: str,
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Resolve the substrate half of one formal COME transfer.

    Selection blocks and the scientific hash are added by the caller, so a
    sparse variant cannot silently fall back to a dense identity.
    """

    if (dataset, source, target) not in TRANSFERS:
        raise ValueError(f"Unsupported formal transfer: {dataset} {source}->{target}")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")

    data_root = Path(_absolute(config["data"]["list_root"], project_root))
    checkpoint_root = Path(_absolute(config["model"]["checkpoint_root"], project_root))
    target_list = data_root / dataset / f"{target}_list.txt"
    class_mapping = data_root / dataset / "class_to_idx.json"
    checkpoint = checkpoint_root / dataset / f"{source}.pth"
    checkpoint_manifest = checkpoint.with_suffix(".manifest.json")
    for path in (target_list, class_mapping, checkpoint, checkpoint_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Required local asset not found: {path}")
    if checkpoint.name.endswith(".last.pth"):
        raise ValueError("*.last.pth cannot be used as source W0")

    dataset_config = config["data"][dataset]
    class_names = _read_class_mapping(
        str(class_mapping), dataset, target, str(target_list)
    )
    with open(checkpoint_manifest, "r", encoding="utf-8") as file_obj:
        checkpoint_manifest_payload = json.load(file_obj)
    checkpoint_record = checkpoint_manifest_payload.get("checkpoint", {})
    checkpoint_sha256 = str(checkpoint_record.get("sha256", "")).lower()
    if len(checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha256
    ):
        raise ValueError("Source manifest checkpoint SHA-256 is invalid")
    if Path(checkpoint_record.get("path", "")).resolve() != checkpoint.resolve():
        raise ValueError("Source manifest checkpoint path differs from W0")
    actual_checkpoint_sha256 = sha256_file(checkpoint)
    if actual_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError(
            "Source checkpoint SHA-256 differs from its manifest: "
            f"{actual_checkpoint_sha256} != {checkpoint_sha256}"
        )

    num_classes = int(DATASETS[dataset]["num_classes"])
    if int(dataset_config["num_classes"]) != num_classes:
        raise ValueError(
            f"data.{dataset}.num_classes must equal the dataset class count "
            f"{num_classes}"
        )
    if num_classes not in (12, 31):
        raise ValueError("COME class count C must be the dataset class count")

    return {
        "schema_version": 1,
        "protocol_revision": PROTOCOL_REVISIONS[variant],
        "protocol_document": PROTOCOL_DOCUMENT,
        "implementation_revision": IMPLEMENTATION_REVISIONS[variant],
        "formal_seed": FORMAL_SEED,
        "method": "come",
        "variant": variant,
        "dataset": dataset,
        "source": source,
        "target": target,
        "transfer": f"{source}->{target}",
        "num_classes": num_classes,
        "class_names": class_names,
        "batch_size": int(dataset_config["batch_size"]),
        "fo_batch_size": int(dataset_config["fo_batch_size"]),
        "workers": int(dataset_config["workers"]),
        "target_list": str(target_list.resolve()),
        "target_list_sha256": sha256_file(target_list),
        "class_mapping_path": str(class_mapping.resolve()),
        "class_mapping_sha256": sha256_file(class_mapping),
        "model_name": MODEL_NAME,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_manifest_path": str(checkpoint_manifest.resolve()),
        "preprocessing": copy.deepcopy(config["data"]["preprocessing"]),
        "stream": copy.deepcopy(config["data"]["stream"]),
        "adaptation": copy.deepcopy(config["adaptation"]),
        "optimization": copy.deepcopy(config["optimization"]),
        "come": copy.deepcopy(config["come"]),
        "come_objective": come_objective_payload(config["come"]),
        "runtime": {**copy.deepcopy(config["runtime"]), "device": device},
        "output_dir": str(Path(output_dir).resolve()),
    }


def finalize_identity(resolved: dict[str, Any], *, variant: str, tag: str = "") -> dict:
    """Attach the scientific hash and experiment key to a resolved config."""

    scientific = {
        key: value
        for key, value in resolved.items()
        if key not in NON_SCIENTIFIC_KEYS
    }
    resolved["scientific_config_sha256"] = canonical_sha256(scientific)
    suffix = f"{tag}_" if tag else ""
    resolved["experiment_key"] = (
        f"{resolved['dataset']}_{resolved['source']}-{resolved['target']}_"
        f"come-{VARIANT_KEYS[variant]}_{suffix}seed-{FORMAL_SEED}_"
        f"{resolved['scientific_config_sha256'][:12]}"
    )
    return resolved


def resolve_transfer_config(
    raw_config: dict[str, Any],
    *,
    project_root: Path,
    dataset: str,
    source: str,
    target: str,
    variant: str,
    budget: float | str | None = None,
    device: str,
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Resolve one formal condition and fail closed on every frozen field.

    ``floor(rho * 6912)`` comes from the SHOT budget helper, so exact-K is the
    same integer as the matched SHOT condition: no ceil, no slack, no per-block
    quota and no minimum-one rule.
    """

    require_supported_variant(variant)
    validate_raw_config(raw_config)
    view = variant_config_view(raw_config, variant)
    validate_variant_config(view, variant=variant)

    if variant in DENSE_VARIANTS:
        if budget is not None:
            raise ValueError("dense COME variants reject structural-group budgets")
    elif budget is None:
        raise ValueError("sparse COME variants require an explicit rho budget")

    resolved = _resolve_substrate(
        view,
        variant=variant,
        project_root=project_root,
        dataset=dataset,
        source=source,
        target=target,
        device=device,
        output_dir=output_dir,
    )
    if variant in DENSE_VARIANTS:
        return finalize_identity(resolved, variant=variant)

    normalized = normalize_budget(budget)
    group_count = budget_group_count(normalized)
    scalars_key = (
        "active_candidate_scalars_per_step"
        if variant in PER_STEP_SCALARS
        else "active_candidate_scalars"
    )
    resolved["selection"] = {
        **copy.deepcopy(view["selection"]),
        "requested_budget": normalized,
        "requested_group_count": group_count,
        scalars_key: group_count * GROUP_SIZE,
    }
    return finalize_identity(
        resolved, variant=variant, tag=budget_tag(normalized)
    )


def select_transfers(selection: str) -> tuple[tuple[str, str, str], ...]:
    if selection == "all":
        return TRANSFERS
    if selection == "office31":
        return tuple(item for item in TRANSFERS if item[0] == "office31")
    if selection == "visda-c":
        return tuple(item for item in TRANSFERS if item[0] == "visda-c")
    raise ValueError("selection must be all, office31, or visda-c")


def parse_variants(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return SUPPORTED_VARIANTS
    variants = tuple(item.strip() for item in value.split(",") if item.strip())
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("variants must be a non-empty unique list")
    for variant in variants:
        require_supported_variant(variant)
    return variants


__all__ = [
    "ADAPTATION_BLOCKS",
    "BASELINE_IMPLEMENTATION_REVISION",
    "BUDGET_TO_K",
    "CANDIDATE_ADAPTATION",
    "CANDIDATE_BLOCKS",
    "CANDIDATE_DENSE",
    "CANDIDATE_SCALAR_COUNT",
    "CANDIDATE_SUFFIXES",
    "CANDIDATE_TENSOR_COUNT",
    "CHILDREN_PER_CONDITION",
    "DENSE_VARIANTS",
    "EXPECTED_COME_BLOCK",
    "EXPECTED_DATA_BLOCKS",
    "EXPECTED_OPTIMIZATION",
    "EXPECTED_PREPROCESSING",
    "EXPECTED_RUNTIME",
    "EXPECTED_STREAM",
    "FORMAL_BUDGETS",
    "FORMAL_SEED",
    "FULL_DENSE",
    "FULL_DENSE_ADAPTATION",
    "GROUP_MAGNITUDE",
    "GROUP_RANDOM",
    "GROUP_SALIENCY",
    "GROUP_SIZE",
    "IMPLEMENTATION_REVISIONS",
    "MASK_SEEDS",
    "MODEL_NAME",
    "NON_SCIENTIFIC_KEYS",
    "NUM_RANDOM_MASKS",
    "PROTOCOL_DOCUMENT",
    "PROTOCOL_REVISIONS",
    "RUN_PREFIXES",
    "SELECTION_BLOCKS",
    "SPARSE_LBI_IMPLEMENTATION_REVISION",
    "SPARSE_VARIANTS",
    "SUPPORTED_VARIANTS",
    "TOTAL_GROUPS",
    "TRANSFERS",
    "VARIANT_KEYS",
    "budget_group_count",
    "budget_key",
    "budget_tag",
    "candidate_parameter_names",
    "come_objective_payload",
    "finalize_identity",
    "load_config",
    "normalize_budget",
    "parse_budgets",
    "parse_variants",
    "require_supported_variant",
    "resolve_transfer_config",
    "select_transfers",
    "validate_raw_config",
    "validate_variant_config",
    "variant_config_view",
    "variant_output_root",
]
