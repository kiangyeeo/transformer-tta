"""Fail-closed IST-Transformer configuration and experiment identity."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

from transformer.candidate_dense.config import (
    CANDIDATE_BLOCKS,
    CANDIDATE_SCALAR_COUNT,
    CANDIDATE_SUFFIXES,
    CANDIDATE_TENSOR_COUNT,
    candidate_parameter_names,
)
from transformer.candidate_dense.model import EXPECTED_SHAPES
from transformer.group_random.config import (
    FORMAL_BUDGETS,
    GROUP_SIZE,
    MASK_SEEDS,
    NUM_RANDOM_MASKS,
    TOTAL_GROUPS,
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
from transformer.structural_budget import format_structural_budget


PROTOCOL_REVISION = "OTTA_IST_TRANSFORMER_LBI_PROTOCOL_20260909_v1"
DENSE_IMPLEMENTATION_REVISION = "ist_transformer_baseline_20260909_v1"
SPARSE_IMPLEMENTATION_REVISION = "ist_transformer_sparse_lbi_20260909_v1"
SOURCE_CHECKPOINT_REVISION = "deit_source_checkpoint_manifest_v1"

FULL_DENSE = "full_dense"
CANDIDATE_DENSE = "candidate_dense"
GROUP_RANDOM = "group_random"
GROUP_MAGNITUDE = "group_magnitude"
GROUP_SALIENCY = "group_saliency"
DENSE_VARIANTS = (FULL_DENSE, CANDIDATE_DENSE)
SPARSE_VARIANTS = (GROUP_RANDOM, GROUP_MAGNITUDE, GROUP_SALIENCY)
SUPPORTED_VARIANTS = DENSE_VARIANTS + SPARSE_VARIANTS


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        result = yaml.safe_load(file_obj)
    if not isinstance(result, dict):
        raise ValueError("IST config must contain a mapping")
    return result


def normalize_budget(value: float | str | None) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError("sparse IST variants require an explicit formal budget")
    parsed = float(value)
    for budget in FORMAL_BUDGETS:
        if math.isclose(parsed, budget, rel_tol=0.0, abs_tol=1.0e-12):
            return float(budget)
    raise ValueError(f"budget must be one of {FORMAL_BUDGETS}")


def parse_budgets(value: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return tuple(FORMAL_BUDGETS)
        values = [item.strip() for item in value.split(",") if item.strip()]
    else:
        values = list(value)
    result = tuple(normalize_budget(item) for item in values)
    if not result or len(set(result)) != len(result):
        raise ValueError("budgets must be a non-empty unique list")
    return result


def budget_group_count(value) -> int:
    return math.floor(normalize_budget(value) * TOTAL_GROUPS)


def budget_tag(value) -> str:
    return "rho-" + format_structural_budget(normalize_budget(value))


def _validate_raw(config: dict) -> None:
    expected_top_level_keys = {
        "schema_version",
        "protocol_revision",
        "method",
        "formal_seed",
        "model",
        "data",
        "ist",
        "optimization",
        "loss",
        "runtime",
        "output",
    }
    if set(config) != expected_top_level_keys:
        raise ValueError(
            "IST config top-level keys differ from the frozen non-LBI schema: "
            f"{sorted(set(config) ^ expected_top_level_keys)}"
        )
    expected_top = {
        "schema_version": 1,
        "protocol_revision": PROTOCOL_REVISION,
        "method": "ist",
        "formal_seed": FORMAL_SEED,
    }
    for name, expected in expected_top.items():
        if config.get(name) != expected:
            raise ValueError(f"{name} must be {expected!r}")
    if config.get("model") != {
        "name": MODEL_NAME,
        "checkpoint_root": "/home/nas3/biod/wangkangyi/checkpoints/source_models",
    }:
        raise ValueError("model config differs from the frozen local DeiT source")
    expected_datasets = {
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
    data = config.get("data", {})
    expected_data_keys = {
        "list_root",
        "office31",
        "visda-c",
        "preprocessing",
        "stream",
    }
    if set(data) != expected_data_keys:
        raise ValueError("IST data config keys differ from the frozen schema")
    for dataset, expected in expected_datasets.items():
        if data.get(dataset) != expected:
            raise ValueError(f"data.{dataset} must be exactly {expected}")
    expected_preprocessing = {
        "resize_size": 256,
        "crop_size": 224,
        "interpolation": "bilinear",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "online_random_crop": True,
        "online_random_horizontal_flip": True,
        "fo_center_crop": True,
    }
    if data.get("preprocessing") != expected_preprocessing:
        raise ValueError("IST preprocessing differs from protocol")
    if data.get("stream") != {
        "order": "fixed_random_permutation",
        "one_pass": True,
        "drop_last": False,
        "amazon_singleton_tail": "merge_into_previous",
        "unexpected_singleton": "fail_before_state_transition",
    }:
        raise ValueError("IST stream config differs from protocol")
    ist = config.get("ist")
    expected_ist = {
        "extend": 8,
        "iters": 1,
        "ema_momentum": 0.9,
        "feature_dim": 384,
        "memory": {"max_len": 10000, "entry_unit": "view"},
        "plca": {
            "repeat": 1,
            "k": 50,
            "gamma": 3,
            "mode": "l2",
            "propagation_alpha": 0.99,
            "solver_max_steps": 20,
            "solver_rtol": 1.0e-6,
            "solver_atol": 0.0,
            "distance_chunk_size": 1024,
        },
        "augmentation": {
            "source": "raw_pre_normalization",
            "resize_size": 256,
            "crop_size": 224,
            "horizontal_flip_probability": 0.5,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "rng": {
            "reference_seed_offset": 10000,
            "adaptation_seed_offset": 20000,
            "inner_order_seed_offset": 30000,
        },
        "pre_inference_chunk_size": 64,
        "native_micro_batch_size": 64,
        "objective_chunk_size": 64,
    }
    if ist != expected_ist:
        raise ValueError("ist mechanism/chunk config differs from protocol")
    if config.get("optimization") != {
        "optimizer": "adamw",
        "lr": 1.0e-5,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.01,
        "scheduler": "none",
    }:
        raise ValueError("IST optimizer config differs from protocol")
    if config.get("loss") != {
        "components": ["hard_ce", "soft_kl"],
        "hard_ce_weight": 1.0,
        "soft_kl_weight": 1.0,
        "target_granularity": "view",
    }:
        raise ValueError("IST objective differs from protocol")
    if config.get("runtime") != {
        "deterministic": True,
        "amp": False,
        "pin_memory": True,
        "save_model": False,
        "stream_checkpoint": False,
        "partial_resume": False,
    }:
        raise ValueError("IST runtime policy differs from protocol")
    output = config.get("output")
    if not isinstance(output, dict) or set(output) != {"root"}:
        raise ValueError("IST output config must contain only root")
    if not Path(output["root"]).is_absolute():
        raise ValueError("IST output root must be absolute")


def resolve_transfer_config(
    raw_config: dict,
    *,
    project_root: Path,
    dataset: str,
    source: str,
    target: str,
    variant: str,
    budget: float | str | None,
    device: str,
    output_dir: str | os.PathLike[str],
    debug_max_outer_batches: int | None = None,
) -> dict:
    _validate_raw(raw_config)
    if variant == "group_lbi":
        raise ValueError("group_lbi is intentionally not implemented in this revision")
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"variant must be one of {SUPPORTED_VARIANTS}")
    if (dataset, source, target) not in TRANSFERS:
        raise ValueError(f"unsupported transfer: {dataset} {source}->{target}")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if variant in DENSE_VARIANTS:
        if budget is not None:
            raise ValueError("dense IST variants reject sparse budgets")
        normalized_budget = None
    else:
        normalized_budget = normalize_budget(budget)
    if debug_max_outer_batches is not None:
        if isinstance(debug_max_outer_batches, bool) or int(debug_max_outer_batches) < 1:
            raise ValueError("debug_max_outer_batches must be a positive integer")
        debug_max_outer_batches = int(debug_max_outer_batches)

    data_root = Path(_absolute(raw_config["data"]["list_root"], project_root))
    checkpoint_root = Path(
        _absolute(raw_config["model"]["checkpoint_root"], project_root)
    )
    target_list = data_root / dataset / f"{target}_list.txt"
    class_mapping = data_root / dataset / "class_to_idx.json"
    checkpoint = checkpoint_root / dataset / f"{source}.pth"
    manifest_path = checkpoint.with_suffix(".manifest.json")
    for path in (target_list, class_mapping, checkpoint, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"required local asset not found: {path}")
    if checkpoint.name.endswith(".last.pth"):
        raise ValueError("*.last.pth cannot be source W0")
    with open(manifest_path, "r", encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    checkpoint_record = manifest.get("checkpoint", {})
    checkpoint_sha256 = str(checkpoint_record.get("sha256", "")).lower()
    if checkpoint_record.get("kind") != "deit_source_checkpoint":
        raise ValueError("source manifest kind is not deit_source_checkpoint")
    if Path(checkpoint_record.get("path", "")).resolve() != checkpoint.resolve():
        raise ValueError("source manifest path differs from selected W0")
    if sha256_file(checkpoint) != checkpoint_sha256:
        raise ValueError("source checkpoint hash differs from manifest")
    class_names = _read_class_mapping(
        str(class_mapping), dataset, target, str(target_list)
    )
    dataset_config = raw_config["data"][dataset]
    selection = None
    if normalized_budget is not None:
        selection = {
            "type": variant.removeprefix("group_"),
            "group_order": "block_then_qk_vo_ffn_then_coordinate",
            "total_groups": TOTAL_GROUPS,
            "group_size": GROUP_SIZE,
            "requested_budget": normalized_budget,
            "requested_group_count": budget_group_count(normalized_budget),
            "active_candidate_scalars": budget_group_count(normalized_budget)
            * GROUP_SIZE,
            "integer_rule": "floor",
            "mask_refresh_policy": (
                "once_per_outer_batch_full_ist_objective"
                if variant == GROUP_SALIENCY
                else "once_before_target_stream"
            ),
        }
        if variant == GROUP_RANDOM:
            selection.update(
                {
                    "num_random_masks": NUM_RANDOM_MASKS,
                    "mask_seeds": list(MASK_SEEDS),
                    "cross_budget_policy": "nested_prefix_same_permutation",
                }
            )
    resolved = {
        "schema_version": 1,
        "protocol_revision": PROTOCOL_REVISION,
        "implementation_revision": (
            SPARSE_IMPLEMENTATION_REVISION
            if variant in SPARSE_VARIANTS
            else DENSE_IMPLEMENTATION_REVISION
        ),
        "source_checkpoint_revision": SOURCE_CHECKPOINT_REVISION,
        "method": "ist",
        "variant": variant,
        "formal_seed": FORMAL_SEED,
        "formal_protocol": debug_max_outer_batches is None,
        "debug_smoke": debug_max_outer_batches is not None,
        "debug_max_outer_batches": debug_max_outer_batches,
        "dataset": dataset,
        "source": source,
        "target": target,
        "transfer": f"{source}->{target}",
        "num_classes": DATASETS[dataset]["num_classes"],
        "class_names": class_names,
        "batch_size": dataset_config["batch_size"],
        "fo_batch_size": dataset_config["fo_batch_size"],
        "workers": dataset_config["workers"],
        "target_list": str(target_list.resolve()),
        "target_list_sha256": sha256_file(target_list),
        "class_mapping_path": str(class_mapping.resolve()),
        "class_mapping_sha256": sha256_file(class_mapping),
        "model_name": MODEL_NAME,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_manifest_path": str(manifest_path.resolve()),
        "source_scientific_config_sha256": manifest.get(
            "scientific_config_sha256"
        ),
        "preprocessing": copy.deepcopy(raw_config["data"]["preprocessing"]),
        "stream": copy.deepcopy(raw_config["data"]["stream"]),
        "adaptation": {
            "model_mode": "eval",
            "precision": "fp32_no_amp",
            "update_scope": (
                "all_deit_parameters_except_classifier_head"
                if variant == FULL_DENSE
                else "last_three_blocks_qkv_proj_mlp_weights"
            ),
            "candidate_blocks": list(CANDIDATE_BLOCKS),
            "candidate_suffixes": list(CANDIDATE_SUFFIXES),
            "candidate_parameter_names": list(candidate_parameter_names()),
            "candidate_shapes": {
                suffix: list(EXPECTED_SHAPES[suffix])
                for suffix in CANDIDATE_SUFFIXES
            },
            "candidate_tensor_count": CANDIDATE_TENSOR_COUNT,
            "candidate_scalar_count": CANDIDATE_SCALAR_COUNT,
            "structural_groups": {
                "order": "block_then_qk_vo_ffn_then_coordinate",
                "blocks": list(CANDIDATE_BLOCKS),
                "definitions": {
                    "QK": "Q_row[p]+K_row[p]",
                    "VO": "V_row[p]+O_column[p]",
                    "FFN": "fc1_row[p]+fc2_column[p]",
                },
                "groups_per_block": {"QK": 384, "VO": 384, "FFN": 1536},
                "total_groups": TOTAL_GROUPS,
                "group_size": GROUP_SIZE,
            },
            "optimizer_lifetime": "persistent_across_inner_and_outer_batches",
            "scientific_step_unit": "one_complete_inner_minibatch",
            "outer_writeback": "single_parameter_ema_0.9_anchor_plus_0.1_adapted",
        },
        "ist": copy.deepcopy(raw_config["ist"]),
        "optimization": copy.deepcopy(raw_config["optimization"]),
        "loss": copy.deepcopy(raw_config["loss"]),
        "runtime": {
            **copy.deepcopy(raw_config["runtime"]),
            "device": device,
        },
        "selection": selection,
        "output_dir": str(Path(output_dir).resolve()),
    }
    scientific = {
        key: value
        for key, value in resolved.items()
        if key not in {"output_dir", "checkpoint_manifest_path"}
    }
    resolved["scientific_config_sha256"] = canonical_sha256(scientific)
    budget_part = "" if normalized_budget is None else f"_{budget_tag(normalized_budget)}"
    mode = "formal" if resolved["formal_protocol"] else "debug"
    resolved["experiment_key"] = (
        f"{dataset}_{source}-{target}_ist-{variant}{budget_part}_"
        f"seed-{FORMAL_SEED}_{mode}_{resolved['scientific_config_sha256'][:12]}"
    )
    return resolved


def select_transfers(selection: str):
    if selection == "all":
        return TRANSFERS
    if selection == "office31":
        return tuple(item for item in TRANSFERS if item[0] == "office31")
    if selection == "visda-c":
        return tuple(item for item in TRANSFERS if item[0] == "visda-c")
    raise ValueError("datasets must be all, office31, or visda-c")
