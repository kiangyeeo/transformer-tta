"""Frozen-field validation and transfer resolution shared by COME variants.

The substrate blocks (model, data, preprocessing, stream, adaptation,
optimization, runtime) are byte-identical to the SHOT-Transformer configs;
``tests/transformer_come_substrate_identity_test.py`` asserts that against the
SHOT YAML files, so a future edit on either side cannot drift silently.  Only
``method``, ``protocol_revision``, the ``loss`` -> ``come`` block swap and the
output root differ.
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

from .identity import (
    EXPECTED_COME_BLOCK,
    IMPLEMENTATION_REVISIONS,
    PROTOCOL_DOCUMENT,
    PROTOCOL_REVISIONS,
    come_objective_payload,
)


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

VARIANT_KEYS = {
    "full_dense": "full-dense",
    "candidate_dense": "candidate-dense",
    "group_random": "group-random",
    "group_magnitude": "group-magnitude",
    "group_saliency": "group-saliency",
}


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def validate_common_fields(
    config: dict[str, Any], *, variant: str, adaptation: dict[str, Any]
) -> None:
    """Fail closed on every frozen non-selection field of a COME config."""

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
    if config.get("adaptation") != adaptation:
        raise ValueError(f"adaptation must be exactly {adaptation}")
    if config.get("optimization") != EXPECTED_OPTIMIZATION:
        raise ValueError(f"optimization must be exactly {EXPECTED_OPTIMIZATION}")
    if config.get("runtime") != EXPECTED_RUNTIME:
        raise ValueError(f"runtime must be exactly {EXPECTED_RUNTIME}")


def resolve_common_transfer_config(
    raw_config: dict[str, Any],
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

    Selection blocks and the scientific hash are added by the variant config
    module, so sparse variants cannot silently fall back to a dense identity.
    """

    config = copy.deepcopy(raw_config)
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


def resolve_sparse_transfer_config(
    raw_config: dict[str, Any],
    *,
    variant: str,
    project_root: Path,
    dataset: str,
    source: str,
    target: str,
    budget,
    device: str,
    output_dir: str | os.PathLike[str],
    per_step: bool = False,
) -> dict[str, Any]:
    """Resolve one sparse COME condition, including its integer budget.

    ``floor(rho * 6912)`` comes from the SHOT budget helper, so exact-K is
    the same integer as the matched SHOT condition: no ceil, no slack, no
    per-block quota and no minimum-one rule.
    """

    from .budget import GROUP_SIZE, budget_group_count, normalize_budget

    normalized = normalize_budget(budget)
    group_count = budget_group_count(normalized)
    resolved = resolve_common_transfer_config(
        raw_config,
        variant=variant,
        project_root=project_root,
        dataset=dataset,
        source=source,
        target=target,
        device=device,
        output_dir=output_dir,
    )
    scalars_key = (
        "active_candidate_scalars_per_step" if per_step else "active_candidate_scalars"
    )
    resolved["selection"] = {
        **copy.deepcopy(raw_config["selection"]),
        "requested_budget": normalized,
        "requested_group_count": group_count,
        scalars_key: group_count * GROUP_SIZE,
    }
    return resolved


def validate_selection_block(config: dict[str, Any], expected: dict[str, Any]) -> None:
    if config.get("selection") != expected:
        raise ValueError(f"selection must be exactly {expected}")


NON_SCIENTIFIC_KEYS = {
    "output_dir",
    "checkpoint_manifest_path",
    "scientific_config_sha256",
    "experiment_key",
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


def select_transfers(selection: str) -> tuple[tuple[str, str, str], ...]:
    if selection == "all":
        return TRANSFERS
    if selection == "office31":
        return tuple(item for item in TRANSFERS if item[0] == "office31")
    if selection == "visda-c":
        return tuple(item for item in TRANSFERS if item[0] == "visda-c")
    raise ValueError("selection must be all, office31, or visda-c")


__all__ = [
    "CANDIDATE_ADAPTATION",
    "EXPECTED_DATA_BLOCKS",
    "EXPECTED_OPTIMIZATION",
    "EXPECTED_PREPROCESSING",
    "EXPECTED_RUNTIME",
    "EXPECTED_STREAM",
    "FULL_DENSE_ADAPTATION",
    "VARIANT_KEYS",
    "finalize_identity",
    "load_config",
    "resolve_common_transfer_config",
    "resolve_sparse_transfer_config",
    "validate_selection_block",
    "select_transfers",
    "validate_common_fields",
]
