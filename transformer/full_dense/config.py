"""Frozen configuration and transfer identity for full-dense SHOT-OTTA."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

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


PROTOCOL_REVISION = "transformer_full_dense_otta_20260822_v1_fc_preprocess"


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def _validate_frozen_fields(config: dict[str, Any]) -> None:
    expected_top = {
        "schema_version": 1,
        "protocol_revision": PROTOCOL_REVISION,
        "formal_seed": FORMAL_SEED,
        "method": "shot",
        "variant": "full_dense",
    }
    for key, expected in expected_top.items():
        if config.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}")
    if config.get("model", {}).get("name") != MODEL_NAME:
        raise ValueError(f"model.name must be {MODEL_NAME}")

    expected_dataset = {
        "office31": {"batch_size": 64, "fo_batch_size": 64, "workers": 4, "num_classes": 31},
        "visda-c": {"batch_size": 256, "fo_batch_size": 256, "workers": 4, "num_classes": 12},
    }
    for dataset, expected in expected_dataset.items():
        if config.get("data", {}).get(dataset) != expected:
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
    if config.get("data", {}).get("preprocessing") != expected_preprocessing:
        raise ValueError("data.preprocessing differs from the frozen FC-aligned protocol")
    expected_stream = {
        "order": "fixed_random_permutation",
        "one_pass": True,
        "drop_last": False,
    }
    if config.get("data", {}).get("stream") != expected_stream:
        raise ValueError(f"data.stream must be exactly {expected_stream}")

    expected_adaptation = {
        "update_scope": "all_except_head",
        "model_mode": "eval",
        "steps_per_online_batch": 1,
    }
    if config.get("adaptation") != expected_adaptation:
        raise ValueError(f"adaptation must be exactly {expected_adaptation}")
    expected_optimization = {
        "optimizer": "adamw",
        "lr": 1.0e-5,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.01,
    }
    if config.get("optimization") != expected_optimization:
        raise ValueError(f"optimization must be exactly {expected_optimization}")
    expected_loss = {
        "components": ["ent", "div", "pseudo"],
        "cls_par": 0.3,
        "ent_par": 1.0,
        "threshold": 0.0,
    }
    if config.get("loss") != expected_loss:
        raise ValueError(f"loss must be exactly {expected_loss}")
    expected_runtime = {"deterministic": True, "amp": False, "pin_memory": True}
    if config.get("runtime") != expected_runtime:
        raise ValueError(f"runtime must be exactly {expected_runtime}")


def resolve_transfer_config(
    raw_config: dict[str, Any],
    *,
    project_root: Path,
    dataset: str,
    source: str,
    target: str,
    device: str,
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Resolve one formal transfer and fail closed on every frozen field."""
    config = copy.deepcopy(raw_config)
    _validate_frozen_fields(config)
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

    resolved = {
        "schema_version": 1,
        "protocol_revision": PROTOCOL_REVISION,
        "formal_seed": FORMAL_SEED,
        "method": "shot",
        "variant": "full_dense",
        "dataset": dataset,
        "source": source,
        "target": target,
        "transfer": f"{source}->{target}",
        "num_classes": int(DATASETS[dataset]["num_classes"]),
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
        "loss": copy.deepcopy(config["loss"]),
        "runtime": {**copy.deepcopy(config["runtime"]), "device": device},
        "output_dir": str(Path(output_dir).resolve()),
    }
    scientific = {
        key: value
        for key, value in resolved.items()
        if key not in {"output_dir", "checkpoint_manifest_path"}
    }
    resolved["scientific_config_sha256"] = canonical_sha256(scientific)
    resolved["experiment_key"] = (
        f"{dataset}_{source}-{target}_full-dense_seed-{FORMAL_SEED}_"
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
