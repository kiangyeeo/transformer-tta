"""Configuration and immutable task definitions for source-only OTTA."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


FORMAL_SEED = 2026
PROTOCOL_REVISION = "transformer_source_only_otta_20260822_v1"
MODEL_NAME = "deit_small_patch16_224.fb_in1k"

DATASETS = {
    "office31": {
        "domains": ("amazon", "dslr", "webcam"),
        "num_classes": 31,
    },
    "visda-c": {
        "domains": ("train", "validation"),
        "num_classes": 12,
    },
}

TRANSFERS = (
    ("office31", "amazon", "dslr"),
    ("office31", "amazon", "webcam"),
    ("office31", "dslr", "amazon"),
    ("office31", "dslr", "webcam"),
    ("office31", "webcam", "amazon"),
    ("office31", "webcam", "dslr"),
    ("visda-c", "train", "validation"),
)

VISDA_CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bus",
    "car",
    "horse",
    "knife",
    "motorcycle",
    "person",
    "plant",
    "skateboard",
    "train",
    "truck",
)


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        while chunk := file_obj.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _absolute(path: str | os.PathLike[str], project_root: Path) -> str:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = project_root / value
    return str(value.resolve())


def _read_class_mapping(
    path: str, dataset: str, target: str, target_list: str
) -> list[str]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if payload.get("dataset") != dataset:
        raise ValueError("class_to_idx.json dataset does not match the task")
    mapping = payload.get("class_to_idx")
    if not isinstance(mapping, dict):
        raise ValueError("class_to_idx.json is missing class_to_idx")
    expected_count = DATASETS[dataset]["num_classes"]
    if sorted(int(value) for value in mapping.values()) != list(
        range(expected_count)
    ):
        raise ValueError("class_to_idx labels are not contiguous fixed classes")
    domain = payload.get("domains", {}).get(target)
    if not isinstance(domain, dict) or "path" not in domain:
        raise ValueError(f"class_to_idx.json is missing target domain {target}")
    if Path(domain["path"]).resolve() != Path(target_list).resolve():
        raise ValueError("class_to_idx target list differs from the configured list")
    class_names: list[str | None] = [None] * expected_count
    for name, label in mapping.items():
        class_names[int(label)] = str(name)
    result = [str(name) for name in class_names]
    if dataset == "visda-c" and tuple(result) != VISDA_CLASS_NAMES:
        raise ValueError("VisDA-C class order is not the canonical 12-class order")
    return result


def _validate_frozen_fields(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("source-only config schema_version must be 1")
    if config.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError(f"protocol_revision must be {PROTOCOL_REVISION}")
    if int(config.get("formal_seed", -1)) != FORMAL_SEED:
        raise ValueError(f"formal_seed must be {FORMAL_SEED}")
    if config.get("model", {}).get("name") != MODEL_NAME:
        raise ValueError(f"model.name must be {MODEL_NAME}")

    stream = config.get("data", {}).get("stream", {})
    expected_stream = {
        "order": "fixed_random_permutation",
        "one_pass": True,
        "drop_last": False,
    }
    if stream != expected_stream:
        raise ValueError(f"data.stream must be exactly {expected_stream}")
    runtime = config.get("runtime", {})
    if runtime.get("deterministic") is not True:
        raise ValueError("runtime.deterministic must be true")
    if runtime.get("amp") is not False:
        raise ValueError("runtime.amp is frozen to false for formal comparison")

    preprocessing = config.get("data", {}).get("preprocessing", {})
    expected_preprocessing = {
        "resize_size": 256,
        "crop_size": 224,
        "interpolation": "bicubic",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "online_random_crop": True,
        "online_random_horizontal_flip": True,
        "fo_center_crop": True,
    }
    if preprocessing != expected_preprocessing:
        raise ValueError(
            "data.preprocessing differs from the frozen source-only protocol"
        )


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
    """Resolve one formal transfer and fail closed on protocol mismatches."""
    config = copy.deepcopy(raw_config)
    _validate_frozen_fields(config)
    if (dataset, source, target) not in TRANSFERS:
        raise ValueError(f"Unsupported formal transfer: {dataset} {source}->{target}")
    if device != "cpu" and device != "cuda":
        raise ValueError("device must be cpu or cuda")

    data_root = Path(_absolute(config["data"]["list_root"], project_root))
    checkpoint_root = Path(
        _absolute(config["model"]["checkpoint_root"], project_root)
    )
    target_list = data_root / dataset / f"{target}_list.txt"
    class_mapping = data_root / dataset / "class_to_idx.json"
    checkpoint = checkpoint_root / dataset / f"{source}.pth"
    checkpoint_manifest = checkpoint.with_suffix(".manifest.json")
    required = (target_list, class_mapping, checkpoint, checkpoint_manifest)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required local asset not found: {path}")
    if checkpoint.name.endswith(".last.pth"):
        raise ValueError("*.last.pth cannot be used as source W0")

    dataset_config = config["data"].get(dataset, {})
    expected_classes = int(DATASETS[dataset]["num_classes"])
    if int(dataset_config.get("num_classes", -1)) != expected_classes:
        raise ValueError(f"{dataset} must use {expected_classes} classes")
    batch_size = int(dataset_config.get("batch_size", 0))
    workers = int(dataset_config.get("workers", -1))
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch_size must be positive and workers non-negative")

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
    manifest_checkpoint_path = Path(checkpoint_record.get("path", ""))
    if manifest_checkpoint_path.resolve() != checkpoint.resolve():
        raise ValueError("Source manifest checkpoint path differs from W0")
    resolved = {
        "schema_version": 1,
        "protocol_revision": PROTOCOL_REVISION,
        "formal_seed": FORMAL_SEED,
        "variant": "source_only",
        "dataset": dataset,
        "source": source,
        "target": target,
        "transfer": f"{source}->{target}",
        "num_classes": expected_classes,
        "class_names": class_names,
        "batch_size": batch_size,
        "workers": workers,
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
        "runtime": {
            **copy.deepcopy(config["runtime"]),
            "device": device,
        },
        "output_dir": str(Path(output_dir).resolve()),
    }
    scientific = {
        key: value
        for key, value in resolved.items()
        if key not in {"output_dir", "checkpoint_manifest_path"}
    }
    resolved["scientific_config_sha256"] = canonical_sha256(scientific)
    resolved["experiment_key"] = (
        f"{dataset}_{source}-{target}_source-only_seed-{FORMAL_SEED}_"
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
