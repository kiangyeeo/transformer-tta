"""Configuration and server-asset contracts for DeiT TTDA source-only."""

import copy
import hashlib
import json
import os.path as osp
import re

import yaml

from experiment_identity import resolve_deit_ttda_source_only_identity


CONFIG_SCHEMA_VERSION = 1
DATASET_SPECS = {
    "office31": {
        "domains": ("amazon", "dslr", "webcam"),
        "num_classes": 31,
    },
    "visda-c": {
        "domains": ("train", "validation"),
        "num_classes": 12,
    },
}
MODEL_NAME = "deit_small_patch16_224.fb_in1k"
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
SOURCE_CHECKPOINT_SCHEMA_VERSION = 1
SOURCE_CHECKPOINT_KIND = "deit_source_checkpoint"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        value = yaml.safe_load(file_obj)
    if not isinstance(value, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return value


def _absolute(path, project_root):
    if osp.isabs(path):
        return osp.normpath(path)
    return osp.abspath(osp.join(project_root, path))


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_target_records(path, num_classes):
    records = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                image_path, label_text = line.rsplit(maxsplit=1)
                label = int(label_text)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid target-list record at {path}:{line_number}"
                ) from error
            if not osp.isabs(image_path):
                raise ValueError(
                    f"Target image path must be absolute at {path}:{line_number}"
                )
            if not 0 <= label < num_classes:
                raise ValueError(
                    f"Target label {label} is outside [0, {num_classes})"
                )
            records.append((image_path, label))
    if not records:
        raise ValueError(f"Target list contains no records: {path}")
    present = {label for _, label in records}
    missing = sorted(set(range(num_classes)) - present)
    if missing:
        raise ValueError(f"Target list is missing labels: {missing}")
    return records


def _load_class_mapping(path, dataset, target_name, target_list, num_classes):
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if payload.get("dataset") != dataset:
        raise ValueError("class_to_idx.json dataset does not match config")
    mapping = payload.get("class_to_idx")
    if not isinstance(mapping, dict):
        raise ValueError("class_to_idx.json must contain class_to_idx")
    labels = sorted(int(value) for value in mapping.values())
    if labels != list(range(num_classes)):
        raise ValueError(
            f"class_to_idx labels must be exactly [0, {num_classes})"
        )
    domain_record = payload.get("domains", {}).get(target_name)
    if not isinstance(domain_record, dict):
        raise ValueError(f"class_to_idx.json has no domain {target_name}")
    mapped_path = osp.normcase(osp.realpath(domain_record.get("path", "")))
    expected_path = osp.normcase(osp.realpath(target_list))
    if mapped_path != expected_path:
        raise ValueError(
            "class_to_idx target list does not match configured target list"
        )
    class_names = [None] * num_classes
    for name, label in mapping.items():
        class_names[int(label)] = str(name)
    if dataset == "visda-c" and class_names != list(VISDA_CLASS_NAMES):
        raise ValueError("VisDA-C class mapping is not in canonical class order")
    return payload, class_names


def _same_path(first, second):
    return osp.normcase(osp.realpath(first)) == osp.normcase(osp.realpath(second))


def load_source_manifest(
    manifest_path,
    checkpoint_path,
    *,
    dataset,
    source_name,
    num_classes,
    model_name,
):
    if checkpoint_path.lower().endswith(".last.pth"):
        raise ValueError("Resume .last.pth files cannot be used as source W0")
    if not checkpoint_path.lower().endswith(".pth"):
        raise ValueError("Source W0 must use the .pth extension")
    if not osp.isfile(checkpoint_path):
        raise FileNotFoundError(f"Source checkpoint not found: {checkpoint_path}")
    if not osp.isfile(manifest_path):
        raise FileNotFoundError(
            f"Adjacent source manifest not found: {manifest_path}"
        )
    with open(manifest_path, "r", encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)
    checkpoint = manifest.get("checkpoint", {})
    if checkpoint.get("schema_version") != SOURCE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported source checkpoint schema in manifest")
    if checkpoint.get("kind") != SOURCE_CHECKPOINT_KIND:
        raise ValueError("Manifest does not describe a DeiT source checkpoint")
    if not _same_path(checkpoint.get("path", ""), checkpoint_path):
        raise ValueError("Manifest checkpoint path does not match configured W0")
    checkpoint_sha256 = str(checkpoint.get("sha256", "")).lower()
    if not SHA256_PATTERN.fullmatch(checkpoint_sha256):
        raise ValueError("Manifest checkpoint SHA-256 is invalid")

    dataset_record = manifest.get("dataset", {})
    if dataset_record.get("name") != dataset:
        raise ValueError("Source manifest dataset does not match target task")
    if dataset_record.get("source_domain") != source_name:
        raise ValueError("Source manifest domain does not match source task")
    if int(dataset_record.get("num_classes", -1)) != num_classes:
        raise ValueError("Source manifest class count is incompatible")

    model_record = manifest.get("model", {})
    if model_record.get("name") != model_name:
        raise ValueError("Source manifest model name is incompatible")
    if int(model_record.get("num_classes", -1)) != num_classes:
        raise ValueError("Source manifest model class count is incompatible")
    expected_head = f"Linear(384,{num_classes})"
    if model_record.get("head_schema") != expected_head:
        raise ValueError(
            f"Source manifest must use direct {expected_head} head"
        )
    if float(model_record.get("drop_rate", -1.0)) != 0.0:
        raise ValueError("Source manifest drop_rate must be 0.0")
    if float(model_record.get("drop_path_rate", -1.0)) != 0.0:
        raise ValueError("Source manifest drop_path_rate must be 0.0")

    training = manifest.get("training", {})
    source_training_seed = training.get("seed")
    if isinstance(source_training_seed, bool) or not isinstance(
        source_training_seed, int
    ):
        raise ValueError("Source manifest training seed is missing or invalid")
    best = manifest.get("best", {})
    if (
        not isinstance(best, dict)
        or "epoch" not in best
        or not isinstance(best.get("source_validation"), dict)
    ):
        raise ValueError("Source manifest best-checkpoint metadata is missing")
    training_git = manifest.get("git")
    if not isinstance(training_git, dict) or "commit" not in training_git:
        raise ValueError("Source manifest training Git metadata is missing")
    training_environment = manifest.get("environment")
    required_environment = {
        "python",
        "cuda",
        "torch",
        "torchvision",
        "timm",
        "safetensors",
    }
    if not isinstance(training_environment, dict) or not required_environment.issubset(
        training_environment
    ):
        raise ValueError("Source manifest training environment is incomplete")
    scientific_config_sha256 = str(
        manifest.get("scientific_config_sha256", "")
    ).lower()
    if not SHA256_PATTERN.fullmatch(scientific_config_sha256):
        raise ValueError("Source manifest scientific config SHA-256 is invalid")
    if not isinstance(manifest.get("effective_config"), dict):
        raise ValueError("Source manifest effective config is missing")
    return {
        "path": checkpoint_path,
        "manifest_path": manifest_path,
        "sha256": checkpoint_sha256,
        "schema_version": checkpoint["schema_version"],
        "kind": checkpoint["kind"],
        "source_training_seed": int(source_training_seed),
        "best": copy.deepcopy(best),
        "training_git": copy.deepcopy(training_git),
        "training_environment": copy.deepcopy(training_environment),
        "scientific_config_sha256": scientific_config_sha256,
    }


def apply_overrides(config, args):
    effective = copy.deepcopy(config)
    for name in ("dataset", "source", "target", "seed"):
        value = getattr(args, name, None)
        if value is not None:
            if name in ("dataset", "source", "target"):
                effective["data"][name] = value
            else:
                effective[name] = value
    if getattr(args, "gpu_id", None) is not None:
        effective["device"]["gpu_id"] = str(args.gpu_id)
    if getattr(args, "batch_size", None) is not None:
        effective["evaluation"]["batch_size"] = int(args.batch_size)
    if getattr(args, "workers", None) is not None:
        effective["evaluation"]["workers"] = int(args.workers)
    if getattr(args, "data_root", None) is not None:
        effective["data"]["list_root"] = args.data_root
    if getattr(args, "source_checkpoint_root", None) is not None:
        effective["model"]["source_checkpoint_root"] = (
            args.source_checkpoint_root
        )
    if getattr(args, "output_root", None) is not None:
        effective["output"]["root"] = args.output_root
    if getattr(args, "run_name", None) is not None:
        effective["output"]["run_name"] = args.run_name
    return effective


def resolve_config(
    config,
    project_root,
    *,
    provided_key=None,
    provided_sha256=None,
):
    effective = copy.deepcopy(config)
    if effective.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    fixed = {
        "method": "no_tta",
        "task": "ttda",
        "variant": "source_only",
    }
    for field, expected in fixed.items():
        if effective.get(field) != expected:
            raise ValueError(f"{field} must be {expected}")
    effective["seed"] = int(effective.get("seed", 2020))

    data = effective.get("data")
    if not isinstance(data, dict):
        raise ValueError("data must be a mapping")
    dataset = data.get("dataset")
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported TTDA dataset: {dataset}")
    spec = DATASET_SPECS[dataset]
    source, target = int(data["source"]), int(data["target"])
    if source == target:
        raise ValueError("Source and target domains must differ")
    if not 0 <= source < len(spec["domains"]):
        raise ValueError(f"Invalid source index {source} for {dataset}")
    if not 0 <= target < len(spec["domains"]):
        raise ValueError(f"Invalid target index {target} for {dataset}")
    if dataset == "visda-c" and (source, target) != (0, 1):
        raise ValueError("VisDA-C only supports train -> validation")
    source_name, target_name = spec["domains"][source], spec["domains"][target]
    list_root = _absolute(data["list_root"], project_root)
    dataset_list_root = osp.join(list_root, dataset)
    target_list = osp.join(dataset_list_root, f"{target_name}_list.txt")
    class_mapping_path = osp.join(dataset_list_root, "class_to_idx.json")
    if not osp.isfile(target_list):
        raise FileNotFoundError(f"Target list not found: {target_list}")
    if not osp.isfile(class_mapping_path):
        raise FileNotFoundError(
            f"Class mapping not found: {class_mapping_path}"
        )
    records = _read_target_records(target_list, spec["num_classes"])
    _, class_names = _load_class_mapping(
        class_mapping_path,
        dataset,
        target_name,
        target_list,
        spec["num_classes"],
    )
    data.update(
        {
            "dataset": dataset,
            "source": source,
            "target": target,
            "domains": list(spec["domains"]),
            "source_name": source_name,
            "target_name": target_name,
            "num_classes": spec["num_classes"],
            "list_root": list_root,
            "target_list": target_list,
            "target_list_sha256": sha256_file(target_list),
            "target_sample_count": len(records),
            "class_mapping_path": class_mapping_path,
            "class_mapping_sha256": sha256_file(class_mapping_path),
            "class_names": class_names,
        }
    )

    model = effective.get("model")
    if not isinstance(model, dict) or model.get("name") != MODEL_NAME:
        raise ValueError(f"model.name must be {MODEL_NAME}")
    checkpoint_root = _absolute(model["source_checkpoint_root"], project_root)
    checkpoint_path = osp.join(checkpoint_root, dataset, f"{source_name}.pth")
    manifest_path = osp.splitext(checkpoint_path)[0] + ".manifest.json"
    checkpoint = load_source_manifest(
        manifest_path,
        checkpoint_path,
        dataset=dataset,
        source_name=source_name,
        num_classes=spec["num_classes"],
        model_name=MODEL_NAME,
    )
    model["source_checkpoint_root"] = checkpoint_root
    model["head_schema"] = f"Linear(384,{spec['num_classes']})"
    effective["source_checkpoint"] = checkpoint

    preprocessing = effective.get("preprocessing")
    expected_preprocessing = {
        "resize_size": 256,
        "crop_size": 224,
        "interpolation": "bicubic",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    if preprocessing != expected_preprocessing:
        raise ValueError(
            "TTDA source-only preprocessing must match source validation v1"
        )

    evaluation = effective.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a mapping")
    for field in ("batch_size", "workers"):
        value = evaluation[field]
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"evaluation.{field} must be an integer")
        evaluation[field] = int(value)
    if evaluation["batch_size"] <= 0 or evaluation["workers"] < 0:
        raise ValueError("Invalid evaluation batch_size or workers")
    for field in ("amp", "deterministic", "pin_memory"):
        if evaluation.get(field) not in (True, False):
            raise ValueError(f"evaluation.{field} must be boolean")
    if evaluation.get("order") != "sequential":
        raise ValueError("TTDA source-only order must be sequential")
    if evaluation.get("drop_last") is not False:
        raise ValueError("TTDA source-only drop_last must be false")

    metrics = effective.get("metrics")
    if metrics != {
        "primary": "macro_class_accuracy",
        "class_denominator": "fixed_dataset_classes",
        "report_overall": True,
        "report_per_class": True,
    }:
        raise ValueError("TTDA source-only metric policy is frozen")

    device = effective.get("device")
    if not isinstance(device, dict) or device.get("type") not in ("cuda", "cpu"):
        raise ValueError("device.type must be cuda or cpu")
    device["gpu_id"] = str(device.get("gpu_id", "0"))
    output = effective.get("output")
    if not isinstance(output, dict):
        raise ValueError("output must be a mapping")
    output["root"] = _absolute(output["root"], project_root)
    output.setdefault("run_name", None)
    effective["task_name"] = f"{source_name[0].upper()}{target_name[0].upper()}"

    identity = resolve_deit_ttda_source_only_identity(
        effective,
        provided_key=provided_key,
        provided_sha256=provided_sha256,
    )
    effective.update(identity)
    return effective


def experiment_output_root(config):
    data = config["data"]
    return osp.join(
        config["output"]["root"],
        "ttda",
        data["dataset"],
        config["task_name"],
        "source_only",
        f"seed_{int(config['seed'])}",
    )
