"""Configuration loading and validation for DeiT source training."""

import copy
import hashlib
import json
import os.path as osp

import yaml


CONFIG_SCHEMA_VERSION = 1
DATASET_SPECS = {
    "office31": {
        "domains": ("amazon", "dslr", "webcam"),
        "num_classes": 31,
        "selection_metrics": ("overall_accuracy",),
    },
    "visda-c": {
        "domains": ("train",),
        "num_classes": 12,
        "selection_metrics": ("macro_class_accuracy",),
    },
}
MODEL_NAME = "deit_small_patch16_224.fb_in1k"


def load_source_config(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    if not isinstance(config, dict):
        raise ValueError(f"Source config must contain a mapping: {path}")
    return config


def dump_source_config(path, config):
    with open(path, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(
            config,
            file_obj,
            sort_keys=False,
            allow_unicode=True,
        )


def _absolute(path, project_root):
    if osp.isabs(path):
        return osp.normpath(path)
    return osp.abspath(osp.join(project_root, path))


def _positive_int(value, field):
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _positive_float(value, field, allow_zero=False):
    value = float(value)
    invalid = value < 0 if allow_zero else value <= 0
    if invalid:
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{field} must be {relation}")
    return value


def canonical_config_sha256(config):
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def apply_source_overrides(
    config,
    *,
    source_domain=None,
    source_list=None,
    pretrained_path=None,
    output_path=None,
    device=None,
):
    effective = copy.deepcopy(config)
    if source_domain is not None:
        effective["data"]["source_domain"] = source_domain
    if source_list is not None:
        effective["data"]["source_list"] = source_list
    if pretrained_path is not None:
        effective["model"]["pretrained_path"] = pretrained_path
    if output_path is not None:
        effective["checkpoint"]["output_path"] = output_path
    if device is not None:
        effective["runtime"]["device"] = device
    return effective


def resolve_source_config(config, project_root):
    """Resolve paths and reject scientifically ambiguous configurations."""
    effective = copy.deepcopy(config)
    if effective.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {CONFIG_SCHEMA_VERSION}"
        )

    data = effective.get("data")
    if not isinstance(data, dict):
        raise ValueError("data must be a mapping")
    dataset = data.get("dataset")
    if dataset not in DATASET_SPECS:
        raise ValueError(
            f"data.dataset must be one of {sorted(DATASET_SPECS)}"
        )
    spec = DATASET_SPECS[dataset]
    source_domain = str(data.get("source_domain", "")).lower()
    if source_domain not in spec["domains"]:
        raise ValueError(
            f"Invalid source domain {source_domain!r} for {dataset}; "
            f"expected one of {list(spec['domains'])}"
        )
    data["source_domain"] = source_domain
    data["num_classes"] = int(data.get("num_classes", spec["num_classes"]))
    if data["num_classes"] != spec["num_classes"]:
        raise ValueError(
            f"{dataset} requires {spec['num_classes']} classes, got "
            f"{data['num_classes']}"
        )
    source_list = str(data["source_list"]).format(
        source_domain=source_domain
    )
    data["source_list"] = _absolute(source_list, project_root)
    class_mapping_path = str(data["class_mapping_path"]).format(
        source_domain=source_domain
    )
    data["class_mapping_path"] = _absolute(
        class_mapping_path, project_root
    )
    split = data.get("split")
    if not isinstance(split, dict):
        raise ValueError("data.split must be a mapping")
    if split.get("strategy") != "stratified_fixed":
        raise ValueError(
            "data.split.strategy must be stratified_fixed"
        )
    validation_fraction = float(split["validation_fraction"])
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "data.split.validation_fraction must be in (0, 1)"
        )
    split["validation_fraction"] = validation_fraction
    split["seed"] = int(split["seed"])

    model = effective.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be a mapping")
    if model.get("name") != MODEL_NAME:
        raise ValueError(f"model.name must be {MODEL_NAME}")
    if model.get("head") != "linear":
        raise ValueError("model.head must be linear")
    if int(model.get("hidden_dim", 0)) != 384:
        raise ValueError("model.hidden_dim must be 384")
    if int(model.get("pretrained_num_classes", 0)) != 1000:
        raise ValueError("model.pretrained_num_classes must be 1000")
    model["pretrained_path"] = _absolute(
        model["pretrained_path"], project_root
    )
    pretrained_config_path = model.get("pretrained_config_path")
    if not pretrained_config_path:
        raise ValueError("model.pretrained_config_path is required")
    model["pretrained_config_path"] = _absolute(
        pretrained_config_path, project_root
    )
    model["head_init_std"] = _positive_float(
        model["head_init_std"], "model.head_init_std"
    )
    for stochastic_field in ("drop_rate", "drop_path_rate"):
        value = float(model[stochastic_field])
        if value != 0.0:
            raise ValueError(
                f"model.{stochastic_field} is frozen to 0.0 for source v1"
            )
        model[stochastic_field] = value

    preprocessing = effective.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("preprocessing must be a mapping")
    if preprocessing.get("interpolation") != "bicubic":
        raise ValueError("preprocessing.interpolation must be bicubic")
    preprocessing["resize_size"] = _positive_int(
        preprocessing["resize_size"], "preprocessing.resize_size"
    )
    preprocessing["crop_size"] = _positive_int(
        preprocessing["crop_size"], "preprocessing.crop_size"
    )
    if preprocessing["crop_size"] != 224:
        raise ValueError("preprocessing.crop_size must be 224")
    flip_probability = float(preprocessing["horizontal_flip_probability"])
    if not 0.0 <= flip_probability <= 1.0:
        raise ValueError(
            "preprocessing.horizontal_flip_probability must be in [0, 1]"
        )
    preprocessing["horizontal_flip_probability"] = flip_probability
    for field in ("mean", "std"):
        values = [float(value) for value in preprocessing[field]]
        if len(values) != 3:
            raise ValueError(f"preprocessing.{field} must have 3 values")
        preprocessing[field] = values
    if preprocessing.get("strong_augmentation") != "disabled":
        raise ValueError(
            "preprocessing.strong_augmentation must be disabled in source v1"
        )

    training = effective.get("training")
    if not isinstance(training, dict):
        raise ValueError("training must be a mapping")
    training["epochs"] = _positive_int(
        training["epochs"], "training.epochs"
    )
    training["batch_size"] = _positive_int(
        training["batch_size"], "training.batch_size"
    )
    training["workers"] = int(training["workers"])
    if training["workers"] < 0:
        raise ValueError("training.workers must be >= 0")
    training["seed"] = int(training["seed"])
    training["label_smoothing"] = float(training["label_smoothing"])
    if not 0.0 <= training["label_smoothing"] < 1.0:
        raise ValueError("training.label_smoothing must be in [0, 1)")
    if training.get("finetune_scope") != "full_model":
        raise ValueError(
            "training.finetune_scope must be full_model; head-only is a "
            "different linear-probe experiment"
        )
    if training.get("amp") not in (True, False):
        raise ValueError("training.amp must be boolean")
    if training.get("gradient_clipping") is not None:
        raise ValueError(
            "training.gradient_clipping must be null in source v1"
        )
    if training.get("model_ema") is not False:
        raise ValueError("training.model_ema must be false in source v1")

    optimizer = training.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("name") != "adamw":
        raise ValueError("training.optimizer.name must be adamw")
    optimizer["backbone_lr"] = _positive_float(
        optimizer["backbone_lr"], "training.optimizer.backbone_lr"
    )
    optimizer["head_lr"] = _positive_float(
        optimizer["head_lr"], "training.optimizer.head_lr"
    )
    optimizer["weight_decay"] = _positive_float(
        optimizer["weight_decay"],
        "training.optimizer.weight_decay",
        allow_zero=True,
    )
    optimizer["betas"] = tuple(float(value) for value in optimizer["betas"])
    if len(optimizer["betas"]) != 2 or not all(
        0.0 <= value < 1.0 for value in optimizer["betas"]
    ):
        raise ValueError("training.optimizer.betas must contain 2 values in [0, 1)")
    optimizer["eps"] = _positive_float(
        optimizer["eps"], "training.optimizer.eps"
    )
    if optimizer.get("no_decay_policy") != "bias_norm_and_tokens":
        raise ValueError(
            "training.optimizer.no_decay_policy must be bias_norm_and_tokens"
        )

    scheduler = training.get("scheduler")
    if not isinstance(scheduler, dict) or scheduler.get("name") != "cosine":
        raise ValueError("training.scheduler.name must be cosine")
    scheduler["warmup_epochs"] = int(scheduler["warmup_epochs"])
    if not 0 <= scheduler["warmup_epochs"] < training["epochs"]:
        raise ValueError(
            "training.scheduler.warmup_epochs must be in [0, epochs)"
        )
    scheduler["min_lr"] = _positive_float(
        scheduler["min_lr"],
        "training.scheduler.min_lr",
        allow_zero=True,
    )
    if scheduler["min_lr"] >= optimizer["backbone_lr"]:
        raise ValueError(
            "training.scheduler.min_lr must be below backbone_lr"
        )

    checkpoint = effective.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping")
    output_path = str(checkpoint["output_path"]).format(
        source_domain=source_domain
    )
    checkpoint["output_path"] = _absolute(output_path, project_root)
    if osp.basename(checkpoint["output_path"]) != f"{source_domain}.pth":
        raise ValueError(
            "checkpoint.output_path must end with "
            f"{source_domain}.pth"
        )
    selection_metric = checkpoint.get("selection_metric")
    if selection_metric not in spec["selection_metrics"]:
        raise ValueError(
            f"checkpoint.selection_metric for {dataset} must be one of "
            f"{list(spec['selection_metrics'])}"
        )
    checkpoint["save_last_each_epoch"] = bool(
        checkpoint.get("save_last_each_epoch", True)
    )

    runtime = effective.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a mapping")
    if runtime.get("device") not in ("cuda", "cpu"):
        raise ValueError("runtime.device must be cuda or cpu")
    if runtime.get("deterministic") is not True:
        raise ValueError("runtime.deterministic must be true")
    runtime["pin_memory"] = bool(runtime.get("pin_memory", True))

    effective["scientific_config_sha256"] = canonical_config_sha256(
        effective
    )
    return effective
