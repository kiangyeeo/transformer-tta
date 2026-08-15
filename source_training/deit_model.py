"""Local-only DeiT-S construction and versioned source checkpoint loading."""

import hashlib
import os.path as osp

import torch
import torch.nn as nn


SOURCE_CHECKPOINT_SCHEMA_VERSION = 1
SOURCE_CHECKPOINT_KIND = "deit_source_checkpoint"


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _default_model_factory(model_name, **kwargs):
    try:
        import timm
    except ImportError as error:
        raise RuntimeError(
            "DeiT source training requires timm; install requirements.txt"
        ) from error
    return timm.create_model(model_name, pretrained=False, **kwargs)


def _default_safetensors_loader(path):
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError(
            "Local DeiT loading requires safetensors; install requirements.txt"
        ) from error
    return load_file(path, device="cpu")


def _strip_module_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def _reset_linear_head(model, num_classes, head_init_std):
    if not hasattr(model, "reset_classifier"):
        raise TypeError("The selected timm model has no reset_classifier method")
    model.reset_classifier(num_classes=num_classes)
    head = model.get_classifier()
    if not isinstance(head, nn.Linear):
        raise TypeError(
            f"Expected a direct nn.Linear DeiT head, got {type(head).__name__}"
        )
    if head.in_features != 384 or head.out_features != num_classes:
        raise ValueError(
            "Unexpected DeiT classifier shape: "
            f"{head.in_features} -> {head.out_features}"
        )
    nn.init.trunc_normal_(head.weight, std=float(head_init_std))
    if head.bias is not None:
        nn.init.zeros_(head.bias)
    return head


def build_deit_source_model(
    *,
    model_name,
    num_classes,
    pretrained_path,
    pretrained_num_classes=1000,
    head_init_std=0.02,
    drop_rate=0.0,
    drop_path_rate=0.0,
    model_factory=None,
    state_loader=None,
):
    """Load the exact local ImageNet state, then replace its 1000-way head.

    ``pretrained=False`` is deliberately hard-coded.  This function never asks
    timm or Hugging Face to download weights.
    """
    if not osp.isfile(pretrained_path):
        raise FileNotFoundError(
            f"Local pretrained checkpoint not found: {pretrained_path}"
        )
    model_factory = model_factory or _default_model_factory
    state_loader = state_loader or _default_safetensors_loader
    model = model_factory(
        model_name,
        num_classes=int(pretrained_num_classes),
        drop_rate=float(drop_rate),
        drop_path_rate=float(drop_path_rate),
    )
    state_dict = _strip_module_prefix(state_loader(pretrained_path))
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict local pretrained load returned incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    head = _reset_linear_head(model, int(num_classes), head_init_std)
    return model, head


def build_deit_for_source_checkpoint(
    *,
    model_name,
    num_classes,
    drop_rate=0.0,
    drop_path_rate=0.0,
    model_factory=None,
):
    """Construct the exact architecture without any pretrained download."""
    model_factory = model_factory or _default_model_factory
    model = model_factory(
        model_name,
        num_classes=int(num_classes),
        drop_rate=float(drop_rate),
        drop_path_rate=float(drop_path_rate),
    )
    head = model.get_classifier()
    if not isinstance(head, nn.Linear):
        raise TypeError("Source checkpoint requires a direct nn.Linear head")
    if head.in_features != 384 or head.out_features != int(num_classes):
        raise ValueError(
            f"Source head must be 384 -> {num_classes}, got "
            f"{head.in_features} -> {head.out_features}"
        )
    return model


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_deit_source_checkpoint(
    path,
    *,
    device="cpu",
    expected_sha256=None,
    model_factory=None,
):
    """Load a versioned source W0 without touching ImageNet weights or network."""
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Source checkpoint SHA-256 mismatch: expected "
                f"{expected_sha256}, got {actual_sha256}"
            )
    payload = _torch_load(path, map_location="cpu")
    if payload.get("schema_version") != SOURCE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported DeiT source checkpoint schema")
    if payload.get("kind") != SOURCE_CHECKPOINT_KIND:
        raise ValueError("Not a DeiT source checkpoint")
    metadata = payload.get("metadata", {})
    model_schema = metadata.get("model", {})
    model = build_deit_for_source_checkpoint(
        model_name=model_schema["name"],
        num_classes=model_schema["num_classes"],
        drop_rate=model_schema.get("drop_rate", 0.0),
        drop_path_rate=model_schema.get("drop_path_rate", 0.0),
        model_factory=model_factory,
    )
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Source checkpoint state_dict is incompatible")
    model.to(device)
    return model, metadata

