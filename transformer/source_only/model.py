"""Strict local source-checkpoint loading and model-state invariants."""

from __future__ import annotations

import hashlib
import json
import os.path as osp

import torch
import torch.nn as nn

from transformer.source_training.deit_model import load_deit_source_checkpoint

from .config import sha256_file


def hash_model_state(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _same_path(first: str, second: str) -> bool:
    return osp.normcase(osp.realpath(first)) == osp.normcase(osp.realpath(second))


def load_frozen_source_model(config: dict, device: torch.device):
    """Verify manifest/W0 provenance, construct locally, and freeze all state."""
    checkpoint_path = config["checkpoint_path"]
    if checkpoint_path.lower().endswith(".last.pth"):
        raise ValueError("*.last.pth cannot be used as source W0")
    with open(
        config["checkpoint_manifest_path"], "r", encoding="utf-8"
    ) as file_obj:
        manifest = json.load(file_obj)
    checkpoint_record = manifest.get("checkpoint", {})
    if checkpoint_record.get("schema_version") != 1:
        raise ValueError("Unsupported source checkpoint schema")
    if checkpoint_record.get("kind") != "deit_source_checkpoint":
        raise ValueError("Manifest does not describe a DeiT source checkpoint")
    if not _same_path(checkpoint_record.get("path", ""), checkpoint_path):
        raise ValueError("Manifest checkpoint path does not match configured W0")
    expected_sha256 = str(checkpoint_record.get("sha256", "")).lower()
    if len(expected_sha256) != 64:
        raise ValueError("Manifest checkpoint SHA-256 is invalid")
    actual_sha256 = sha256_file(checkpoint_path)
    if expected_sha256 != config["checkpoint_sha256"]:
        raise ValueError("Resolved checkpoint SHA-256 differs from manifest")
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Source checkpoint hash mismatch: {actual_sha256} != {expected_sha256}"
        )

    dataset_record = manifest.get("dataset", {})
    expected_dataset = {
        "name": config["dataset"],
        "source_domain": config["source"],
        "num_classes": config["num_classes"],
    }
    for field, expected in expected_dataset.items():
        if dataset_record.get(field) != expected:
            raise ValueError(f"Source manifest dataset.{field} mismatch")
    model_record = manifest.get("model", {})
    if model_record.get("name") != config["model_name"]:
        raise ValueError("Source manifest model name mismatch")
    if int(model_record.get("num_classes", -1)) != config["num_classes"]:
        raise ValueError("Source manifest classifier size mismatch")
    if model_record.get("head_schema") != f"Linear(384,{config['num_classes']})":
        raise ValueError("Source manifest does not use the required direct head")
    if float(model_record.get("drop_rate", -1)) != 0.0:
        raise ValueError("Source manifest drop_rate must be 0.0")
    if float(model_record.get("drop_path_rate", -1)) != 0.0:
        raise ValueError("Source manifest drop_path_rate must be 0.0")

    model, checkpoint_metadata = load_deit_source_checkpoint(
        checkpoint_path,
        device=device,
        expected_sha256=None,
    )
    loaded_dataset = checkpoint_metadata.get("dataset", {})
    for field, expected in expected_dataset.items():
        if loaded_dataset.get(field) != expected:
            raise ValueError(f"Loaded checkpoint dataset.{field} mismatch")
    loaded_model = checkpoint_metadata.get("model", {})
    if loaded_model.get("name") != config["model_name"]:
        raise ValueError("Loaded checkpoint model name mismatch")
    if checkpoint_metadata.get("scientific_config_sha256") != manifest.get(
        "scientific_config_sha256"
    ):
        raise ValueError("Checkpoint and manifest scientific config differ")

    head = model.get_classifier()
    if not isinstance(head, nn.Linear):
        raise TypeError("DeiT source-only requires a direct Linear head")
    if head.in_features != 384 or head.out_features != config["num_classes"]:
        raise ValueError("Loaded DeiT classifier has the wrong shape")
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Failed to freeze every source-only parameter")
    return model, {
        "path": checkpoint_path,
        "sha256": actual_sha256,
        "manifest_path": config["checkpoint_manifest_path"],
        "best": manifest.get("best"),
        "source_training_seed": manifest.get("training", {}).get("seed"),
        "source_training_config_sha256": manifest.get(
            "scientific_config_sha256"
        ),
        "source_training_git": manifest.get("git"),
        "source_training_environment": manifest.get("environment"),
    }
