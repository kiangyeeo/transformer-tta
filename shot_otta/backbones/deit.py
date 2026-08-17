"""Strict local DeiT source-checkpoint adapter for TTA evaluation."""

import hashlib

import torch
import torch.nn as nn

from source_training.deit_model import load_deit_source_checkpoint

# Proposal-frozen candidate space: the weight tensors of the last 3 blocks.
# Bias, LayerNorm, class/position tokens, patch embed, final norm, and the
# head are deliberately outside the structural candidate denominator.
DEFAULT_CANDIDATE_BLOCKS = (9, 10, 11)
CANDIDATE_WEIGHT_COMPONENTS = (
    "attn.qkv.weight",
    "attn.proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
)


def hash_model_state(model):
    """Hash parameter and buffer values in a stable name/dtype/shape order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu_tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu_tensor.dtype).encode("ascii"))
        digest.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
        digest.update(cpu_tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_loaded_metadata(metadata, config):
    data = config["data"]
    expected = {
        "name": data["dataset"],
        "source_domain": data["source_name"],
        "num_classes": data["num_classes"],
    }
    dataset_record = metadata.get("dataset", {})
    for field, value in expected.items():
        if dataset_record.get(field) != value:
            raise ValueError(
                f"Loaded checkpoint metadata dataset.{field} mismatch"
            )
    model_record = metadata.get("model", {})
    if model_record.get("name") != config["model"]["name"]:
        raise ValueError("Loaded checkpoint model name mismatch")
    if int(model_record.get("num_classes", -1)) != data["num_classes"]:
        raise ValueError("Loaded checkpoint model class count mismatch")
    if model_record.get("head_schema") != config["model"]["head_schema"]:
        raise ValueError("Loaded checkpoint head schema mismatch")
    training_seed = metadata.get("training", {}).get("seed")
    if training_seed != config["source_checkpoint"]["source_training_seed"]:
        raise ValueError("Loaded checkpoint source-training seed mismatch")
    if metadata.get("best") != config["source_checkpoint"]["best"]:
        raise ValueError("Loaded checkpoint best metadata differs from manifest")
    if metadata.get("scientific_config_sha256") != config[
        "source_checkpoint"
    ]["scientific_config_sha256"]:
        raise ValueError(
            "Loaded checkpoint scientific config differs from manifest"
        )


def _check_direct_head(model, config):
    head = model.get_classifier()
    if not isinstance(head, nn.Linear):
        raise TypeError("DeiT TTA requires a direct Linear head")
    expected_classes = config["data"]["num_classes"]
    if head.in_features != 384 or head.out_features != expected_classes:
        raise ValueError(
            f"Unexpected DeiT head shape {head.in_features}->{head.out_features}"
        )
    return head


def candidate_parameter_names(candidate_blocks=DEFAULT_CANDIDATE_BLOCKS):
    """Return the exact weight-tensor names updated by candidate-dense."""
    names = set()
    for block in candidate_blocks:
        for component in CANDIDATE_WEIGHT_COMPONENTS:
            names.add(f"blocks.{int(block)}.{component}")
    return names


def apply_update_scope(model, update_scope, candidate_blocks=None):
    """Set ``requires_grad`` according to the frozen adaptation scope.

    Returns the exact set of trainable parameter names.
    """
    if update_scope == "all_parameters":
        model.requires_grad_(True)
        return {name for name, _ in model.named_parameters()}
    if update_scope == "last_3_block_weights":
        if candidate_blocks is None:
            candidate_blocks = DEFAULT_CANDIDATE_BLOCKS
        trainable = candidate_parameter_names(candidate_blocks)
        model.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if name in trainable:
                parameter.requires_grad_(True)
        return trainable
    raise ValueError(f"Unsupported update_scope: {update_scope}")


def load_frozen_deit_source(config, device, *, model_factory=None):
    """Load W0, validate its metadata/head, freeze it, and select eval mode."""
    checkpoint = config["source_checkpoint"]
    model, metadata = load_deit_source_checkpoint(
        checkpoint["path"],
        device=device,
        expected_sha256=checkpoint["sha256"],
        model_factory=model_factory,
    )
    _validate_loaded_metadata(metadata, config)
    _check_direct_head(model, config)
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Source-only model freezing failed")
    return model, metadata


def load_deit_source_for_adaptation(config, device, *, model_factory=None):
    """Load W0, validate its metadata/head, and enable scope-limited gradients.

    ``adaptation.update_scope`` selects either all parameters (full-dense) or
    exactly the candidate weight tensors of ``adaptation.candidate_blocks``
    (candidate-dense).  The model stays in eval mode so the frozen
    drop_rate/drop_path_rate of 0.0 keep the stream deterministic.
    """
    checkpoint = config["source_checkpoint"]
    model, metadata = load_deit_source_checkpoint(
        checkpoint["path"],
        device=device,
        expected_sha256=checkpoint["sha256"],
        model_factory=model_factory,
    )
    _validate_loaded_metadata(metadata, config)
    _check_direct_head(model, config)
    adaptation = config["adaptation"]
    expected_trainable = apply_update_scope(
        model,
        adaptation["update_scope"],
        candidate_blocks=adaptation.get("candidate_blocks"),
    )
    model.eval()
    actual_trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not actual_trainable:
        raise RuntimeError("Adaptation selected no trainable parameters")
    if actual_trainable != set(expected_trainable):
        raise RuntimeError(
            "Adaptation trainable set does not match the configured scope"
        )
    return model, metadata
