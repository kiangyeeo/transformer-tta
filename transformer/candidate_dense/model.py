"""Strictly expose only the DeiT-S candidate-dense parameter tensors."""

from __future__ import annotations

import hashlib

import torch.nn as nn

from transformer.source_only.model import load_frozen_source_model

from .config import (
    CANDIDATE_BLOCKS,
    CANDIDATE_SCALAR_COUNT,
    CANDIDATE_SUFFIXES,
    CANDIDATE_TENSOR_COUNT,
    candidate_parameter_names,
)


EXPECTED_SHAPES = {
    "attn.qkv.weight": (1152, 384),
    "attn.proj.weight": (384, 384),
    "mlp.fc1.weight": (1536, 384),
    "mlp.fc2.weight": (384, 1536),
}


def hash_tensors(named_tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def frozen_named_state(model: nn.Module, candidate_names: set[str]):
    for name, parameter in model.named_parameters():
        if name not in candidate_names:
            yield f"parameter:{name}", parameter
    for name, buffer in model.named_buffers():
        yield f"buffer:{name}", buffer


def configure_candidate_dense_scope(model: nn.Module):
    """Freeze the model, validate the exact DeiT-S scope, and expose candidates."""
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) != 12:
        raise ValueError("Candidate-dense requires a 12-block DeiT-S model")

    expected_names = set(candidate_parameter_names())
    named_parameters = dict(model.named_parameters())
    missing = sorted(expected_names - set(named_parameters))
    if missing:
        raise ValueError(f"Candidate parameter tensors are missing: {missing}")

    model.requires_grad_(False)
    trainable = []
    for name in candidate_parameter_names():
        parameter = named_parameters[name]
        suffix = name.split(".", maxsplit=2)[2]
        expected_shape = EXPECTED_SHAPES[suffix]
        if tuple(parameter.shape) != expected_shape:
            raise ValueError(
                f"Candidate tensor {name} must have shape {expected_shape}, "
                f"got {tuple(parameter.shape)}"
            )
        parameter.requires_grad_(True)
        trainable.append((name, parameter))

    actual_trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if actual_trainable != expected_names:
        raise RuntimeError(
            "Trainable parameter scope differs from the exact candidate set: "
            f"{sorted(actual_trainable ^ expected_names)}"
        )
    scalar_count = sum(parameter.numel() for _, parameter in trainable)
    if len(trainable) != CANDIDATE_TENSOR_COUNT:
        raise RuntimeError(
            f"Expected {CANDIDATE_TENSOR_COUNT} candidate tensors, got {len(trainable)}"
        )
    if scalar_count != CANDIDATE_SCALAR_COUNT:
        raise RuntimeError(
            f"Expected {CANDIDATE_SCALAR_COUNT} candidate scalars, got {scalar_count}"
        )

    frozen = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name not in expected_names
    ]
    model.eval()
    return trainable, frozen, {
        "candidate_blocks": list(CANDIDATE_BLOCKS),
        "candidate_parameter_suffixes": list(CANDIDATE_SUFFIXES),
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_tensor_count": len(trainable),
        "trainable_scalars": scalar_count,
        "frozen_parameter_names": [name for name, _ in frozen],
        "frozen_parameter_scalars": sum(parameter.numel() for _, parameter in frozen),
        "total_parameter_scalars": sum(parameter.numel() for parameter in model.parameters()),
    }


def load_candidate_dense_model(config: dict, device):
    model, checkpoint_record = load_frozen_source_model(config, device)
    trainable, frozen, scope_record = configure_candidate_dense_scope(model)
    return model, checkpoint_record, trainable, frozen, scope_record

