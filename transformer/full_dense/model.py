"""Load a verified source W0 and expose every DeiT parameter except its head."""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

from transformer.source_only.model import load_frozen_source_model


def hash_tensors(named_tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def hash_head_state(model: nn.Module) -> str:
    head = model.get_classifier()
    return hash_tensors(head.state_dict().items())


def load_full_dense_model(config: dict, device: torch.device):
    model, checkpoint_record = load_frozen_source_model(config, device)
    head = model.get_classifier()
    if not isinstance(head, nn.Linear):
        raise TypeError("Full-dense DeiT requires a direct Linear classifier head")
    head_parameter_ids = {id(parameter) for parameter in head.parameters()}
    model.requires_grad_(True)
    head.requires_grad_(False)
    model.eval()

    trainable = []
    frozen = []
    for name, parameter in model.named_parameters():
        if id(parameter) in head_parameter_ids:
            frozen.append((name, parameter))
            if parameter.requires_grad:
                raise RuntimeError(f"Classifier parameter unexpectedly trainable: {name}")
        else:
            trainable.append((name, parameter))
            if not parameter.requires_grad:
                raise RuntimeError(f"Non-head parameter unexpectedly frozen: {name}")
    if not trainable or not frozen:
        raise RuntimeError("Full-dense scope requires trainable body and frozen head parameters")
    return model, checkpoint_record, trainable, {
        "trainable_parameter_names": [name for name, _ in trainable],
        "frozen_head_parameter_names": [name for name, _ in frozen],
        "trainable_scalars": sum(parameter.numel() for _, parameter in trainable),
        "frozen_head_scalars": sum(parameter.numel() for _, parameter in frozen),
        "total_parameter_scalars": sum(parameter.numel() for parameter in model.parameters()),
    }
