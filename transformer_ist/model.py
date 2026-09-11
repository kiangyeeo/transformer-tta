"""DeiT source, feature/logit adapter, and exact IST variant scopes."""

from __future__ import annotations

import torch
import torch.nn as nn

from transformer.candidate_dense.model import (
    configure_candidate_dense_scope,
    frozen_named_state,
    hash_tensors,
)
from transformer.source_only.model import hash_model_state, load_frozen_source_model

from .config import FULL_DENSE, SUPPORTED_VARIANTS


class DeiTISTAdapter:
    """Expose the classifier-input representation without changing the model."""

    feature_dim = 384

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        head = model.get_classifier()
        if not isinstance(head, nn.Linear):
            raise TypeError("IST requires the non-distilled direct Linear DeiT head")
        if head.in_features != self.feature_dim:
            raise ValueError("IST DeiT classifier input must be 384-dimensional")

    def features_and_logits(self, inputs):
        tokens = self.model.forward_features(inputs)
        features = self.model.forward_head(tokens, pre_logits=True)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"IST pre-logits features must be [N,384], got {tuple(features.shape)}"
            )
        logits = self.model.get_classifier()(features)
        return features, logits

    def logits(self, inputs):
        return self.features_and_logits(inputs)[1]


def validate_adapter_contract(model: nn.Module, device) -> dict:
    """Validate source logits on synthetic input without consuming target data/RNG."""
    state_before = hash_model_state(model)
    sample = torch.zeros((1, 3, 224, 224), device=device, dtype=torch.float32)
    with torch.inference_mode():
        adapter = DeiTISTAdapter(model)
        features, adapter_logits = adapter.features_and_logits(sample)
        direct_logits = model(sample)
    if not torch.equal(adapter_logits, direct_logits):
        if not torch.allclose(adapter_logits, direct_logits, rtol=1.0e-6, atol=1.0e-7):
            raise RuntimeError("IST adapter logits differ from model(x)")
    if hash_model_state(model) != state_before:
        raise RuntimeError("adapter contract check mutated source model state")
    return {
        "feature_point": "forward_head(forward_features(x),pre_logits=True)",
        "feature_shape": [None, int(features.shape[1])],
        "adapter_logits_match_model": True,
        "adapter_logits_max_abs_error": float(
            (adapter_logits - direct_logits).abs().max().item()
        ),
    }


def _configure_full_dense(model):
    head_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
    model.requires_grad_(True)
    model.get_classifier().requires_grad_(False)
    trainable, frozen = [], []
    for name, parameter in model.named_parameters():
        (frozen if id(parameter) in head_ids else trainable).append((name, parameter))
    if not trainable or not frozen:
        raise RuntimeError("full_dense requires a trainable body and frozen head")
    return trainable, frozen, {
        "update_scope": "all_except_head",
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_scalars": sum(parameter.numel() for _, parameter in trainable),
        "frozen_parameter_names": [name for name, _ in frozen],
    }


def load_ist_model(config: dict, device):
    variant = config["variant"]
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported non-LBI IST variant: {variant}")
    model, checkpoint = load_frozen_source_model(config, device)
    if variant == FULL_DENSE:
        trainable, frozen, scope = _configure_full_dense(model)
    else:
        trainable, frozen, scope = configure_candidate_dense_scope(model)
        scope["update_scope"] = "last_three_blocks_qkv_proj_mlp_weights"
    model.eval()
    if model.training:
        raise RuntimeError("IST Transformer must remain in eval mode")
    adapter_record = validate_adapter_contract(model, device)
    candidate_names = {name for name, _ in trainable}
    frozen_hash = hash_tensors(frozen_named_state(model, candidate_names))
    return (
        model,
        DeiTISTAdapter(model),
        checkpoint,
        trainable,
        frozen,
        {
            **scope,
            **adapter_record,
            "model_mode": "eval",
            "amp": False,
            "initial_frozen_state_sha256": frozen_hash,
        },
    )


__all__ = [
    "DeiTISTAdapter",
    "frozen_named_state",
    "hash_model_state",
    "hash_tensors",
    "load_ist_model",
    "validate_adapter_contract",
]
