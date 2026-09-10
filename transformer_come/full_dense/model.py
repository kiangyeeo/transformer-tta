"""Source W0 loading and full-dense scope, shared with SHOT-Transformer.

COME must use the same source checkpoint, the same manifest/SHA-256
verification and the same ``all_except_head`` scope as the SHOT full-dense
baseline, so the loader is imported instead of re-implemented; the wrapper
only also returns the frozen head parameters so the shared dense runner can
audit them explicitly.
"""

from __future__ import annotations

from transformer.candidate_dense.model import frozen_named_state, hash_tensors
from transformer.full_dense.model import hash_head_state
from transformer.full_dense.model import load_full_dense_model as _load_full_dense_model


def load_full_dense_model(config: dict, device):
    model, checkpoint_record, trainable, scope_record = _load_full_dense_model(
        config, device
    )
    head_parameter_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
    frozen = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if id(parameter) in head_parameter_ids
    ]
    scope_record = {
        **scope_record,
        "update_scope": "all_except_head",
        "host_objective": "come",
    }
    return model, checkpoint_record, trainable, frozen, scope_record


__all__ = [
    "frozen_named_state",
    "hash_head_state",
    "hash_tensors",
    "load_full_dense_model",
]
