"""Candidate scope (12 tensors, 5,308,416 scalars), shared with SHOT.

The exact candidate universe comes from
``transformer.candidate_dense.model.configure_candidate_dense_scope``: same
names, shapes, order and parameter identity as the SHOT-Transformer track.
"""

from __future__ import annotations

from transformer.candidate_dense.model import (
    EXPECTED_SHAPES,
    configure_candidate_dense_scope,
    frozen_named_state,
    hash_tensors,
)
from transformer.source_only.model import load_frozen_source_model


def load_candidate_dense_model(config: dict, device):
    model, checkpoint_record = load_frozen_source_model(config, device)
    candidates, frozen, scope_record = configure_candidate_dense_scope(model)
    scope_record = {
        **scope_record,
        "candidate_scope": "last_three_blocks_qkv_proj_mlp_weights",
        "host_objective": "come",
    }
    return model, checkpoint_record, candidates, frozen, scope_record


__all__ = [
    "EXPECTED_SHAPES",
    "configure_candidate_dense_scope",
    "frozen_named_state",
    "hash_tensors",
    "load_candidate_dense_model",
]
