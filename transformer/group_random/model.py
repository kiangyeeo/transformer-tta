"""Load DeiT-S and expose the exact controlled candidate tensors."""

from transformer.candidate_dense.model import (
    configure_candidate_dense_scope,
    frozen_named_state,
    hash_tensors,
)
from transformer.source_only.model import load_frozen_source_model


def load_group_random_model(config: dict, device):
    model, checkpoint_record = load_frozen_source_model(config, device)
    candidates, frozen, scope_record = configure_candidate_dense_scope(model)
    scope_record = {
        **scope_record,
        "candidate_scope": "last_three_blocks_qkv_proj_mlp_weights",
        "selection_method": "uniform_structural_group_random",
    }
    return model, checkpoint_record, candidates, frozen, scope_record


__all__ = [
    "frozen_named_state",
    "hash_tensors",
    "load_group_random_model",
]
