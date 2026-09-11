"""Source W0 loading and the five COME update scopes, shared with SHOT.

COME must use the same source checkpoint, the same manifest/SHA-256
verification and the same scopes as the matched SHOT-Transformer baselines, so
every loader is imported from ``transformer`` and only annotated here with the
host objective and the support policy that will be recorded in the artifacts.
"""

from __future__ import annotations

from transformer.candidate_dense.model import (
    EXPECTED_SHAPES,
    configure_candidate_dense_scope,
    frozen_named_state,
    hash_tensors,
)
from transformer.full_dense.model import hash_head_state
from transformer.full_dense.model import load_full_dense_model as _load_full_dense_model
from transformer.source_only.model import load_frozen_source_model

from .config import (
    CANDIDATE_DENSE,
    FULL_DENSE,
    GROUP_MAGNITUDE,
    GROUP_RANDOM,
    GROUP_SALIENCY,
)


# The support policy each sparse variant records in its scope block.
SELECTION_METHODS = {
    GROUP_RANDOM: "uniform_structural_group_random",
    GROUP_MAGNITUDE: "source_w0_structural_group_l2",
    GROUP_SALIENCY: "dynamic_abs_weight_times_gradient_group_l2",
}


def load_full_dense_model(config: dict, device):
    """Everything except the classifier head, as in SHOT full-dense.

    The wrapper only also returns the frozen head parameters, so the dense
    runner can audit them explicitly.
    """

    model, checkpoint_record, trainable, scope_record = _load_full_dense_model(
        config, device
    )
    head_parameter_ids = {
        id(parameter) for parameter in model.get_classifier().parameters()
    }
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


def _load_candidate_scope_model(config: dict, device, *, selection_method=None):
    """The exact 12-tensor, 5,308,416-scalar candidate universe."""

    model, checkpoint_record = load_frozen_source_model(config, device)
    candidates, frozen, scope_record = configure_candidate_dense_scope(model)
    scope_record = {
        **scope_record,
        "candidate_scope": "last_three_blocks_qkv_proj_mlp_weights",
        **({} if selection_method is None else {"selection_method": selection_method}),
        "host_objective": "come",
    }
    return model, checkpoint_record, candidates, frozen, scope_record


def load_candidate_dense_model(config: dict, device):
    return _load_candidate_scope_model(config, device)


def load_group_random_model(config: dict, device):
    return _load_candidate_scope_model(
        config, device, selection_method=SELECTION_METHODS[GROUP_RANDOM]
    )


def load_group_magnitude_model(config: dict, device):
    return _load_candidate_scope_model(
        config, device, selection_method=SELECTION_METHODS[GROUP_MAGNITUDE]
    )


def load_group_saliency_model(config: dict, device):
    return _load_candidate_scope_model(
        config, device, selection_method=SELECTION_METHODS[GROUP_SALIENCY]
    )


MODEL_LOADERS = {
    FULL_DENSE: load_full_dense_model,
    CANDIDATE_DENSE: load_candidate_dense_model,
    GROUP_RANDOM: load_group_random_model,
    GROUP_MAGNITUDE: load_group_magnitude_model,
    GROUP_SALIENCY: load_group_saliency_model,
}


__all__ = [
    "EXPECTED_SHAPES",
    "MODEL_LOADERS",
    "SELECTION_METHODS",
    "configure_candidate_dense_scope",
    "frozen_named_state",
    "hash_head_state",
    "hash_tensors",
    "load_candidate_dense_model",
    "load_frozen_source_model",
    "load_full_dense_model",
    "load_group_magnitude_model",
    "load_group_random_model",
    "load_group_saliency_model",
]
