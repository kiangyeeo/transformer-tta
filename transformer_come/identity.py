"""Independent COME-Transformer protocol/implementation revisions and payload.

Protocol section 20: COME must carry its own Transformer protocol revision,
baseline implementation revision and sparse/LBI implementation revision.  It
must not reuse the SHOT-Transformer revisions or the ResNet/FC COME revisions
(``come_otta_baseline_20260908_v2`` / ``come_otta_sparse_lbi_20260908_v3``),
which describe a different substrate.
"""

from __future__ import annotations

from transformer_come.objective import (
    OFFICIAL_COME_COMMIT,
    OFFICIAL_ENTROPY_EPSILON,
    OFFICIAL_P,
    OFFICIAL_TAU,
)


PROTOCOL_DOCUMENT = "OTTA_COME_TRANSFORMER_LBI_PROTOCOL_20260909_v1"

BASELINE_IMPLEMENTATION_REVISION = "come_transformer_baseline_20260909_v1"
SPARSE_LBI_IMPLEMENTATION_REVISION = "come_transformer_sparse_lbi_20260909_v1"

PROTOCOL_REVISIONS = {
    "full_dense": "come_transformer_full_dense_otta_20260909_v1",
    "candidate_dense": "come_transformer_candidate_dense_otta_20260909_v1",
    "group_random": "come_transformer_group_random_otta_20260909_v1",
    "group_magnitude": "come_transformer_group_magnitude_otta_20260909_v1",
    "group_saliency": "come_transformer_group_saliency_otta_20260909_v1",
}

IMPLEMENTATION_REVISIONS = {
    "full_dense": BASELINE_IMPLEMENTATION_REVISION,
    "candidate_dense": BASELINE_IMPLEMENTATION_REVISION,
    "group_random": SPARSE_LBI_IMPLEMENTATION_REVISION,
    "group_magnitude": SPARSE_LBI_IMPLEMENTATION_REVISION,
    "group_saliency": SPARSE_LBI_IMPLEMENTATION_REVISION,
}

# The frozen ``come:`` config block.  Its own namespace keeps ``come.tau``
# (the logit-constraint parameter, 1.0) separate from the future
# ``lbi.tau_g`` (the normalized Gamma support threshold, 1e-4).
EXPECTED_COME_BLOCK = {
    "objective": "come_entropy_of_opinion",
    "official_come_commit": OFFICIAL_COME_COMMIT,
    "p": OFFICIAL_P,
    "tau": OFFICIAL_TAU,
    "entropy_epsilon": OFFICIAL_ENTROPY_EPSILON,
    "class_count_source": "dataset_classifier_output_dimension",
    "norm_epsilon": "none",
    "norm_clamp": "none",
    "stop_gradient_on_norm": True,
    "opinion_implementation": "stable_log_domain",
    "renormalize_after_epsilon": False,
    "tunable_hyperparameters": [],
}


def come_objective_payload(come_block: dict) -> dict:
    """COME-specific scientific payload recorded in every artifact."""

    return {
        "host_objective": "come",
        "come_official_commit": come_block["official_come_commit"],
        "come_p": float(come_block["p"]),
        "come_tau": float(come_block["tau"]),
        "come_entropy_epsilon": float(come_block["entropy_epsilon"]),
        "come_class_count_source": come_block["class_count_source"],
        "come_norm_epsilon": come_block["norm_epsilon"],
        "come_norm_clamp": come_block["norm_clamp"],
        "come_stop_gradient_on_norm": bool(come_block["stop_gradient_on_norm"]),
        "come_opinion_implementation": come_block["opinion_implementation"],
        "come_renormalize_after_epsilon": bool(
            come_block["renormalize_after_epsilon"]
        ),
        "come_objective_chunking": "none_single_full_batch_objective",
        "come_pseudo_label_or_teacher_or_memory": "none",
        "come_tunable_hyperparameters": list(come_block["tunable_hyperparameters"]),
    }


__all__ = [
    "BASELINE_IMPLEMENTATION_REVISION",
    "EXPECTED_COME_BLOCK",
    "IMPLEMENTATION_REVISIONS",
    "PROTOCOL_DOCUMENT",
    "PROTOCOL_REVISIONS",
    "SPARSE_LBI_IMPLEMENTATION_REVISION",
    "come_objective_payload",
]
