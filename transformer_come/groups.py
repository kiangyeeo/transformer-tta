"""Paired structural groups and the three support scorers, shared with SHOT.

The canonical 6912-group partition, the deterministic Random permutation per
seed, the source-W0 magnitude score and the |W * grad| saliency score are all
imported from the SHOT-Transformer substrate.  The scorers only read candidate
values and their ``.grad``, so they are objective-agnostic: the COME gradient
flows through the SHOT implementation unchanged and only the objective that
produced ``.grad`` differs.
"""

from transformer.group_magnitude.groups import (
    compute_group_l2_scores,
    magnitude_mask_record,
    score_vector_sha256,
    select_magnitude_group_ids,
)
from transformer.group_random.groups import (
    BLOCKS,
    GROUP_KINDS,
    HIDDEN_DIM,
    MLP_DIM,
    STRUCTURAL_GROUPS,
    TOTAL_GROUPS,
    StructuralGroup,
    build_masks,
    hash_masked_values,
    mask_record,
    mask_sha256,
    selected_group_ids,
    structural_groups,
)
from transformer.group_saliency.groups import (
    compute_group_saliency_scores,
    dynamic_mask_record,
    mask_history_sha256,
    select_saliency_group_ids,
)

__all__ = [
    "BLOCKS",
    "GROUP_KINDS",
    "HIDDEN_DIM",
    "MLP_DIM",
    "STRUCTURAL_GROUPS",
    "TOTAL_GROUPS",
    "StructuralGroup",
    "build_masks",
    "compute_group_l2_scores",
    "compute_group_saliency_scores",
    "dynamic_mask_record",
    "hash_masked_values",
    "magnitude_mask_record",
    "mask_history_sha256",
    "mask_record",
    "mask_sha256",
    "score_vector_sha256",
    "select_magnitude_group_ids",
    "select_saliency_group_ids",
    "selected_group_ids",
    "structural_groups",
]
