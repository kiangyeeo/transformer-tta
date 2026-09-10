"""Paired groups and source-W0 magnitude scoring, shared with SHOT.

The COME objective plays no role in Magnitude support construction, so the
scoring and the group/mask machinery are imported unchanged.
"""

from transformer.group_magnitude.groups import (
    STRUCTURAL_GROUPS,
    StructuralGroup,
    build_masks,
    compute_group_l2_scores,
    hash_masked_values,
    magnitude_mask_record,
    score_vector_sha256,
    select_magnitude_group_ids,
)

__all__ = [
    "STRUCTURAL_GROUPS",
    "StructuralGroup",
    "build_masks",
    "compute_group_l2_scores",
    "hash_masked_values",
    "magnitude_mask_record",
    "score_vector_sha256",
    "select_magnitude_group_ids",
]
