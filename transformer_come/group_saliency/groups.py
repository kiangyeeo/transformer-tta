"""Paired groups and |W*grad| saliency scoring, shared with SHOT.

The scorer only reads the current candidate values and their ``.grad``; it is
objective-agnostic, so the COME gradient flows through the SHOT-Transformer
implementation unchanged.  Only the objective that produced ``.grad`` differs.
"""

from transformer.group_saliency.groups import (
    STRUCTURAL_GROUPS,
    StructuralGroup,
    build_masks,
    compute_group_saliency_scores,
    dynamic_mask_record,
    mask_history_sha256,
    select_saliency_group_ids,
)

__all__ = [
    "STRUCTURAL_GROUPS",
    "StructuralGroup",
    "build_masks",
    "compute_group_saliency_scores",
    "dynamic_mask_record",
    "mask_history_sha256",
    "select_saliency_group_ids",
]
